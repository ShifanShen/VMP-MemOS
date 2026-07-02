"""Deterministic dev-only tuning for the VMP retrieval ranker."""

from __future__ import annotations

import logging
import random
import tempfile
from pathlib import Path
from time import perf_counter

from pydantic import Field, JsonValue

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.evaluation import aggregate_retrieval_metrics, compute_retrieval_metrics
from vmp_memos.frameworks.text import parse_date
from vmp_memos.frameworks.vmp_memos import VMPRuleAdapter
from vmp_memos.frameworks.vmp_tuned import (
    BASELINE_VMP_WEIGHTS,
    VMP_TUNED_FEATURES,
    VMPTunedModel,
    is_near_duplicate,
    superseded_candidate_indices,
    vmp_tuned_feature_values,
)
from vmp_memos.longmemeval.converter import sample_to_session_events
from vmp_memos.longmemeval.schema import LongMemEvalSample
from vmp_memos.longmemeval.splits import load_split_samples, sha256_file, sha256_json
from vmp_memos.longmemeval.validation import validate_longmemeval_dates
from vmp_memos.schemas import PolicyFeatures
from vmp_memos.schemas.base import NonEmptyStr, NonNegativeInt, SchemaModel

LOGGER = logging.getLogger(__name__)

DEFAULT_OBJECTIVE_WEIGHTS: dict[str, float] = {
    "recall_all@5": 1.0,
    "mrr": 0.5,
    "normalized_token_cost": -0.05,
    "memory_growth": -0.05,
    "stale_retrieval_rate": -0.10,
    "conflict_retrieval_rate": -0.10,
}

_WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "semantic_relevance": (0.10, 0.60),
    "importance": (0.00, 0.35),
    "scope_match": (0.00, 0.25),
    "confidence": (0.00, 0.20),
    "success_contribution": (0.00, 0.20),
    "recency": (0.00, 0.40),
    "contradiction": (-0.30, 0.15),
    "redundancy": (-0.20, 0.10),
    "token_cost": (-0.25, 0.00),
    "staleness": (-0.30, 0.00),
    "update_signal": (0.00, 0.50),
    "action_signal": (0.00, 0.25),
}


class VMPTuningCandidate(SchemaModel):
    """One session and its query-dependent VMP features."""

    memory_id: NonEmptyStr
    session_id: NonEmptyStr
    content: NonEmptyStr
    source_date: str | None = None
    token_count: NonNegativeInt
    policy_features: PolicyFeatures


class VMPTuningExample(SchemaModel):
    """One answerable dev question used by the optimizer."""

    question_id: NonEmptyStr
    question_type: NonEmptyStr
    gold_session_ids: list[NonEmptyStr]
    candidates: list[VMPTuningCandidate]
    memory_count: NonNegativeInt
    memory_tokens: NonNegativeInt


class VMPTuningResult(SchemaModel):
    """Frozen model plus an auditable search report."""

    model: VMPTunedModel
    trials_evaluated: NonNegativeInt
    candidate_examples: NonNegativeInt
    skipped_examples: NonNegativeInt
    trial_summaries: list[dict[str, JsonValue]] = Field(default_factory=list)


def train_vmp_tuned(
    data_path: str | Path,
    split_manifest_path: str | Path,
    *,
    embedder: BaseEmbedder | None,
    trials: int = 64,
    tuning_seed: int = 2025,
    retrieval_depth: int = 10,
    qa_top_k: int = 5,
    token_budget: int = 2048,
) -> VMPTuningResult:
    """Tune on manifest ``dev`` IDs and freeze the best deterministic trial."""

    if trials < 1:
        raise ValueError("trials must be at least 1")
    if retrieval_depth < qa_top_k:
        raise ValueError("retrieval_depth must be at least qa_top_k")
    if qa_top_k < 1 or token_budget < 1:
        raise ValueError("qa_top_k and token_budget must be positive")

    samples, manifest = load_split_samples(data_path, split_manifest_path, "dev")
    date_validation = validate_longmemeval_dates(samples)
    LOGGER.info("Loaded %d Dev samples; building reusable feature rows.", len(samples))
    examples, skipped = build_vmp_tuning_examples(
        samples,
        embedder=embedder,
        token_budget=token_budget,
    )
    if not examples:
        raise ValueError("dev split has no answerable examples with gold session IDs")

    trial_parameters = _trial_parameters(trials, tuning_seed)
    max_memory_count = max(example.memory_count for example in examples)
    summaries: list[dict[str, JsonValue]] = []
    best_parameters: tuple[dict[str, float], float] | None = None
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, float, float, float, str] | None = None
    search_started = perf_counter()
    for index, (weights, threshold) in enumerate(trial_parameters):
        metrics = evaluate_vmp_parameters(
            examples,
            weights=weights,
            retrieve_threshold=threshold,
            retrieval_depth=retrieval_depth,
            qa_top_k=qa_top_k,
            token_budget=token_budget,
            max_memory_count=max_memory_count,
        )
        objective = sum(
            DEFAULT_OBJECTIVE_WEIGHTS[name] * metrics[name]
            for name in DEFAULT_OBJECTIVE_WEIGHTS
        )
        parameter_hash = sha256_json({"weights": weights, "threshold": threshold})
        key = (
            objective,
            metrics["recall_all@5"],
            metrics["mrr"],
            -metrics["normalized_token_cost"],
            parameter_hash,
        )
        summaries.append(
            {
                "trial": index,
                "objective": objective,
                "retrieve_threshold": threshold,
                "parameter_sha256": parameter_hash,
                "metrics": {
                    name: float(value) for name, value in metrics.items()
                },
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_parameters = (weights, threshold)
            best_metrics = metrics
        completed = index + 1
        if completed == 1 or completed % 8 == 0 or completed == len(trial_parameters):
            LOGGER.info(
                "Parameter search %d/%d: objective=%.6f best=%.6f elapsed=%.1fs",
                completed,
                len(trial_parameters),
                objective,
                best_key[0],
                perf_counter() - search_started,
            )

    if best_parameters is None or best_metrics is None or best_key is None:
        raise RuntimeError("VMP-Tuned search produced no model")
    best_weights, best_threshold = best_parameters
    manifest_path = Path(split_manifest_path).expanduser().resolve()
    model = VMPTunedModel(
        weights=best_weights,
        retrieve_threshold=best_threshold,
        split_id=manifest.split_id,
        split_manifest_sha256=sha256_file(manifest_path),
        dataset_sha256=manifest.dataset_sha256,
        embedding_identifier=embedder.identifier if embedder else None,
        objective=DEFAULT_OBJECTIVE_WEIGHTS,
        best_objective=best_key[0],
        dev_metrics=best_metrics,
        metadata={
            "tuning_seed": tuning_seed,
            "trials": trials,
            "dev_question_count": len(samples),
            "answerable_dev_question_count": len(examples),
            "skipped_dev_question_count": skipped,
            "retrieval_depth": retrieval_depth,
            "qa_top_k": qa_top_k,
            "token_budget": token_budget,
            "search": "seeded_uniform_random_with_rule_baseline",
            "feature_semantics_version": "2",
            "retrieval_objective_metric": "recall_all@5",
            "abstention_rule": "question_id_suffix_abs",
            "date_format": "longmemeval_timestamp_or_iso8601",
            "date_validation": date_validation,
            "test_labels_used": False,
            "operation_policy": (
                "update-aware scoring, superseded-evidence archive, "
                "near-duplicate merge"
            ),
            "memory_growth_note": (
                "Per-question reset stores every input session; the normalized "
                "storage term is constant across ranking trials."
            ),
            "stale_conflict_proxy": (
                "For knowledge-update questions, older non-gold sessions are stale; "
                "those with contradiction >= 0.45 are conflicting."
            ),
        },
    )
    return VMPTuningResult(
        model=model,
        trials_evaluated=len(summaries),
        candidate_examples=len(examples),
        skipped_examples=skipped,
        trial_summaries=summaries,
    )


def build_vmp_tuning_examples(
    samples: list[LongMemEvalSample],
    *,
    embedder: BaseEmbedder | None,
    token_budget: int,
) -> tuple[list[VMPTuningExample], int]:
    """Precompute features once; no trial can inspect gold beyond scoring."""

    examples: list[VMPTuningExample] = []
    skipped = 0
    adapter = VMPRuleAdapter(embedder=embedder)
    build_started = perf_counter()
    total_samples = len(samples)
    try:
        with tempfile.TemporaryDirectory(prefix="vmp_tuning_") as workspace:
            workspace_root = Path(workspace)
            for sample_index, sample in enumerate(samples):
                completed = sample_index + 1
                if sample.is_abstention or not sample.answer_session_ids:
                    skipped += 1
                    LOGGER.info(
                        "Dev feature progress %d/%d: skipped question_id=%s",
                        completed,
                        total_samples,
                        sample.question_id,
                    )
                    continue
                sample_started = perf_counter()
                LOGGER.info(
                    "Dev feature progress %d/%d: embedding question_id=%s sessions=%d",
                    completed,
                    total_samples,
                    sample.question_id,
                    sample.session_count,
                )
                adapter.reset(workspace_root / f"sample_{sample_index:04d}")
                for events in sample_to_session_events(sample):
                    adapter.ingest_session(events)
                adapter.finalize_ingestion()
                rows = adapter.feature_rows(
                    sample.question,
                    question_date=sample.question_date,
                    metadata={
                        "question_id": sample.question_id,
                        "question_type": sample.question_type,
                        "token_budget": token_budget,
                    },
                )
                candidates = [
                    VMPTuningCandidate(
                        memory_id=chunk.memory_id,
                        session_id=chunk.source_session_id or chunk.memory_id,
                        content=chunk.content,
                        source_date=chunk.source_date,
                        token_count=chunk.token_count,
                        policy_features=features,
                    )
                    for chunk, features in rows
                ]
                examples.append(
                    VMPTuningExample(
                        question_id=sample.question_id,
                        question_type=sample.question_type,
                        gold_session_ids=list(sample.answer_session_ids),
                        candidates=candidates,
                        memory_count=adapter.memory_count,
                        memory_tokens=adapter.total_tokens,
                    )
                )
                LOGGER.info(
                    "Dev feature progress %d/%d: completed candidates=%d sample=%.1fs total=%.1fs",
                    completed,
                    total_samples,
                    len(candidates),
                    perf_counter() - sample_started,
                    perf_counter() - build_started,
                )
    finally:
        adapter.close()
    return examples, skipped


def evaluate_vmp_parameters(
    examples: list[VMPTuningExample],
    *,
    weights: dict[str, float],
    retrieve_threshold: float,
    retrieval_depth: int,
    qa_top_k: int,
    token_budget: int,
    max_memory_count: int,
) -> dict[str, float]:
    """Evaluate one parameter set against dev gold labels."""

    metric_rows: list[dict[str, float]] = []
    token_costs: list[float] = []
    memory_growth: list[float] = []
    stale_rates: list[float] = []
    conflict_rates: list[float] = []
    search_model = VMPTunedModel(
        weights=weights,
        split_id="dev_search",
        split_manifest_sha256="dev_search",
        dataset_sha256="dev_search",
        best_objective=0.0,
    )
    for example in examples:
        archived = set(
            superseded_candidate_indices(
                [
                    (
                        candidate.content,
                        candidate.source_date,
                        candidate.policy_features,
                    )
                    for candidate in example.candidates
                ],
                model=search_model,
            )
        )
        ranked = sorted(
            (
                (
                    _parameter_score(
                        vmp_tuned_feature_values(candidate.policy_features),
                        weights,
                    ),
                    candidate,
                )
                for index, candidate in enumerate(example.candidates)
                if index not in archived
            ),
            key=lambda pair: (-pair[0], pair[1].memory_id),
        )
        retrieved: list[VMPTuningCandidate] = []
        active_contents: list[str] = []
        active_count = 0
        for score, candidate in ranked:
            if is_near_duplicate(
                candidate.content,
                active_contents,
                threshold=search_model.merge_similarity_threshold,
            ):
                continue
            active_count += 1
            active_contents.append(candidate.content)
            if score >= retrieve_threshold and len(retrieved) < retrieval_depth:
                retrieved.append(candidate)
        retrieved_ids = [candidate.session_id for candidate in retrieved]
        metric_rows.append(
            compute_retrieval_metrics(retrieved_ids, example.gold_session_ids)
        )
        qa_evidence = retrieved[:qa_top_k]
        token_costs.append(
            min(1.0, sum(candidate.token_count for candidate in qa_evidence) / token_budget)
        )
        memory_growth.append(active_count / max(1, max_memory_count))
        stale_rate, conflict_rate = _update_error_rates(example, qa_evidence)
        stale_rates.append(stale_rate)
        conflict_rates.append(conflict_rate)

    retrieval_metrics = aggregate_retrieval_metrics(metric_rows)
    return {
        "recall_all@5": retrieval_metrics["recall_all@5"],
        "mrr": retrieval_metrics["mrr"],
        "normalized_token_cost": _mean(token_costs),
        "memory_growth": _mean(memory_growth),
        "stale_retrieval_rate": _mean(stale_rates),
        "conflict_retrieval_rate": _mean(conflict_rates),
    }


def _trial_parameters(
    trials: int,
    seed: int,
) -> list[tuple[dict[str, float], float]]:
    parameters = [(dict(BASELINE_VMP_WEIGHTS), 0.05)]
    rng = random.Random(seed)
    for _ in range(trials - 1):
        weights = {
            name: rng.uniform(*_WEIGHT_BOUNDS[name])
            for name in VMP_TUNED_FEATURES
        }
        parameters.append((weights, rng.uniform(0.0, 0.20)))
    return parameters


def _parameter_score(features: dict[str, float], weights: dict[str, float]) -> float:
    raw_score = sum(weights[name] * features[name] for name in VMP_TUNED_FEATURES)
    return min(1.0, max(0.0, raw_score))


def _update_error_rates(
    example: VMPTuningExample,
    retrieved: list[VMPTuningCandidate],
) -> tuple[float, float]:
    if "update" not in example.question_type.casefold() or not retrieved:
        return 0.0, 0.0
    gold_dates = [
        parsed
        for candidate in example.candidates
        if candidate.session_id in example.gold_session_ids
        if (parsed := parse_date(candidate.source_date)) is not None
    ]
    newest_gold_date = max(gold_dates, default=None)
    stale = [
        candidate
        for candidate in retrieved
        if candidate.session_id not in example.gold_session_ids
        and newest_gold_date is not None
        and (candidate_date := parse_date(candidate.source_date)) is not None
        and candidate_date < newest_gold_date
    ]
    conflict = [
        candidate
        for candidate in stale
        if candidate.policy_features.contradiction >= 0.45
    ]
    denominator = len(retrieved)
    return len(stale) / denominator, len(conflict) / denominator


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
