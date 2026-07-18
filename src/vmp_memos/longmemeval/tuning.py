"""Deterministic dev-only tuning for the VMP retrieval ranker."""

from __future__ import annotations

import logging
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

from pydantic import Field, JsonValue

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.evaluation import aggregate_retrieval_metrics, compute_retrieval_metrics
from vmp_memos.frameworks.text import parse_date
from vmp_memos.frameworks.vmp_memos import VMPRuleAdapter
from vmp_memos.frameworks.vmp_tuned import (
    VMP_TUNED_FEATURES,
    VMPTunedModel,
    normalized_bm25_scores,
    question_has_temporal_intent,
    superseded_candidate_indices,
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
    # Physical memory growth is reported, but V3 never rewards destructive
    # deletion during query-time ranking.
    "memory_growth": 0.0,
    "stale_retrieval_rate": -0.10,
    "conflict_retrieval_rate": -0.10,
}

_WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "semantic_relevance": (0.0, 0.0),
    "importance": (0.0, 0.0),
    "scope_match": (0.0, 0.0),
    "confidence": (0.0, 0.0),
    "success_contribution": (0.0, 0.0),
    "recency": (0.00, 0.40),
    "contradiction": (-0.20, 0.10),
    "redundancy": (-0.15, 0.00),
    "token_cost": (-0.10, 0.00),
    "staleness": (-0.20, 0.00),
    "update_signal": (0.00, 0.50),
    "action_signal": (0.00, 0.20),
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
    question: NonEmptyStr
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


@dataclass(frozen=True)
class VMPTuningParameters:
    """One safe hybrid parameter set evaluated on Dev."""

    weights: dict[str, float]
    retrieve_threshold: float
    semantic_anchor_weight: float
    lexical_anchor_weight: float
    policy_adjustment_limit: float
    archive_score_penalty: float

    def as_payload(self) -> dict[str, object]:
        """Return deterministic JSON-compatible search provenance."""

        return {
            "weights": dict(self.weights),
            "retrieve_threshold": self.retrieve_threshold,
            "semantic_anchor_weight": self.semantic_anchor_weight,
            "lexical_anchor_weight": self.lexical_anchor_weight,
            "policy_adjustment_limit": self.policy_adjustment_limit,
            "archive_score_penalty": self.archive_score_penalty,
        }


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
    best_parameters: VMPTuningParameters | None = None
    best_metrics: dict[str, float] | None = None
    baseline_metrics: dict[str, float] | None = None
    best_objective: float | None = None
    best_key: tuple[float, float, float, float, str] | None = None
    search_started = perf_counter()
    for index, parameters in enumerate(trial_parameters):
        weights = parameters.weights
        threshold = parameters.retrieve_threshold
        metrics = evaluate_vmp_parameters(
            examples,
            weights=weights,
            retrieve_threshold=threshold,
            semantic_anchor_weight=parameters.semantic_anchor_weight,
            lexical_anchor_weight=parameters.lexical_anchor_weight,
            policy_adjustment_limit=parameters.policy_adjustment_limit,
            archive_score_penalty=parameters.archive_score_penalty,
            retrieval_depth=retrieval_depth,
            qa_top_k=qa_top_k,
            token_budget=token_budget,
            max_memory_count=max_memory_count,
        )
        objective = sum(
            DEFAULT_OBJECTIVE_WEIGHTS[name] * metrics[name]
            for name in DEFAULT_OBJECTIVE_WEIGHTS
        )
        if index == 0:
            baseline_metrics = dict(metrics)
        parameter_hash = sha256_json(parameters.as_payload())
        key = (
            metrics["recall_all@5"],
            objective,
            metrics["mrr"],
            -metrics["normalized_token_cost"],
            parameter_hash,
        )
        summaries.append(
            {
                "trial": index,
                "objective": objective,
                "retrieve_threshold": threshold,
                "semantic_anchor_weight": parameters.semantic_anchor_weight,
                "lexical_anchor_weight": parameters.lexical_anchor_weight,
                "policy_adjustment_limit": parameters.policy_adjustment_limit,
                "archive_score_penalty": parameters.archive_score_penalty,
                "parameter_sha256": parameter_hash,
                "metrics": cast(
                    JsonValue,
                    {name: float(value) for name, value in metrics.items()},
                ),
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_parameters = parameters
            best_metrics = metrics
            best_objective = objective
        completed = index + 1
        if completed == 1 or completed % 8 == 0 or completed == len(trial_parameters):
            LOGGER.info(
                "Parameter search %d/%d: objective=%.6f best=%.6f elapsed=%.1fs",
                completed,
                len(trial_parameters),
                objective,
                best_key[1],
                perf_counter() - search_started,
            )

    if (
        best_parameters is None
        or best_metrics is None
        or best_key is None
        or best_objective is None
        or baseline_metrics is None
    ):
        raise RuntimeError("VMP-Tuned search produced no model")
    best_weights = best_parameters.weights
    best_threshold = best_parameters.retrieve_threshold
    manifest_path = Path(split_manifest_path).expanduser().resolve()
    model = VMPTunedModel(
        weights=best_weights,
        retrieve_threshold=best_threshold,
        semantic_anchor_weight=best_parameters.semantic_anchor_weight,
        lexical_anchor_weight=best_parameters.lexical_anchor_weight,
        policy_adjustment_limit=best_parameters.policy_adjustment_limit,
        archive_score_penalty=best_parameters.archive_score_penalty,
        split_id=manifest.split_id,
        split_manifest_sha256=sha256_file(manifest_path),
        dataset_sha256=manifest.dataset_sha256,
        embedding_identifier=embedder.identifier if embedder else None,
        objective=DEFAULT_OBJECTIVE_WEIGHTS,
        best_objective=best_objective,
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
            "search": "seeded_bounded_policy_search_with_dense_safety_baseline",
            "feature_semantics_version": "3",
            "retrieval_objective_metric": "recall_all@5",
            "dense_safety_baseline_metrics": cast(JsonValue, baseline_metrics),
            "dev_recall_all_at_5_delta_vs_dense": (
                best_metrics["recall_all@5"]
                - float(baseline_metrics.get("recall_all@5", 0.0))
            ),
            "abstention_rule": "question_id_suffix_abs",
            "date_format": "longmemeval_timestamp_or_iso8601",
            "date_validation": date_validation,
            "test_labels_used": False,
            "ranking_pipeline": (
                "hybrid_candidate_generation -> bounded_policy_rerank -> "
                "non_destructive_lifecycle"
            ),
            "operation_policy": (
                "query-independent lifecycle annotations with bounded score penalties; "
                "source chunks are never deleted during retrieval"
            ),
            "memory_growth_note": (
                "Physical memory is preserved during retrieval. Compression and "
                "consolidation must be evaluated separately from ranking."
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
                        question=sample.question,
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
    semantic_anchor_weight: float = 0.80,
    lexical_anchor_weight: float = 0.20,
    policy_adjustment_limit: float = 0.10,
    archive_score_penalty: float = 0.03,
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
        retrieve_threshold=retrieve_threshold,
        semantic_anchor_weight=semantic_anchor_weight,
        lexical_anchor_weight=lexical_anchor_weight,
        policy_adjustment_limit=policy_adjustment_limit,
        archive_score_penalty=archive_score_penalty,
    )
    for example in examples:
        superseded = set(
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
        lexical_scores = normalized_bm25_scores(
            example.question,
            [candidate.content for candidate in example.candidates],
        )
        temporal_intent = question_has_temporal_intent(example.question)
        anchor_ranked = sorted(
            (
                (
                    search_model.anchor_score(
                        float(candidate.policy_features.semantic_relevance),
                        lexical_score,
                    ),
                    index,
                    candidate,
                )
                for index, (candidate, lexical_score) in enumerate(
                    zip(example.candidates, lexical_scores, strict=True)
                )
            ),
            key=lambda row: (-row[0], row[2].memory_id),
        )
        candidate_pool = anchor_ranked[
            : max(retrieval_depth, search_model.candidate_pool_size)
        ]
        ranked = sorted(
            (
                (
                    search_model.score(
                        candidate.policy_features,
                        anchor_score=anchor_score,
                        temporal_intent=temporal_intent,
                        lifecycle_status=(
                            "superseded" if index in superseded else "active"
                        ),
                    ),
                    anchor_score,
                    candidate,
                )
                for anchor_score, index, candidate in candidate_pool
            ),
            key=lambda row: (-row[0], -row[1], row[2].memory_id),
        )
        retrieved = [
            candidate
            for score, _, candidate in ranked[:retrieval_depth]
            if score >= retrieve_threshold
        ]
        retrieved_ids = [candidate.session_id for candidate in retrieved]
        metric_rows.append(
            compute_retrieval_metrics(retrieved_ids, example.gold_session_ids)
        )
        qa_evidence = retrieved[:qa_top_k]
        token_costs.append(
            min(1.0, sum(candidate.token_count for candidate in qa_evidence) / token_budget)
        )
        memory_growth.append(example.memory_count / max(1, max_memory_count))
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
) -> list[VMPTuningParameters]:
    parameters = [
        VMPTuningParameters(
            weights={name: 0.0 for name in VMP_TUNED_FEATURES},
            retrieve_threshold=0.0,
            semantic_anchor_weight=1.0,
            lexical_anchor_weight=0.0,
            policy_adjustment_limit=0.0,
            archive_score_penalty=0.0,
        )
    ]
    # Deterministic fusion sweep gives every run strong dense/BM25 anchors
    # before exploring policy deltas.
    for lexical_weight in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        if len(parameters) >= trials:
            return parameters
        parameters.append(
            VMPTuningParameters(
                weights={name: 0.0 for name in VMP_TUNED_FEATURES},
                retrieve_threshold=0.0,
                semantic_anchor_weight=1.0 - lexical_weight,
                lexical_anchor_weight=lexical_weight,
                policy_adjustment_limit=0.0,
                archive_score_penalty=0.0,
            )
        )
    rng = random.Random(seed)
    while len(parameters) < trials:
        weights = {
            name: rng.uniform(*_WEIGHT_BOUNDS[name])
            for name in VMP_TUNED_FEATURES
        }
        semantic_anchor_weight = rng.uniform(0.60, 1.0)
        parameters.append(
            VMPTuningParameters(
                weights=weights,
                retrieve_threshold=0.0,
                semantic_anchor_weight=semantic_anchor_weight,
                lexical_anchor_weight=1.0 - semantic_anchor_weight,
                policy_adjustment_limit=rng.uniform(0.0, 0.15),
                archive_score_penalty=rng.uniform(0.0, 0.05),
            )
        )
    return parameters


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
