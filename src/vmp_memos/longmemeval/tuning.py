"""Deterministic dev-only tuning for the VMP retrieval ranker."""

from __future__ import annotations

import logging
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

from pydantic import Field, JsonValue

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.evaluation import aggregate_retrieval_metrics, compute_retrieval_metrics
from vmp_memos.frameworks.text import clamp01, parse_date
from vmp_memos.frameworks.vmp_memos import VMPRuleAdapter
from vmp_memos.frameworks.vmp_tuned import (
    VMP_TUNED_FEATURES,
    VMPTunedModel,
    build_static_vmp_features,
    build_vmp_feature_rows,
    guarded_ranked_indices,
    normalized_bm25_scores,
    question_has_temporal_intent,
    superseded_candidate_indices,
    vmp_tuned_feature_values,
)
from vmp_memos.longmemeval.converter import sample_to_session_events
from vmp_memos.longmemeval.schema import LongMemEvalSample
from vmp_memos.longmemeval.splits import load_split_samples, sha256_file, sha256_json
from vmp_memos.longmemeval.validation import validate_longmemeval_dates
from vmp_memos.schemas import PolicyFeatures
from vmp_memos.schemas.base import NonEmptyStr, NonNegativeInt, SchemaModel, Score

LOGGER = logging.getLogger(__name__)

DEFAULT_OBJECTIVE_WEIGHTS: dict[str, float] = {
    "recall_all@5": 0.80,
    "macro_type_recall_all@5": 0.35,
    "worst_type_recall_all@5": 0.15,
    "mrr": 0.40,
    "fold_recall_stddev": -0.20,
    "normalized_token_cost": -0.05,
    # Physical memory growth is reported, but V4 never rewards destructive
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
    lexical_score: Score = 0.0
    lifecycle_status: NonEmptyStr = "active"
    policy_values: dict[NonEmptyStr, Score] = Field(default_factory=dict)


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
    protected_dense_count: int
    promotion_margin: float

    def as_payload(self) -> dict[str, object]:
        """Return deterministic JSON-compatible search provenance."""

        return {
            "weights": dict(self.weights),
            "retrieve_threshold": self.retrieve_threshold,
            "semantic_anchor_weight": self.semantic_anchor_weight,
            "lexical_anchor_weight": self.lexical_anchor_weight,
            "policy_adjustment_limit": self.policy_adjustment_limit,
            "archive_score_penalty": self.archive_score_penalty,
            "protected_dense_count": self.protected_dense_count,
            "promotion_margin": self.promotion_margin,
        }


TrialSelectionKey = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    str,
]


def vmp_trial_selection_key(
    metrics: dict[str, float],
    *,
    baseline_metrics: dict[str, float],
    objective: float,
    policy_adjustment_limit: float,
    parameter_hash: str,
    min_required_recall_all_at_5: float,
    min_required_delta_vs_dense: float = 0.02,
    min_required_macro_delta_vs_dense: float = 0.0,
    min_required_worst_type_delta_vs_dense: float = -0.03,
    max_allowed_fold_recall_stddev: float = 0.20,
) -> tuple[TrialSelectionKey, bool, bool]:
    """Rank robust trials without preferring a gate-failing absolute recall.

    Robust non-regression remains the first constraint. Within that feasible
    region, a trial that clears the configured absolute Dev gate always outranks
    one that does not. Fold stability then breaks ties between trials with the
    same discrete Recall-All@5 rather than silently overriding the gate.
    """

    baseline_recall = float(baseline_metrics.get("recall_all@5", 0.0))
    baseline_macro = float(
        baseline_metrics.get("macro_type_recall_all@5", 0.0)
    )
    baseline_worst = float(
        baseline_metrics.get("worst_type_recall_all@5", 0.0)
    )
    robust_non_regression = (
        metrics["recall_all@5"] >= baseline_recall
        and metrics["macro_type_recall_all@5"] >= baseline_macro
        and metrics["worst_type_recall_all@5"] >= baseline_worst - 0.03
    )
    clears_quality_gate = (
        metrics["recall_all@5"] >= min_required_recall_all_at_5
        and metrics["recall_all@5"] - baseline_recall
        >= min_required_delta_vs_dense
        and metrics["macro_type_recall_all@5"] - baseline_macro
        >= min_required_macro_delta_vs_dense
        and metrics["worst_type_recall_all@5"] - baseline_worst
        >= min_required_worst_type_delta_vs_dense
        and metrics.get("fold_recall_stddev", 0.0)
        <= max_allowed_fold_recall_stddev
    )
    key: TrialSelectionKey = (
        float(clears_quality_gate),
        float(robust_non_regression),
        metrics["recall_all@5"],
        metrics["macro_type_recall_all@5"],
        metrics["min_fold_recall_all@5"],
        -metrics.get("fold_recall_stddev", 0.0),
        objective,
        metrics["mrr"],
        -policy_adjustment_limit,
        parameter_hash,
    )
    return key, robust_non_regression, clears_quality_gate


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
    stability_folds: int = 5,
    min_required_recall_all_at_5: float = 0.90,
    min_required_delta_vs_dense: float = 0.02,
    min_required_macro_delta_vs_dense: float = 0.0,
    min_required_worst_type_delta_vs_dense: float = -0.03,
    max_allowed_fold_recall_stddev: float = 0.20,
) -> VMPTuningResult:
    """Tune on manifest ``dev`` IDs and freeze the best deterministic trial."""

    if trials < 1:
        raise ValueError("trials must be at least 1")
    if retrieval_depth < qa_top_k:
        raise ValueError("retrieval_depth must be at least qa_top_k")
    if qa_top_k < 1 or token_budget < 1 or stability_folds < 2:
        raise ValueError("qa_top_k and token_budget must be positive")
    if not 0.0 <= min_required_recall_all_at_5 <= 1.0:
        raise ValueError("min_required_recall_all_at_5 must be in [0, 1]")

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
    oracle_metrics = dense_guard_oracle_metrics(
        examples,
        safety_top_k=qa_top_k,
        preserve_dense_top_n=retrieval_depth,
        protected_dense_count=max(1, qa_top_k - 1),
    )
    LOGGER.info(
        "Dense guard oracle: dense@5=%.6f dense@10=%.6f guarded@5=%.6f required=%.6f",
        oracle_metrics["dense_recall_all@5"],
        oracle_metrics["dense_recall_all@10"],
        oracle_metrics["guarded_recall_all@5_ceiling"],
        min_required_recall_all_at_5,
    )

    trial_parameters = _trial_parameters(trials, tuning_seed)
    max_memory_count = max(example.memory_count for example in examples)
    summaries: list[dict[str, JsonValue]] = []
    best_parameters: VMPTuningParameters | None = None
    best_metrics: dict[str, float] | None = None
    baseline_metrics: dict[str, float] | None = None
    best_objective: float | None = None
    best_key: TrialSelectionKey | None = None
    selected_trial: int | None = None
    max_recall_key: tuple[float, float, float, str] | None = None
    max_recall_trial: int | None = None
    max_recall_metrics: dict[str, float] | None = None
    max_recall_objective: float | None = None
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
            protected_dense_count=parameters.protected_dense_count,
            promotion_margin=parameters.promotion_margin,
            retrieval_depth=retrieval_depth,
            qa_top_k=qa_top_k,
            token_budget=token_budget,
            max_memory_count=max_memory_count,
            stability_folds=stability_folds,
        )
        objective = sum(
            DEFAULT_OBJECTIVE_WEIGHTS[name] * metrics[name]
            for name in DEFAULT_OBJECTIVE_WEIGHTS
        )
        if index == 0:
            baseline_metrics = dict(metrics)
        parameter_hash = sha256_json(parameters.as_payload())
        key, robust_non_regression, clears_quality_gate = vmp_trial_selection_key(
            metrics,
            baseline_metrics=baseline_metrics or metrics,
            objective=objective,
            policy_adjustment_limit=parameters.policy_adjustment_limit,
            parameter_hash=parameter_hash,
            min_required_recall_all_at_5=min_required_recall_all_at_5,
            min_required_delta_vs_dense=min_required_delta_vs_dense,
            min_required_macro_delta_vs_dense=(
                min_required_macro_delta_vs_dense
            ),
            min_required_worst_type_delta_vs_dense=(
                min_required_worst_type_delta_vs_dense
            ),
            max_allowed_fold_recall_stddev=max_allowed_fold_recall_stddev,
        )
        recall_key = (
            metrics["recall_all@5"],
            objective,
            metrics["mrr"],
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
                "protected_dense_count": parameters.protected_dense_count,
                "promotion_margin": parameters.promotion_margin,
                "robust_non_regression": robust_non_regression,
                "clears_quality_gate": clears_quality_gate,
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
            selected_trial = index
        if max_recall_key is None or recall_key > max_recall_key:
            max_recall_key = recall_key
            max_recall_trial = index
            max_recall_metrics = dict(metrics)
            max_recall_objective = objective
        completed = index + 1
        if completed == 1 or completed % 8 == 0 or completed == len(trial_parameters):
            if (
                best_objective is None
                or best_metrics is None
                or max_recall_metrics is None
            ):
                raise RuntimeError("trial diagnostics were not initialized")
            LOGGER.info(
                "Parameter search %d/%d: objective=%.6f selected=%.6f "
                "selected_recall=%.6f max_recall=%.6f elapsed=%.1fs",
                completed,
                len(trial_parameters),
                objective,
                best_objective,
                best_metrics["recall_all@5"],
                max_recall_metrics["recall_all@5"],
                perf_counter() - search_started,
            )

    if (
        best_parameters is None
        or best_metrics is None
        or best_key is None
        or best_objective is None
        or baseline_metrics is None
        or selected_trial is None
        or max_recall_trial is None
        or max_recall_metrics is None
        or max_recall_objective is None
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
        protected_dense_count=best_parameters.protected_dense_count,
        promotion_margin=best_parameters.promotion_margin,
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
            "stability_folds": stability_folds,
            "min_required_recall_all_at_5": min_required_recall_all_at_5,
            "min_required_delta_vs_dense": min_required_delta_vs_dense,
            "min_required_macro_delta_vs_dense": (
                min_required_macro_delta_vs_dense
            ),
            "min_required_worst_type_delta_vs_dense": (
                min_required_worst_type_delta_vs_dense
            ),
            "max_allowed_fold_recall_stddev": max_allowed_fold_recall_stddev,
            "search": "seeded_guarded_policy_search_with_fold_and_group_stability",
            "feature_semantics_version": "4",
            "retrieval_objective_metric": "recall_all@5",
            "dense_safety_baseline_metrics": cast(JsonValue, baseline_metrics),
            "dev_oracle_ceiling_metrics": cast(JsonValue, oracle_metrics),
            "selected_trial": selected_trial,
            "max_recall_trial": max_recall_trial,
            "max_dev_recall_all_at_5_seen": max_recall_metrics["recall_all@5"],
            "max_recall_trial_objective": max_recall_objective,
            "max_recall_trial_metrics": cast(JsonValue, max_recall_metrics),
            "dev_recall_all_at_5_delta_vs_dense": (
                best_metrics["recall_all@5"]
                - float(baseline_metrics.get("recall_all@5", 0.0))
            ),
            "abstention_rule": "question_id_suffix_abs",
            "date_format": "longmemeval_timestamp_or_iso8601",
            "date_validation": date_validation,
            "test_labels_used": False,
            "ranking_pipeline": (
                "dense_top10_safety_set -> guarded_top5_policy_rerank -> "
                "cached_non_destructive_lifecycle"
            ),
            "dense_safety_guarantee": (
                "Returned Top-10 preserves the dense Top-10 set; Top-5 retains at "
                "least protected_dense_count dense-head items."
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
                query_embedding = (
                    embedder.embed_one(sample.question) if embedder is not None else None
                )
                static_features = build_static_vmp_features(adapter.chunks)
                rows = build_vmp_feature_rows(
                    adapter.chunks,
                    sample.question,
                    query_embedding=query_embedding,
                    question_date=sample.question_date,
                    metadata={
                        "question_id": sample.question_id,
                        "question_type": sample.question_type,
                        "token_budget": token_budget,
                    },
                    static_features=static_features,
                )
                lexical_scores = normalized_bm25_scores(
                    sample.question,
                    [chunk.content for chunk, _ in rows],
                )
                lifecycle_model = VMPTunedModel(
                    weights={name: 0.0 for name in VMP_TUNED_FEATURES},
                    split_id="dev_features",
                    split_manifest_sha256="dev_features",
                    dataset_sha256="dev_features",
                    best_objective=0.0,
                )
                superseded = set(
                    superseded_candidate_indices(
                        [
                            (chunk.content, chunk.source_date, features)
                            for chunk, features in rows
                        ],
                        model=lifecycle_model,
                    )
                )
                temporal_intent = question_has_temporal_intent(sample.question)
                candidates = [
                    VMPTuningCandidate(
                        memory_id=chunk.memory_id,
                        session_id=chunk.source_session_id or chunk.memory_id,
                        content=chunk.content,
                        source_date=chunk.source_date,
                        token_count=chunk.token_count,
                        policy_features=features,
                        lexical_score=lexical_score,
                        lifecycle_status=(
                            "superseded" if index in superseded else "active"
                        ),
                        policy_values=vmp_tuned_feature_values(
                            features,
                            temporal_intent=temporal_intent,
                        ),
                    )
                    for index, ((chunk, features), lexical_score) in enumerate(
                        zip(rows, lexical_scores, strict=True)
                    )
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


def dense_guard_oracle_metrics(
    examples: list[VMPTuningExample],
    *,
    safety_top_k: int,
    preserve_dense_top_n: int,
    protected_dense_count: int,
) -> dict[str, float]:
    """Report label-aware Dev ceilings imposed by the dense guard structure.

    This diagnostic never participates in candidate scoring. It answers whether
    the configured guard can theoretically reach the absolute Dev gate when an
    oracle chooses the open Top-5 slots from the preserved Dense Top-10 set.
    """

    if safety_top_k < 1 or preserve_dense_top_n < safety_top_k:
        raise ValueError("invalid dense guard sizes")
    if not 0 <= protected_dense_count <= safety_top_k:
        raise ValueError("protected_dense_count must be within safety_top_k")
    if not examples:
        return {
            "dense_recall_all@5": 0.0,
            "dense_recall_all@10": 0.0,
            "guarded_recall_all@5_ceiling": 0.0,
        }

    dense_head_successes = 0
    dense_set_successes = 0
    guarded_successes = 0
    open_slots = safety_top_k - protected_dense_count
    for example in examples:
        dense_ranked = sorted(
            example.candidates,
            key=lambda candidate: (
                -float(candidate.policy_features.semantic_relevance),
                candidate.memory_id,
            ),
        )
        gold = set(example.gold_session_ids)
        dense_head = {
            candidate.session_id for candidate in dense_ranked[:safety_top_k]
        }
        preserved = {
            candidate.session_id
            for candidate in dense_ranked[:preserve_dense_top_n]
        }
        protected = {
            candidate.session_id
            for candidate in dense_ranked[:protected_dense_count]
        }
        dense_head_successes += int(gold.issubset(dense_head))
        dense_set_successes += int(gold.issubset(preserved))
        remaining_gold = gold - protected
        guarded_successes += int(
            remaining_gold.issubset(preserved)
            and len(remaining_gold) <= open_slots
        )

    denominator = len(examples)
    return {
        "dense_recall_all@5": dense_head_successes / denominator,
        "dense_recall_all@10": dense_set_successes / denominator,
        "guarded_recall_all@5_ceiling": guarded_successes / denominator,
    }


def evaluate_vmp_parameters(
    examples: list[VMPTuningExample],
    *,
    weights: dict[str, float],
    retrieve_threshold: float,
    semantic_anchor_weight: float = 0.80,
    lexical_anchor_weight: float = 0.20,
    policy_adjustment_limit: float = 0.06,
    archive_score_penalty: float = 0.02,
    protected_dense_count: int = 4,
    promotion_margin: float = 0.02,
    retrieval_depth: int,
    qa_top_k: int,
    token_budget: int,
    max_memory_count: int,
    stability_folds: int = 5,
) -> dict[str, float]:
    """Evaluate one parameter set with dense guards and stability slices."""

    metric_rows: list[dict[str, float]] = []
    token_costs: list[float] = []
    memory_growth: list[float] = []
    stale_rates: list[float] = []
    conflict_rates: list[float] = []
    retained_dense_head: list[float] = []
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
        protected_dense_count=protected_dense_count,
        promotion_margin=promotion_margin,
    )
    for example in examples:
        temporal_intent = question_has_temporal_intent(example.question)
        dense_ranked = sorted(
            range(len(example.candidates)),
            key=lambda candidate_index: (
                -float(
                    example.candidates[
                        candidate_index
                    ].policy_features.semantic_relevance
                ),
                example.candidates[candidate_index].memory_id,
            ),
        )
        anchor_ranked = sorted(
            (
                (
                    search_model.anchor_score(
                        float(candidate.policy_features.semantic_relevance),
                        float(candidate.lexical_score),
                    ),
                    index,
                )
                for index, candidate in enumerate(example.candidates)
            ),
            key=lambda row: (-row[0], example.candidates[row[1]].memory_id),
        )
        pool_size = max(
            retrieval_depth,
            search_model.candidate_pool_size,
            search_model.preserve_dense_top_n,
        )
        pool_indices = list(
            dict.fromkeys(
                [*dense_ranked[:pool_size], *[row[1] for row in anchor_ranked[:pool_size]]]
            )
        )
        anchor_scores = {index: score for score, index in anchor_ranked}
        policy_scores: dict[int, float] = {}
        for index in pool_indices:
            candidate = example.candidates[index]
            values = candidate.policy_values or vmp_tuned_feature_values(
                candidate.policy_features,
                temporal_intent=temporal_intent,
            )
            raw_policy_score = sum(
                float(weights[name]) * float(values[name])
                for name in VMP_TUNED_FEATURES
            )
            policy_delta = policy_adjustment_limit * math.tanh(raw_policy_score)
            policy_scores[index] = clamp01(
                anchor_scores[index]
                + policy_delta
                - search_model.lifecycle_penalty(candidate.lifecycle_status)
            )
        selected_indices = guarded_ranked_indices(
            dense_ranked_indices=dense_ranked,
            policy_scores=policy_scores,
            anchor_scores=anchor_scores,
            requested_top_k=retrieval_depth,
            model=search_model,
        )
        retrieved = [
            example.candidates[index]
            for index in selected_indices
            if policy_scores[index] >= retrieve_threshold
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
        dense_head = set(dense_ranked[:qa_top_k])
        selected_head = set(selected_indices[:qa_top_k])
        retained_dense_head.append(
            len(dense_head & selected_head) / max(1, len(dense_head))
        )

    retrieval_metrics = aggregate_retrieval_metrics(metric_rows)
    robust_metrics = _stability_metrics(
        examples,
        metric_rows,
        folds=stability_folds,
    )
    return {
        "recall_all@5": retrieval_metrics["recall_all@5"],
        "mrr": retrieval_metrics["mrr"],
        **robust_metrics,
        "dense_head_retention@5": _mean(retained_dense_head),
        "normalized_token_cost": _mean(token_costs),
        "memory_growth": _mean(memory_growth),
        "stale_retrieval_rate": _mean(stale_rates),
        "conflict_retrieval_rate": _mean(conflict_rates),
    }


def _stability_metrics(
    examples: list[VMPTuningExample],
    metric_rows: list[dict[str, float]],
    *,
    folds: int,
) -> dict[str, float]:
    by_type: dict[str, list[dict[str, float]]] = {}
    by_fold: dict[int, list[dict[str, float]]] = {}
    for example, row in zip(examples, metric_rows, strict=True):
        by_type.setdefault(example.question_type, []).append(row)
        fold = int(sha256_json(example.question_id)[:8], 16) % folds
        by_fold.setdefault(fold, []).append(row)
    type_recalls = [
        aggregate_retrieval_metrics(rows)["recall_all@5"]
        for rows in by_type.values()
    ]
    fold_recalls = [
        aggregate_retrieval_metrics(rows)["recall_all@5"]
        for _, rows in sorted(by_fold.items())
    ]
    fold_mean = _mean(fold_recalls)
    fold_variance = _mean([(value - fold_mean) ** 2 for value in fold_recalls])
    return {
        "macro_type_recall_all@5": _mean(type_recalls),
        "worst_type_recall_all@5": min(type_recalls, default=0.0),
        "min_fold_recall_all@5": min(fold_recalls, default=0.0),
        "fold_recall_stddev": math.sqrt(fold_variance),
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
            protected_dense_count=5,
            promotion_margin=0.0,
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
                protected_dense_count=5,
                promotion_margin=0.0,
            )
        )
    rng = random.Random(seed)
    while len(parameters) < trials:
        weights = {
            name: rng.uniform(*_WEIGHT_BOUNDS[name])
            for name in VMP_TUNED_FEATURES
        }
        semantic_anchor_weight = rng.uniform(0.70, 1.0)
        parameters.append(
            VMPTuningParameters(
                weights=weights,
                retrieve_threshold=0.0,
                semantic_anchor_weight=semantic_anchor_weight,
                lexical_anchor_weight=1.0 - semantic_anchor_weight,
                # Dense Top-10 and four Top-5 items remain structurally guarded,
                # so V4 can recover the policy range that V3 needed for >=0.90
                # Dev recall without permitting destructive multi-item churn.
                policy_adjustment_limit=rng.uniform(0.0, 0.15),
                archive_score_penalty=rng.uniform(0.0, 0.02),
                protected_dense_count=rng.choice((4, 5)),
                promotion_margin=rng.uniform(0.01, 0.06),
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
