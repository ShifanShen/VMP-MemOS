"""Dev-only tuning for VMP-v5 hierarchical session/turn fusion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

from pydantic import Field, JsonValue

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks.text import dense_cosine, sparse_cosine, term_counts
from vmp_memos.frameworks.vmp_hierarchical import (
    VMPHierarchicalModel,
    contextual_turn_chunk,
    hierarchical_fusion_score,
    pool_top_scores,
)
from vmp_memos.frameworks.vmp_tuned import VMPTunedModel, normalized_bm25_scores
from vmp_memos.longmemeval.converter import sample_to_events
from vmp_memos.longmemeval.schema import LongMemEvalSample
from vmp_memos.longmemeval.splits import (
    load_split_samples,
    sha256_file,
    sha256_json,
    split_assignment_sha256,
)
from vmp_memos.longmemeval.tuning import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    VMPTuningExample,
    build_vmp_tuning_examples,
    dense_guard_oracle_metrics,
    evaluate_vmp_parameters,
)
from vmp_memos.longmemeval.validation import validate_longmemeval_dates
from vmp_memos.schemas.base import NonNegativeInt, SchemaModel

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HierarchicalFusionParameters:
    """One deterministic point on the three-signal fusion simplex."""

    session_semantic_weight: float
    turn_semantic_weight: float
    turn_lexical_weight: float
    turn_pooling_top_n: int

    def as_payload(self) -> dict[str, float | int]:
        return {
            "session_semantic_weight": self.session_semantic_weight,
            "turn_semantic_weight": self.turn_semantic_weight,
            "turn_lexical_weight": self.turn_lexical_weight,
            "turn_pooling_top_n": self.turn_pooling_top_n,
        }


@dataclass(frozen=True)
class HierarchicalTuningExample:
    """Base VMP features plus query-to-turn scores for one Dev question."""

    base: VMPTuningExample
    turn_semantic_scores: tuple[tuple[float, ...], ...]
    turn_lexical_scores: tuple[tuple[float, ...], ...]
    turn_count: int
    turn_tokens: int


class VMPHierarchicalTuningResult(SchemaModel):
    """Frozen V5 model plus the complete deterministic grid report."""

    model: VMPHierarchicalModel
    trials_evaluated: NonNegativeInt
    candidate_examples: NonNegativeInt
    skipped_examples: NonNegativeInt
    trial_summaries: list[dict[str, JsonValue]] = Field(default_factory=list)


def train_vmp_hierarchical(
    data_path: str | Path,
    split_manifest_path: str | Path,
    base_model_path: str | Path,
    *,
    embedder: BaseEmbedder | None,
    grid_step: float = 0.2,
    turn_pooling_options: tuple[int, ...] = (1, 2, 3),
    retrieval_depth: int = 10,
    qa_top_k: int = 5,
    token_budget: int = 2048,
    stability_folds: int = 5,
) -> VMPHierarchicalTuningResult:
    """Tune hierarchy weights on Dev while keeping Test completely inaccessible."""

    if retrieval_depth < 10 or retrieval_depth < qa_top_k:
        raise ValueError("retrieval_depth must be at least max(10, qa_top_k)")
    if qa_top_k != 5:
        raise ValueError("VMP-v5 currently requires qa_top_k=5")
    if token_budget < 1 or stability_folds < 2:
        raise ValueError("token_budget must be positive and stability_folds >= 2")

    samples, manifest = load_split_samples(data_path, split_manifest_path, "dev")
    validate_longmemeval_dates(samples)
    manifest_path = Path(split_manifest_path).expanduser().resolve()
    manifest_sha256 = sha256_file(manifest_path)
    base_model = VMPTunedModel.load(base_model_path)
    _validate_base_model(
        base_model,
        dataset_sha256=manifest.dataset_sha256,
        split_id=manifest.split_id,
        embedding_identifier=embedder.identifier if embedder else None,
    )
    assignment_sha256 = split_assignment_sha256(manifest)
    if manifest_sha256 != base_model.split_manifest_sha256:
        LOGGER.warning(
            "Base V4.3 manifest file SHA-256 differs from the current file, "
            "but dataset SHA-256 and canonical split_id match. Continuing "
            "with semantic split assignment SHA-256 %s.",
            assignment_sha256,
        )

    LOGGER.info(
        "Loaded %d V5 Dev samples; building session features and turn signals.",
        len(samples),
    )
    base_examples, skipped = build_vmp_tuning_examples(
        samples,
        embedder=embedder,
        token_budget=token_budget,
    )
    if not base_examples:
        raise ValueError("dev split has no answerable examples with gold sessions")
    hierarchical_examples = build_hierarchical_tuning_examples(
        samples,
        base_examples=base_examples,
        embedder=embedder,
    )
    parameters = hierarchical_parameter_grid(
        step=grid_step,
        turn_pooling_options=turn_pooling_options,
    )
    max_memory_count = max(
        example.base.memory_count + example.turn_count
        for example in hierarchical_examples
    )

    best_parameters: HierarchicalFusionParameters | None = None
    best_metrics: dict[str, float] | None = None
    best_objective: float | None = None
    best_key: tuple[float, ...] | None = None
    best_examples: list[VMPTuningExample] | None = None
    baseline_metrics: dict[str, float] | None = None
    trial_summaries: list[dict[str, JsonValue]] = []
    started_at = perf_counter()
    for trial, candidate_parameters in enumerate(parameters):
        transformed = apply_hierarchical_fusion(
            hierarchical_examples,
            parameters=candidate_parameters,
        )
        metrics = _evaluate_frozen_base(
            transformed,
            base_model=base_model,
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
        parameter_hash = sha256_json(candidate_parameters.as_payload())
        key = hierarchical_trial_selection_key(
            metrics,
            objective=objective,
            parameter_hash=parameter_hash,
        )
        if (
            candidate_parameters.session_semantic_weight == 1.0
            and candidate_parameters.turn_semantic_weight == 0.0
            and candidate_parameters.turn_lexical_weight == 0.0
        ):
            baseline_metrics = dict(metrics)
        trial_summaries.append(
            {
                "trial": trial,
                "parameters": cast(
                    JsonValue,
                    candidate_parameters.as_payload(),
                ),
                "parameter_sha256": parameter_hash,
                "objective": objective,
                "metrics": cast(JsonValue, metrics),
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_parameters = candidate_parameters
            best_metrics = metrics
            best_objective = objective
            best_examples = transformed
        completed = trial + 1
        if completed == 1 or completed % 10 == 0 or completed == len(parameters):
            LOGGER.info(
                "Hierarchical search %d/%d: recall@5=%.6f best=%.6f "
                "elapsed=%.1fs",
                completed,
                len(parameters),
                metrics["recall_all@5"],
                best_metrics["recall_all@5"] if best_metrics else 0.0,
                perf_counter() - started_at,
            )

    if (
        best_parameters is None
        or best_metrics is None
        or best_objective is None
        or best_examples is None
        or baseline_metrics is None
    ):
        raise RuntimeError("VMP-v5 hierarchical search produced no model")

    oracle_metrics = dense_guard_oracle_metrics(
        best_examples,
        safety_top_k=qa_top_k,
        preserve_dense_top_n=retrieval_depth,
        protected_dense_count=max(1, qa_top_k - 1),
    )
    safe_base_model = _hierarchical_base_model(
        base_model,
        dev_metrics=best_metrics,
    )
    model = VMPHierarchicalModel(
        base_model=safe_base_model,
        session_semantic_weight=best_parameters.session_semantic_weight,
        turn_semantic_weight=best_parameters.turn_semantic_weight,
        turn_lexical_weight=best_parameters.turn_lexical_weight,
        turn_pooling_top_n=best_parameters.turn_pooling_top_n,
        split_id=manifest.split_id,
        split_manifest_sha256=base_model.split_manifest_sha256,
        split_assignment_sha256=assignment_sha256,
        dataset_sha256=manifest.dataset_sha256,
        embedding_identifier=embedder.identifier if embedder else None,
        best_objective=best_objective,
        dev_metrics=best_metrics,
        metadata={
            "training_split": "dev",
            "dev_question_count": len(samples),
            "answerable_dev_question_count": len(hierarchical_examples),
            "skipped_dev_question_count": skipped,
            "grid_step": grid_step,
            "turn_pooling_options": list(turn_pooling_options),
            "trials": len(parameters),
            "retrieval_depth": retrieval_depth,
            "qa_top_k": qa_top_k,
            "token_budget": token_budget,
            "stability_folds": stability_folds,
            "turn_representation": "role_prefixed_raw_turn",
            "promotion_ranker": "disabled_after_semantic_geometry_change",
            "membership_strategy": "hierarchical_dense_top5",
            "ordering_strategy": "frozen_vmp_base_policy_score",
            "baseline_session_only_metrics": cast(JsonValue, baseline_metrics),
            "base_v43_dev_metrics": cast(JsonValue, base_model.dev_metrics),
            "dev_oracle_ceiling_metrics": cast(JsonValue, oracle_metrics),
            "dev_recall_all_at_5_delta_vs_session_only": (
                best_metrics["recall_all@5"]
                - baseline_metrics["recall_all@5"]
            ),
            "dev_recall_all_at_5_delta_vs_v43": (
                best_metrics["recall_all@5"]
                - float(base_model.dev_metrics.get("recall_all@5", 0.0))
            ),
            "base_model_path": str(Path(base_model_path)),
            "base_model_sha256": sha256_file(base_model_path),
            "current_split_manifest_sha256": manifest_sha256,
            "split_manifest_file_sha256_matches_base": (
                manifest_sha256 == base_model.split_manifest_sha256
            ),
            "split_assignment_sha256": assignment_sha256,
            "ranking_pipeline": (
                "session_embedding + pooled_turn_embedding + turn_bm25 -> "
                "hierarchical_dense_top10 -> frozen_vmp_policy_ordering -> "
                "cached_non_destructive_lifecycle"
            ),
            "test_labels_used": False,
        },
    )
    return VMPHierarchicalTuningResult(
        model=model,
        trials_evaluated=len(parameters),
        candidate_examples=len(hierarchical_examples),
        skipped_examples=skipped,
        trial_summaries=trial_summaries,
    )


def build_hierarchical_tuning_examples(
    samples: list[LongMemEvalSample],
    *,
    base_examples: list[VMPTuningExample],
    embedder: BaseEmbedder | None,
) -> list[HierarchicalTuningExample]:
    """Attach grouped query-to-turn scores to precomputed session features."""

    samples_by_id = {sample.question_id: sample for sample in samples}
    examples: list[HierarchicalTuningExample] = []
    started_at = perf_counter()
    for index, base_example in enumerate(base_examples, start=1):
        sample = samples_by_id[base_example.question_id]
        turn_chunks = [
            chunk
            for event in sample_to_events(sample)
            if (chunk := contextual_turn_chunk(event)) is not None
        ]
        query_embedding = embedder.embed_one(sample.question) if embedder else None
        if embedder is not None and turn_chunks:
            vectors = embedder.embed([chunk.content for chunk in turn_chunks])
            for chunk, vector in zip(turn_chunks, vectors, strict=True):
                chunk.content_embedding = list(vector)
        query_counts = term_counts(sample.question)
        semantic_scores = [
            (
                dense_cosine(query_embedding, chunk.content_embedding)
                if query_embedding is not None and chunk.content_embedding
                else sparse_cosine(query_counts, term_counts(chunk.content))
            )
            for chunk in turn_chunks
        ]
        lexical_scores = normalized_bm25_scores(
            sample.question,
            [chunk.content for chunk in turn_chunks],
        )
        semantic_by_session: dict[str, list[float]] = {}
        lexical_by_session: dict[str, list[float]] = {}
        for chunk, semantic, lexical in zip(
            turn_chunks,
            semantic_scores,
            lexical_scores,
            strict=True,
        ):
            session_id = chunk.source_session_id or chunk.memory_id
            semantic_by_session.setdefault(session_id, []).append(semantic)
            lexical_by_session.setdefault(session_id, []).append(lexical)
        examples.append(
            HierarchicalTuningExample(
                base=base_example,
                turn_semantic_scores=tuple(
                    tuple(semantic_by_session.get(candidate.session_id, ()))
                    for candidate in base_example.candidates
                ),
                turn_lexical_scores=tuple(
                    tuple(lexical_by_session.get(candidate.session_id, ()))
                    for candidate in base_example.candidates
                ),
                turn_count=len(turn_chunks),
                turn_tokens=sum(chunk.token_count for chunk in turn_chunks),
            )
        )
        if index == 1 or index % 10 == 0 or index == len(base_examples):
            LOGGER.info(
                "Turn feature progress %d/%d: question_id=%s turns=%d "
                "elapsed=%.1fs",
                index,
                len(base_examples),
                sample.question_id,
                len(turn_chunks),
                perf_counter() - started_at,
            )
    return examples


def apply_hierarchical_fusion(
    examples: list[HierarchicalTuningExample],
    *,
    parameters: HierarchicalFusionParameters,
) -> list[VMPTuningExample]:
    """Materialize fused semantic features for one cheap grid evaluation."""

    transformed: list[VMPTuningExample] = []
    for example in examples:
        candidates = []
        for index, candidate in enumerate(example.base.candidates):
            turn_semantic = pool_top_scores(
                example.turn_semantic_scores[index],
                top_n=parameters.turn_pooling_top_n,
            )
            turn_lexical = pool_top_scores(
                example.turn_lexical_scores[index],
                top_n=parameters.turn_pooling_top_n,
            )
            fused = hierarchical_fusion_score(
                session_semantic=float(
                    candidate.policy_features.semantic_relevance
                ),
                turn_semantic=turn_semantic,
                turn_lexical=turn_lexical,
                session_semantic_weight=parameters.session_semantic_weight,
                turn_semantic_weight=parameters.turn_semantic_weight,
                turn_lexical_weight=parameters.turn_lexical_weight,
            )
            candidates.append(
                candidate.model_copy(
                    update={
                        "policy_features": candidate.policy_features.model_copy(
                            update={"semantic_relevance": fused}
                        )
                    }
                )
            )
        transformed.append(
            example.base.model_copy(
                update={
                    "candidates": candidates,
                    "memory_count": (
                        example.base.memory_count + example.turn_count
                    ),
                    "memory_tokens": (
                        example.base.memory_tokens + example.turn_tokens
                    ),
                }
            )
        )
    return transformed


def hierarchical_parameter_grid(
    *,
    step: float,
    turn_pooling_options: tuple[int, ...],
) -> list[HierarchicalFusionParameters]:
    """Generate an exact deterministic simplex grid with the session baseline."""

    if not 0.0 < step <= 1.0:
        raise ValueError("grid step must be in (0, 1]")
    units = round(1.0 / step)
    if units < 1 or abs(units * step - 1.0) > 1e-9:
        raise ValueError("grid step must divide 1.0 exactly")
    if not turn_pooling_options or any(value < 1 for value in turn_pooling_options):
        raise ValueError("turn pooling options must contain positive integers")
    parameters: list[HierarchicalFusionParameters] = []
    for top_n in sorted(set(turn_pooling_options)):
        for session_units in range(units, -1, -1):
            for turn_units in range(units - session_units, -1, -1):
                lexical_units = units - session_units - turn_units
                parameters.append(
                    HierarchicalFusionParameters(
                        session_semantic_weight=session_units / units,
                        turn_semantic_weight=turn_units / units,
                        turn_lexical_weight=lexical_units / units,
                        turn_pooling_top_n=top_n,
                    )
                )
    return parameters


def hierarchical_trial_selection_key(
    metrics: dict[str, float],
    *,
    objective: float,
    parameter_hash: str,
) -> tuple[float, ...]:
    """Prefer robust recall before ranking quality and deterministic hash order."""

    hash_tiebreaker = int(parameter_hash[:13], 16) / float(16**13)
    return (
        metrics["recall_all@5"],
        metrics["macro_type_recall_all@5"],
        metrics["worst_type_recall_all@5"],
        metrics["min_fold_recall_all@5"],
        -metrics["fold_recall_stddev"],
        metrics["mrr"],
        objective,
        hash_tiebreaker,
    )


def _evaluate_frozen_base(
    examples: list[VMPTuningExample],
    *,
    base_model: VMPTunedModel,
    retrieval_depth: int,
    qa_top_k: int,
    token_budget: int,
    max_memory_count: int,
    stability_folds: int,
) -> dict[str, float]:
    return evaluate_vmp_parameters(
        examples,
        weights={
            name: float(value) for name, value in base_model.weights.items()
        },
        retrieve_threshold=float(base_model.retrieve_threshold),
        semantic_anchor_weight=float(base_model.semantic_anchor_weight),
        lexical_anchor_weight=float(base_model.lexical_anchor_weight),
        policy_adjustment_limit=float(base_model.policy_adjustment_limit),
        archive_score_penalty=float(base_model.archive_score_penalty),
        protected_dense_count=qa_top_k,
        promotion_margin=0.0,
        promotion_ranker=None,
        retrieval_depth=retrieval_depth,
        qa_top_k=qa_top_k,
        token_budget=token_budget,
        max_memory_count=max_memory_count,
        stability_folds=stability_folds,
    )


def _hierarchical_base_model(
    base_model: VMPTunedModel,
    *,
    dev_metrics: dict[str, float],
) -> VMPTunedModel:
    payload = base_model.model_dump(mode="python")
    metadata = dict(base_model.metadata)
    metadata.update(
        {
            "ranking_semantics_version": "5.0",
            "promotion_ranker": "disabled_after_semantic_geometry_change",
            "membership_strategy": "hierarchical_dense_top5",
            "test_labels_used": False,
        }
    )
    payload.update(
        {
            "protected_dense_count": base_model.safety_top_k,
            "promotion_margin": 0.0,
            "promotion_ranker": None,
            "dev_metrics": dev_metrics,
            "metadata": metadata,
        }
    )
    return VMPTunedModel.model_validate(payload)


def _validate_base_model(
    model: VMPTunedModel,
    *,
    dataset_sha256: str,
    split_id: str,
    embedding_identifier: str | None,
) -> None:
    if model.training_split != "dev":
        raise ValueError("VMP-v5 base model must be trained on Dev")
    if model.dataset_sha256 != dataset_sha256:
        raise ValueError("VMP-v5 base model dataset differs from split manifest")
    if model.split_id != split_id:
        raise ValueError("VMP-v5 base model split ID differs")
    if model.embedding_identifier != embedding_identifier:
        raise ValueError("VMP-v5 base model embedding identifier differs")
    if model.metadata.get("test_labels_used") is not False:
        raise ValueError("VMP-v5 base model does not prove Test isolation")
