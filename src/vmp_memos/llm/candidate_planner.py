"""Deterministic candidate planning before shared local-vLLM selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from pydantic import Field, JsonValue

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeInt,
    SchemaModel,
    Score,
)

LONGMEMEVAL_IDENTITY_CANDIDATE_PLANNER_VERSION = "identity_v1"
LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION = (
    "vmp_v55_dual_view_rrf_v1"
)
LONGMEMEVAL_CANDIDATE_PLANNER_VERSIONS = frozenset(
    {
        LONGMEMEVAL_IDENTITY_CANDIDATE_PLANNER_VERSION,
        LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION,
    }
)


class LongMemEvalCandidatePlan(SchemaModel):
    """One auditable, label-free candidate-planning result."""

    planner_version: NonEmptyStr
    applied: bool
    input_candidate_count: NonNegativeInt
    output_candidate_count: NonNegativeInt
    rrf_k: NonNegativeInt
    hierarchical_weight: Score
    candidates: list[RetrievedMemory] = Field(default_factory=list)


def plan_longmemeval_rerank_candidates(
    candidates: Sequence[RetrievedMemory],
    *,
    candidate_count: int,
    planner_version: str = LONGMEMEVAL_IDENTITY_CANDIDATE_PLANNER_VERSION,
    rrf_k: int = 60,
    hierarchical_weight: float = 0.8,
) -> LongMemEvalCandidatePlan:
    """Plan a fixed shortlist without reading question types or gold labels."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    if planner_version not in LONGMEMEVAL_CANDIDATE_PLANNER_VERSIONS:
        raise ValueError("unsupported LongMemEval candidate planner version")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if not math.isfinite(hierarchical_weight) or not 0.0 <= hierarchical_weight <= 1.0:
        raise ValueError("hierarchical_weight must be finite and in [0, 1]")

    unique = _unique_session_memories(candidates)
    if planner_version == LONGMEMEVAL_IDENTITY_CANDIDATE_PLANNER_VERSION:
        planned = unique[:candidate_count]
        return LongMemEvalCandidatePlan(
            planner_version=planner_version,
            applied=False,
            input_candidate_count=len(unique),
            output_candidate_count=len(planned),
            rrf_k=rrf_k,
            hierarchical_weight=hierarchical_weight,
            candidates=planned,
        )

    observed_session_scores = [
        _metadata_float(memory, "session_semantic_score") for memory in unique
    ]
    if any(score is None for score in observed_session_scores):
        planned = unique[:candidate_count]
        return LongMemEvalCandidatePlan(
            planner_version=planner_version,
            applied=False,
            input_candidate_count=len(unique),
            output_candidate_count=len(planned),
            rrf_k=rrf_k,
            hierarchical_weight=hierarchical_weight,
            candidates=planned,
        )
    session_scores = [cast(float, score) for score in observed_session_scores]

    original_rank = {
        _session_id(memory): index
        for index, memory in enumerate(unique, start=1)
    }
    session_order = sorted(
        zip(unique, session_scores, strict=True),
        key=lambda item: (
            -float(item[1]),
            _session_id(item[0]),
        ),
    )
    session_rank = {
        _session_id(memory): index
        for index, (memory, _) in enumerate(session_order, start=1)
    }
    session_weight = 1.0 - hierarchical_weight
    scored: list[tuple[float, int, int, RetrievedMemory]] = []
    for memory in unique:
        session_id = _session_id(memory)
        hierarchy_rank = original_rank[session_id]
        semantic_rank = session_rank[session_id]
        score = (
            hierarchical_weight / (rrf_k + hierarchy_rank)
            + session_weight / (rrf_k + semantic_rank)
        )
        metadata = dict(memory.metadata)
        metadata["candidate_plan"] = cast(
            JsonValue,
            {
                "planner_version": planner_version,
                "original_rank": hierarchy_rank,
                "original_score": float(memory.score),
                "session_view_rank": semantic_rank,
                "session_semantic_score": _metadata_float(
                    memory,
                    "session_semantic_score",
                ),
                "rrf_score": score,
                "rrf_k": rrf_k,
                "hierarchical_weight": hierarchical_weight,
                "test_labels_used": False,
            },
        )
        scored.append(
            (
                score,
                hierarchy_rank,
                semantic_rank,
                memory.model_copy(
                    update={
                        "score": score,
                        "metadata": metadata,
                    }
                ),
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3].memory_id))
    planned = [item[3] for item in scored[:candidate_count]]
    return LongMemEvalCandidatePlan(
        planner_version=planner_version,
        applied=True,
        input_candidate_count=len(unique),
        output_candidate_count=len(planned),
        rrf_k=rrf_k,
        hierarchical_weight=hierarchical_weight,
        candidates=planned,
    )


def _unique_session_memories(
    candidates: Sequence[RetrievedMemory],
) -> list[RetrievedMemory]:
    unique: list[RetrievedMemory] = []
    seen: set[str] = set()
    for memory in candidates:
        session_id = _session_id(memory)
        if session_id in seen:
            continue
        seen.add(session_id)
        unique.append(memory)
    return unique


def _session_id(memory: RetrievedMemory) -> str:
    return memory.source_session_id or memory.memory_id


def _metadata_float(memory: RetrievedMemory, key: str) -> float | None:
    value = memory.metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    observed = float(value)
    return observed if math.isfinite(observed) else None
