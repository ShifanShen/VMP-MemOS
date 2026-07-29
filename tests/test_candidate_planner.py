from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.llm.candidate_planner import (
    LONGMEMEVAL_IDENTITY_CANDIDATE_PLANNER_VERSION,
    LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION,
    plan_longmemeval_rerank_candidates,
)


def test_v55_dual_view_planner_promotes_session_view_evidence() -> None:
    memories = [
        RetrievedMemory(
            memory_id=f"s{index}",
            source_session_id=f"s{index}",
            content=f"session {index}",
            score=1.0 / index,
            token_count=4,
            metadata={
                "hierarchical_fused_score": 1.0 / index,
                "session_semantic_score": (
                    1.0 if index == 11 else 0.5 - index / 100.0
                ),
            },
        )
        for index in range(1, 13)
    ]

    result = plan_longmemeval_rerank_candidates(
        memories,
        candidate_count=10,
        planner_version=LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION,
        rrf_k=60,
        hierarchical_weight=0.8,
    )

    planned_ids = [memory.source_session_id for memory in result.candidates]
    assert result.applied is True
    assert result.input_candidate_count == 12
    assert result.output_candidate_count == 10
    assert "s11" in planned_ids
    assert "s10" not in planned_ids
    promoted = next(memory for memory in result.candidates if memory.source_session_id == "s11")
    assert promoted.metadata["candidate_plan"]["original_rank"] == 11
    assert promoted.metadata["candidate_plan"]["session_view_rank"] == 1


def test_candidate_planner_is_identity_without_hierarchical_metadata() -> None:
    memories = [
        RetrievedMemory(
            memory_id=f"s{index}",
            source_session_id=f"s{index}",
            content=f"session {index}",
            score=1.0 / index,
            token_count=4,
        )
        for index in range(1, 13)
    ]

    result = plan_longmemeval_rerank_candidates(
        memories,
        candidate_count=10,
        planner_version=LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION,
        rrf_k=60,
        hierarchical_weight=0.8,
    )

    assert result.applied is False
    assert [memory.source_session_id for memory in result.candidates] == [
        f"s{index}" for index in range(1, 11)
    ]

    identity = plan_longmemeval_rerank_candidates(
        memories,
        candidate_count=10,
        planner_version=LONGMEMEVAL_IDENTITY_CANDIDATE_PLANNER_VERSION,
    )
    assert identity.applied is False
    assert identity.candidates == memories[:10]
