"""Tests for session-level LongMemEval retrieval metrics."""

from __future__ import annotations

import pytest

from vmp_memos.evaluation import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
    ranked_unique_session_ids,
)


def test_ranked_unique_session_ids_preserves_first_rank() -> None:
    assert ranked_unique_session_ids(["s1", "s1", None, "s2", "s1"]) == ["s1", "s2"]


def test_compute_retrieval_metrics_handles_multiple_gold_sessions() -> None:
    metrics = compute_retrieval_metrics(
        ["noise", "gold_b", "gold_a"],
        ["gold_a", "gold_b"],
    )

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["precision_at_5"] == pytest.approx(0.4)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert 0.0 < metrics["ndcg_at_5"] < 1.0


def test_aggregate_retrieval_metrics_macro_averages_rows() -> None:
    assert aggregate_retrieval_metrics([{"mrr": 1.0}, {"mrr": 0.0}]) == {"mrr": 0.5}
    assert aggregate_retrieval_metrics([]) == {}


def test_retrieval_metrics_requires_gold_evidence() -> None:
    with pytest.raises(ValueError, match="gold_session_ids"):
        compute_retrieval_metrics(["s1"], [])
