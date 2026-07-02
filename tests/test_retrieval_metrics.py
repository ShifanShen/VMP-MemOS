"""Tests for session-level LongMemEval retrieval metrics."""

from __future__ import annotations

import math

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

    assert metrics["recall_any@1"] == 0.0
    assert metrics["recall_all@1"] == 0.0
    assert metrics["recall_any@3"] == 1.0
    assert metrics["recall_all@3"] == 1.0
    assert metrics["fractional_recall@3"] == 1.0
    assert metrics["precision@5"] == pytest.approx(0.4)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["ndcg_any@5"] == pytest.approx(
        (1.0 + 1.0 / math.log2(3)) / 2.0
    )
    assert 0.0 < metrics["standard_ndcg@5"] < 1.0


def test_official_recall_all_requires_every_gold_session() -> None:
    metrics = compute_retrieval_metrics(
        ["gold_a", "noise", "noise_2"],
        ["gold_a", "gold_b"],
    )

    assert metrics["recall_any@5"] == 1.0
    assert metrics["recall_all@5"] == 0.0
    assert metrics["fractional_recall@5"] == 0.5


def test_aggregate_retrieval_metrics_macro_averages_rows() -> None:
    assert aggregate_retrieval_metrics([{"mrr": 1.0}, {"mrr": 0.0}]) == {"mrr": 0.5}
    assert aggregate_retrieval_metrics([]) == {}


def test_retrieval_metrics_requires_gold_evidence() -> None:
    with pytest.raises(ValueError, match="gold_session_ids"):
        compute_retrieval_metrics(["s1"], [])
