"""Tests for local reproducible LongMemEval QA metrics."""

from __future__ import annotations

import pytest

from vmp_memos.evaluation import (
    aggregate_qa_metrics,
    compute_qa_metrics,
    is_abstention_answer,
    normalize_answer,
)


def test_normalized_qa_metrics_support_multiple_answers() -> None:
    metrics = compute_qa_metrics(
        "Alex now prefers swimming.",
        ["swimming", "swim"],
        is_abstention=False,
    )

    assert metrics["normalized_exact_match"] == 0.0
    assert metrics["contains_answer"] == 1.0
    assert metrics["token_f1"] == pytest.approx(0.4)
    assert normalize_answer("The SWIMMING!") == "swimming"


def test_abstention_metric_recognizes_fixed_answer() -> None:
    assert is_abstention_answer("I don't know.") is True
    assert compute_qa_metrics(
        "I don't know.",
        "I don't know",
        is_abstention=True,
    ) == {"abstention_accuracy": 1.0}


def test_aggregate_qa_metrics_uses_valid_subsets() -> None:
    aggregate = aggregate_qa_metrics(
        [
            (
                {
                    "normalized_exact_match": 1.0,
                    "token_f1": 1.0,
                    "contains_answer": 1.0,
                },
                False,
            ),
            ({"abstention_accuracy": 0.0}, True),
        ]
    )
    assert aggregate["normalized_exact_match"] == 1.0
    assert aggregate["abstention_accuracy"] == 0.0
