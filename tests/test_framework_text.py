"""Regression tests for LongMemEval date handling."""

from datetime import UTC, datetime

from vmp_memos.frameworks.text import parse_date, recency_score


def test_parse_date_supports_official_longmemeval_timestamp() -> None:
    parsed = parse_date("2023/10/30 (Mon) 16:38")

    assert parsed == datetime(2023, 10, 30, 16, 38, tzinfo=UTC)
    assert recency_score(
        "2023/10/30 (Mon) 16:38",
        "2023/11/29 (Wed) 16:38",
    ) == 0.5

