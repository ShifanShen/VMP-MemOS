"""Validation guards for paper-grade LongMemEval runs."""

from __future__ import annotations

from pydantic import JsonValue

from vmp_memos.frameworks.text import parse_date
from vmp_memos.longmemeval.schema import LongMemEvalSample


def validate_longmemeval_dates(
    samples: list[LongMemEvalSample],
) -> dict[str, JsonValue]:
    """Reject non-empty timestamps that would disable temporal features."""

    invalid: list[tuple[str, str]] = []
    question_dates = 0
    source_dates = 0
    for sample in samples:
        if sample.question_date:
            question_dates += 1
            if parse_date(sample.question_date) is None:
                invalid.append((sample.question_id, sample.question_date))
        for source_date in sample.haystack_dates:
            if source_date:
                source_dates += 1
                if parse_date(source_date) is None:
                    invalid.append((sample.question_id, source_date))
        if len(invalid) >= 5:
            break
    if invalid:
        examples = ", ".join(
            f"{question_id}={value!r}" for question_id, value in invalid
        )
        raise ValueError(
            "LongMemEval contains non-empty dates that cannot be parsed; "
            f"examples: {examples}"
        )
    return {
        "status": "passed",
        "question_dates_parsed": question_dates,
        "source_dates_parsed": source_dates,
        "accepted_formats": [
            "YYYY/MM/DD (Day) HH:MM[:SS]",
            "ISO-8601",
        ],
    }
