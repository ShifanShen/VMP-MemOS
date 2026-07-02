"""Small deterministic text utilities used by local retrieval adapters."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

_TOKEN_PATTERN = re.compile(r"[\w-]+", flags=re.UNICODE)
_LONGMEMEVAL_DATE_PATTERN = re.compile(
    r"^(?P<date>\d{4}/\d{1,2}/\d{1,2})\s+"
    r"\([A-Za-z]{3}\)\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)$"
)


def terms(text: str) -> list[str]:
    """Tokenize text into case-folded lexical terms."""

    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


def term_counts(text: str) -> Counter[str]:
    """Return token frequency counts."""

    return Counter(terms(text))


def lexical_jaccard(left: str, right: str) -> float:
    """Return a simple set Jaccard similarity."""

    left_terms = set(terms(left))
    right_terms = set(terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def sparse_cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Cosine similarity over sparse dictionaries."""

    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return clamp01(numerator / (left_norm * right_norm))


def dense_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity over dense vectors."""

    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return clamp01(numerator / (left_norm * right_norm))


def estimate_tokens(text: str) -> int:
    """Approximate LLM token count without external tokenizers."""

    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_chars = sum(1 for char in text if ord(char) < 128 and not char.isspace())
    other_chars = max(0, len(text) - cjk_chars - ascii_chars)
    return max(1, math.ceil(cjk_chars / 1.8 + ascii_chars / 4.0 + other_chars / 3.0))


def recency_score(source_date: str | None, question_date: str | None) -> float:
    """Return a bounded recency score relative to the question date."""

    source = parse_date(source_date)
    question = parse_date(question_date)
    if source is None:
        return 0.5
    if question is None:
        return 0.5
    age_days = max(0.0, (question - source).total_seconds() / 86_400.0)
    return clamp01(1.0 / (1.0 + age_days / 30.0))


def heuristic_importance(text: str) -> float:
    """Estimate content importance without using gold labels."""

    token_count = len(terms(text))
    if token_count == 0:
        return 0.0
    length_signal = min(1.0, math.log1p(token_count) / math.log(160))
    lowered = text.casefold()
    preference_signal = 0.2 if any(
        keyword in lowered
        for keyword in (
            "prefer",
            "favorite",
            "remember",
            "important",
            "changed",
            "now",
            "不再",
            "现在",
            "偏好",
            "记住",
            "重要",
            "改为",
        )
    ) else 0.0
    return clamp01(0.75 * length_signal + preference_signal)


def parse_date(value: str | None) -> datetime | None:
    """Parse common dataset date strings."""

    if not value:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    longmemeval_match = _LONGMEMEVAL_DATE_PATTERN.fullmatch(normalized)
    if longmemeval_match is not None:
        normalized = (
            longmemeval_match.group("date").replace("/", "-")
            + "T"
            + longmemeval_match.group("time")
        )
    for candidate in (
        normalized,
        normalized.replace("Z", "+00:00"),
        normalized[:10],
    ):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def clamp01(value: float) -> float:
    """Clamp non-finite or out-of-range scores to [0, 1]."""

    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))
