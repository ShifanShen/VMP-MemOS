"""Dependency-free local QA metrics for LongMemEval."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence

_ASCII_OR_CJK_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_ARTICLES = {"a", "an", "the"}
_ABSTENTION_ANSWERS = {
    "i do not know",
    "i dont know",
    "unknown",
    "no answer",
    "not answerable",
    "cannot answer",
    "无法回答",
    "不知道",
}


def normalize_answer(text: str) -> str:
    """Normalize case, punctuation, English articles, and whitespace."""

    lowered = text.casefold().replace("’", "'")
    lowered = lowered.replace("don't", "do not").replace("dont", "do not")
    without_punctuation = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in lowered
    )
    tokens = [
        token
        for token in _ASCII_OR_CJK_TOKEN.findall(without_punctuation)
        if token not in _ARTICLES
    ]
    return " ".join(tokens)


def is_abstention_answer(text: str) -> bool:
    """Recognize the fixed no-answer forms accepted by the local evaluator."""

    normalized = normalize_answer(text)
    if normalized in _ABSTENTION_ANSWERS:
        return True
    return normalized.startswith("i do not know") or normalized.startswith("i dont know")


def compute_qa_metrics(
    prediction: str,
    gold_answers: str | Sequence[str],
    *,
    is_abstention: bool,
) -> dict[str, float]:
    """Score one prediction against one or more accepted gold answers."""

    answers = [gold_answers] if isinstance(gold_answers, str) else list(gold_answers)
    if not answers:
        raise ValueError("gold_answers cannot be empty")
    if is_abstention:
        return {"abstention_accuracy": float(is_abstention_answer(prediction))}

    normalized_prediction = normalize_answer(prediction)
    normalized_gold = [normalize_answer(answer) for answer in answers]
    return {
        "normalized_exact_match": max(
            float(normalized_prediction == answer) for answer in normalized_gold
        ),
        "token_f1": max(_token_f1(normalized_prediction, answer) for answer in normalized_gold),
        "contains_answer": max(
            _contains_token_sequence(normalized_prediction, answer)
            for answer in normalized_gold
        ),
    }


def aggregate_qa_metrics(
    rows: Sequence[tuple[Mapping[str, float], bool]],
) -> dict[str, float]:
    """Aggregate answerable and abstention metrics over their valid subsets."""

    answerable = [metrics for metrics, abstention in rows if not abstention]
    abstentions = [metrics for metrics, abstention in rows if abstention]
    aggregate: dict[str, float] = {}
    for name in ("normalized_exact_match", "token_f1", "contains_answer"):
        values = [float(metrics[name]) for metrics in answerable if name in metrics]
        aggregate[name] = _mean(values)
    abstention_values = [
        float(metrics["abstention_accuracy"])
        for metrics in abstentions
        if "abstention_accuracy" in metrics
    ]
    aggregate["abstention_accuracy"] = _mean(abstention_values)
    return aggregate


def _token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = prediction.split()
    gold_tokens = gold.split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    overlap = Counter(prediction_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _contains_token_sequence(prediction: str, gold: str) -> float:
    prediction_tokens = prediction.split()
    gold_tokens = gold.split()
    if not gold_tokens:
        return float(not prediction_tokens)
    width = len(gold_tokens)
    return float(
        any(
            prediction_tokens[index : index + width] == gold_tokens
            for index in range(len(prediction_tokens) - width + 1)
        )
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
