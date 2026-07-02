"""Session-level retrieval metrics for LongMemEval."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

DEFAULT_RECALL_CUTOFFS = (1, 3, 5, 10)


def ranked_unique_session_ids(session_ids: Iterable[str | None]) -> list[str]:
    """Deduplicate a ranked list while preserving the first rank of each session."""

    ranked: list[str] = []
    seen: set[str] = set()
    for session_id in session_ids:
        if not session_id or session_id in seen:
            continue
        seen.add(session_id)
        ranked.append(session_id)
    return ranked


def compute_retrieval_metrics(
    retrieved_session_ids: Sequence[str],
    gold_session_ids: Sequence[str],
    *,
    recall_cutoffs: Sequence[int] = DEFAULT_RECALL_CUTOFFS,
    precision_k: int = 5,
    ndcg_k: int = 5,
) -> dict[str, float]:
    """Compute binary-relevance session retrieval metrics for one question."""

    _validate_cutoffs((*recall_cutoffs, precision_k, ndcg_k))
    ranked = ranked_unique_session_ids(retrieved_session_ids)
    gold = set(gold_session_ids)
    if not gold:
        raise ValueError("gold_session_ids cannot be empty for retrieval evaluation")

    metrics = {
        f"recall_at_{cutoff}": len(gold.intersection(ranked[:cutoff])) / len(gold)
        for cutoff in recall_cutoffs
    }
    metrics[f"precision_at_{precision_k}"] = (
        len(gold.intersection(ranked[:precision_k])) / precision_k
    )
    metrics["mrr"] = _reciprocal_rank(ranked, gold)
    metrics[f"ndcg_at_{ndcg_k}"] = _ndcg_at_k(ranked, gold, ndcg_k)
    return metrics


def aggregate_retrieval_metrics(
    metric_rows: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Macro-average identically shaped per-question metric dictionaries."""

    if not metric_rows:
        return {}
    metric_names = tuple(metric_rows[0])
    expected = set(metric_names)
    for row in metric_rows:
        if set(row) != expected:
            raise ValueError("all metric rows must contain the same metric names")
    return {
        name: sum(float(row[name]) for row in metric_rows) / len(metric_rows)
        for name in metric_names
    }


def _reciprocal_rank(ranked: Sequence[str], gold: set[str]) -> float:
    for rank, session_id in enumerate(ranked, start=1):
        if session_id in gold:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(ranked: Sequence[str], gold: set[str], cutoff: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, session_id in enumerate(ranked[:cutoff], start=1)
        if session_id in gold
    )
    ideal_hits = min(len(gold), cutoff)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _validate_cutoffs(cutoffs: Sequence[int]) -> None:
    if any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("metric cutoffs must be positive")
