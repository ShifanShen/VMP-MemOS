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
    """Compute official and supplementary session retrieval metrics."""

    _validate_cutoffs((*recall_cutoffs, precision_k, ndcg_k))
    ranked = ranked_unique_session_ids(retrieved_session_ids)
    gold = set(gold_session_ids)
    if not gold:
        raise ValueError("gold_session_ids cannot be empty for retrieval evaluation")

    metrics: dict[str, float] = {}
    for cutoff in recall_cutoffs:
        recalled = gold.intersection(ranked[:cutoff])
        metrics[f"recall_any@{cutoff}"] = float(bool(recalled))
        metrics[f"recall_all@{cutoff}"] = float(recalled == gold)
        metrics[f"fractional_recall@{cutoff}"] = len(recalled) / len(gold)
    metrics[f"precision@{precision_k}"] = (
        len(gold.intersection(ranked[:precision_k])) / precision_k
    )
    metrics["mrr"] = _reciprocal_rank(ranked, gold)
    for cutoff in sorted(set((*recall_cutoffs, ndcg_k))):
        metrics[f"ndcg_any@{cutoff}"] = _official_ndcg_at_k(
            ranked,
            gold,
            cutoff,
        )
        metrics[f"standard_ndcg@{cutoff}"] = _standard_ndcg_at_k(
            ranked,
            gold,
            cutoff,
        )
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


def _official_ndcg_at_k(
    ranked: Sequence[str],
    gold: set[str],
    cutoff: int,
) -> float:
    """Mirror LongMemEval's released ``eval_utils.ndcg`` implementation."""

    relevances = [float(session_id in gold) for session_id in ranked[:cutoff]]
    ideal = [1.0] * min(len(gold), cutoff)
    actual_dcg = _official_dcg(relevances)
    ideal_dcg = _official_dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg else 0.0


def _official_dcg(relevances: Sequence[float]) -> float:
    if not relevances:
        return 0.0
    return float(relevances[0]) + sum(
        relevance / math.log2(index)
        for index, relevance in enumerate(relevances[1:], start=2)
    )


def _standard_ndcg_at_k(ranked: Sequence[str], gold: set[str], cutoff: int) -> float:
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
