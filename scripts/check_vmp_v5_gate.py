#!/usr/bin/env python3
"""Fail fast unless frozen VMP-v5 improves robust Dev retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.frameworks import VMPHierarchicalModel


def _metric(payload: object, name: str) -> float:
    if not isinstance(payload, dict):
        return 0.0
    value = payload.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--min-recall-all-at-5", type=float, default=0.91)
    parser.add_argument("--min-delta-vs-v43", type=float, default=0.0)
    parser.add_argument("--min-delta-vs-session-only", type=float, default=0.0)
    parser.add_argument("--min-turn-weight", type=float, default=0.2)
    parser.add_argument("--max-fold-recall-stddev", type=float, default=0.20)
    args = parser.parse_args()

    model = VMPHierarchicalModel.load(args.model)
    dev = model.dev_metrics
    base_v43 = model.metadata.get("base_v43_dev_metrics", {})
    session_only = model.metadata.get("baseline_session_only_metrics", {})
    recall = _metric(dev, "recall_all@5")
    delta_v43 = recall - _metric(base_v43, "recall_all@5")
    delta_session = recall - _metric(session_only, "recall_all@5")
    turn_weight = (
        float(model.turn_semantic_weight)
        + float(model.turn_lexical_weight)
    )
    fold_stddev = _metric(dev, "fold_recall_stddev")
    passed = all(
        (
            recall >= args.min_recall_all_at_5,
            delta_v43 >= args.min_delta_vs_v43,
            delta_session >= args.min_delta_vs_session_only,
            turn_weight >= args.min_turn_weight,
            fold_stddev <= args.max_fold_recall_stddev,
            model.metadata.get("test_labels_used") is False,
        )
    )
    print(
        json.dumps(
            {
                "status": "passed" if passed else "failed",
                "model": str(args.model.expanduser().resolve()),
                "schema_version": model.schema_version,
                "split_assignment_sha256": model.split_assignment_sha256,
                "dev_recall_all@5": recall,
                "delta_vs_v43": delta_v43,
                "delta_vs_session_only": delta_session,
                "fold_recall_stddev": fold_stddev,
                "fusion": {
                    "session_semantic_weight": (
                        model.session_semantic_weight
                    ),
                    "turn_semantic_weight": model.turn_semantic_weight,
                    "turn_lexical_weight": model.turn_lexical_weight,
                    "turn_pooling_top_n": model.turn_pooling_top_n,
                },
                "dev_oracle_ceiling_metrics": model.metadata.get(
                    "dev_oracle_ceiling_metrics"
                ),
                "required": {
                    "min_recall_all@5": args.min_recall_all_at_5,
                    "min_delta_vs_v43": args.min_delta_vs_v43,
                    "min_delta_vs_session_only": (
                        args.min_delta_vs_session_only
                    ),
                    "min_turn_weight": args.min_turn_weight,
                    "max_fold_recall_stddev": (
                        args.max_fold_recall_stddev
                    ),
                },
                "test_labels_used": model.metadata.get("test_labels_used"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
