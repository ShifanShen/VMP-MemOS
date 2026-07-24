#!/usr/bin/env python3
"""Fail fast unless a frozen VMP-v4 model passes robust Dev safety gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.frameworks import VMPTunedModel


def _metric(payload: object, name: str) -> float:
    if not isinstance(payload, dict):
        return 0.0
    value = payload.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--min-recall-all-at-5", type=float, default=0.90)
    parser.add_argument("--min-delta-vs-dense", type=float, default=0.02)
    parser.add_argument("--min-macro-delta-vs-dense", type=float, default=0.0)
    parser.add_argument("--min-worst-type-delta-vs-dense", type=float, default=-0.03)
    parser.add_argument("--max-fold-recall-stddev", type=float, default=0.20)
    args = parser.parse_args()

    model = VMPTunedModel.load(args.model)
    dev = model.dev_metrics
    baseline = model.metadata.get("dense_safety_baseline_metrics", {})
    recall = _metric(dev, "recall_all@5")
    recall_delta = recall - _metric(baseline, "recall_all@5")
    macro_delta = _metric(dev, "macro_type_recall_all@5") - _metric(
        baseline, "macro_type_recall_all@5"
    )
    worst_delta = _metric(dev, "worst_type_recall_all@5") - _metric(
        baseline, "worst_type_recall_all@5"
    )
    fold_stddev = _metric(dev, "fold_recall_stddev")
    passed = all(
        (
            recall >= args.min_recall_all_at_5,
            recall_delta >= args.min_delta_vs_dense,
            macro_delta >= args.min_macro_delta_vs_dense,
            worst_delta >= args.min_worst_type_delta_vs_dense,
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
                "dev_recall_all@5": recall,
                "delta_vs_dense": recall_delta,
                "macro_delta_vs_dense": macro_delta,
                "worst_type_delta_vs_dense": worst_delta,
                "fold_recall_stddev": fold_stddev,
                "dense_head_retention@5": _metric(dev, "dense_head_retention@5"),
                "required": {
                    "min_recall_all@5": args.min_recall_all_at_5,
                    "min_delta_vs_dense": args.min_delta_vs_dense,
                    "min_macro_delta_vs_dense": args.min_macro_delta_vs_dense,
                    "min_worst_type_delta_vs_dense": (
                        args.min_worst_type_delta_vs_dense
                    ),
                    "max_fold_recall_stddev": args.max_fold_recall_stddev,
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
