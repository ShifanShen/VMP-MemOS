#!/usr/bin/env python3
"""Fail fast unless a frozen VMP-v3 model beats its Dev dense safety baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.frameworks import VMPTunedModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--min-recall-all-at-5", type=float, default=0.90)
    parser.add_argument("--min-delta-vs-dense", type=float, default=0.02)
    args = parser.parse_args()

    model = VMPTunedModel.load(args.model)
    recall = float(model.dev_metrics.get("recall_all@5", 0.0))
    metadata = model.metadata
    baseline_payload = metadata.get("dense_safety_baseline_metrics", {})
    baseline_value = (
        baseline_payload.get("recall_all@5", 0.0)
        if isinstance(baseline_payload, dict)
        else 0.0
    )
    baseline = (
        float(baseline_value)
        if isinstance(baseline_value, int | float)
        else 0.0
    )
    delta = recall - baseline
    passed = (
        recall >= args.min_recall_all_at_5
        and delta >= args.min_delta_vs_dense
    )
    print(
        json.dumps(
            {
                "status": "passed" if passed else "failed",
                "model": str(args.model.expanduser().resolve()),
                "schema_version": model.schema_version,
                "dev_recall_all@5": recall,
                "dense_safety_baseline_recall_all@5": baseline,
                "delta_vs_dense": delta,
                "required_min_recall_all@5": args.min_recall_all_at_5,
                "required_min_delta_vs_dense": args.min_delta_vs_dense,
                "test_labels_used": model.metadata.get("test_labels_used"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
