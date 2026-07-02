#!/usr/bin/env python3
"""Create the fixed, leakage-safe LongMemEval dev/test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.longmemeval.splits import create_longmemeval_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/longmemeval/splits/dev_test_seed42.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=400)
    args = parser.parse_args()

    manifest = create_longmemeval_split(
        args.data,
        dev_size=args.dev_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    output_path = manifest.save(args.output)
    print(
        json.dumps(
            {
                "split_id": manifest.split_id,
                "output": str(output_path),
                "dataset_sha256": manifest.dataset_sha256,
                "counts": {
                    name: len(question_ids)
                    for name, question_ids in manifest.splits.items()
                },
                "question_type_counts": manifest.question_type_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
