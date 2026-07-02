#!/usr/bin/env python3
"""Export LongMemEval Table 5 cost and efficiency artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.longmemeval.cost import export_longmemeval_cost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-qa",
        action="store_true",
        help="Export a retrieval-only preview with QA fields left at zero.",
    )
    args = parser.parse_args()

    outputs = export_longmemeval_cost(
        args.retrieval_run,
        output_dir=args.output_dir,
        require_qa=not args.allow_missing_qa,
    )
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
