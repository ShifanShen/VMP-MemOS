#!/usr/bin/env python3
"""Export the frozen-model LongMemEval VMP ablation table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.longmemeval.ablation import export_longmemeval_ablation_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    outputs = export_longmemeval_ablation_table(
        args.retrieval_run,
        output_dir=args.output_dir,
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
