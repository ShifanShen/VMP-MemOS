"""Export LongMemEval retrieval tables from one completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.longmemeval.tables import export_retrieval_tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    outputs = export_retrieval_tables(
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
