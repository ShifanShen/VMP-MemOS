"""Merge comparable official-prompt judge runs without calling an LLM again."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.longmemeval.official_qa_merge import merge_official_judge_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-run",
        type=Path,
        action="append",
        required=True,
        help="Completed judge directory; repeat for every independent run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = merge_official_judge_runs(args.judge_run, output_dir=args.output)
    print(
        json.dumps(
            {
                "status": "completed",
                "comparison_run": str(result),
                "manifest": str(result / "manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
