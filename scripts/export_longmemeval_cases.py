#!/usr/bin/env python3
"""Export four auditable qualitative cases for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.longmemeval.cases import export_longmemeval_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--ablation-run", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vmp-method", default=None)
    parser.add_argument("--vector-method", default=None)
    parser.add_argument(
        "--allow-missing-qa",
        action="store_true",
        help="Allow retrieval-only selection; VMP error selection may be weaker.",
    )
    args = parser.parse_args()

    outputs = export_longmemeval_cases(
        args.retrieval_run,
        ablation_run=args.ablation_run,
        output_dir=args.output_dir,
        vmp_method=args.vmp_method,
        vector_method=args.vector_method,
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
