"""Merge independent framework runs and export unified LongMemEval paper tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.longmemeval.cost import export_longmemeval_cost
from vmp_memos.longmemeval.paper_comparison import merge_longmemeval_paper_runs
from vmp_memos.longmemeval.paper_efficiency import export_official_judge_efficiency
from vmp_memos.longmemeval.qa_statistics import (
    LongMemEvalQAReportConfig,
    export_longmemeval_qa_report,
)
from vmp_memos.longmemeval.tables import export_retrieval_tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-run",
        type=Path,
        action="append",
        required=True,
        help="Completed rerank run; repeat once for each framework run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-subdir", default="qa_v21_test")
    parser.add_argument("--judge-subdir", default="official_judge_local_vllm_v1")
    parser.add_argument(
        "--reference-method",
        default="vmp_hierarchical__vllm_boundary",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    comparison = merge_longmemeval_paper_runs(
        args.retrieval_run,
        output_dir=args.output,
        qa_subdir=args.qa_subdir,
        judge_subdir=args.judge_subdir,
    )
    paper_dir = comparison / "paper"
    outputs = export_retrieval_tables(comparison, output_dir=paper_dir)
    outputs.update(
        export_longmemeval_cost(
            comparison,
            output_dir=paper_dir,
            qa_subdir=args.qa_subdir,
            reference_method=args.reference_method,
        )
    )
    qa_report = export_longmemeval_qa_report(
        LongMemEvalQAReportConfig(
            judge_run=comparison / args.qa_subdir / args.judge_subdir,
            reference_method=args.reference_method,
            output_dir=paper_dir,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    )
    outputs.update(qa_report.outputs)
    outputs.update(
        export_official_judge_efficiency(
            comparison,
            qa_subdir=args.qa_subdir,
            judge_subdir=args.judge_subdir,
            reference_method=args.reference_method,
            output_dir=paper_dir,
        )
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "comparison_run": str(comparison),
                "paper_outputs": {
                    name: str(path) for name, path in sorted(outputs.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
