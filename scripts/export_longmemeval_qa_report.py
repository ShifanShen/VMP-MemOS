"""Export paper QA tables and paired statistics from a completed judge run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.longmemeval.qa_statistics import (
    LongMemEvalQAReportConfig,
    export_longmemeval_qa_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-run", type=Path, required=True)
    parser.add_argument("--reference-method", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = export_longmemeval_qa_report(
        LongMemEvalQAReportConfig(
            judge_run=args.judge_run,
            reference_method=args.reference_method,
            output_dir=args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )
    )
    print(
        json.dumps(
            {
                "report": str(result.report_path),
                "outputs": {name: str(path) for name, path in result.outputs.items()},
                "comparisons": [
                    comparison.model_dump(mode="json")
                    for comparison in result.comparisons
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
