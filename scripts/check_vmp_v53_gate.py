#!/usr/bin/env python3
"""Block VMP-v5.3 Test evaluation unless strict Dev boundary gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.llm import (
    LONGMEMEVAL_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_RERANK_PROMPT_VERSION,
)
from vmp_memos.longmemeval.boundary_gate import evaluate_v53_gate
from vmp_memos.longmemeval.rerank_gate import write_gate_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--rerank-run", type=Path, required=True)
    parser.add_argument("--vmp-method", default="vmp_hierarchical")
    parser.add_argument("--baseline-method", default="vmp_tuned")
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--min-recall-all-at-5", type=float, default=0.93)
    parser.add_argument("--min-delta-vs-raw-v5", type=float, default=0.025)
    parser.add_argument("--min-delta-vs-shared-v43", type=float, default=0.03)
    parser.add_argument("--min-macro-delta-vs-raw-v5", type=float, default=0.0)
    parser.add_argument("--max-type-regression-vs-raw-v5", type=float, default=0.03)
    parser.add_argument("--max-parse-fallback-rate", type=float, default=0.02)
    parser.add_argument("--max-boundary-fallback-rate", type=float, default=0.02)
    parser.add_argument("--max-selector-call-fallback-rate", type=float, default=0.02)
    parser.add_argument("--max-regressed-questions", type=int, default=0)
    parser.add_argument("--min-recovered-questions", type=int, default=3)
    parser.add_argument("--min-candidate-count", type=int, default=30)
    parser.add_argument(
        "--expected-selector-prompt-version",
        default=LONGMEMEVAL_RERANK_PROMPT_VERSION,
    )
    parser.add_argument(
        "--expected-boundary-prompt-version",
        default=LONGMEMEVAL_BOUNDARY_PROMPT_VERSION,
    )
    parser.add_argument(
        "--expected-candidate-planner-version",
        default=None,
    )
    parser.add_argument(
        "--expected-candidate-excerpt-version",
        default=None,
    )
    args = parser.parse_args()

    report = evaluate_v53_gate(
        args.candidate_run,
        args.rerank_run,
        vmp_method=args.vmp_method,
        baseline_method=args.baseline_method,
        min_recall_all_at_5=args.min_recall_all_at_5,
        min_delta_vs_raw_v5=args.min_delta_vs_raw_v5,
        min_delta_vs_shared_v43=args.min_delta_vs_shared_v43,
        min_macro_delta_vs_raw_v5=args.min_macro_delta_vs_raw_v5,
        max_type_regression_vs_raw_v5=args.max_type_regression_vs_raw_v5,
        max_parse_fallback_rate=args.max_parse_fallback_rate,
        max_boundary_fallback_rate=args.max_boundary_fallback_rate,
        max_selector_call_fallback_rate=args.max_selector_call_fallback_rate,
        max_regressed_questions=args.max_regressed_questions,
        min_recovered_questions=args.min_recovered_questions,
        min_candidate_count=args.min_candidate_count,
        expected_selector_prompt_version=args.expected_selector_prompt_version,
        expected_boundary_prompt_version=args.expected_boundary_prompt_version,
        expected_candidate_planner_version=args.expected_candidate_planner_version,
        expected_candidate_excerpt_version=args.expected_candidate_excerpt_version,
    )
    if report["status"] == "passed" and args.receipt is not None:
        report["receipt"] = str(write_gate_receipt(report, args.receipt))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
