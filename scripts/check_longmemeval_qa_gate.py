"""Enforce the frozen Dev-only LongMemEval QA reader gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.longmemeval.qa_gate import (
    LongMemEvalQAGateConfig,
    evaluate_longmemeval_qa_gate,
    write_longmemeval_qa_gate_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--qa-subdir", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--expected-prompt-version", required=True)
    parser.add_argument("--expected-evidence-mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-answerable-refusal-rate", type=float, default=0.25)
    parser.add_argument("--min-answerable-fact-coverage", type=float, default=0.90)
    parser.add_argument("--min-token-f1", type=float, default=0.25)
    parser.add_argument("--min-contains-answer", type=float, default=0.10)
    parser.add_argument("--min-abstention-accuracy", type=float, default=0.50)
    args = parser.parse_args()

    result = evaluate_longmemeval_qa_gate(
        LongMemEvalQAGateConfig(
            retrieval_run=args.retrieval_run,
            qa_subdir=args.qa_subdir,
            methods=[value.strip() for value in args.methods.split(",") if value.strip()],
            expected_prompt_version=args.expected_prompt_version,
            expected_evidence_mode=args.expected_evidence_mode,
            max_answerable_refusal_rate=args.max_answerable_refusal_rate,
            min_answerable_fact_coverage=args.min_answerable_fact_coverage,
            min_token_f1=args.min_token_f1,
            min_contains_answer=args.min_contains_answer,
            min_abstention_accuracy=args.min_abstention_accuracy,
        )
    )
    output = write_longmemeval_qa_gate_result(result, args.output)
    print(
        json.dumps(
            {
                **result.model_dump(mode="json"),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.status == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
