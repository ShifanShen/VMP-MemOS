"""Audit a completed official Mem0 run for extraction and retrieval integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp_memos.longmemeval.mem0_protocol import audit_mem0_protocol_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--method", default="mem0_official")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-unrecovered-failure-rate",
        type=float,
        default=0.0,
    )
    parser.add_argument("--max-initial-invalid-rate", type=float, default=0.02)
    parser.add_argument("--allow-bm25-disabled", action="store_true")
    parser.add_argument("--allow-spacy-disabled", action="store_true")
    parser.add_argument("--expected-llm-max-tokens", type=int, default=2048)
    parser.add_argument("--expected-llm-retry-max-tokens", type=int, default=4096)
    parser.add_argument("--expected-llm-context-window", type=int, default=32768)
    args = parser.parse_args()

    report = audit_mem0_protocol_run(
        args.retrieval_run,
        method=args.method,
        max_unrecovered_failure_rate=args.max_unrecovered_failure_rate,
        max_initial_invalid_rate=args.max_initial_invalid_rate,
        require_bm25=not args.allow_bm25_disabled,
        require_spacy=not args.allow_spacy_disabled,
        expected_llm_max_tokens=args.expected_llm_max_tokens,
        expected_llm_retry_max_tokens=args.expected_llm_retry_max_tokens,
        expected_llm_context_window=args.expected_llm_context_window,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
