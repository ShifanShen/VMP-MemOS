"""Inspect a LongMemEval JSON / JSONL file."""

from __future__ import annotations

import argparse
import json

from vmp_memos.longmemeval import inspect_longmemeval, load_longmemeval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a LongMemEval JSON/JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples.")
    parser.add_argument("--show", type=int, default=3, help="Number of examples to print.")
    args = parser.parse_args()

    stats = inspect_longmemeval(args.data, limit=args.limit)
    print(json.dumps(stats.model_dump(mode="json"), ensure_ascii=False, indent=2))

    for sample in load_longmemeval(args.data, limit=args.show):
        print()
        print(f"[{sample.question_id}] {sample.question_type}")
        print(f"question: {sample.question}")
        print(f"answer: {sample.answer}")
        print(
            "sessions: "
            f"{sample.session_count}, turns: {sample.turn_count}, "
            f"answer_session_ids: {sample.answer_session_ids}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
