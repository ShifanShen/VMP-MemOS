"""Audit external memory frameworks before allowing them into main tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from vmp_memos.frameworks import audit_known_frameworks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frameworks",
        default="mem0,letta,langmem,graphiti",
        help="Comma-separated framework names.",
    )
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument(
        "--llm-model",
        default=os.getenv("VMP_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument(
        "--official-llm-max-tokens",
        type=int,
        default=int(os.getenv("VMP_OFFICIAL_LLM_MAX_TOKENS", "512")),
    )
    parser.add_argument(
        "--official-llm-temperature",
        type=float,
        default=float(os.getenv("VMP_OFFICIAL_LLM_TEMPERATURE", "0.0")),
    )
    parser.add_argument("--output-dir", default="outputs/longmemeval")
    parser.add_argument(
        "--verification-dir",
        default="outputs/longmemeval/audit",
        help="Directory containing official adapter smoke JSON files.",
    )
    args = parser.parse_args()

    names = [item.strip() for item in args.frameworks.split(",") if item.strip()]
    reports = audit_known_frameworks(
        names,
        vllm_base_url=args.vllm_base_url,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        official_llm_max_tokens=args.official_llm_max_tokens,
        official_llm_temperature=args.official_llm_temperature,
        verification_dir=args.verification_dir,
    )
    output_dir = Path(args.output_dir)
    audit_dir = output_dir / "audit"
    tables_dir = output_dir / "tables"
    audit_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = audit_dir / "framework_controllability.json"
    json_path.write_text(
        json.dumps(
            [report.model_dump(mode="json") for report in reports],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = tables_dir / "table6_fairness.csv"
    rows = [report.model_dump(mode="json") for report in reports]
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote: {json_path}")
    print(f"wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
