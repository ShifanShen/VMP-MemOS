#!/usr/bin/env python3
"""Analyze a completed Dev rerank run without touching sealed Test artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def analyze_dev_rerank(
    run_dir: Path,
    *,
    method: str,
) -> dict[str, Any]:
    """Summarize transitions, coverage decisions, and recoverable Dev failures."""

    resolved_run = run_dir.expanduser().resolve()
    manifest = _read_object(resolved_run / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("rerank run must be completed before analysis")
    split = manifest.get("split")
    split_name = split.get("name") if isinstance(split, dict) else split
    if split_name != "dev":
        raise ValueError("diagnostic analysis is restricted to the Dev split")
    records = _read_jsonl(resolved_run / method / "retrieval.jsonl")
    transitions: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    coverage_gains: list[float] = []
    recoverable_failures: list[dict[str, Any]] = []
    for record in records:
        metadata = _mapping(record.get("rerank_metadata"))
        transition = metadata.get("transition_vs_source")
        if isinstance(transition, str):
            transitions[transition] += 1
        plan = _mapping(metadata.get("question_evidence_plan"))
        operator = plan.get("operator")
        if isinstance(operator, str):
            operators[operator] += 1
        for value in _list(metadata.get("selector_span_binding_failures")):
            if isinstance(value, str):
                failure_reasons[value.split(":", maxsplit=1)[-1]] += 1
        gain = metadata.get("coverage_gain")
        if isinstance(gain, int | float):
            coverage_gains.append(float(gain))
        if (
            transition == "stable_failure"
            and metadata.get("candidate_oracle_recoverable") is True
        ):
            label_map = _mapping(metadata.get("candidate_label_session_ids"))
            gold = _strings(record.get("gold_session_ids"))
            selected = _strings(metadata.get("selected_session_ids"))
            missing = [session_id for session_id in gold if session_id not in selected]
            recoverable_failures.append(
                {
                    "question_id": record.get("question_id"),
                    "question_type": record.get("question_type"),
                    "question": record.get("question"),
                    "operator": operator,
                    "gold_session_ids": gold,
                    "selected_session_ids": selected,
                    "missing_gold_session_ids": missing,
                    "missing_gold_candidate_labels": [
                        label
                        for label, session_id in label_map.items()
                        if session_id in missing
                    ],
                    "coverage_selection": metadata.get("coverage_selection"),
                    "selector_span_binding_failures": metadata.get(
                        "selector_span_binding_failures"
                    ),
                }
            )
    mean_gain = sum(coverage_gains) / len(coverage_gains) if coverage_gains else 0.0
    return {
        "schema_version": "1.0",
        "run": str(resolved_run),
        "method": method,
        "split": "dev",
        "records": len(records),
        "transitions": dict(sorted(transitions.items())),
        "question_operators": dict(sorted(operators.items())),
        "mean_coverage_gain": mean_gain,
        "positive_coverage_gain_questions": sum(gain > 0 for gain in coverage_gains),
        "extraction_failure_reasons": dict(sorted(failure_reasons.items())),
        "candidate_oracle_recoverable_stable_failures": recoverable_failures,
        "candidate_oracle_recoverable_stable_failure_count": len(
            recoverable_failures
        ),
        "uses_dev_gold_labels_for_diagnostics": True,
        "test_labels_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--method",
        default="vmp_hierarchical__vllm_boundary",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = analyze_dev_rerank(args.run, method=args.method)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        payload
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        if isinstance((payload := json.loads(line)), dict)
    ]


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _strings(value: object) -> list[str]:
    return [item for item in _list(value) if isinstance(item, str)]


if __name__ == "__main__":
    raise SystemExit(main())
