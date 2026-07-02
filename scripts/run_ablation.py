#!/usr/bin/env python3
"""Run Phase 10 policy-feature ablations and export a Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Final

from vmp_memos.benchmark import (
    DEFAULT_ABLATION_FEATURES,
    AblationRunConfig,
    run_ablation,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "benchmark.yaml",
        help="Benchmark config YAML used to locate dataset and output paths.",
    )
    parser.add_argument(
        "--disable",
        action="append",
        choices=DEFAULT_ABLATION_FEATURES,
        default=[],
        help="Policy feature to zero out. Repeat to run multiple ablations.",
    )
    parser.add_argument(
        "--baselines",
        default="no_memory,vector_rag,vmp_rule",
        help="Comma-separated baseline names for comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "ablation.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for raw ablation JSONL results.",
    )
    parser.add_argument("--run-id", default=None, help="Optional deterministic run ID.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k.")
    parser.add_argument(
        "--max-error-cases",
        type=int,
        default=10,
        help="Maximum incorrect/error rows to show in the report.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    raw_config = _load_yaml_mapping(args.config.expanduser().resolve())
    dataset_path = _resolve_project_path(raw_config.get("dataset_path", ""))
    output_dir = (
        _resolve_project_path(args.output_dir)
        if args.output_dir is not None
        else _resolve_project_path(raw_config.get("output_dir", "outputs/runs"))
    )
    disabled_features = args.disable or list(DEFAULT_ABLATION_FEATURES)
    config = AblationRunConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        report_path=_resolve_project_path(args.output),
        baseline_names=[
            name.strip()
            for name in args.baselines.split(",")
            if name.strip()
        ],
        disabled_features=list(dict.fromkeys(disabled_features)),
        top_k=args.top_k or int(raw_config.get("top_k", 3)),
        run_id=args.run_id,
        max_error_cases=args.max_error_cases,
    )
    summary = run_ablation(config)
    print(f"Run ID: {summary.run_id}")
    print(f"Samples: {summary.num_samples}")
    print(f"Systems: {', '.join(summary.systems)}")
    print(f"Disabled features: {', '.join(summary.disabled_features)}")
    print(f"Results: {summary.result_path}")
    print(f"Report: {summary.report_path}")
    return 0


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml_mapping(text)
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Benchmark config must be a mapping: {path}")
    return dict(loaded)


def _parse_simple_yaml_mapping(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError("YAML list item without a preceding key")
            values.setdefault(current_list_key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        current_list_key = None
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, raw_value = stripped.split(":", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not raw_value:
            values[key] = []
            current_list_key = key
        else:
            values[key] = _parse_scalar(raw_value)
    return values


def _parse_scalar(value: str) -> Any:
    lowered = value.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.isdigit():
        return int(value)
    return value.strip('"').strip("'")


if __name__ == "__main__":
    raise SystemExit(main())
