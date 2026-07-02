#!/usr/bin/env python3
"""Run the toy memory-policy benchmark and export JSONL plus Markdown artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Final

from vmp_memos.benchmark import BenchmarkRunConfig, BenchmarkRunner

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> BenchmarkRunConfig:
    """Load benchmark YAML config and resolve relative paths from project root."""

    raw = _load_yaml_mapping(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark config must be a mapping: {path}")
    values: dict[str, Any] = dict(raw)
    for key in ("dataset_path", "output_dir", "report_dir", "policy_model_path"):
        if key in values and values[key] is not None:
            values[key] = _resolve_project_path(values[key])
    return BenchmarkRunConfig.model_validate(values)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "benchmark.yaml",
        help="Benchmark config YAML path.",
    )
    parser.add_argument(
        "--baselines",
        default=None,
        help="Comma-separated baseline names overriding the config.",
    )
    parser.add_argument(
        "--policy",
        choices=["rule", "learned"],
        default=None,
        help="Convenience selector for the VMP policy baseline.",
    )
    parser.add_argument(
        "--policy-model-path",
        type=Path,
        default=None,
        help="Path to a trained learned-policy JSON model.",
    )
    parser.add_argument("--run-id", default=None, help="Optional deterministic run ID.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config.expanduser().resolve())
    updates: dict[str, Any] = {}
    if args.baselines:
        updates["baselines"] = [
            name.strip()
            for name in args.baselines.split(",")
            if name.strip()
        ]
    if args.policy == "learned":
        requested = list(updates.get("baselines", config.baselines))
        if not args.baselines:
            requested = ["learned_policy"]
        elif "learned_policy" not in requested and "learned" not in requested:
            requested.append("learned_policy")
        updates["baselines"] = requested
    elif args.policy == "rule" and not args.baselines:
        updates["baselines"] = ["vmp_rule"]
    if args.policy_model_path is not None:
        updates["policy_model_path"] = _resolve_project_path(args.policy_model_path)
    if args.run_id:
        updates["run_id"] = args.run_id
    if args.top_k is not None:
        updates["top_k"] = args.top_k
    if updates:
        config = config.model_copy(update=updates)

    summary = BenchmarkRunner(config).run()
    print(f"Run ID: {summary.run_id}")
    print(f"Samples: {summary.num_samples}")
    print(f"Baselines: {', '.join(summary.baselines)}")
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
    """Parse the tiny YAML subset used by configs/benchmark.yaml.

    This fallback keeps the benchmark CLI usable in lightweight local
    environments where PyYAML has not been installed yet. It supports top-level
    scalar keys and top-level lists written with ``- item``.
    """

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
