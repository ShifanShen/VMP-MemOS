#!/usr/bin/env python3
"""Train the lightweight learned memory-operation policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Final

from vmp_memos.benchmark import (
    build_policy_training_examples,
    load_benchmark_samples,
    load_policy_training_examples_from_operation_logs,
    write_policy_training_examples,
)
from vmp_memos.policy import LogisticPolicyModel

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "benchmark.yaml",
        help="Benchmark config YAML used to locate the default dataset.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Override the benchmark dataset path.",
    )
    parser.add_argument(
        "--operation-log",
        type=Path,
        action="append",
        default=[],
        help="Optional operations.jsonl path. Can be provided more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "learned_policy.json",
        help="Output JSON model path.",
    )
    parser.add_argument(
        "--training-data-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "learned_policy_examples.jsonl",
        help="Audit JSONL path for generated training examples.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.4)
    parser.add_argument("--l2", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    config = _load_yaml_mapping(args.config.expanduser().resolve())
    dataset_path = args.dataset_path or Path(str(config["dataset_path"]))
    dataset_path = _resolve_project_path(dataset_path)
    samples = load_benchmark_samples(dataset_path)

    examples = build_policy_training_examples(samples)
    examples.extend(
        load_policy_training_examples_from_operation_logs(
            _resolve_project_path(path) for path in args.operation_log
        )
    )
    training_data_path = write_policy_training_examples(
        _resolve_project_path(args.training_data_output),
        examples,
    )
    model = LogisticPolicyModel.train(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    model_path = model.save(_resolve_project_path(args.output))

    raw_label_counts = model.metadata.get("label_counts", {})
    label_count_items = raw_label_counts.items() if isinstance(raw_label_counts, dict) else []
    label_counts = ", ".join(
        f"{label}={count}"
        for label, count in sorted(label_count_items)
    )
    print(f"Dataset: {dataset_path}")
    print(f"Training examples: {len(examples)}")
    print(f"Label counts: {label_counts}")
    print(f"Training data: {training_data_path}")
    print(f"Model: {model_path}")
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
