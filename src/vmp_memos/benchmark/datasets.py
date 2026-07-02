"""Dataset loading and result writing helpers for memory-policy benchmarks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from vmp_memos.schemas import BenchmarkResult, BenchmarkSample


def load_benchmark_samples(path: str | Path) -> list[BenchmarkSample]:
    """Load a JSONL benchmark file into validated samples."""

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Benchmark dataset not found: {dataset_path}")

    samples: list[BenchmarkSample] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            samples.append(BenchmarkSample.model_validate_json(stripped))
        except Exception as exc:
            raise ValueError(
                f"Invalid benchmark sample at {dataset_path}:{line_number}: {exc}"
            ) from exc
    if not samples:
        raise ValueError(f"Benchmark dataset contains no samples: {dataset_path}")
    return samples


def write_benchmark_results(
    path: str | Path,
    results: Iterable[BenchmarkResult],
) -> Path:
    """Write benchmark results as JSONL and return the resolved path."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for result in results:
            stream.write(result.to_json_line())
            stream.write("\n")
    return output_path
