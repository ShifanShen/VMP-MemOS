"""Load and inspect LongMemEval JSON / JSONL files."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from vmp_memos.longmemeval.schema import (
    LongMemEvalDatasetStats,
    LongMemEvalSample,
)


def load_longmemeval(path: str | Path, *, limit: int | None = None) -> list[LongMemEvalSample]:
    """Load a LongMemEval file into typed samples."""

    return list(iter_longmemeval(path, limit=limit))


def iter_longmemeval(path: str | Path, *, limit: int | None = None) -> Iterator[LongMemEvalSample]:
    """Yield typed LongMemEval samples from JSON array or JSONL input."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return
    count = 0
    for record in _iter_records(Path(path)):
        yield LongMemEvalSample.model_validate(record)
        count += 1
        if limit is not None and count >= limit:
            return


def inspect_longmemeval(
    path: str | Path,
    *,
    limit: int | None = None,
    first_n: int = 5,
) -> LongMemEvalDatasetStats:
    """Return a compact dataset summary without requiring model or GPU dependencies."""

    samples = load_longmemeval(path, limit=limit)
    question_types: Counter[str] = Counter(sample.question_type for sample in samples)
    session_counts = [sample.session_count for sample in samples]
    turn_counts = [sample.turn_count for sample in samples]
    return LongMemEvalDatasetStats(
        path=str(path),
        sample_count=len(samples),
        question_types=dict(sorted(question_types.items())),
        abstention_count=sum(1 for sample in samples if sample.is_abstention),
        session_count_min=min(session_counts, default=0),
        session_count_max=max(session_counts, default=0),
        session_count_avg=_average(session_counts),
        turn_count_min=min(turn_counts, default=0),
        turn_count_max=max(turn_counts, default=0),
        turn_count_avg=_average(turn_counts),
        first_question_ids=[sample.question_id for sample in samples[:first_n]],
    )


def _iter_records(path: Path) -> Iterator[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        yield from _iter_jsonl_records(path)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            yield _ensure_mapping(item, path=path, index=index)
        return
    if isinstance(payload, dict):
        records = payload.get("data") or payload.get("samples") or payload.get("records")
        if isinstance(records, list):
            for index, item in enumerate(records):
                yield _ensure_mapping(item, path=path, index=index)
            return
    raise ValueError(f"{path} must contain a JSON array, JSONL, or an object with data/samples")


def _iter_jsonl_records(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            yield _ensure_mapping(
                json.loads(line),
                path=path,
                index=line_number,
            )


def _ensure_mapping(value: object, *, path: Path, index: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} record {index} must be a JSON object")
    return value


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
