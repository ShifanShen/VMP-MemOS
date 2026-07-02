"""Leakage-safe deterministic splits for LongMemEval."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field, JsonValue, model_validator

from vmp_memos.longmemeval.loader import load_longmemeval
from vmp_memos.longmemeval.schema import LongMemEvalSample
from vmp_memos.schemas.base import NonEmptyStr, NonNegativeInt, SchemaModel


class LongMemEvalSplitManifest(SchemaModel):
    """Auditable question-level dev/test assignment."""

    schema_version: NonEmptyStr = "1.0"
    split_id: NonEmptyStr
    dataset_path: NonEmptyStr
    dataset_sha256: NonEmptyStr
    dataset_question_ids_sha256: NonEmptyStr
    seed: NonNegativeInt
    strategy: NonEmptyStr = "sha256_rank"
    created_at: datetime
    splits: dict[NonEmptyStr, list[NonEmptyStr]]
    question_type_counts: dict[NonEmptyStr, dict[NonEmptyStr, NonNegativeInt]] = Field(
        default_factory=dict
    )
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_disjoint_splits(self) -> "LongMemEvalSplitManifest":
        """Reject duplicate IDs within or across splits."""

        seen: set[str] = set()
        for split_name, question_ids in self.splits.items():
            if len(question_ids) != len(set(question_ids)):
                raise ValueError(f"duplicate question IDs in split {split_name!r}")
            overlap = seen.intersection(question_ids)
            if overlap:
                raise ValueError(
                    f"question IDs occur in more than one split: {sorted(overlap)[:3]}"
                )
            seen.update(question_ids)
        if "dev" not in self.splits or "test" not in self.splits:
            raise ValueError("split manifest must contain dev and test")
        return self

    def save(self, path: str | Path) -> Path:
        """Write the manifest as stable UTF-8 JSON."""

        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "LongMemEvalSplitManifest":
        """Load a split manifest."""

        manifest_path = Path(path).expanduser().resolve()
        return cls.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def create_longmemeval_split(
    data_path: str | Path,
    *,
    dev_size: int = 100,
    test_size: int | None = None,
    seed: int = 42,
) -> LongMemEvalSplitManifest:
    """Create an exact-size split by seeded SHA-256 rank.

    Ranking question IDs rather than relying on a library RNG makes the split
    stable across Python and operating-system versions.
    """

    source_path = Path(data_path).expanduser().resolve()
    samples = load_longmemeval(source_path)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if dev_size < 1:
        raise ValueError("dev_size must be at least 1")
    resolved_test_size = len(samples) - dev_size if test_size is None else test_size
    if resolved_test_size < 1:
        raise ValueError("test_size must be at least 1")
    if dev_size + resolved_test_size > len(samples):
        raise ValueError(
            f"requested {dev_size + resolved_test_size} questions from "
            f"a dataset containing {len(samples)}"
        )

    by_id = _samples_by_id(samples)
    ranked_ids = sorted(
        by_id,
        key=lambda question_id: (
            hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest(),
            question_id,
        ),
    )
    dev_ids = ranked_ids[:dev_size]
    test_ids = ranked_ids[dev_size : dev_size + resolved_test_size]
    unused_ids = ranked_ids[dev_size + resolved_test_size :]
    split_payload = {"dev": dev_ids, "test": test_ids}
    split_id = _split_id(
        dataset_sha256=sha256_file(source_path),
        seed=seed,
        splits=split_payload,
    )
    return LongMemEvalSplitManifest(
        split_id=split_id,
        dataset_path=str(source_path),
        dataset_sha256=sha256_file(source_path),
        dataset_question_ids_sha256=_question_ids_sha256(by_id),
        seed=seed,
        created_at=datetime.now(UTC),
        splits=split_payload,
        question_type_counts={
            name: dict(
                sorted(Counter(by_id[question_id].question_type for question_id in ids).items())
            )
            for name, ids in split_payload.items()
        },
        metadata={
            "dataset_sample_count": len(samples),
            "dev_size": dev_size,
            "test_size": resolved_test_size,
            "unused_question_ids": unused_ids,
            "assignment_uses_labels": False,
        },
    )


def load_split_samples(
    data_path: str | Path,
    manifest_path: str | Path,
    split_name: str,
) -> tuple[list[LongMemEvalSample], LongMemEvalSplitManifest]:
    """Load one split after checking source bytes and exact question membership."""

    source_path = Path(data_path).expanduser().resolve()
    manifest = LongMemEvalSplitManifest.load(manifest_path)
    expected_split_id = _split_id(
        dataset_sha256=manifest.dataset_sha256,
        seed=manifest.seed,
        splits={name: list(ids) for name, ids in manifest.splits.items()},
    )
    if manifest.split_id != expected_split_id:
        raise ValueError("split manifest assignments do not match split_id")
    if split_name not in manifest.splits:
        known = ", ".join(sorted(manifest.splits))
        raise ValueError(f"unknown split {split_name!r}; known splits: {known}")
    actual_sha256 = sha256_file(source_path)
    if actual_sha256 != manifest.dataset_sha256:
        raise ValueError(
            "LongMemEval dataset SHA-256 does not match split manifest: "
            f"expected {manifest.dataset_sha256}, got {actual_sha256}"
        )

    by_id = _samples_by_id(load_longmemeval(source_path))
    actual_ids_sha256 = _question_ids_sha256(by_id)
    if actual_ids_sha256 != manifest.dataset_question_ids_sha256:
        raise ValueError("LongMemEval question IDs do not match split manifest")
    missing = [
        question_id
        for question_id in manifest.splits[split_name]
        if question_id not in by_id
    ]
    if missing:
        raise ValueError(f"split references missing question IDs: {missing[:3]}")
    return [by_id[question_id] for question_id in manifest.splits[split_name]], manifest


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    """Hash a JSON-compatible payload using canonical serialization."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _samples_by_id(samples: list[LongMemEvalSample]) -> dict[str, LongMemEvalSample]:
    by_id: dict[str, LongMemEvalSample] = {}
    for sample in samples:
        if sample.question_id in by_id:
            raise ValueError(f"duplicate LongMemEval question_id: {sample.question_id}")
        by_id[sample.question_id] = sample
    return by_id


def _question_ids_sha256(by_id: dict[str, LongMemEvalSample]) -> str:
    return sha256_json(sorted(by_id))


def _split_id(
    *,
    dataset_sha256: str,
    seed: int,
    splits: dict[str, list[str]],
) -> str:
    digest = sha256_json(
        {
            "dataset_sha256": dataset_sha256,
            "seed": seed,
            "strategy": "sha256_rank",
            "splits": splits,
        }
    )
    return f"lme_seed{seed}_{digest[:12]}"
