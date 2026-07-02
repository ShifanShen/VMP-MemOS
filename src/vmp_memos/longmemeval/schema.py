"""Pydantic schemas for the LongMemEval cleaned dataset."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, JsonValue, model_validator

from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
)


class LongMemEvalRawModel(SchemaModel):
    """Base model for public dataset records.

    LongMemEval is an external dataset and may add bookkeeping fields over time.
    The integration keeps those fields instead of rejecting the record, while
    still normalizing the fields needed by the benchmark pipeline.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class LongMemEvalTurn(LongMemEvalRawModel):
    """One turn inside a LongMemEval history session."""

    role: NonEmptyStr
    content: str = ""
    has_answer: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_turn(cls, value: object) -> object:
        """Accept dict-like records and coerce common aliases."""

        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "role" not in data:
            data["role"] = data.get("speaker") or data.get("author") or "unknown"
        if "content" not in data:
            data["content"] = data.get("text") or data.get("message") or ""
        return data


class LongMemEvalSession(SchemaModel):
    """A normalized LongMemEval history session."""

    session_id: NonEmptyStr
    date: str | None = None
    turns: list[LongMemEvalTurn] = Field(default_factory=list)

    @property
    def turn_count(self) -> int:
        """Return the number of turns in the session."""

        return len(self.turns)


class LongMemEvalSample(LongMemEvalRawModel):
    """One LongMemEval question with its candidate history sessions."""

    question_id: NonEmptyStr
    question_type: NonEmptyStr = "unknown"
    question: NonEmptyStr
    answer: NonEmptyStr | list[NonEmptyStr]
    question_date: str | None = None
    haystack_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    haystack_dates: list[str | None] = Field(default_factory=list)
    haystack_sessions: list[list[LongMemEvalTurn]] = Field(default_factory=list)
    answer_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    has_answer: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_sample(cls, value: object) -> object:
        """Normalize aliases and make missing external fields safe."""

        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "question_id" not in data:
            data["question_id"] = data.get("id") or data.get("qid")
        if "answer_session_ids" not in data:
            data["answer_session_ids"] = data.get("gold_session_ids") or []
        if "haystack_session_ids" not in data:
            sessions = data.get("haystack_sessions") or []
            data["haystack_session_ids"] = [
                f"session_{index}" for index, _ in enumerate(sessions)
            ]
        if "haystack_dates" not in data:
            data["haystack_dates"] = [None] * len(data.get("haystack_sessions") or [])
        if "answer" not in data or data.get("answer") in (None, "", []):
            data["answer"] = "I don't know"
        if "question_type" not in data or data.get("question_type") in (None, ""):
            data["question_type"] = "unknown"
        return data

    @model_validator(mode="after")
    def validate_parallel_session_fields(self) -> "LongMemEvalSample":
        """Ensure session ids, dates, and sessions can be zipped safely."""

        session_count = len(self.haystack_sessions)
        if not self.haystack_session_ids:
            self.haystack_session_ids = [
                f"session_{index}" for index in range(session_count)
            ]
        if len(self.haystack_session_ids) != session_count:
            raise ValueError(
                "haystack_session_ids and haystack_sessions must have the same length"
            )
        if len(self.haystack_dates) < session_count:
            self.haystack_dates.extend([None] * (session_count - len(self.haystack_dates)))
        elif len(self.haystack_dates) > session_count:
            self.haystack_dates = self.haystack_dates[:session_count]
        return self

    @property
    def sessions(self) -> list[LongMemEvalSession]:
        """Return normalized sessions preserving official ids and dates."""

        return [
            LongMemEvalSession(session_id=session_id, date=date, turns=turns)
            for session_id, date, turns in zip(
                self.haystack_session_ids,
                self.haystack_dates,
                self.haystack_sessions,
                strict=True,
            )
        ]

    @property
    def session_count(self) -> int:
        """Return the number of candidate history sessions."""

        return len(self.haystack_sessions)

    @property
    def turn_count(self) -> int:
        """Return the number of turns across all candidate sessions."""

        return sum(len(session) for session in self.haystack_sessions)

    @property
    def is_abstention(self) -> bool:
        """Return whether the sample is an abstention / no-answer case."""

        if self.has_answer is False:
            return True
        if self.answer_session_ids:
            return False
        answers = self.answer if isinstance(self.answer, list) else [self.answer]
        normalized = {" ".join(answer.casefold().split()) for answer in answers}
        return bool(
            normalized
            & {
                "i don't know",
                "unknown",
                "not answerable",
                "no answer",
                "无法回答",
                "不知道",
            }
        )


class LongMemEvalDatasetStats(SchemaModel):
    """Inspection summary for one LongMemEval file."""

    path: str
    sample_count: NonNegativeInt
    question_types: dict[str, NonNegativeInt] = Field(default_factory=dict)
    abstention_count: NonNegativeInt = 0
    session_count_min: NonNegativeInt = 0
    session_count_max: NonNegativeInt = 0
    session_count_avg: NonNegativeFloat = 0.0
    turn_count_min: NonNegativeInt = 0
    turn_count_max: NonNegativeInt = 0
    turn_count_avg: NonNegativeFloat = 0.0
    first_question_ids: list[NonEmptyStr] = Field(default_factory=list)


class LongMemEvalRunConfig(SchemaModel):
    """Minimal reproducible config for LongMemEval retrieval/QA runs."""

    data_path: Path
    methods: list[NonEmptyStr] = Field(default_factory=list)
    top_k: NonNegativeInt = 5
    retrieval_depth: NonNegativeInt = 10
    limit: NonNegativeInt | None = None
    output_dir: Path = Path("outputs/longmemeval")
    ingestion_granularity: NonEmptyStr = "session"
    skip_abstention_for_retrieval: bool = True
    split_manifest_path: Path | None = None
    split_name: str | None = None
    vmp_tuned_model_path: Path | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_run_config(self) -> "LongMemEvalRunConfig":
        """Keep CLI-created configs honest."""

        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.retrieval_depth < self.top_k:
            raise ValueError("retrieval_depth must be greater than or equal to top_k")
        if self.retrieval_depth < 10:
            raise ValueError("retrieval_depth must be at least 10 for Recall@10")
        if self.ingestion_granularity not in {"session", "turn"}:
            raise ValueError("ingestion_granularity must be 'session' or 'turn'")
        if bool(self.split_manifest_path) != bool(self.split_name):
            raise ValueError("split_manifest_path and split_name must be provided together")
        if self.split_name is not None and self.split_name not in {"dev", "test"}:
            raise ValueError("split_name must be 'dev' or 'test'")
        normalized_methods = {
            method.casefold().replace("-", "_") for method in self.methods
        }
        uses_vmp_tuned = any(
            method == "vmp_full" or method.startswith("vmp_tuned")
            for method in normalized_methods
        )
        if uses_vmp_tuned and self.vmp_tuned_model_path is None:
            raise ValueError("vmp_tuned requires vmp_tuned_model_path")
        return self


def raw_record_metadata(record: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Return JSON-safe extra metadata from a raw LongMemEval record."""

    metadata: dict[str, JsonValue] = {}
    reserved = set(LongMemEvalSample.model_fields)
    for key, value in record.items():
        if key not in reserved and _is_json_value(value):
            metadata[key] = value
    return metadata


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False
