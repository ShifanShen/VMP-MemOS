"""Shared schema primitives and JSONL serialization helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Annotated, Self
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Create an opaque ID that remains stable across serialization round trips."""

    return f"{prefix}_{uuid4().hex}"


class SchemaModel(BaseModel):
    """Strict base model used by every public Phase 1 schema."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class TimestampedSchema(SchemaModel):
    """Base for schemas that can be serialized or appended as JSONL."""

    timestamp: AwareDatetime = Field(default_factory=utc_now, frozen=True)

    def to_json_line(self) -> str:
        """Serialize this object as one compact JSONL record without a newline."""

        return self.model_dump_json(by_alias=True)

    @classmethod
    def from_json_line(cls, line: str) -> Self:
        """Validate one JSONL record and return the typed object."""

        if not line.strip():
            raise ValueError("JSONL record cannot be empty")
        return cls.model_validate_json(line)

    def append_jsonl(self, path: str | PathLike[str]) -> Path:
        """Append this object to a UTF-8 JSONL file, creating parents as needed."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(self.to_json_line())
            stream.write("\n")
        return output_path
