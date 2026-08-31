"""End-to-end LongMemEval QA runner over saved retrieval artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pydantic import Field, JsonValue, ValidationError, field_validator

from vmp_memos.evaluation import (
    aggregate_qa_metrics,
    compute_qa_metrics,
    is_abstention_answer,
)
from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.llm import (
    CandidateEvidenceProfile,
    LongMemEvalReader,
    build_query_evidence_windows,
)
from vmp_memos.longmemeval.retrieval_runner import RetrievalSampleRecord
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
)

LOGGER = logging.getLogger(__name__)


class LongMemEvalQARunConfig(SchemaModel):
    """Reproducible QA settings applied to one completed retrieval run."""

    retrieval_run: Path
    methods: list[NonEmptyStr] = Field(default_factory=list)
    top_k: NonNegativeInt = 5
    limit: NonNegativeInt | None = None
    resume: bool = False
    qa_subdir: NonEmptyStr = "qa"
    reader_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("qa_subdir")
    @classmethod
    def validate_qa_subdir(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("qa_subdir must be one safe directory name")
        return value


class QASampleRecord(SchemaModel):
    """One generated answer and its local reproducible metrics."""

    question_id: NonEmptyStr
    question_type: NonEmptyStr
    method: NonEmptyStr
    question: NonEmptyStr
    gold_answer: NonEmptyStr | list[NonEmptyStr]
    prediction: str
    is_abstention: bool
    metrics: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    evidence_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    evidence_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    reader_provider: NonEmptyStr
    reader_model: NonEmptyStr
    reader_finish_reason: str | None = None
    reader_prompt_version: NonEmptyStr = "longmemeval_full_session_reader_v1"
    reader_evidence_mode: NonEmptyStr = "full_sessions"
    prompt_sha256: NonEmptyStr
    evidence_profile_count: NonNegativeInt = 0
    evidence_fact_count: NonNegativeInt = 0
    evidence_window_count: NonNegativeInt = 0
    evidence_span_count: NonNegativeInt = 0
    evidence_window_chars: NonNegativeInt = 0
    retrieved_tokens: NonNegativeInt = 0
    reader_input_tokens: NonNegativeInt = 0
    reader_output_tokens: NonNegativeInt = 0
    ingest_latency_ms: NonNegativeFloat = 0.0
    retrieval_latency_ms: NonNegativeFloat = 0.0
    reader_latency_ms: NonNegativeFloat = 0.0
    end_to_end_latency_ms: NonNegativeFloat = 0.0
    reader_usage: dict[str, JsonValue] = Field(default_factory=dict)


class QAMethodSummary(SchemaModel):
    """Aggregate answer quality and reader cost for one memory method."""

    method: NonEmptyStr
    processed_questions: NonNegativeInt
    answerable_questions: NonNegativeInt
    abstention_questions: NonNegativeInt
    metrics: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    by_question_type: dict[str, dict[str, NonNegativeFloat]] = Field(default_factory=dict)
    answerable_refusal_rate: NonNegativeFloat = 0.0
    answerable_fact_coverage_rate: NonNegativeFloat = 0.0
    answerable_evidence_coverage_rate: NonNegativeFloat = 0.0
    mean_evidence_fact_count: NonNegativeFloat = 0.0
    mean_evidence_window_count: NonNegativeFloat = 0.0
    mean_evidence_span_count: NonNegativeFloat = 0.0
    mean_evidence_window_chars: NonNegativeFloat = 0.0
    mean_retrieved_tokens: NonNegativeFloat = 0.0
    mean_reader_input_tokens: NonNegativeFloat = 0.0
    mean_reader_output_tokens: NonNegativeFloat = 0.0
    mean_ingest_latency_ms: NonNegativeFloat = 0.0
    mean_retrieval_latency_ms: NonNegativeFloat = 0.0
    mean_reader_latency_ms: NonNegativeFloat = 0.0
    mean_end_to_end_latency_ms: NonNegativeFloat = 0.0


class QARunResult(SchemaModel):
    """Artifacts produced by a QA run."""

    retrieval_run: Path
    qa_dir: Path
    manifest_path: Path
    summaries: dict[str, QAMethodSummary]


def run_longmemeval_qa(
    config: LongMemEvalQARunConfig,
    *,
    reader: LongMemEvalReader,
) -> QARunResult:
    """Generate answers for saved retrieval records with resumable JSONL writes."""

    retrieval_run = config.retrieval_run.expanduser().resolve()
    retrieval_manifest_path = retrieval_run / "manifest.json"
    retrieval_manifest = _read_json_object(retrieval_manifest_path)
    if retrieval_manifest.get("status") != "completed":
        raise ValueError(f"Retrieval run is not completed: {retrieval_run}")
    methods = _resolve_methods(retrieval_run, config.methods)
    if not methods:
        raise ValueError("no retrieval methods found for QA")
    if config.top_k < 1:
        raise ValueError("top_k must be at least 1")
    if config.top_k != reader.config.top_k:
        raise ValueError("QA config top_k must equal reader config top_k")
    LOGGER.info(
        "Resolved %d QA methods: %s prompt=%s evidence=%s output=%s",
        len(methods),
        ",".join(methods),
        reader.config.prompt_version,
        reader.config.evidence_mode,
        config.qa_subdir,
    )

    retrieval_records_by_method = {
        method: _load_retrieval_records(
            retrieval_run / method / "retrieval.jsonl",
            limit=config.limit,
        )
        for method in methods
    }
    if reader.config.evidence_mode in {
        "reranker_facts",
        "reranker_facts_with_query_windows",
    }:
        for method, retrieval_records in retrieval_records_by_method.items():
            fact_questions, fact_count = _preflight_evidence_profiles(
                retrieval_records,
                top_k=config.top_k,
            )
            LOGGER.info(
                "Grounded-fact preflight passed for %s: questions=%d "
                "questions_with_facts=%d facts=%d",
                method,
                len(retrieval_records),
                fact_questions,
                fact_count,
            )
            if reader.config.evidence_mode == "reranker_facts_with_query_windows":
                window_questions, window_count, span_count, window_chars = (
                    _preflight_query_windows(
                        retrieval_records,
                        reader=reader,
                        top_k=config.top_k,
                    )
                )
                LOGGER.info(
                    "Query-window preflight passed for %s: questions=%d "
                    "questions_with_windows=%d windows=%d spans=%d chars=%d",
                    method,
                    len(retrieval_records),
                    window_questions,
                    window_count,
                    span_count,
                    window_chars,
                )

    qa_dir = retrieval_run / config.qa_subdir
    hypothesis_dir = (
        retrieval_run / "hypotheses"
        if config.qa_subdir == "qa"
        else qa_dir / "hypotheses"
    )
    qa_dir.mkdir(parents=True, exist_ok=True)
    hypothesis_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = qa_dir / "manifest.json"
    signature = _qa_signature(config, reader=reader, methods=methods)
    manifest = _prepare_manifest(
        manifest_path,
        signature=signature,
        resume=config.resume,
        retrieval_manifest_sha256=_sha256(retrieval_manifest_path),
    )
    wall_started = perf_counter()
    summaries: dict[str, QAMethodSummary] = {}
    observed_readers: set[tuple[str, str]] = set()
    try:
        for method in methods:
            method_started = perf_counter()
            retrieval_records = retrieval_records_by_method[method]
            qa_path = qa_dir / f"{method}.jsonl"
            existing = _load_qa_records(qa_path) if config.resume else []
            existing_ids = {record.question_id for record in existing}
            if len(existing_ids) != len(existing):
                raise ValueError(f"Duplicate question_id in existing QA file: {qa_path}")
            pending_count = sum(
                record.question_id not in existing_ids
                for record in retrieval_records
            )
            LOGGER.info(
                "QA method %s started: total=%d existing=%d pending=%d",
                method,
                len(retrieval_records),
                len(existing),
                pending_count,
            )
            observed_readers.update(
                (record.reader_provider, record.reader_model) for record in existing
            )
            _validate_one_reader(observed_readers)

            with qa_path.open("a", encoding="utf-8", newline="\n") as stream:
                completed_pending = 0
                for retrieval_record in retrieval_records:
                    if retrieval_record.question_id in existing_ids:
                        continue
                    qa_record = _answer_one(
                        method,
                        retrieval_record=retrieval_record,
                        reader=reader,
                        top_k=config.top_k,
                    )
                    observed_readers.add(
                        (qa_record.reader_provider, qa_record.reader_model)
                    )
                    _validate_one_reader(observed_readers)
                    stream.write(qa_record.model_dump_json())
                    stream.write("\n")
                    stream.flush()
                    existing.append(qa_record)
                    existing_ids.add(qa_record.question_id)
                    completed_pending += 1
                    if (
                        completed_pending == 1
                        or completed_pending % 10 == 0
                        or completed_pending == pending_count
                    ):
                        LOGGER.info(
                            "QA method %s progress %d/%d: question_id=%s "
                            "reader_latency=%.1fms elapsed=%.1fs",
                            method,
                            completed_pending,
                            pending_count,
                            qa_record.question_id,
                            qa_record.reader_latency_ms,
                            perf_counter() - method_started,
                        )

            ordered = _order_records(existing, retrieval_records)
            _write_hypotheses(
                hypothesis_dir / f"{method}.jsonl",
                ordered,
            )
            summary = summarize_qa_method(method, ordered)
            summaries[method] = summary
            _write_json(
                qa_dir / f"{method}.summary.json",
                summary.model_dump(mode="json"),
            )
            LOGGER.info(
                "QA method %s completed in %.1fs: questions=%d exact_match=%.4f",
                method,
                perf_counter() - method_started,
                summary.processed_questions,
                float(summary.metrics.get("normalized_exact_match", 0.0)),
            )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "wall_duration_seconds": perf_counter() - wall_started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(manifest_path, manifest)
        raise

    _write_json(
        qa_dir / "summary.json",
        {
            "retrieval_run": str(retrieval_run),
            "methods": {
                method: summary.model_dump(mode="json")
                for method, summary in summaries.items()
            },
        },
    )
    manifest.update(
        {
            "status": "completed",
            "finished_at": datetime.now(UTC).isoformat(),
            "wall_duration_seconds": perf_counter() - wall_started,
            "observed_reader": (
                {
                    "provider": next(iter(observed_readers))[0],
                    "model": next(iter(observed_readers))[1],
                }
                if observed_readers
                else None
            ),
        }
    )
    _write_json(manifest_path, manifest)
    return QARunResult(
        retrieval_run=retrieval_run,
        qa_dir=qa_dir,
        manifest_path=manifest_path,
        summaries=summaries,
    )


def summarize_qa_method(
    method: str,
    records: list[QASampleRecord],
) -> QAMethodSummary:
    """Aggregate one method's answer records."""

    rows = [(record.metrics, record.is_abstention) for record in records]
    answerable_records = [record for record in records if not record.is_abstention]
    by_type: dict[str, list[tuple[dict[str, float], bool]]] = defaultdict(list)
    for record in records:
        by_type[record.question_type].append(
            (
                {name: float(value) for name, value in record.metrics.items()},
                record.is_abstention,
            )
        )
    return QAMethodSummary(
        method=method,
        processed_questions=len(records),
        answerable_questions=sum(not record.is_abstention for record in records),
        abstention_questions=sum(record.is_abstention for record in records),
        metrics=aggregate_qa_metrics(
            [
                (
                    {name: float(value) for name, value in metrics.items()},
                    is_abstention,
                )
                for metrics, is_abstention in rows
            ]
        ),
        by_question_type={
            question_type: aggregate_qa_metrics(question_rows)
            for question_type, question_rows in sorted(by_type.items())
        },
        answerable_refusal_rate=_mean(
            [float(is_abstention_answer(record.prediction)) for record in answerable_records]
        ),
        answerable_fact_coverage_rate=_mean(
            [float(record.evidence_fact_count > 0) for record in answerable_records]
        ),
        answerable_evidence_coverage_rate=_mean(
            [
                float(
                    record.evidence_fact_count > 0
                    or record.evidence_window_count > 0
                )
                for record in answerable_records
            ]
        ),
        mean_evidence_fact_count=_mean(
            [record.evidence_fact_count for record in records]
        ),
        mean_evidence_window_count=_mean(
            [record.evidence_window_count for record in records]
        ),
        mean_evidence_span_count=_mean(
            [record.evidence_span_count for record in records]
        ),
        mean_evidence_window_chars=_mean(
            [record.evidence_window_chars for record in records]
        ),
        mean_retrieved_tokens=_mean([record.retrieved_tokens for record in records]),
        mean_reader_input_tokens=_mean(
            [record.reader_input_tokens for record in records]
        ),
        mean_reader_output_tokens=_mean(
            [record.reader_output_tokens for record in records]
        ),
        mean_ingest_latency_ms=_mean(
            [record.ingest_latency_ms for record in records]
        ),
        mean_retrieval_latency_ms=_mean(
            [record.retrieval_latency_ms for record in records]
        ),
        mean_reader_latency_ms=_mean(
            [record.reader_latency_ms for record in records]
        ),
        mean_end_to_end_latency_ms=_mean(
            [record.end_to_end_latency_ms for record in records]
        ),
    )


def _answer_one(
    method: str,
    *,
    retrieval_record: RetrievalSampleRecord,
    reader: LongMemEvalReader,
    top_k: int,
) -> QASampleRecord:
    memories = retrieval_record.retrieved_memories[:top_k]
    evidence_profiles = _selected_evidence_profiles(
        retrieval_record,
        memories=memories,
    )
    started_at = perf_counter()
    output = reader.answer(
        question=retrieval_record.question,
        question_date=retrieval_record.question_date,
        memories=memories,
        evidence_profiles=evidence_profiles,
    )
    reader_latency_ms = (perf_counter() - started_at) * 1000.0
    ingest_latency_ms = _adapter_stat(
        retrieval_record,
        "total_ingest_latency_ms",
    )
    retrieval_latency_ms = _adapter_stat(
        retrieval_record,
        "total_retrieval_latency_ms",
    )
    return QASampleRecord(
        question_id=retrieval_record.question_id,
        question_type=retrieval_record.question_type,
        method=method,
        question=retrieval_record.question,
        gold_answer=retrieval_record.answer,
        prediction=output.answer,
        is_abstention=retrieval_record.is_abstention,
        metrics=compute_qa_metrics(
            output.answer,
            retrieval_record.answer,
            is_abstention=retrieval_record.is_abstention,
        ),
        evidence_memory_ids=[memory.memory_id for memory in memories],
        evidence_session_ids=[
            memory.source_session_id
            for memory in memories
            if memory.source_session_id is not None
        ],
        reader_provider=output.provider,
        reader_model=output.model,
        reader_finish_reason=output.finish_reason,
        reader_prompt_version=output.prompt_version,
        reader_evidence_mode=output.evidence_mode,
        prompt_sha256=hashlib.sha256(
            (output.system_prompt + "\n" + output.prompt).encode("utf-8")
        ).hexdigest(),
        evidence_profile_count=output.evidence_profile_count,
        evidence_fact_count=output.evidence_fact_count,
        evidence_window_count=output.evidence_window_count,
        evidence_span_count=output.evidence_span_count,
        evidence_window_chars=output.evidence_window_chars,
        retrieved_tokens=sum(memory.token_count for memory in memories),
        reader_input_tokens=output.input_tokens,
        reader_output_tokens=output.output_tokens,
        ingest_latency_ms=ingest_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        reader_latency_ms=reader_latency_ms,
        end_to_end_latency_ms=(
            ingest_latency_ms + retrieval_latency_ms + reader_latency_ms
        ),
        reader_usage=output.usage,
    )


def _prepare_manifest(
    path: Path,
    *,
    signature: dict[str, JsonValue],
    resume: bool,
    retrieval_manifest_sha256: str,
) -> dict[str, JsonValue]:
    existing: dict[str, JsonValue] | None = None
    if path.exists():
        existing = _read_json_object(path)
        if not resume:
            raise FileExistsError(
                f"QA manifest already exists: {path}. Use --resume to continue."
            )
        if existing.get("signature") != signature:
            raise ValueError("Existing QA manifest does not match the requested configuration")
    elif any(path.parent.glob("*.jsonl")):
        raise FileExistsError(
            f"QA records exist without a manifest in {path.parent}; "
            "move them aside before starting a new run."
        )
    now = datetime.now(UTC).isoformat()
    manifest: dict[str, JsonValue] = (
        {
            **existing,
            "status": "running",
            "resumed_at": now,
            "finished_at": None,
            "error_type": None,
            "error": None,
        }
        if existing is not None
        else {
            "schema_version": "2.0",
            "status": "running",
            "signature": signature,
            "retrieval_manifest_sha256": retrieval_manifest_sha256,
            "started_at": now,
        }
    )
    _write_json(path, manifest)
    return manifest


def _qa_signature(
    config: LongMemEvalQARunConfig,
    *,
    reader: LongMemEvalReader,
    methods: list[str],
) -> dict[str, JsonValue]:
    method_values: list[JsonValue] = list(methods)
    protocol: dict[str, JsonValue] = {
        "prompt_version": reader.config.prompt_version,
        "evidence_mode": reader.config.evidence_mode,
        "gold_answers_visible_to_reader": False,
    }
    if reader.config.evidence_mode == "reranker_facts_with_query_windows":
        protocol["query_windows"] = {
            "version": reader.config.query_window_version,
            "max_memories": reader.config.query_window_max_memories,
            "max_spans_per_memory": (
                reader.config.query_window_max_spans_per_memory
            ),
            "radius": reader.config.query_window_radius,
            "max_chars_per_memory": (
                reader.config.query_window_max_chars_per_memory
            ),
            "total_max_chars": reader.config.query_window_total_max_chars,
        }
    return {
        "retrieval_run": str(config.retrieval_run.expanduser().resolve()),
        "methods": method_values,
        "top_k": config.top_k,
        "limit": config.limit,
        "qa_subdir": config.qa_subdir,
        "generation": reader.config.generation.model_dump(mode="json"),
        "protocol": protocol,
        "reader": config.reader_metadata,
    }


def _selected_evidence_profiles(
    record: RetrievalSampleRecord,
    *,
    memories: list[RetrievedMemory],
) -> list[CandidateEvidenceProfile]:
    """Load validated V6 fact profiles for only the selected Top-k sessions."""

    raw_profiles = record.rerank_metadata.get("selector_evidence_selections", [])
    if not isinstance(raw_profiles, list):
        raise ValueError(
            "selector_evidence_selections must be a list for "
            f"question {record.question_id}"
        )
    selected_session_ids = {
        memory.source_session_id
        for memory in memories
        if memory.source_session_id is not None
    }
    profiles: list[CandidateEvidenceProfile] = []
    observed_sessions: set[str] = set()
    for index, value in enumerate(raw_profiles, start=1):
        try:
            profile = CandidateEvidenceProfile.model_validate(value)
        except ValidationError as exc:
            raise ValueError(
                "Invalid selector evidence profile "
                f"{index} for question {record.question_id}: {exc}"
            ) from exc
        if profile.session_id not in selected_session_ids:
            continue
        if profile.session_id in observed_sessions:
            raise ValueError(
                "Duplicate selector evidence profile for session "
                f"{profile.session_id} in question {record.question_id}"
            )
        profiles.append(profile)
        observed_sessions.add(profile.session_id)
    return profiles


def _preflight_evidence_profiles(
    records: list[RetrievalSampleRecord],
    *,
    top_k: int,
) -> tuple[int, int]:
    """Validate every saved profile before the first vLLM answer call."""

    questions_with_facts = 0
    fact_count = 0
    for record in records:
        profiles = _selected_evidence_profiles(
            record,
            memories=record.retrieved_memories[:top_k],
        )
        observed = sum(len(profile.facts) for profile in profiles)
        if observed:
            questions_with_facts += 1
            fact_count += observed
    return questions_with_facts, fact_count


def _preflight_query_windows(
    records: list[RetrievalSampleRecord],
    *,
    reader: LongMemEvalReader,
    top_k: int,
) -> tuple[int, int, int, int]:
    """Build every deterministic window before the first vLLM answer call."""

    questions_with_windows = 0
    window_count = 0
    span_count = 0
    window_chars = 0
    for record in records:
        windows = build_query_evidence_windows(
            question=record.question,
            memories=record.retrieved_memories[:top_k],
            max_memories=reader.config.query_window_max_memories,
            max_spans_per_memory=reader.config.query_window_max_spans_per_memory,
            radius=reader.config.query_window_radius,
            max_chars_per_memory=reader.config.query_window_max_chars_per_memory,
            total_max_chars=reader.config.query_window_total_max_chars,
        )
        if windows:
            questions_with_windows += 1
        window_count += len(windows)
        span_count += sum(len(window.spans) for window in windows)
        window_chars += sum(window.char_count for window in windows)
    return questions_with_windows, window_count, span_count, window_chars


def _resolve_methods(retrieval_run: Path, requested: list[str]) -> list[str]:
    if requested:
        candidates = [_normalize_method(method) for method in requested]
    else:
        candidates = sorted(
            path.parent.name
            for path in retrieval_run.glob("*/retrieval.jsonl")
        )
    methods: list[str] = []
    seen: set[str] = set()
    for method in candidates:
        if method in seen:
            continue
        retrieval_path = retrieval_run / method / "retrieval.jsonl"
        if not retrieval_path.exists():
            raise FileNotFoundError(retrieval_path)
        methods.append(method)
        seen.add(method)
    return methods


def _load_retrieval_records(
    path: Path,
    *,
    limit: int | None,
) -> list[RetrievalSampleRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[RetrievalSampleRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if limit is not None and len(records) >= limit:
                break
            records.append(RetrievalSampleRecord.model_validate_json(line))
    return records


def _load_qa_records(path: Path) -> list[QASampleRecord]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [
            QASampleRecord.model_validate_json(line)
            for line in stream
            if line.strip()
        ]


def _order_records(
    records: list[QASampleRecord],
    retrieval_records: list[RetrievalSampleRecord],
) -> list[QASampleRecord]:
    by_id = {record.question_id: record for record in records}
    return [
        by_id[retrieval.question_id]
        for retrieval in retrieval_records
        if retrieval.question_id in by_id
    ]


def _write_hypotheses(path: Path, records: list[QASampleRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    {
                        "question_id": record.question_id,
                        "hypothesis": record.prediction,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")


def _adapter_stat(record: RetrievalSampleRecord, name: str) -> float:
    value = record.adapter_stats.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _validate_one_reader(observed: set[tuple[str, str]]) -> None:
    if len(observed) > 1:
        raise ValueError(
            "QA records contain multiple reader provider/model pairs: "
            f"{sorted(observed)}"
        )


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_method(method: str) -> str:
    return method.strip().casefold().replace("-", "_")


def _mean(values: list[int | float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0
