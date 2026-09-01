"""Reproducible session-retrieval runner for LongMemEval."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

from pydantic import Field, JsonValue

from vmp_memos.embeddings import BaseEmbedder, CachedEmbedder
from vmp_memos.evaluation import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
    ranked_unique_session_ids,
)
from vmp_memos.frameworks import (
    FrameworkRuntimeConfig,
    RetrievedMemory,
    adapter_for_name,
)
from vmp_memos.longmemeval.converter import (
    sample_to_events,
    sample_to_session_events,
    session_to_text,
)
from vmp_memos.longmemeval.loader import load_longmemeval
from vmp_memos.longmemeval.schema import LongMemEvalRunConfig, LongMemEvalSample
from vmp_memos.longmemeval.splits import (
    LongMemEvalSplitManifest,
    load_split_samples,
    sha256_file,
    split_assignment_sha256,
)
from vmp_memos.longmemeval.validation import validate_longmemeval_dates
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
)

_SAFE_PATH_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
LOGGER = logging.getLogger(__name__)


class RetrievalSampleRecord(SchemaModel):
    """One method's retrieval output for one LongMemEval question."""

    question_id: NonEmptyStr
    question_type: NonEmptyStr
    question: NonEmptyStr
    answer: NonEmptyStr | list[NonEmptyStr]
    question_date: str | None = None
    method: NonEmptyStr
    is_abstention: bool
    evaluation_skipped: bool = False
    skip_reason: str | None = None
    gold_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    retrieved_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    retrieved_memories: list[RetrievedMemory] = Field(default_factory=list)
    metrics: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    retrieved_tokens: NonNegativeInt = 0
    adapter_stats: dict[str, JsonValue] = Field(default_factory=dict)
    rerank_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RetrievalMethodSummary(SchemaModel):
    """Aggregate metrics and costs for one retrieval method."""

    method: NonEmptyStr
    processed_questions: NonNegativeInt
    evaluated_questions: NonNegativeInt
    skipped_questions: NonNegativeInt
    metrics: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    by_question_type: dict[str, dict[str, NonNegativeFloat]] = Field(default_factory=dict)
    mean_retrieved_tokens: NonNegativeFloat = 0.0
    mean_memory_count: NonNegativeFloat = 0.0
    mean_memory_tokens: NonNegativeFloat = 0.0
    mean_storage_size_bytes: NonNegativeFloat = 0.0
    mean_ingest_latency_ms: NonNegativeFloat = 0.0
    mean_retrieval_latency_ms: NonNegativeFloat = 0.0
    embedding_cache_requests: NonNegativeInt = 0
    embedding_cache_hits: NonNegativeInt = 0
    embedding_cache_misses: NonNegativeInt = 0
    embedding_cache_generated: NonNegativeInt = 0
    embedding_cache_hit_rate: NonNegativeFloat = 0.0


class RetrievalRunResult(SchemaModel):
    """Paths and summaries produced by a complete retrieval run."""

    run_id: NonEmptyStr
    run_dir: Path
    manifest_path: Path
    summaries: dict[str, RetrievalMethodSummary]


def run_longmemeval_retrieval(
    config: LongMemEvalRunConfig,
    *,
    embedder: BaseEmbedder | None = None,
    framework_runtime: FrameworkRuntimeConfig | None = None,
    run_id: str | None = None,
    resume: bool = False,
) -> RetrievalRunResult:
    """Run every configured method and write replayable JSON/JSONL artifacts."""

    methods = _unique_methods(config.methods)
    if not methods:
        raise ValueError("at least one retrieval method is required")

    resolved_run_id = _safe_component(run_id or _default_run_id())
    run_dir = config.output_dir / "runs" / resolved_run_id
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose a new --run-id "
            "or use --resume."
        )
    started_at = datetime.now(UTC)
    wall_started = perf_counter()
    samples, split_manifest = _load_run_samples(config)
    date_validation = validate_longmemeval_dates(samples)
    LOGGER.info(
        "Loaded %d retrieval samples for %d methods: %s",
        len(samples),
        len(methods),
        ",".join(methods),
    )
    _validate_vmp_provenance(
        config,
        split_manifest=split_manifest,
        embedder=embedder,
    )
    run_dir.mkdir(parents=True, exist_ok=resume)
    manifest_path = run_dir / "manifest.json"
    expected_manifest = _build_manifest(
        config,
        run_id=resolved_run_id,
        methods=methods,
        embedder=embedder,
        framework_runtime=framework_runtime,
        sample_count=len(samples),
        started_at=started_at,
        split_manifest=split_manifest,
    )
    expected_manifest["resume_signature"] = _resume_signature(expected_manifest)
    previous_wall_duration = 0.0
    if resume:
        manifest = _load_resume_manifest(manifest_path, expected_manifest)
        previous_wall_duration = _json_number(manifest.get("wall_duration_seconds"))
        resume_history = manifest.get("resume_history", [])
        if not isinstance(resume_history, list):
            resume_history = []
        resume_history.append(started_at.isoformat())
        manifest.update(
            {
                "status": "running",
                "resume_history": resume_history,
                "date_validation": date_validation,
            }
        )
        for key in ("finished_at", "error_type", "error"):
            manifest.pop(key, None)
        LOGGER.info(
            "Resuming retrieval run %s from %s.",
            resolved_run_id,
            run_dir,
        )
    else:
        manifest = expected_manifest
        manifest["date_validation"] = date_validation
    _write_json(manifest_path, manifest)

    summaries: dict[str, RetrievalMethodSummary] = {}
    try:
        if not (resume and isinstance(manifest.get("embedding_prewarm"), dict)):
            manifest["embedding_prewarm"] = _prewarm_embeddings(
                samples,
                embedder=embedder,
                ingestion_granularity=config.ingestion_granularity,
                methods=methods,
                enabled=config.prewarm_embeddings,
            )
        else:
            LOGGER.info("Reusing completed embedding-prewarm receipt.")
        _write_json(manifest_path, manifest)
        for method in methods:
            method_started = perf_counter()
            method_dir = run_dir / method
            records_path = method_dir / "retrieval.jsonl"
            existing_records = (
                _load_resume_records(records_path, method=method, samples=samples)
                if resume
                else []
            )
            LOGGER.info(
                "Method %s started: total=%d existing=%d pending=%d.",
                method,
                len(samples),
                len(existing_records),
                len(samples) - len(existing_records),
            )
            records = _run_method(
                method,
                samples=samples,
                config=config,
                embedder=embedder,
                framework_runtime=framework_runtime,
                run_dir=run_dir,
                records_path=records_path,
                existing_records=existing_records,
            )
            summary = summarize_method(method, records)
            summaries[method] = summary
            _write_json(method_dir / "summary.json", summary.model_dump(mode="json"))
            LOGGER.info(
                "Method %s completed in %.1fs: evaluated=%d skipped=%d recall_all@5=%.4f",
                method,
                perf_counter() - method_started,
                summary.evaluated_questions,
                summary.skipped_questions,
                float(summary.metrics.get("recall_all@5", 0.0)),
            )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "wall_duration_seconds": (
                    previous_wall_duration + perf_counter() - wall_started
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(manifest_path, manifest)
        raise

    finished_at = datetime.now(UTC)
    combined_summary = {
        "run_id": resolved_run_id,
        "methods": {
            name: summary.model_dump(mode="json") for name, summary in summaries.items()
        },
    }
    _write_json(run_dir / "summary.json", combined_summary)
    manifest.update(
        {
            "status": "completed",
            "finished_at": finished_at.isoformat(),
            "wall_duration_seconds": (
                previous_wall_duration + perf_counter() - wall_started
            ),
        }
    )
    _write_json(manifest_path, manifest)
    return RetrievalRunResult(
        run_id=resolved_run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        summaries=summaries,
    )


def summarize_method(
    method: str,
    records: list[RetrievalSampleRecord],
) -> RetrievalMethodSummary:
    """Aggregate one method's retrieval records."""

    evaluated = [record for record in records if not record.evaluation_skipped]
    by_type_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in evaluated:
        by_type_rows[record.question_type].append(
            {name: float(value) for name, value in record.metrics.items()}
        )
    cache_requests = sum(
        int(_numeric_stat(record, "embedding_cache_requests"))
        for record in records
    )
    cache_hits = sum(
        int(_numeric_stat(record, "embedding_cache_hits"))
        for record in records
    )
    cache_misses = sum(
        int(_numeric_stat(record, "embedding_cache_misses"))
        for record in records
    )
    cache_generated = sum(
        int(_numeric_stat(record, "embedding_cache_generated"))
        for record in records
    )
    return RetrievalMethodSummary(
        method=method,
        processed_questions=len(records),
        evaluated_questions=len(evaluated),
        skipped_questions=len(records) - len(evaluated),
        metrics=aggregate_retrieval_metrics(
            [{name: float(value) for name, value in record.metrics.items()} for record in evaluated]
        ),
        by_question_type={
            question_type: aggregate_retrieval_metrics(rows)
            for question_type, rows in sorted(by_type_rows.items())
        },
        mean_retrieved_tokens=_mean([record.retrieved_tokens for record in records]),
        mean_memory_count=_mean(
            [_numeric_stat(record, "memory_count") for record in records]
        ),
        mean_memory_tokens=_mean(
            [_numeric_stat(record, "total_tokens") for record in records]
        ),
        mean_storage_size_bytes=_mean(
            [_numeric_stat(record, "storage_size_bytes") for record in records]
        ),
        mean_ingest_latency_ms=_mean(
            [_numeric_stat(record, "total_ingest_latency_ms") for record in records]
        ),
        mean_retrieval_latency_ms=_mean(
            [_numeric_stat(record, "total_retrieval_latency_ms") for record in records]
        ),
        embedding_cache_requests=cache_requests,
        embedding_cache_hits=cache_hits,
        embedding_cache_misses=cache_misses,
        embedding_cache_generated=cache_generated,
        embedding_cache_hit_rate=(
            cache_hits / cache_requests if cache_requests else 0.0
        ),
    )


def _run_method(
    method: str,
    *,
    samples: list[LongMemEvalSample],
    config: LongMemEvalRunConfig,
    embedder: BaseEmbedder | None,
    framework_runtime: FrameworkRuntimeConfig | None,
    run_dir: Path,
    records_path: Path,
    existing_records: list[RetrievalSampleRecord],
) -> list[RetrievalSampleRecord]:
    records = list(existing_records)
    completed_count = len(records)
    if completed_count == len(samples):
        LOGGER.info("Method %s already has all %d records.", method, len(samples))
        return records
    if method in {
        "mem0",
        "mem0_official",
        "letta",
        "letta_official",
    } and embedder is not None:
        embedder.release()
    adapter = adapter_for_name(
        method,
        embedder=embedder,
        runtime=framework_runtime,
        vmp_tuned_model_path=(
            str(config.vmp_tuned_model_path)
            if config.vmp_tuned_model_path is not None
            else None
        ),
        vmp_hierarchical_model_path=(
            str(config.vmp_hierarchical_model_path)
            if config.vmp_hierarchical_model_path is not None
            else None
        ),
    )
    method_dir = run_dir / method
    workspace_root = method_dir / "workspaces"
    method_started = perf_counter()
    sample_count = len(samples)
    try:
        for sample_index, sample in enumerate(
            samples[completed_count:],
            start=completed_count + 1,
        ):
            if sample_index == 1 or sample_index % 10 == 0 or sample_index == sample_count:
                LOGGER.info(
                    "Method %s progress %d/%d: question_id=%s elapsed=%.1fs",
                    method,
                    sample_index,
                    sample_count,
                    sample.question_id,
                    perf_counter() - method_started,
                )
            cache_before = _cache_stats(embedder)
            adapter.reset(workspace_root / _safe_component(sample.question_id))
            if config.ingestion_granularity == "session":
                for events in sample_to_session_events(sample):
                    adapter.ingest_session(events)
            else:
                for event in sample_to_events(sample):
                    adapter.ingest_event(event)
            adapter.finalize_ingestion()
            retrieved = adapter.retrieve(
                sample.question,
                top_k=config.retrieval_depth,
                question_date=sample.question_date,
                metadata={
                    "question_id": sample.question_id,
                    "question_type": sample.question_type,
                    "token_budget": _token_budget(config.metadata),
                },
            )
            adapter_stats = adapter.stats()
            adapter_stats.update(
                _cache_stats_delta(cache_before, _cache_stats(embedder))
            )
            record = _sample_record(
                sample,
                method=method,
                retrieved=retrieved,
                adapter_stats=adapter_stats,
                skip_abstention=config.skip_abstention_for_retrieval,
            )
            _append_jsonl_record(records_path, record)
            records.append(record)
    finally:
        adapter.close()
    return records


def _sample_record(
    sample: LongMemEvalSample,
    *,
    method: str,
    retrieved: list[RetrievedMemory],
    adapter_stats: dict[str, JsonValue],
    skip_abstention: bool,
) -> RetrievalSampleRecord:
    ranked_sessions = ranked_unique_session_ids(
        memory.source_session_id for memory in retrieved
    )
    skip_reason: str | None = None
    if skip_abstention and sample.is_abstention:
        skip_reason = "abstention"
    elif not sample.answer_session_ids:
        skip_reason = "missing_gold_session_ids"
    metrics = (
        {}
        if skip_reason
        else compute_retrieval_metrics(ranked_sessions, sample.answer_session_ids)
    )
    return RetrievalSampleRecord(
        question_id=sample.question_id,
        question_type=sample.question_type,
        question=sample.question,
        answer=sample.answer,
        question_date=sample.question_date,
        method=method,
        is_abstention=sample.is_abstention,
        evaluation_skipped=skip_reason is not None,
        skip_reason=skip_reason,
        gold_session_ids=list(sample.answer_session_ids),
        retrieved_session_ids=ranked_sessions,
        retrieved_memories=retrieved,
        metrics=metrics,
        retrieved_tokens=sum(memory.token_count for memory in retrieved),
        adapter_stats=adapter_stats,
    )


def _build_manifest(
    config: LongMemEvalRunConfig,
    *,
    run_id: str,
    methods: list[str],
    embedder: BaseEmbedder | None,
    framework_runtime: FrameworkRuntimeConfig | None,
    sample_count: int,
    started_at: datetime,
    split_manifest: LongMemEvalSplitManifest | None,
) -> dict[str, JsonValue]:
    config_payload = config.model_dump(mode="json")
    config_payload["methods"] = methods
    return {
        "schema_version": "1.1",
        "status": "running",
        "run_id": run_id,
        "dataset": "longmemeval-cleaned",
        "data_sha256": _sha256(config.data_path),
        "sample_count": sample_count,
        "split": (
            {
                "name": config.split_name,
                "split_id": split_manifest.split_id,
                "manifest_path": str(config.split_manifest_path),
                "manifest_sha256": sha256_file(config.split_manifest_path),
                "question_count": len(split_manifest.splits[str(config.split_name)]),
            }
            if split_manifest is not None and config.split_manifest_path is not None
            else None
        ),
        "vmp_tuned_model": (
            {
                "path": str(config.vmp_tuned_model_path),
                "sha256": sha256_file(config.vmp_tuned_model_path),
            }
            if config.vmp_tuned_model_path is not None
            else None
        ),
        "vmp_hierarchical_model": (
            {
                "path": str(config.vmp_hierarchical_model_path),
                "sha256": sha256_file(
                    config.vmp_hierarchical_model_path
                ),
            }
            if config.vmp_hierarchical_model_path is not None
            else None
        ),
        "embedding_identifier": embedder.identifier if embedder else None,
        "embedding_runtime": _embedding_runtime_metadata(embedder),
        "official_framework_runtime": (
            framework_runtime.public_metadata()
            if framework_runtime is not None
            else None
        ),
        "official_framework_versions": _official_framework_versions(methods),
        "config": config_payload,
        "started_at": started_at.isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[RetrievalSampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(record.model_dump_json())
            stream.write("\n")


def _append_jsonl_record(path: Path, record: RetrievalSampleRecord) -> None:
    """Durably checkpoint one completed question before continuing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(record.model_dump_json())
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_resume_records(
    path: Path,
    *,
    method: str,
    samples: list[LongMemEvalSample],
) -> list[RetrievalSampleRecord]:
    """Load a valid ordered prefix, repairing only a torn final JSONL line."""

    if not path.exists():
        return []
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    nonempty = [(index, line) for index, line in enumerate(lines) if line.strip()]
    records: list[RetrievalSampleRecord] = []
    repaired = bool(raw_text and not raw_text.endswith("\n"))
    for position, (line_index, line) in enumerate(nonempty):
        try:
            record = RetrievalSampleRecord.model_validate_json(line)
        except Exception as exc:
            if position == len(nonempty) - 1:
                LOGGER.warning(
                    "Discarding torn final JSONL record from %s (line %d).",
                    path,
                    line_index + 1,
                )
                repaired = True
                break
            raise ValueError(
                f"Cannot resume: malformed JSONL record in {path} "
                f"at line {line_index + 1}"
            ) from exc
        records.append(record)
    if len(records) > len(samples):
        raise ValueError(
            f"Cannot resume {method}: {len(records)} records exceed "
            f"the expected {len(samples)} samples"
        )
    expected_ids = [sample.question_id for sample in samples[: len(records)]]
    observed_ids = [record.question_id for record in records]
    if observed_ids != expected_ids:
        raise ValueError(
            f"Cannot resume {method}: existing records are not the exact "
            "ordered prefix of the configured samples"
        )
    if any(record.method != method for record in records):
        raise ValueError(
            f"Cannot resume {method}: an existing record has a different method"
        )
    if repaired:
        _write_jsonl(path, records)
    return records


def _resume_signature(manifest: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Hash immutable inputs while excluding timestamps and cache counters."""

    embedding_runtime = manifest.get("embedding_runtime")
    if isinstance(embedding_runtime, dict):
        embedding_runtime = {
            key: value
            for key, value in embedding_runtime.items()
            if key != "cache_entries_at_start"
        }
    payload = {
        key: manifest.get(key)
        for key in (
            "schema_version",
            "run_id",
            "dataset",
            "data_sha256",
            "sample_count",
            "split",
            "vmp_tuned_model",
            "vmp_hierarchical_model",
            "embedding_identifier",
            "official_framework_runtime",
            "official_framework_versions",
            "config",
        )
    }
    payload["embedding_runtime"] = embedding_runtime
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "version": "retrieval-resume-v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _load_resume_manifest(
    path: Path,
    expected: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot resume retrieval run: manifest is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot resume retrieval run: manifest is unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Cannot resume retrieval run: manifest must be an object")
    observed = payload.get("resume_signature")
    wanted = expected.get("resume_signature")
    if observed != wanted:
        raise ValueError(
            "Cannot resume retrieval run: immutable inputs differ from the "
            "existing manifest (or it predates resumable schema 1.1)"
        )
    return payload


def _json_number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_stat(record: RetrievalSampleRecord, name: str) -> float:
    value = record.adapter_stats.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _token_budget(metadata: dict[str, JsonValue]) -> int:
    value = metadata.get("token_budget", 2048)
    return value if isinstance(value, int) and value > 0 else 2048


def _official_framework_versions(methods: list[str]) -> dict[str, JsonValue]:
    distributions = {
        "mem0": "mem0ai",
        "mem0_official": "mem0ai",
        "langmem": "langmem",
        "langmem_official": "langmem",
        "graphiti": "graphiti-core",
        "graphiti_official": "graphiti-core",
        "letta": "letta-client",
        "letta_official": "letta-client",
    }
    versions: dict[str, JsonValue] = {}
    for method in methods:
        distribution = distributions.get(method)
        if distribution is None:
            continue
        try:
            versions[method] = version(distribution)
        except PackageNotFoundError:
            versions[method] = None
    return versions


def _mean(values: list[int | float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _unique_methods(methods: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for method in methods:
        value = method.strip().casefold().replace("-", "_")
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def _safe_component(value: str) -> str:
    normalized = _SAFE_PATH_PATTERN.sub("_", value.strip()).strip("._")
    return normalized or "run"


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("retrieval_%Y%m%dT%H%M%SZ")


def _load_run_samples(
    config: LongMemEvalRunConfig,
) -> tuple[list[LongMemEvalSample], LongMemEvalSplitManifest | None]:
    if config.split_manifest_path is None or config.split_name is None:
        return load_longmemeval(config.data_path, limit=config.limit), None
    samples, manifest = load_split_samples(
        config.data_path,
        config.split_manifest_path,
        config.split_name,
    )
    if config.limit is not None:
        samples = samples[: config.limit]
    return samples, manifest


def _validate_vmp_provenance(
    config: LongMemEvalRunConfig,
    *,
    split_manifest: LongMemEvalSplitManifest | None,
    embedder: BaseEmbedder | None,
) -> None:
    normalized_methods = {
        method.casefold().replace("-", "_") for method in config.methods
    }
    uses_vmp_tuned = any(
        method == "vmp_full" or method.startswith("vmp_tuned")
        for method in normalized_methods
    )
    uses_vmp_hierarchical = bool(
        normalized_methods & {"vmp_hierarchical", "vmp_v5"}
    )
    if not uses_vmp_tuned and not uses_vmp_hierarchical:
        return
    if (
        split_manifest is None
        or config.split_name is None
        or config.split_manifest_path is None
    ):
        raise ValueError("VMP evaluation requires a checked split manifest")
    actual_embedding = embedder.identifier if embedder else None
    manifest_sha256 = sha256_file(config.split_manifest_path)
    assignment_sha256 = split_assignment_sha256(split_manifest)
    if uses_vmp_tuned:
        if config.vmp_tuned_model_path is None:
            raise ValueError("vmp_tuned_model_path is required")
        from vmp_memos.frameworks.vmp_tuned import VMPTunedModel

        model = VMPTunedModel.load(config.vmp_tuned_model_path)
        _validate_frozen_vmp_model(
            model_name="VMP-Tuned",
            dataset_sha256=model.dataset_sha256,
            split_id=model.split_id,
            split_manifest_sha256=model.split_manifest_sha256,
            split_assignment_sha256=None,
            training_split=model.training_split,
            embedding_identifier=model.embedding_identifier,
            expected_dataset_sha256=split_manifest.dataset_sha256,
            expected_split_id=split_manifest.split_id,
            expected_manifest_sha256=manifest_sha256,
            expected_assignment_sha256=assignment_sha256,
            evaluation_split=config.split_name,
            allow_training_split=config.allow_dev_model_selection,
            actual_embedding_identifier=actual_embedding,
        )
    if uses_vmp_hierarchical:
        if config.vmp_hierarchical_model_path is None:
            raise ValueError("vmp_hierarchical_model_path is required")
        from vmp_memos.frameworks.vmp_hierarchical import (
            VMPHierarchicalModel,
        )

        hierarchical = VMPHierarchicalModel.load(
            config.vmp_hierarchical_model_path
        )
        _validate_frozen_vmp_model(
            model_name="VMP-v5",
            dataset_sha256=hierarchical.dataset_sha256,
            split_id=hierarchical.split_id,
            split_manifest_sha256=hierarchical.split_manifest_sha256,
            split_assignment_sha256=(
                hierarchical.split_assignment_sha256
            ),
            training_split=hierarchical.training_split,
            embedding_identifier=hierarchical.embedding_identifier,
            expected_dataset_sha256=split_manifest.dataset_sha256,
            expected_split_id=split_manifest.split_id,
            expected_manifest_sha256=manifest_sha256,
            expected_assignment_sha256=assignment_sha256,
            evaluation_split=config.split_name,
            allow_training_split=config.allow_dev_model_selection,
            actual_embedding_identifier=actual_embedding,
        )


def _prewarm_embeddings(
    samples: list[LongMemEvalSample],
    *,
    embedder: BaseEmbedder | None,
    ingestion_granularity: str,
    methods: list[str],
    enabled: bool,
) -> dict[str, JsonValue]:
    if not enabled:
        return {"enabled": False, "performed": False, "reason": "disabled"}
    if not isinstance(embedder, CachedEmbedder):
        return {
            "enabled": True,
            "performed": False,
            "reason": "persistent_cache_required",
        }

    started = perf_counter()
    before_stats = embedder.cache_stats()
    entries_before = embedder.cache.count(embedder.identifier)
    total = len(samples)
    include_hierarchical_turns = bool(
        set(methods) & {"vmp_hierarchical", "vmp_v5"}
    )
    for index, sample in enumerate(samples, start=1):
        if ingestion_granularity == "session":
            texts = [
                session_to_text(events)
                for events in sample_to_session_events(sample)
            ]
            if include_hierarchical_turns:
                from vmp_memos.frameworks.vmp_hierarchical import (
                    contextual_turn_chunk,
                )

                texts.extend(
                    chunk.content
                    for event in sample_to_events(sample)
                    if (chunk := contextual_turn_chunk(event)) is not None
                )
        else:
            texts = [str(event.content) for event in sample_to_events(sample)]
        texts.append(sample.question)
        embedder.embed(list(dict.fromkeys(texts)))
        if index == 1 or index % 10 == 0 or index == total:
            LOGGER.info(
                "Embedding prewarm progress %d/%d: question_id=%s elapsed=%.1fs",
                index,
                total,
                sample.question_id,
                perf_counter() - started,
            )
    delta = _raw_cache_stats_delta(before_stats, embedder.cache_stats())
    return {
        "enabled": True,
        "performed": True,
        "duration_seconds": perf_counter() - started,
        "cache_entries_before": entries_before,
        "cache_entries_after": embedder.cache.count(embedder.identifier),
        "hierarchical_turns": include_hierarchical_turns,
        **delta,
    }


def _validate_frozen_vmp_model(
    *,
    model_name: str,
    dataset_sha256: str,
    split_id: str,
    split_manifest_sha256: str,
    split_assignment_sha256: str | None,
    training_split: str,
    embedding_identifier: str | None,
    expected_dataset_sha256: str,
    expected_split_id: str,
    expected_manifest_sha256: str,
    expected_assignment_sha256: str,
    evaluation_split: str,
    allow_training_split: bool,
    actual_embedding_identifier: str | None,
) -> None:
    if dataset_sha256 != expected_dataset_sha256:
        raise ValueError(
            f"{model_name} model dataset SHA-256 differs from split manifest"
        )
    if split_id != expected_split_id:
        raise ValueError(
            f"{model_name} model and evaluation split manifest differ"
        )
    if (
        split_assignment_sha256 is not None
        and split_assignment_sha256 != expected_assignment_sha256
    ):
        raise ValueError(
            f"{model_name} model semantic split assignment differs"
        )
    if split_manifest_sha256 != expected_manifest_sha256:
        LOGGER.warning(
            "%s split-manifest file SHA-256 differs, but the checked dataset "
            "and canonical split assignments match; accepting volatile "
            "path/timestamp differences.",
            model_name,
        )
    if evaluation_split == training_split and not allow_training_split:
        raise ValueError(
            f"Refusing to report {model_name} on its training split; "
            "use --split test"
        )
    if evaluation_split == training_split:
        LOGGER.warning(
            "%s is running on its training split for explicitly marked "
            "Dev-only model selection; this run must not be reported as Test.",
            model_name,
        )
    if embedding_identifier != actual_embedding_identifier:
        raise ValueError(
            f"{model_name} embedding differs from evaluation embedding: "
            f"expected {embedding_identifier!r}, "
            f"got {actual_embedding_identifier!r}"
        )


def _cache_stats(embedder: BaseEmbedder | None) -> dict[str, int]:
    if not isinstance(embedder, CachedEmbedder):
        return {"requests": 0, "hits": 0, "misses": 0, "generated": 0}
    return embedder.cache_stats()


def _cache_stats_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    return {
        f"embedding_cache_{name}": value
        for name, value in _raw_cache_stats_delta(before, after).items()
    }


def _raw_cache_stats_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    return {
        name: max(0, after[name] - before[name])
        for name in ("requests", "hits", "misses", "generated")
    }


def _embedding_runtime_metadata(
    embedder: BaseEmbedder | None,
) -> dict[str, JsonValue] | None:
    if embedder is None:
        return None
    base = embedder.embedder if isinstance(embedder, CachedEmbedder) else embedder
    batch_size = getattr(base, "batch_size", None)
    return {
        "identifier": embedder.identifier,
        "batch_size": batch_size if isinstance(batch_size, int) else None,
        "persistent_cache": isinstance(embedder, CachedEmbedder),
        "cache_path": (
            str(embedder.cache.path)
            if isinstance(embedder, CachedEmbedder)
            else None
        ),
        "cache_entries_at_start": (
            embedder.cache.count(embedder.identifier)
            if isinstance(embedder, CachedEmbedder)
            else None
        ),
    }
