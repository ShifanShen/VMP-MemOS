"""Reproducible session-retrieval runner for LongMemEval."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

from pydantic import Field, JsonValue

from vmp_memos.embeddings import BaseEmbedder
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
from vmp_memos.longmemeval.converter import sample_to_events, sample_to_session_events
from vmp_memos.longmemeval.loader import load_longmemeval
from vmp_memos.longmemeval.schema import LongMemEvalRunConfig, LongMemEvalSample
from vmp_memos.longmemeval.splits import (
    LongMemEvalSplitManifest,
    load_split_samples,
    sha256_file,
)
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
) -> RetrievalRunResult:
    """Run every configured method and write replayable JSON/JSONL artifacts."""

    methods = _unique_methods(config.methods)
    if not methods:
        raise ValueError("at least one retrieval method is required")

    resolved_run_id = _safe_component(run_id or _default_run_id())
    run_dir = config.output_dir / "runs" / resolved_run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose a new --run-id."
        )
    started_at = datetime.now(UTC)
    wall_started = perf_counter()
    samples, split_manifest = _load_run_samples(config)
    LOGGER.info(
        "Loaded %d retrieval samples for %d methods: %s",
        len(samples),
        len(methods),
        ",".join(methods),
    )
    _validate_vmp_tuned_provenance(
        config,
        split_manifest=split_manifest,
        embedder=embedder,
    )
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest = _build_manifest(
        config,
        run_id=resolved_run_id,
        methods=methods,
        embedder=embedder,
        framework_runtime=framework_runtime,
        sample_count=len(samples),
        started_at=started_at,
        split_manifest=split_manifest,
    )
    _write_json(manifest_path, manifest)

    summaries: dict[str, RetrievalMethodSummary] = {}
    try:
        for method in methods:
            method_started = perf_counter()
            LOGGER.info("Method %s started (%d samples).", method, len(samples))
            records = _run_method(
                method,
                samples=samples,
                config=config,
                embedder=embedder,
                framework_runtime=framework_runtime,
                run_dir=run_dir,
            )
            method_dir = run_dir / method
            _write_jsonl(method_dir / "retrieval.jsonl", records)
            summary = summarize_method(method, records)
            summaries[method] = summary
            _write_json(method_dir / "summary.json", summary.model_dump(mode="json"))
            LOGGER.info(
                "Method %s completed in %.1fs: evaluated=%d skipped=%d recall@5=%.4f",
                method,
                perf_counter() - method_started,
                summary.evaluated_questions,
                summary.skipped_questions,
                float(summary.metrics.get("recall_at_5", 0.0)),
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
            "wall_duration_seconds": perf_counter() - wall_started,
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
    )


def _run_method(
    method: str,
    *,
    samples: list[LongMemEvalSample],
    config: LongMemEvalRunConfig,
    embedder: BaseEmbedder | None,
    framework_runtime: FrameworkRuntimeConfig | None,
    run_dir: Path,
) -> list[RetrievalSampleRecord]:
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
    )
    method_dir = run_dir / method
    workspace_root = method_dir / "workspaces"
    records: list[RetrievalSampleRecord] = []
    method_started = perf_counter()
    sample_count = len(samples)
    try:
        for sample_index, sample in enumerate(samples, start=1):
            if sample_index == 1 or sample_index % 10 == 0 or sample_index == sample_count:
                LOGGER.info(
                    "Method %s progress %d/%d: question_id=%s elapsed=%.1fs",
                    method,
                    sample_index,
                    sample_count,
                    sample.question_id,
                    perf_counter() - method_started,
                )
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
            records.append(
                _sample_record(
                    sample,
                    method=method,
                    retrieved=retrieved,
                    adapter_stats=adapter.stats(),
                    skip_abstention=config.skip_abstention_for_retrieval,
                )
            )
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
        "schema_version": "1.0",
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
        "embedding_identifier": embedder.identifier if embedder else None,
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


def _validate_vmp_tuned_provenance(
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
    if not uses_vmp_tuned:
        return
    if split_manifest is None or config.split_name is None:
        raise ValueError("vmp_tuned evaluation requires a checked split manifest")
    if config.vmp_tuned_model_path is None:
        raise ValueError("vmp_tuned_model_path is required")

    from vmp_memos.frameworks.vmp_tuned import VMPTunedModel

    model = VMPTunedModel.load(config.vmp_tuned_model_path)
    if model.dataset_sha256 != split_manifest.dataset_sha256:
        raise ValueError("VMP-Tuned model dataset SHA-256 differs from split manifest")
    if model.split_id != split_manifest.split_id:
        raise ValueError("VMP-Tuned model and evaluation split manifest differ")
    if model.split_manifest_sha256 != sha256_file(config.split_manifest_path):
        raise ValueError("VMP-Tuned model split-manifest SHA-256 differs")
    if config.split_name == model.training_split:
        raise ValueError(
            "Refusing to report VMP-Tuned on its training split; use --split test"
        )
    actual_embedding = embedder.identifier if embedder else None
    if model.embedding_identifier != actual_embedding:
        raise ValueError(
            "VMP-Tuned embedding differs from evaluation embedding: "
            f"expected {model.embedding_identifier!r}, got {actual_embedding!r}"
        )
