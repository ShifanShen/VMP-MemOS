"""Resumable shared-vLLM reranking over saved LongMemEval candidates."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from pydantic import Field, JsonValue

from vmp_memos.evaluation import (
    compute_retrieval_metrics,
    ranked_unique_session_ids,
)
from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.llm import (
    LongMemEvalRerankDecision,
    LongMemEvalRerankerConfig,
    reorder_memories,
)
from vmp_memos.longmemeval.retrieval_runner import (
    RetrievalMethodSummary,
    RetrievalSampleRecord,
    summarize_method,
)
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
)

LOGGER = logging.getLogger(__name__)


class EvidenceReranker(Protocol):
    """Runtime interface implemented by the local-vLLM evidence reranker."""

    config: LongMemEvalRerankerConfig

    def rerank(
        self,
        *,
        question: str,
        question_date: str | None,
        candidates: Sequence[RetrievedMemory],
    ) -> LongMemEvalRerankDecision:
        """Return one label-free evidence ranking."""


class LongMemEvalRerankRunConfig(SchemaModel):
    """Configuration for one replayable shared-reranker run."""

    source_run: Path
    methods: list[NonEmptyStr] = Field(default_factory=list)
    output_dir: Path = Path("outputs/longmemeval")
    resume: bool = False
    limit: NonNegativeInt | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RerankMethodSummary(RetrievalMethodSummary):
    """Retrieval metrics plus local-vLLM parsing and cost statistics."""

    source_method: NonEmptyStr
    reranker_provider: str | None = None
    reranker_model: str | None = None
    prompt_version: NonEmptyStr
    candidate_count: NonNegativeInt
    min_observed_candidate_count: NonNegativeInt
    output_top_k: NonNegativeInt
    protected_top_n: NonNegativeInt
    boundary_verification: bool = False
    boundary_prompt_version: str | None = None
    boundary_protected_top_n: NonNegativeInt = 0
    boundary_calls: NonNegativeInt = 0
    boundary_skips: NonNegativeInt = 0
    boundary_parse_fallbacks: NonNegativeInt = 0
    boundary_parse_fallback_rate: NonNegativeFloat = 0.0
    boundary_invalid_session_id_count: NonNegativeInt = 0
    boundary_policy_rejections: NonNegativeInt = 0
    boundary_replacements_accepted: NonNegativeInt = 0
    mean_boundary_latency_ms: NonNegativeFloat = 0.0
    parse_fallbacks: NonNegativeInt = 0
    parse_fallback_rate: NonNegativeFloat = 0.0
    invalid_session_id_count: NonNegativeInt = 0
    recovered_questions: NonNegativeInt = 0
    regressed_questions: NonNegativeInt = 0
    stable_success_questions: NonNegativeInt = 0
    stable_failure_questions: NonNegativeInt = 0
    candidate_oracle_recoverable_questions: NonNegativeInt = 0
    mean_rerank_latency_ms: NonNegativeFloat = 0.0
    mean_reranker_input_tokens: NonNegativeFloat = 0.0
    mean_reranker_output_tokens: NonNegativeFloat = 0.0


class LongMemEvalRerankRunResult(SchemaModel):
    """Artifacts produced by one shared-reranking pass."""

    run_id: NonEmptyStr
    run_dir: Path
    manifest_path: Path
    summaries: dict[str, RerankMethodSummary]


def reranked_method_name(
    source_method: str,
    *,
    boundary_verification: bool = False,
) -> str:
    """Return the stable method name used by QA and paper tables."""

    normalized = _normalize_method(source_method)
    suffix = "vllm_boundary" if boundary_verification else "vllm_rerank"
    return f"{normalized}__{suffix}"


def run_longmemeval_rerank(
    config: LongMemEvalRerankRunConfig,
    *,
    reranker: EvidenceReranker,
    run_id: str,
) -> LongMemEvalRerankRunResult:
    """Rerank saved candidates with one shared, resumable local-vLLM layer."""

    source_run = config.source_run.expanduser().resolve()
    source_manifest_path = source_run / "manifest.json"
    source_manifest = _read_json_object(source_manifest_path)
    if source_manifest.get("status") != "completed":
        raise ValueError(f"Source retrieval run is not completed: {source_run}")
    methods = _resolve_methods(source_run, config.methods)
    if not methods:
        raise ValueError("no source retrieval methods selected for reranking")
    resolved_run_id = _safe_run_id(run_id)
    run_dir = config.output_dir.expanduser().resolve() / "runs" / resolved_run_id
    if run_dir == source_run:
        raise ValueError("rerank output run must differ from the source run")
    signature = _run_signature(
        config,
        methods=methods,
        reranker_config=reranker.config,
        source_manifest_sha256=_sha256(source_manifest_path),
    )
    manifest_path = run_dir / "manifest.json"
    manifest = _prepare_manifest(
        manifest_path,
        run_id=resolved_run_id,
        source_run=source_run,
        source_manifest=source_manifest,
        signature=signature,
        resume=config.resume,
    )
    wall_started = perf_counter()
    summaries: dict[str, RerankMethodSummary] = {}
    observed_rerankers: set[tuple[str, str]] = set()
    try:
        for source_method in methods:
            output_method = reranked_method_name(
                source_method,
                boundary_verification=reranker.config.boundary_verification,
            )
            method_started = perf_counter()
            source_records = _load_records(
                source_run / source_method / "retrieval.jsonl",
                limit=config.limit,
            )
            method_dir = run_dir / output_method
            output_path = method_dir / "retrieval.jsonl"
            existing = _load_resume_records(output_path) if config.resume else []
            existing_by_id = {record.question_id: record for record in existing}
            if len(existing_by_id) != len(existing):
                raise ValueError(f"Duplicate question_id in existing rerank file: {output_path}")
            pending = [
                record for record in source_records if record.question_id not in existing_by_id
            ]
            LOGGER.info(
                "Rerank method %s -> %s started: total=%d existing=%d pending=%d",
                source_method,
                output_method,
                len(source_records),
                len(existing),
                len(pending),
            )
            for record in existing:
                observed = _observed_reranker(record)
                if observed is not None:
                    observed_rerankers.add(observed)
            _validate_one_reranker(observed_rerankers)

            method_dir.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8", newline="\n") as stream:
                for index, source_record in enumerate(pending, start=1):
                    output_record = _rerank_record(
                        source_record,
                        output_method=output_method,
                        reranker=reranker,
                    )
                    observed = _observed_reranker(output_record)
                    if observed is not None:
                        observed_rerankers.add(observed)
                    _validate_one_reranker(observed_rerankers)
                    stream.write(output_record.model_dump_json())
                    stream.write("\n")
                    stream.flush()
                    existing.append(output_record)
                    existing_by_id[output_record.question_id] = output_record
                    if index == 1 or index % 10 == 0 or index == len(pending):
                        LOGGER.info(
                            "Rerank method %s progress %d/%d: question_id=%s "
                            "fallback=%s latency=%.1fms elapsed=%.1fs",
                            source_method,
                            index,
                            len(pending),
                            output_record.question_id,
                            output_record.rerank_metadata.get("parse_fallback"),
                            _metadata_number(output_record, "rerank_latency_ms"),
                            perf_counter() - method_started,
                        )

            ordered = _order_records(existing, source_records)
            summary = summarize_rerank_method(
                source_method,
                output_method=output_method,
                records=ordered,
                config=reranker.config,
            )
            summaries[output_method] = summary
            _write_json(
                method_dir / "summary.json",
                summary.model_dump(mode="json"),
            )
            LOGGER.info(
                "Rerank method %s completed in %.1fs: recall_all@5=%.4f "
                "recovered=%d regressed=%d fallbacks=%d/%d",
                source_method,
                perf_counter() - method_started,
                float(summary.metrics.get("recall_all@5", 0.0)),
                summary.recovered_questions,
                summary.regressed_questions,
                summary.parse_fallbacks,
                summary.processed_questions,
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
        run_dir / "summary.json",
        {
            "run_id": resolved_run_id,
            "methods": {
                method: summary.model_dump(mode="json") for method, summary in summaries.items()
            },
        },
    )
    manifest.update(
        {
            "status": "completed",
            "finished_at": datetime.now(UTC).isoformat(),
            "wall_duration_seconds": perf_counter() - wall_started,
            "observed_reranker": (
                {
                    "provider": next(iter(observed_rerankers))[0],
                    "model": next(iter(observed_rerankers))[1],
                }
                if observed_rerankers
                else None
            ),
        }
    )
    _write_json(manifest_path, manifest)
    return LongMemEvalRerankRunResult(
        run_id=resolved_run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        summaries=summaries,
    )


def summarize_rerank_method(
    source_method: str,
    *,
    output_method: str,
    records: list[RetrievalSampleRecord],
    config: LongMemEvalRerankerConfig,
) -> RerankMethodSummary:
    """Aggregate retrieval, parser, latency, and token statistics."""

    base = summarize_method(output_method, records)
    observations = {
        observed for record in records if (observed := _observed_reranker(record)) is not None
    }
    _validate_one_reranker(observations)
    provider, model = next(iter(observations)) if observations else (None, None)
    fallback_count = sum(record.rerank_metadata.get("parse_fallback") is True for record in records)
    boundary_calls = sum(_boundary_bool(record, "call_made") for record in records)
    boundary_fallbacks = sum(
        _boundary_bool(record, "parse_fallback") for record in records
    )
    payload = base.model_dump(mode="python")
    payload.update(
        {
            "source_method": _normalize_method(source_method),
            "reranker_provider": provider,
            "reranker_model": model,
            "prompt_version": config.prompt_version,
            "candidate_count": config.candidate_count,
            "min_observed_candidate_count": min(
                (int(_metadata_number(record, "candidate_count_observed")) for record in records),
                default=0,
            ),
            "output_top_k": config.output_top_k,
            "protected_top_n": config.protected_top_n,
            "boundary_verification": config.boundary_verification,
            "boundary_prompt_version": (
                config.boundary_prompt_version if config.boundary_verification else None
            ),
            "boundary_protected_top_n": (
                config.boundary_protected_top_n if config.boundary_verification else 0
            ),
            "boundary_calls": boundary_calls,
            "boundary_skips": sum(
                not _boundary_bool(record, "call_made") for record in records
            )
            if config.boundary_verification
            else 0,
            "boundary_parse_fallbacks": boundary_fallbacks,
            "boundary_parse_fallback_rate": (
                boundary_fallbacks / boundary_calls
                if boundary_calls
                else 0.0
            ),
            "boundary_invalid_session_id_count": sum(
                len(_boundary_list(record, "invalid_session_ids")) for record in records
            ),
            "boundary_policy_rejections": sum(
                _boundary_bool(record, "policy_rejected") for record in records
            ),
            "boundary_replacements_accepted": sum(
                _boundary_bool(record, "replacement_accepted") for record in records
            ),
            "mean_boundary_latency_ms": _mean(
                [_boundary_number(record, "latency_ms") for record in records]
            ),
            "parse_fallbacks": fallback_count,
            "parse_fallback_rate": fallback_count / len(records) if records else 0.0,
            "invalid_session_id_count": sum(
                len(_metadata_list(record, "invalid_session_ids")) for record in records
            ),
            "recovered_questions": sum(
                record.rerank_metadata.get("transition_vs_source") == "recovered"
                for record in records
            ),
            "regressed_questions": sum(
                record.rerank_metadata.get("transition_vs_source") == "regressed"
                for record in records
            ),
            "stable_success_questions": sum(
                record.rerank_metadata.get("transition_vs_source") == "stable_success"
                for record in records
            ),
            "stable_failure_questions": sum(
                record.rerank_metadata.get("transition_vs_source") == "stable_failure"
                for record in records
            ),
            "candidate_oracle_recoverable_questions": sum(
                record.rerank_metadata.get("candidate_oracle_recoverable") is True
                for record in records
            ),
            "mean_rerank_latency_ms": _mean(
                [_metadata_number(record, "rerank_latency_ms") for record in records]
            ),
            "mean_reranker_input_tokens": _mean(
                [_metadata_number(record, "input_tokens") for record in records]
            ),
            "mean_reranker_output_tokens": _mean(
                [_metadata_number(record, "output_tokens") for record in records]
            ),
        }
    )
    return RerankMethodSummary.model_validate(payload)


def _rerank_record(
    source: RetrievalSampleRecord,
    *,
    output_method: str,
    reranker: EvidenceReranker,
) -> RetrievalSampleRecord:
    candidates = source.retrieved_memories[: reranker.config.candidate_count]
    started_at = perf_counter()
    decision = reranker.rerank(
        question=source.question,
        question_date=source.question_date,
        candidates=candidates,
    )
    rerank_latency_ms = (perf_counter() - started_at) * 1000.0
    ranked_memories = reorder_memories(
        candidates,
        decision.ranked_session_ids,
    )
    ranked_sessions = ranked_unique_session_ids(
        memory.source_session_id or memory.memory_id for memory in ranked_memories
    )
    metrics = (
        {}
        if source.evaluation_skipped
        else compute_retrieval_metrics(ranked_sessions, source.gold_session_ids)
    )
    source_recall = _metric_value(source.metrics, "recall_all@5")
    reranked_recall = _metric_value(metrics, "recall_all@5")
    transition = _transition(
        before=source_recall == 1.0,
        after=reranked_recall == 1.0,
    )
    candidate_session_ids = {memory.source_session_id or memory.memory_id for memory in candidates}
    candidate_oracle_recoverable = (
        not source.evaluation_skipped
        and len(source.gold_session_ids) <= reranker.config.output_top_k
        and set(source.gold_session_ids).issubset(candidate_session_ids)
    )
    adapter_stats = dict(source.adapter_stats)
    source_retrieval_latency = _adapter_number(
        source,
        "total_retrieval_latency_ms",
    )
    adapter_stats.update(
        {
            "source_method": source.method,
            "source_retrieval_latency_ms": source_retrieval_latency,
            "rerank_calls": 1 + int(
                decision.boundary is not None and decision.boundary.call_made
            ),
            "total_rerank_latency_ms": rerank_latency_ms,
            "total_retrieval_latency_ms": source_retrieval_latency + rerank_latency_ms,
            "reranker_provider": decision.provider,
            "reranker_model": decision.model,
            "reranker_prompt_version": decision.prompt_version,
        }
    )
    rerank_metadata: dict[str, JsonValue] = {
        "schema_version": "2.0" if decision.boundary is not None else "1.0",
        "source_method": source.method,
        "prompt_version": decision.prompt_version,
        "prompt_sha256": decision.prompt_sha256,
        "provider": decision.provider,
        "model": decision.model,
        "finish_reason": decision.finish_reason,
        "candidate_count_requested": reranker.config.candidate_count,
        "candidate_count_observed": len(candidates),
        "candidate_oracle_recoverable": candidate_oracle_recoverable,
        "output_top_k": reranker.config.output_top_k,
        "protected_top_n": reranker.config.protected_top_n,
        "source_top5_session_ids": cast(
            JsonValue,
            source.retrieved_session_ids[: reranker.config.output_top_k],
        ),
        "source_recall_all@5": source_recall,
        "reranked_recall_all@5": reranked_recall,
        "transition_vs_source": transition,
        "evidence_needs": cast(JsonValue, decision.evidence_needs),
        "raw_selected_session_ids": cast(
            JsonValue,
            decision.raw_selected_session_ids,
        ),
        "raw_ranked_session_ids": cast(
            JsonValue,
            decision.raw_ranked_session_ids,
        ),
        "selector_selected_session_ids": cast(
            JsonValue,
            decision.selector_selected_session_ids,
        ),
        "selected_session_ids": cast(JsonValue, decision.selected_session_ids),
        "invalid_session_ids": cast(JsonValue, decision.invalid_session_ids),
        "parse_fallback": decision.parse_fallback,
        "parse_fallback_reason": decision.parse_fallback_reason,
        "input_tokens": decision.input_tokens,
        "output_tokens": decision.output_tokens,
        "usage": cast(JsonValue, decision.usage),
        "response_text": decision.response_text,
        "rerank_latency_ms": rerank_latency_ms,
        "boundary": cast(
            JsonValue,
            decision.boundary.model_dump(mode="json")
            if decision.boundary is not None
            else None,
        ),
        "test_labels_used": False,
    }
    return source.model_copy(
        update={
            "method": output_method,
            "retrieved_session_ids": ranked_sessions,
            "retrieved_memories": ranked_memories,
            "metrics": metrics,
            "retrieved_tokens": sum(memory.token_count for memory in ranked_memories),
            "adapter_stats": adapter_stats,
            "rerank_metadata": rerank_metadata,
        }
    )


def _prepare_manifest(
    path: Path,
    *,
    run_id: str,
    source_run: Path,
    source_manifest: dict[str, JsonValue],
    signature: dict[str, JsonValue],
    resume: bool,
) -> dict[str, JsonValue]:
    if path.exists():
        existing = _read_json_object(path)
        if not resume:
            raise FileExistsError(
                f"Rerank run already exists: {path.parent}. Use --resume to continue."
            )
        if existing.get("signature") != signature:
            raise ValueError("Existing rerank manifest does not match requested settings")
        manifest = existing
        manifest.update(
            {
                "status": "running",
                "resumed_at": datetime.now(UTC).isoformat(),
            }
        )
    else:
        if path.parent.exists() and any(path.parent.iterdir()):
            raise FileExistsError(
                f"Rerank directory exists without a compatible manifest: {path.parent}"
            )
        reranker_signature = signature.get("reranker")
        if not isinstance(reranker_signature, dict):
            raise ValueError("rerank signature is missing its reranker config")
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "2.0",
            "status": "running",
            "run_id": run_id,
            "dataset": source_manifest.get("dataset"),
            "data_sha256": source_manifest.get("data_sha256"),
            "sample_count": source_manifest.get("sample_count"),
            "split": source_manifest.get("split"),
            "source_retrieval_run": str(source_run),
            "source_retrieval_manifest_sha256": signature["source_manifest_sha256"],
            "signature": signature,
            "config": {
                "methods": signature["reranked_methods"],
                "source_methods": signature["methods"],
                "top_k": reranker_signature["output_top_k"],
                "retrieval_depth": reranker_signature["candidate_count"],
                "reranker": reranker_signature,
                "limit": signature["limit"],
            },
            "fairness": {
                "shared_across_methods": True,
                "provider": "vllm",
                "same_prompt": True,
                "same_generation": True,
                "two_stage_boundary_verification": bool(
                    reranker_signature.get("boundary_verification")
                ),
                "gold_labels_visible_to_reranker": False,
            },
            "test_labels_used": False,
            "started_at": datetime.now(UTC).isoformat(),
        }
    _write_json(path, manifest)
    return manifest


def _run_signature(
    config: LongMemEvalRerankRunConfig,
    *,
    methods: list[str],
    reranker_config: LongMemEvalRerankerConfig,
    source_manifest_sha256: str,
) -> dict[str, JsonValue]:
    return {
        "source_manifest_sha256": source_manifest_sha256,
        "methods": cast(JsonValue, methods),
        "reranked_methods": cast(
            JsonValue,
            [
                reranked_method_name(
                    method,
                    boundary_verification=reranker_config.boundary_verification,
                )
                for method in methods
            ],
        ),
        "limit": config.limit,
        "reranker": cast(JsonValue, reranker_config.model_dump(mode="json")),
        "metadata": cast(JsonValue, config.metadata),
        "test_labels_used": False,
    }


def _resolve_methods(source_run: Path, requested: list[str]) -> list[str]:
    candidates = (
        [_normalize_method(method) for method in requested]
        if requested
        else sorted(path.parent.name for path in source_run.glob("*/retrieval.jsonl"))
    )
    methods: list[str] = []
    for method in candidates:
        if method in methods:
            continue
        path = source_run / method / "retrieval.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        methods.append(method)
    return methods


def _load_records(path: Path, *, limit: int | None = None) -> list[RetrievalSampleRecord]:
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


def _load_resume_records(path: Path) -> list[RetrievalSampleRecord]:
    """Load completed JSONL rows and repair only an interrupted trailing write."""

    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    records: list[RetrievalSampleRecord] = []
    valid_prefix: list[str] = []
    for index, line in enumerate(lines):
        if not line.strip():
            valid_prefix.append(line)
            continue
        try:
            records.append(RetrievalSampleRecord.model_validate_json(line))
        except ValueError:
            is_incomplete_tail = index == len(lines) - 1 and not line.endswith(("\n", "\r"))
            if not is_incomplete_tail:
                raise
            LOGGER.warning(
                "Removing an interrupted trailing JSONL record before resume: %s",
                path,
            )
            path.write_text("".join(valid_prefix), encoding="utf-8", newline="\n")
            return records
        valid_prefix.append(line)
    return records


def _order_records(
    records: list[RetrievalSampleRecord],
    source_records: list[RetrievalSampleRecord],
) -> list[RetrievalSampleRecord]:
    by_id = {record.question_id: record for record in records}
    return [by_id[source.question_id] for source in source_records if source.question_id in by_id]


def _observed_reranker(record: RetrievalSampleRecord) -> tuple[str, str] | None:
    provider = record.rerank_metadata.get("provider")
    model = record.rerank_metadata.get("model")
    if isinstance(provider, str) and provider and isinstance(model, str) and model:
        return provider, model
    return None


def _validate_one_reranker(observed: set[tuple[str, str]]) -> None:
    if len(observed) > 1:
        raise ValueError(
            f"Rerank records contain multiple provider/model pairs: {sorted(observed)}"
        )


def _metadata_number(record: RetrievalSampleRecord, name: str) -> float:
    value = record.rerank_metadata.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _metadata_list(record: RetrievalSampleRecord, name: str) -> list[object]:
    value = record.rerank_metadata.get(name, [])
    return list(value) if isinstance(value, list) else []


def _boundary_payload(record: RetrievalSampleRecord) -> dict[str, JsonValue]:
    value = record.rerank_metadata.get("boundary")
    return value if isinstance(value, dict) else {}


def _boundary_bool(record: RetrievalSampleRecord, name: str) -> bool:
    return _boundary_payload(record).get(name) is True


def _boundary_number(record: RetrievalSampleRecord, name: str) -> float:
    value = _boundary_payload(record).get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _boundary_list(record: RetrievalSampleRecord, name: str) -> list[JsonValue]:
    value = _boundary_payload(record).get(name, [])
    return value if isinstance(value, list) else []


def _adapter_number(record: RetrievalSampleRecord, name: str) -> float:
    value = record.adapter_stats.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _metric_value(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _transition(*, before: bool, after: bool) -> str:
    if not before and after:
        return "recovered"
    if before and not after:
        return "regressed"
    return "stable_success" if before else "stable_failure"


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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_method(method: str) -> str:
    return method.strip().casefold().replace("-", "_")


def _safe_run_id(run_id: str) -> str:
    normalized = re_sub_unsafe(run_id)
    if not normalized:
        raise ValueError("run_id must contain at least one safe character")
    return normalized


def re_sub_unsafe(value: str) -> str:
    """Normalize one output directory component without importing runner internals."""

    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value.strip()
    ).strip("._")


def _mean(values: list[int | float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0
