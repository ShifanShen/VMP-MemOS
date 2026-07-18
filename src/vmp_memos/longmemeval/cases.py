"""Deterministic, auditable paper-case export from completed experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field, JsonValue

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.frameworks.text import lexical_jaccard, parse_date
from vmp_memos.longmemeval.qa_runner import QASampleRecord
from vmp_memos.longmemeval.retrieval_runner import RetrievalSampleRecord
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeInt,
    SchemaModel,
)

_VMP_METHOD_PRIORITY = ("vmp_tuned", "vmp_full", "vmp_rule")
_VECTOR_METHOD_PRIORITY = (
    "naive_vector",
    "naive_vector_rag",
    "vector_rag",
)
_NO_ARCHIVE_METHOD = "vmp_tuned__no_archive_operation"


class CaseEvidence(SchemaModel):
    """One retrieved evidence item shown in a paper case."""

    rank: NonNegativeInt
    memory_id: NonEmptyStr
    source_session_id: str | None = None
    source_date: str | None = None
    score: float
    token_count: NonNegativeInt
    is_gold: bool
    content: str
    content_truncated: bool = False
    policy_features: dict[str, JsonValue] = Field(default_factory=dict)
    policy_contributions: dict[str, JsonValue] = Field(default_factory=dict)


class MethodCaseView(SchemaModel):
    """Retrieval, QA, and operation evidence for one compared method."""

    method: NonEmptyStr
    retrieved_session_ids: list[str] = Field(default_factory=list)
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    retrieved_tokens: NonNegativeInt = 0
    evidence: list[CaseEvidence] = Field(default_factory=list)
    prediction: str | None = None
    qa_metrics: dict[str, float] = Field(default_factory=dict)
    locally_correct: bool | None = None
    operation_counts: dict[str, NonNegativeInt] = Field(default_factory=dict)
    active_memory_count: float | None = None
    active_memory_tokens: float | None = None
    storage_size_bytes: float | None = None


class PaperCase(SchemaModel):
    """One selected qualitative case with machine-readable provenance."""

    case_id: NonEmptyStr
    title: NonEmptyStr
    question_id: NonEmptyStr
    question_type: NonEmptyStr
    question: NonEmptyStr
    gold_answer: NonEmptyStr | list[NonEmptyStr]
    gold_session_ids: list[str] = Field(default_factory=list)
    source_run: NonEmptyStr
    selection_reason: NonEmptyStr
    methods: dict[NonEmptyStr, MethodCaseView]
    analysis: list[NonEmptyStr] = Field(default_factory=list)


class CaseExportManifest(SchemaModel):
    """Hashes and selected IDs for one case-export operation."""

    schema_version: NonEmptyStr = "1.0"
    generated_at: datetime
    retrieval_run: NonEmptyStr
    retrieval_manifest_sha256: NonEmptyStr
    qa_manifest_sha256: str | None = None
    ablation_run: NonEmptyStr
    ablation_manifest_sha256: NonEmptyStr
    data_sha256: NonEmptyStr
    split_id: NonEmptyStr
    vmp_method: NonEmptyStr
    vector_method: NonEmptyStr
    source_artifact_sha256: dict[NonEmptyStr, NonEmptyStr] = Field(
        default_factory=dict
    )
    selected_question_ids: dict[NonEmptyStr, NonEmptyStr]


def export_longmemeval_cases(
    retrieval_run: str | Path,
    *,
    ablation_run: str | Path | None = None,
    output_dir: str | Path | None = None,
    vmp_method: str | None = None,
    vector_method: str | None = None,
    require_qa: bool = True,
) -> dict[str, Path]:
    """Select and export the four paper cases defined in the experiment plan."""

    main_dir = Path(retrieval_run).expanduser().resolve()
    ablation_dir = (
        Path(ablation_run).expanduser().resolve()
        if ablation_run is not None
        else main_dir
    )
    main_manifest_path = main_dir / "manifest.json"
    ablation_manifest_path = ablation_dir / "manifest.json"
    main_manifest = _completed_test_manifest(main_manifest_path)
    ablation_manifest = _completed_test_manifest(ablation_manifest_path)
    _validate_compatible_runs(main_manifest, ablation_manifest)

    main_methods = _manifest_methods(main_manifest)
    ablation_methods = _manifest_methods(ablation_manifest)
    resolved_vmp = _resolve_method(
        vmp_method,
        main_methods,
        _VMP_METHOD_PRIORITY,
        role="VMP",
    )
    resolved_vector = _resolve_method(
        vector_method,
        main_methods,
        _VECTOR_METHOD_PRIORITY,
        role="vector baseline",
    )
    if resolved_vmp not in ablation_methods:
        raise ValueError(
            f"ablation run does not contain full VMP method {resolved_vmp!r}"
        )
    if _NO_ARCHIVE_METHOD not in ablation_methods:
        raise ValueError(
            f"ablation run does not contain {_NO_ARCHIVE_METHOD!r}"
        )

    main_vmp = _records_by_id(main_dir, resolved_vmp)
    main_vector = _records_by_id(main_dir, resolved_vector)
    _validate_question_sets(resolved_vmp, main_vmp, resolved_vector, main_vector)
    qa_vmp = _qa_by_id(main_dir, resolved_vmp, require=require_qa)
    qa_vector = _qa_by_id(main_dir, resolved_vector, require=require_qa)
    if require_qa:
        _validate_qa_set(resolved_vmp, main_vmp, qa_vmp)
        _validate_qa_set(resolved_vector, main_vector, qa_vector)

    ablation_vmp = _records_by_id(ablation_dir, resolved_vmp)
    no_archive = _records_by_id(ablation_dir, _NO_ARCHIVE_METHOD)
    _validate_question_sets(
        resolved_vmp,
        ablation_vmp,
        _NO_ARCHIVE_METHOD,
        no_archive,
    )

    case1_id = _select_knowledge_update(main_vmp, main_vector, qa_vmp)
    case2_id = _select_stale_vector(main_vmp, main_vector, qa_vmp, qa_vector)
    case3_id = _select_archive_case(ablation_vmp, no_archive)
    case4_id = _select_vmp_error(main_vmp, qa_vmp)

    cases = [
        _build_case(
            "case1_knowledge_update",
            "VMP correctly handles a knowledge update",
            case1_id,
            source_run=main_dir,
            records={
                resolved_vector: main_vector[case1_id],
                resolved_vmp: main_vmp[case1_id],
            },
            qa={resolved_vector: qa_vector.get(case1_id), resolved_vmp: qa_vmp.get(case1_id)},
            selection_reason=(
                "Knowledge-update question where VMP is locally correct and "
                "outperforms the vector baseline on gold-session retrieval."
            ),
        ),
        _build_case(
            "case2_stale_vector_retrieval",
            "NaiveVectorRAG retrieves stale conflicting evidence",
            case2_id,
            source_run=main_dir,
            records={
                resolved_vector: main_vector[case2_id],
                resolved_vmp: main_vmp[case2_id],
            },
            qa={resolved_vector: qa_vector.get(case2_id), resolved_vmp: qa_vmp.get(case2_id)},
            selection_reason=(
                "Knowledge-update question maximizing the excess number of "
                "older non-gold sessions retrieved by the vector baseline."
            ),
        ),
        _build_case(
            "case3_archive_suppression",
            "VMP softly reranks superseded evidence",
            case3_id,
            source_run=ablation_dir,
            records={
                resolved_vmp: ablation_vmp[case3_id],
                _NO_ARCHIVE_METHOD: no_archive[case3_id],
            },
            qa={},
            selection_reason=(
                "Question where non-destructive archive annotations change the "
                "ranking of evidence retained by the no-archive ablation."
            ),
        ),
        _build_case(
            "case4_vmp_error",
            "VMP failure case",
            case4_id,
            source_run=main_dir,
            records={
                resolved_vector: main_vector[case4_id],
                resolved_vmp: main_vmp[case4_id],
            },
            qa={resolved_vector: qa_vector.get(case4_id), resolved_vmp: qa_vmp.get(case4_id)},
            selection_reason=(
                "VMP is locally incorrect or misses at least one gold session; "
                "the lowest retrieval/QA score is selected deterministically."
            ),
        ),
    ]

    target = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else main_dir.parent.parent / "cases"
    )
    target.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for case in cases:
        path = target / f"{case.case_id}.json"
        path.write_text(case.model_dump_json(indent=2) + "\n", encoding="utf-8")
        outputs[case.case_id] = path
    combined_path = target / "cases.json"
    combined_path.write_text(
        json.dumps(
            [case.model_dump(mode="json") for case in cases],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path = target / "paper_cases.md"
    markdown_path.write_text(_render_markdown(cases), encoding="utf-8")
    split = main_manifest["split"]
    if not isinstance(split, dict):
        raise ValueError("retrieval manifest split is invalid")
    manifest = CaseExportManifest(
        generated_at=datetime.now(UTC),
        retrieval_run=str(main_dir),
        retrieval_manifest_sha256=_sha256(main_manifest_path),
        qa_manifest_sha256=(
            _sha256(main_dir / "qa" / "manifest.json")
            if (main_dir / "qa" / "manifest.json").exists()
            else None
        ),
        ablation_run=str(ablation_dir),
        ablation_manifest_sha256=_sha256(ablation_manifest_path),
        data_sha256=str(main_manifest["data_sha256"]),
        split_id=str(split["split_id"]),
        vmp_method=resolved_vmp,
        vector_method=resolved_vector,
        source_artifact_sha256=_source_artifact_hashes(
            main_dir,
            ablation_dir,
            resolved_vmp,
            resolved_vector,
        ),
        selected_question_ids={
            case.case_id: case.question_id for case in cases
        },
    )
    manifest_path = target / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return {
        **outputs,
        "combined_json": combined_path,
        "paper_markdown": markdown_path,
        "manifest": manifest_path,
    }


def _select_knowledge_update(
    vmp: dict[str, RetrievalSampleRecord],
    vector: dict[str, RetrievalSampleRecord],
    qa_vmp: dict[str, QASampleRecord],
) -> str:
    candidates: list[tuple[tuple[float, float, float], str]] = []
    for question_id, vmp_record in vmp.items():
        if "update" not in vmp_record.question_type.casefold():
            continue
        vmp_correct = _qa_correct(qa_vmp.get(question_id))
        vmp_recall = _metric(vmp_record, "recall_all@5")
        vector_recall = _metric(vector[question_id], "recall_all@5")
        if vmp_correct is False or vmp_recall <= vector_recall:
            continue
        candidates.append(
            (
                (
                    vmp_recall - vector_recall,
                    _metric(vmp_record, "mrr"),
                    -vector_recall,
                ),
                question_id,
            )
        )
    return _best_candidate(candidates, "knowledge-update success")


def _select_stale_vector(
    vmp: dict[str, RetrievalSampleRecord],
    vector: dict[str, RetrievalSampleRecord],
    qa_vmp: dict[str, QASampleRecord],
    qa_vector: dict[str, QASampleRecord],
) -> str:
    candidates: list[tuple[tuple[float, float, float], str]] = []
    for question_id, vector_record in vector.items():
        if "update" not in vector_record.question_type.casefold():
            continue
        gold_date, gold_contents = _newest_gold_context(
            vector_record,
            vmp[question_id],
        )
        if gold_date is None:
            continue
        vector_stale = _stale_count(
            vector_record,
            gold_date,
            gold_contents,
        )
        vmp_stale = _stale_count(
            vmp[question_id],
            gold_date,
            gold_contents,
        )
        if vector_stale <= vmp_stale:
            continue
        qa_advantage = float(
            _qa_correct(qa_vmp.get(question_id)) is True
            and _qa_correct(qa_vector.get(question_id)) is False
        )
        candidates.append(
            (
                (
                    float(vector_stale - vmp_stale),
                    qa_advantage,
                    _metric(vmp[question_id], "recall_all@5"),
                ),
                question_id,
            )
        )
    return _best_candidate(candidates, "stale vector retrieval")


def _select_archive_case(
    full: dict[str, RetrievalSampleRecord],
    no_archive: dict[str, RetrievalSampleRecord],
) -> str:
    candidates: list[tuple[tuple[float, float, float], str]] = []
    for question_id, full_record in full.items():
        no_archive_record = no_archive[question_id]
        archive_count = _operation_count(full_record, "archive")
        ranking_delta = len(
            set(no_archive_record.retrieved_session_ids)
            - set(full_record.retrieved_session_ids)
        )
        memory_delta = (
            _adapter_number(no_archive_record, "memory_count")
            - _adapter_number(full_record, "memory_count")
        )
        extra_non_gold = _non_gold_count(no_archive_record) - _non_gold_count(full_record)
        if archive_count <= 0:
            continue
        candidates.append(
            (
                (
                    float(archive_count),
                    float(max(ranking_delta, extra_non_gold)),
                    memory_delta,
                ),
                question_id,
            )
        )
    return _best_candidate(candidates, "archive suppression")


def _select_vmp_error(
    vmp: dict[str, RetrievalSampleRecord],
    qa_vmp: dict[str, QASampleRecord],
) -> str:
    candidates: list[tuple[tuple[float, float, float], str]] = []
    for question_id, record in vmp.items():
        qa_correct = _qa_correct(qa_vmp.get(question_id))
        recall = _metric(record, "recall_all@5")
        if qa_correct is not False and (record.evaluation_skipped or recall >= 1.0):
            continue
        candidates.append(
            (
                (
                    float(qa_correct is False),
                    1.0 - recall,
                    1.0 - _metric(record, "mrr"),
                ),
                question_id,
            )
        )
    return _best_candidate(candidates, "VMP error")


def _best_candidate(
    candidates: Sequence[tuple[tuple[float, ...], str]],
    case_name: str,
) -> str:
    if not candidates:
        raise ValueError(f"no qualifying sample found for {case_name}")
    return sorted(
        candidates,
        key=lambda item: (*(-value for value in item[0]), item[1]),
    )[0][1]


def _build_case(
    case_id: str,
    title: str,
    question_id: str,
    *,
    source_run: Path,
    records: dict[str, RetrievalSampleRecord],
    qa: dict[str, QASampleRecord | None],
    selection_reason: str,
) -> PaperCase:
    first = next(iter(records.values()))
    views = {
        method: _method_view(record, qa.get(method))
        for method, record in records.items()
    }
    analysis = _analysis_lines(case_id, views)
    return PaperCase(
        case_id=case_id,
        title=title,
        question_id=question_id,
        question_type=first.question_type,
        question=first.question,
        gold_answer=first.answer,
        gold_session_ids=list(first.gold_session_ids),
        source_run=str(source_run),
        selection_reason=selection_reason,
        methods=views,
        analysis=analysis,
    )


def _method_view(
    record: RetrievalSampleRecord,
    qa: QASampleRecord | None,
) -> MethodCaseView:
    operation_counts = {
        operation: _operation_count(record, operation)
        for operation in ("update", "merge", "archive")
    }
    return MethodCaseView(
        method=record.method,
        retrieved_session_ids=list(record.retrieved_session_ids),
        retrieval_metrics={
            name: float(value) for name, value in record.metrics.items()
        },
        retrieved_tokens=record.retrieved_tokens,
        evidence=[
            _case_evidence(rank, memory, set(record.gold_session_ids))
            for rank, memory in enumerate(record.retrieved_memories, start=1)
        ],
        prediction=qa.prediction if qa is not None else None,
        qa_metrics=(
            {name: float(value) for name, value in qa.metrics.items()}
            if qa is not None
            else {}
        ),
        locally_correct=_qa_correct(qa),
        operation_counts=operation_counts,
        active_memory_count=(
            _adapter_optional_number(record, "active_memory_count")
            if "active_memory_count" in record.adapter_stats
            else _adapter_optional_number(record, "memory_count")
        ),
        active_memory_tokens=_adapter_optional_number(record, "total_tokens"),
        storage_size_bytes=_adapter_optional_number(record, "storage_size_bytes"),
    )


def _case_evidence(
    rank: int,
    memory: RetrievedMemory,
    gold_session_ids: set[str],
) -> CaseEvidence:
    content, truncated = _truncate(memory.content, 1200)
    features = memory.metadata.get("policy_features")
    contributions = memory.metadata.get("policy_contributions")
    return CaseEvidence(
        rank=rank,
        memory_id=memory.memory_id,
        source_session_id=memory.source_session_id,
        source_date=memory.source_date,
        score=float(memory.score),
        token_count=memory.token_count,
        is_gold=memory.source_session_id in gold_session_ids,
        content=content,
        content_truncated=truncated,
        policy_features=features if isinstance(features, dict) else {},
        policy_contributions=contributions if isinstance(contributions, dict) else {},
    )


def _analysis_lines(
    case_id: str,
    views: dict[str, MethodCaseView],
) -> list[str]:
    methods = list(views)
    first, second = views[methods[0]], views[methods[1]]
    if case_id == "case1_knowledge_update":
        return [
            f"{second.method} Recall-All@5="
            f"{second.retrieval_metrics.get('recall_all@5', 0):.3f}; "
            f"{first.method} Recall-All@5="
            f"{first.retrieval_metrics.get('recall_all@5', 0):.3f}.",
            "The VMP policy ranks newer update-bearing evidence using recency and "
            "contradiction-aware signals.",
        ]
    if case_id == "case2_stale_vector_retrieval":
        return [
            f"{first.method} returns older non-gold evidence that {second.method} suppresses.",
            "This case isolates stale retrieval rather than attributing the result "
            "to a different reader or embedding model.",
        ]
    if case_id == "case3_archive_suppression":
        return [
            f"Archive operations={first.operation_counts.get('archive', 0)}; "
            f"active memories {first.active_memory_count} vs {second.active_memory_count}.",
            "Both variants retain the same physical source memories. The no-archive "
            "variant disables only the bounded superseded-status score penalty.",
        ]
    return [
        f"VMP local correctness={second.locally_correct}; "
        f"Recall-All@5={second.retrieval_metrics.get('recall_all@5', 0):.3f}.",
        "This failure is retained to bound the method's claims and support error analysis.",
    ]


def _render_markdown(cases: list[PaperCase]) -> str:
    lines = ["# LongMemEval Qualitative Cases", ""]
    for case in cases:
        lines.extend(
            [
                f"## {case.title}",
                "",
                f"- Case ID: `{case.case_id}`",
                f"- Question ID: `{case.question_id}`",
                f"- Question type: `{case.question_type}`",
                f"- Source run: `{case.source_run}`",
                f"- Selection: {case.selection_reason}",
                "",
                f"**Question:** {case.question}",
                "",
                f"**Gold answer:** {_answer_text(case.gold_answer)}",
                "",
                f"**Gold sessions:** `{', '.join(case.gold_session_ids)}`",
                "",
            ]
        )
        for method, view in case.methods.items():
            lines.extend(
                [
                    f"### {method}",
                    "",
                    f"- Retrieved sessions: `{', '.join(view.retrieved_session_ids)}`",
                    f"- Retrieval metrics: `{json.dumps(view.retrieval_metrics, sort_keys=True)}`",
                    f"- Prediction: {_markdown_text(view.prediction or '(not available)')}",
                    f"- Locally correct: `{view.locally_correct}`",
                    f"- Operations: `{json.dumps(view.operation_counts, sort_keys=True)}`",
                    f"- Active memories: `{view.active_memory_count}`",
                    "",
                ]
            )
            for evidence in view.evidence:
                lines.extend(
                    [
                        f"#### Rank {evidence.rank}: session `{evidence.source_session_id}`",
                        "",
                        f"- Date: `{evidence.source_date}`",
                        f"- Score: `{evidence.score:.4f}`",
                        f"- Gold evidence: `{evidence.is_gold}`",
                        "",
                        "> " + evidence.content.replace("\n", "\n> "),
                        "",
                    ]
                )
        lines.extend(["### Analysis", ""])
        lines.extend(f"- {item}" for item in case.analysis)
        lines.extend(["", "---", ""])
    return "\n".join(lines)


def _completed_test_manifest(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    if payload.get("status") != "completed":
        raise ValueError(f"retrieval run is not completed: {path.parent}")
    split = payload.get("split")
    if not isinstance(split, dict) or split.get("name") != "test":
        raise ValueError("paper cases must be exported from the test split")
    return payload


def _validate_compatible_runs(
    main: dict[str, object],
    ablation: dict[str, object],
) -> None:
    if main.get("data_sha256") != ablation.get("data_sha256"):
        raise ValueError("retrieval and ablation runs use different dataset bytes")
    main_split = main.get("split")
    ablation_split = ablation.get("split")
    if not isinstance(main_split, dict) or not isinstance(ablation_split, dict):
        raise ValueError("run split metadata is missing")
    if main_split.get("split_id") != ablation_split.get("split_id"):
        raise ValueError("retrieval and ablation runs use different splits")
    main_model = main.get("vmp_tuned_model")
    ablation_model = ablation.get("vmp_tuned_model")
    if (
        isinstance(main_model, dict)
        and isinstance(ablation_model, dict)
        and main_model.get("sha256") != ablation_model.get("sha256")
    ):
        raise ValueError("retrieval and ablation runs use different VMP models")


def _manifest_methods(manifest: dict[str, object]) -> list[str]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("retrieval manifest is missing config")
    methods = config.get("methods")
    if not isinstance(methods, list) or not all(
        isinstance(method, str) for method in methods
    ):
        raise ValueError("retrieval manifest config is missing methods")
    return [str(method) for method in methods]


def _resolve_method(
    requested: str | None,
    methods: list[str],
    priority: tuple[str, ...],
    *,
    role: str,
) -> str:
    if requested is not None:
        normalized = requested.strip().casefold().replace("-", "_")
        if normalized not in methods:
            raise ValueError(f"{role} method {normalized!r} is not in the run")
        return normalized
    for candidate in priority:
        if candidate in methods:
            return candidate
    raise ValueError(f"run contains no supported {role} method")


def _records_by_id(
    run_dir: Path,
    method: str,
) -> dict[str, RetrievalSampleRecord]:
    path = run_dir / method / "retrieval.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    records = [
        RetrievalSampleRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record.question_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"duplicate retrieval question IDs for {method}")
    return by_id


def _qa_by_id(
    run_dir: Path,
    method: str,
    *,
    require: bool,
) -> dict[str, QASampleRecord]:
    qa_manifest_path = run_dir / "qa" / "manifest.json"
    if require:
        manifest = _read_json(qa_manifest_path)
        if manifest.get("status") != "completed":
            raise ValueError(f"QA run is not completed: {run_dir}")
    path = run_dir / "qa" / f"{method}.jsonl"
    if not path.exists():
        if require:
            raise FileNotFoundError(path)
        return {}
    records = [
        QASampleRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record.question_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"duplicate QA question IDs for {method}")
    return by_id


def _validate_question_sets(
    left_name: str,
    left: dict[str, RetrievalSampleRecord],
    right_name: str,
    right: dict[str, RetrievalSampleRecord],
) -> None:
    if set(left) != set(right):
        raise ValueError(f"question sets differ for {left_name} and {right_name}")


def _validate_qa_set(
    method: str,
    retrieval: dict[str, RetrievalSampleRecord],
    qa: dict[str, QASampleRecord],
) -> None:
    if set(retrieval) != set(qa):
        raise ValueError(f"retrieval and QA question sets differ for {method}")


def _newest_gold_context(
    *records: RetrievalSampleRecord,
) -> tuple[datetime | None, list[str]]:
    evidence = [
        (parsed, memory.content)
        for record in records
        for memory in record.retrieved_memories
        if memory.source_session_id in record.gold_session_ids
        if (parsed := parse_date(memory.source_date)) is not None
    ]
    return (
        max((date for date, _ in evidence), default=None),
        [content for _, content in evidence],
    )


def _stale_count(
    record: RetrievalSampleRecord,
    gold_date: datetime,
    gold_contents: list[str],
) -> int:
    return sum(
        1
        for memory in record.retrieved_memories[:5]
        if memory.source_session_id not in record.gold_session_ids
        and (memory_date := parse_date(memory.source_date)) is not None
        and memory_date < gold_date
        and any(
            lexical_jaccard(memory.content, gold_content) >= 0.10
            for gold_content in gold_contents
        )
    )


def _non_gold_count(record: RetrievalSampleRecord) -> int:
    gold = set(record.gold_session_ids)
    return sum(
        memory.source_session_id not in gold
        for memory in record.retrieved_memories[:5]
    )


def _metric(record: RetrievalSampleRecord, name: str) -> float:
    return float(record.metrics.get(name, 0.0))


def _qa_correct(record: QASampleRecord | None) -> bool | None:
    if record is None:
        return None
    metric = (
        "abstention_accuracy"
        if record.is_abstention
        else "normalized_exact_match"
    )
    return float(record.metrics.get(metric, 0.0)) >= 1.0


def _operation_count(record: RetrievalSampleRecord, operation: str) -> int:
    payload = record.adapter_stats.get("policy_operation_counts")
    if not isinstance(payload, dict):
        return 0
    value = payload.get(operation, 0)
    return int(value) if isinstance(value, int | float) and value >= 0 else 0


def _adapter_number(record: RetrievalSampleRecord, name: str) -> float:
    value = record.adapter_stats.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _adapter_optional_number(
    record: RetrievalSampleRecord,
    name: str,
) -> float | None:
    value = record.adapter_stats.get(name)
    return float(value) if isinstance(value, int | float) else None


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[: limit - 1].rstrip() + "…", True


def _answer_text(answer: str | list[str]) -> str:
    return " / ".join(answer) if isinstance(answer, list) else answer


def _markdown_text(value: str) -> str:
    return value.replace("\n", " ").strip()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_artifact_hashes(
    main_dir: Path,
    ablation_dir: Path,
    vmp_method: str,
    vector_method: str,
) -> dict[str, str]:
    paths = [
        main_dir / vmp_method / "retrieval.jsonl",
        main_dir / vector_method / "retrieval.jsonl",
        main_dir / "qa" / f"{vmp_method}.jsonl",
        main_dir / "qa" / f"{vector_method}.jsonl",
        ablation_dir / vmp_method / "retrieval.jsonl",
        ablation_dir / _NO_ARCHIVE_METHOD / "retrieval.jsonl",
    ]
    return {
        str(path): _sha256(path)
        for path in paths
        if path.exists()
    }
