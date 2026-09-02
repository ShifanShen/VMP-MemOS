"""Build one immutable paper-comparison run from independent framework runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from vmp_memos.longmemeval.official_qa_merge import merge_official_judge_runs
from vmp_memos.longmemeval.qa_runner import QASampleRecord, summarize_qa_method
from vmp_memos.longmemeval.retrieval_runner import (
    RetrievalSampleRecord,
    summarize_method,
)


def merge_longmemeval_paper_runs(
    retrieval_runs: list[Path],
    *,
    output_dir: Path,
    qa_subdir: str = "qa_v21_test",
    judge_subdir: str = "official_judge_local_vllm_v1",
) -> Path:
    """Merge retrieval, QA, and judge artifacts after strict protocol checks."""

    if len(retrieval_runs) < 2:
        raise ValueError("at least two retrieval runs are required")
    _safe_subdir(qa_subdir, name="qa_subdir")
    _safe_subdir(judge_subdir, name="judge_subdir")
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Paper comparison already exists: {output}. Choose a new path."
        )

    sources = [_load_source(run, qa_subdir, judge_subdir) for run in retrieval_runs]
    compatibility = sources[0]["compatibility"]
    canonical_coverage = sources[0]["coverage"]
    seen_methods: set[str] = set()
    for source in sources:
        if source["compatibility"] != compatibility:
            raise ValueError(
                "Runs are not comparable: dataset, split, reranker, reader, "
                "generation, or observed model provenance differs"
            )
        if source["coverage"] != canonical_coverage:
            raise ValueError(
                "Runs do not have identical ordered question coverage"
            )
        duplicate = seen_methods.intersection(source["methods"])
        if duplicate:
            raise ValueError(
                f"Duplicate methods across comparison runs: {sorted(duplicate)}"
            )
        seen_methods.update(source["methods"])

    temp = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    temp.mkdir(parents=True)
    try:
        methods = _write_merged_retrieval(
            temp,
            sources,
            compatibility,
            run_id=output.name,
        )
        _write_merged_qa(
            temp,
            sources,
            compatibility,
            methods=methods,
            qa_subdir=qa_subdir,
            logical_target=output,
        )
        merge_official_judge_runs(
            [source["judge_dir"] for source in sources],
            output_dir=temp / qa_subdir / judge_subdir,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temp.replace(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return output


def _load_source(
    raw_run: Path,
    qa_subdir: str,
    judge_subdir: str,
) -> dict[str, Any]:
    run = raw_run.expanduser().resolve()
    manifest_path = run / "manifest.json"
    manifest = _completed_manifest(manifest_path, artifact="retrieval/rerank run")
    methods = _manifest_methods(manifest, manifest_path)
    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        raise ValueError(f"Rerank manifest is missing signature: {manifest_path}")
    reranker = signature.get("reranker")
    if not isinstance(reranker, dict):
        raise ValueError(
            f"Paper comparison requires a shared rerank run: {manifest_path}"
        )
    if signature.get("test_labels_used") is not False:
        raise ValueError(f"Rerank run does not seal Test labels: {manifest_path}")

    qa_dir = run / qa_subdir
    qa_manifest_path = qa_dir / "manifest.json"
    qa_manifest = _completed_manifest(qa_manifest_path, artifact="QA run")
    if qa_manifest.get("retrieval_manifest_sha256") != _sha256(manifest_path):
        raise ValueError(f"QA-to-rerank provenance mismatch: {qa_dir}")
    qa_signature = qa_manifest.get("signature")
    if not isinstance(qa_signature, dict):
        raise ValueError(f"QA manifest is missing signature: {qa_manifest_path}")
    qa_methods = qa_signature.get("methods")
    if qa_methods != methods:
        raise ValueError(f"QA methods differ from rerank methods: {qa_dir}")
    qa_protocol = qa_signature.get("protocol")
    if not isinstance(qa_protocol, dict):
        raise ValueError(f"QA manifest is missing protocol: {qa_manifest_path}")
    if qa_protocol.get("gold_answers_visible_to_reader") is not False:
        raise ValueError(f"QA reader was exposed to gold answers: {qa_dir}")

    records: dict[str, list[RetrievalSampleRecord]] = {}
    qa_records: dict[str, list[QASampleRecord]] = {}
    coverage: list[tuple[str, str, str]] | None = None
    for method in methods:
        _safe_subdir(method, name="method")
        method_records = _read_retrieval_records(
            run / method / "retrieval.jsonl"
        )
        method_qa_records = _read_qa_records(qa_dir / f"{method}.jsonl")
        _validate_method_records(method, method_records, method_qa_records)
        observed_coverage = [
            (record.question_id, record.question_type, record.question)
            for record in method_records
        ]
        if coverage is None:
            coverage = observed_coverage
        elif coverage != observed_coverage:
            raise ValueError(
                f"Methods within {run} do not have identical ordered coverage"
            )
        records[method] = method_records
        qa_records[method] = method_qa_records

    split = manifest.get("split")
    if not isinstance(split, dict):
        raise ValueError(f"Rerank manifest is missing split: {manifest_path}")
    compatibility = {
        "dataset": manifest.get("dataset"),
        "data_sha256": manifest.get("data_sha256"),
        "sample_count": manifest.get("sample_count"),
        "split": {
            "name": split.get("name"),
            "split_id": split.get("split_id"),
            "question_count": split.get("question_count"),
        },
        "reranker": reranker,
        "rerank_limit": signature.get("limit"),
        "rerank_question_ids": signature.get("question_ids"),
        "observed_reranker": manifest.get("observed_reranker"),
        "qa": {
            "top_k": qa_signature.get("top_k"),
            "limit": qa_signature.get("limit"),
            "qa_subdir": qa_signature.get("qa_subdir"),
            "generation": qa_signature.get("generation"),
            "protocol": qa_protocol,
            "reader": qa_signature.get("reader"),
            "observed_reader": qa_manifest.get("observed_reader"),
        },
        "test_labels_used": False,
    }
    return {
        "run": run,
        "manifest_path": manifest_path,
        "qa_manifest_path": qa_manifest_path,
        "judge_dir": qa_dir / judge_subdir,
        "methods": methods,
        "records": records,
        "qa_records": qa_records,
        "coverage": coverage or [],
        "compatibility": compatibility,
    }


def _write_merged_retrieval(
    target: Path,
    sources: list[dict[str, Any]],
    compatibility: dict[str, Any],
    *,
    run_id: str,
) -> list[str]:
    methods: list[str] = []
    source_receipts: list[dict[str, Any]] = []
    summaries: dict[str, object] = {}
    for source in sources:
        source_methods: list[str] = source["methods"]
        methods.extend(source_methods)
        source_receipts.append(
            {
                "retrieval_run": str(source["run"]),
                "retrieval_manifest_sha256": _sha256(source["manifest_path"]),
                "qa_manifest_sha256": _sha256(source["qa_manifest_path"]),
                "methods": source_methods,
            }
        )
        for method in source_methods:
            records = source["records"][method]
            method_dir = target / method
            method_dir.mkdir(parents=True)
            _write_jsonl(method_dir / "retrieval.jsonl", records)
            summary = summarize_method(method, records)
            summaries[method] = summary.model_dump(mode="json")
            _write_json(method_dir / "summary.json", summaries[method])

    _write_json(
        target / "summary.json",
        {"run_id": run_id, "methods": summaries},
    )
    _write_json(
        target / "manifest.json",
        {
            "schema_version": "paper-comparison-v1",
            "status": "completed",
            "run_id": run_id,
            "dataset": compatibility["dataset"],
            "data_sha256": compatibility["data_sha256"],
            "sample_count": compatibility["sample_count"],
            "split": compatibility["split"],
            "signature": {
                "methods": methods,
                "reranker": compatibility["reranker"],
                "test_labels_used": False,
                "source_runs": source_receipts,
                "merged_without_rerunning_models": True,
            },
            "config": {
                "methods": methods,
                "top_k": compatibility["reranker"].get("output_top_k"),
                "retrieval_depth": compatibility["reranker"].get("candidate_count"),
            },
            "observed_reranker": compatibility["observed_reranker"],
            "test_labels_used": False,
        },
    )
    return methods


def _write_merged_qa(
    target: Path,
    sources: list[dict[str, Any]],
    compatibility: dict[str, Any],
    *,
    methods: list[str],
    qa_subdir: str,
    logical_target: Path,
) -> None:
    qa_dir = target / qa_subdir
    qa_dir.mkdir()
    summaries: dict[str, object] = {}
    for source in sources:
        for method in source["methods"]:
            records = source["qa_records"][method]
            _write_qa_jsonl(qa_dir / f"{method}.jsonl", records)
            summary = summarize_qa_method(method, records)
            summaries[method] = summary.model_dump(mode="json")
            _write_json(qa_dir / f"{method}.summary.json", summaries[method])
    _write_json(
        qa_dir / "summary.json",
        {"retrieval_run": str(logical_target), "methods": summaries},
    )
    qa = compatibility["qa"]
    _write_json(
        qa_dir / "manifest.json",
        {
            "schema_version": "paper-comparison-v1",
            "status": "completed",
            "signature": {
                "methods": methods,
                "top_k": qa["top_k"],
                "limit": qa["limit"],
                "qa_subdir": qa_subdir,
                "generation": qa["generation"],
                "protocol": qa["protocol"],
                "reader": qa["reader"],
                "merged_without_rerunning_models": True,
            },
            "retrieval_manifest_sha256": _sha256(target / "manifest.json"),
            "observed_reader": qa["observed_reader"],
        },
    )


def _validate_method_records(
    method: str,
    records: list[RetrievalSampleRecord],
    qa_records: list[QASampleRecord],
) -> None:
    if not records or not qa_records:
        raise ValueError(f"Method has incomplete retrieval or QA records: {method}")
    if any(record.method != method for record in records):
        raise ValueError(f"Retrieval record method mismatch: {method}")
    if any(record.method != method for record in qa_records):
        raise ValueError(f"QA record method mismatch: {method}")
    retrieval_ids = [record.question_id for record in records]
    qa_ids = [record.question_id for record in qa_records]
    if len(retrieval_ids) != len(set(retrieval_ids)):
        raise ValueError(f"Duplicate retrieval question IDs: {method}")
    if retrieval_ids != qa_ids:
        raise ValueError(f"Retrieval and QA question order differs: {method}")
    for retrieval, qa in zip(records, qa_records, strict=True):
        if (
            retrieval.question_type != qa.question_type
            or retrieval.question != qa.question
        ):
            raise ValueError(f"Retrieval and QA content differs: {method}")


def _manifest_methods(manifest: dict[str, Any], path: Path) -> list[str]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Retrieval manifest is missing config: {path}")
    methods = config.get("methods")
    if not isinstance(methods, list) or not methods or not all(
        isinstance(method, str) and method for method in methods
    ):
        raise ValueError(f"Retrieval manifest has invalid methods: {path}")
    return list(dict.fromkeys(methods))


def _completed_manifest(path: Path, *, artifact: str) -> dict[str, Any]:
    payload = _read_object(path)
    if payload.get("status") != "completed":
        raise ValueError(f"{artifact} is not completed: {path}")
    return payload


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_retrieval_records(path: Path) -> list[RetrievalSampleRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        RetrievalSampleRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_qa_records(path: Path) -> list[QASampleRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        QASampleRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[RetrievalSampleRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(record.model_dump_json())
            stream.write("\n")


def _write_qa_jsonl(path: Path, records: list[QASampleRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(record.model_dump_json())
            stream.write("\n")


def _safe_subdir(value: str, *, name: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{name} must be one safe directory name")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
