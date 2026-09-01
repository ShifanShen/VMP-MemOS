"""Strictly merge independently judged methods into one comparison artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vmp_memos.longmemeval.official_qa import (
    LOCAL_JUDGE_SCORE_KIND,
    OfficialJudgeMethodSummary,
    OfficialJudgeRecord,
    summarize_official_judge_method,
)


def merge_official_judge_runs(
    judge_runs: list[Path],
    *,
    output_dir: Path,
) -> Path:
    """Merge completed runs only when judge and question coverage are identical."""

    if len(judge_runs) < 2:
        raise ValueError("at least two judge runs are required for a comparison")
    if output_dir.exists():
        raise FileExistsError(
            f"Comparison output already exists: {output_dir}. Choose a new path."
        )

    canonical_compatibility: dict[str, Any] | None = None
    canonical_coverage: list[tuple[str, str]] | None = None
    records_by_method: dict[str, list[OfficialJudgeRecord]] = {}
    summaries: dict[str, OfficialJudgeMethodSummary] = {}
    sources: list[dict[str, Any]] = []

    for raw_run in judge_runs:
        judge_run = raw_run.expanduser().resolve()
        manifest_path = judge_run / "manifest.json"
        manifest = _read_object(manifest_path)
        if manifest.get("status") != "completed":
            raise ValueError(f"Judge run is not completed: {judge_run}")
        signature = manifest.get("signature")
        if not isinstance(signature, dict):
            raise ValueError(f"Judge manifest is missing signature: {manifest_path}")
        methods = signature.get("methods")
        if not isinstance(methods, list) or not methods or not all(
            isinstance(method, str) and method for method in methods
        ):
            raise ValueError(f"Judge manifest has invalid methods: {manifest_path}")
        qa_manifest_path = judge_run.parent / "manifest.json"
        qa_manifest = _read_completed_manifest(
            qa_manifest_path,
            artifact="QA run",
        )
        rerank_manifest_path = judge_run.parent.parent / "manifest.json"
        rerank_manifest = _read_completed_manifest(
            rerank_manifest_path,
            artifact="rerank run",
        )
        qa_manifest_sha256 = _sha256(qa_manifest_path)
        if signature.get("qa_manifest_sha256") != qa_manifest_sha256:
            raise ValueError(
                f"Judge-to-QA manifest provenance mismatch: {judge_run}"
            )
        rerank_manifest_sha256 = _sha256(rerank_manifest_path)
        if qa_manifest.get("retrieval_manifest_sha256") != rerank_manifest_sha256:
            raise ValueError(
                f"QA-to-rerank manifest provenance mismatch: {judge_run.parent}"
            )
        compatibility = _compatibility_signature(
            signature,
            manifest,
            qa_manifest=qa_manifest,
            rerank_manifest=rerank_manifest,
        )
        if canonical_compatibility is None:
            canonical_compatibility = compatibility
        elif compatibility != canonical_compatibility:
            raise ValueError(
                "Judge runs are not comparable: reranker, reader, judge, data, "
                "coverage, or generation provenance differs"
            )

        source_methods: list[str] = []
        for method in methods:
            if method in records_by_method:
                raise ValueError(f"Duplicate method across judge runs: {method}")
            records = _read_records(judge_run / f"{method}.jsonl")
            if not records:
                raise ValueError(f"Judge method has no records: {judge_run} / {method}")
            if any(record.method != method for record in records):
                raise ValueError(f"Judge record method mismatch: {judge_run} / {method}")
            coverage = [
                (record.question_id, record.question_type) for record in records
            ]
            if len({question_id for question_id, _ in coverage}) != len(coverage):
                raise ValueError(f"Duplicate question_id in judge method: {method}")
            if canonical_coverage is None:
                canonical_coverage = coverage
            elif coverage != canonical_coverage:
                raise ValueError(
                    "Judge methods do not have identical ordered question coverage"
                )
            records_by_method[method] = records
            summaries[method] = summarize_official_judge_method(method, records)
            source_methods.append(method)
        sources.append(
            {
                "judge_run": str(judge_run),
                "manifest_sha256": _sha256(manifest_path),
                "qa_manifest_sha256": qa_manifest_sha256,
                "rerank_manifest_sha256": rerank_manifest_sha256,
                "methods": source_methods,
            }
        )

    assert canonical_compatibility is not None
    assert canonical_coverage is not None
    output_dir.mkdir(parents=True)
    for method, records in records_by_method.items():
        _write_jsonl(output_dir / f"{method}.jsonl", records)
        _write_json(
            output_dir / f"{method}.summary.json",
            summaries[method].model_dump(mode="json"),
        )
    _write_json(
        output_dir / "summary.json",
        {
            "score_kind": LOCAL_JUDGE_SCORE_KIND,
            "directly_comparable_to_published_gpt4o_scores": False,
            "methods": {
                method: summary.model_dump(mode="json")
                for method, summary in summaries.items()
            },
        },
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "1.0",
            "status": "completed",
            "signature": {
                **canonical_compatibility,
                "methods": list(records_by_method),
                "source_runs": sources,
                "question_count": len(canonical_coverage),
                "merged_without_rejudging": True,
            },
            "observed_judge": canonical_compatibility["observed_judge"],
        },
    )
    return output_dir


def _compatibility_signature(
    signature: dict[str, Any],
    manifest: dict[str, Any],
    *,
    qa_manifest: dict[str, Any],
    rerank_manifest: dict[str, Any],
) -> dict[str, Any]:
    score_kind = signature.get("score_kind")
    if score_kind != LOCAL_JUDGE_SCORE_KIND:
        raise ValueError(f"Unsupported judge score kind: {score_kind!r}")
    if signature.get("directly_comparable_to_published_gpt4o_scores") is not False:
        raise ValueError("Judge provenance must explicitly reject GPT-4o comparability")
    observed = manifest.get("observed_judge")
    if not isinstance(observed, dict):
        raise ValueError("Judge manifest is missing observed_judge")
    qa_signature = qa_manifest.get("signature")
    if not isinstance(qa_signature, dict):
        raise ValueError("QA manifest is missing signature")
    rerank_signature = rerank_manifest.get("signature")
    if not isinstance(rerank_signature, dict):
        raise ValueError("Rerank manifest is missing signature")
    split = rerank_manifest.get("split")
    if not isinstance(split, dict):
        raise ValueError("Rerank manifest is missing split provenance")
    return {
        "reference_data_sha256": signature.get("reference_data_sha256"),
        "limit": signature.get("limit"),
        "score_kind": score_kind,
        "directly_comparable_to_published_gpt4o_scores": False,
        "protocol": signature.get("protocol"),
        "generation": signature.get("generation"),
        "configured_judge": signature.get("judge"),
        "observed_judge": observed,
        "reader_protocol": {
            "top_k": qa_signature.get("top_k"),
            "limit": qa_signature.get("limit"),
            "generation": qa_signature.get("generation"),
            "protocol": qa_signature.get("protocol"),
            "reader": qa_signature.get("reader"),
            "observed_reader": qa_manifest.get("observed_reader"),
        },
        "reranker_protocol": {
            "dataset": rerank_manifest.get("dataset"),
            "data_sha256": rerank_manifest.get("data_sha256"),
            "sample_count": rerank_manifest.get("sample_count"),
            "split": {
                "name": split.get("name"),
                "split_id": split.get("split_id"),
                "question_count": split.get("question_count"),
            },
            "limit": rerank_signature.get("limit"),
            "question_ids": rerank_signature.get("question_ids"),
            "reranker": rerank_signature.get("reranker"),
            "test_labels_used": rerank_signature.get("test_labels_used"),
            "observed_reranker": rerank_manifest.get("observed_reranker"),
        },
    }


def _read_completed_manifest(path: Path, *, artifact: str) -> dict[str, Any]:
    manifest = _read_object(path)
    if manifest.get("status") != "completed":
        raise ValueError(f"{artifact} is not completed: {path}")
    return manifest


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_records(path: Path) -> list[OfficialJudgeRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        OfficialJudgeRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[OfficialJudgeRecord]) -> None:
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
