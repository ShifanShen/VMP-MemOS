"""Dev-only quality gate for the grounded LongMemEval QA reader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import Field, JsonValue

from vmp_memos.longmemeval.qa_runner import QASampleRecord
from vmp_memos.schemas.base import NonEmptyStr, SchemaModel, Score


class LongMemEvalQAGateConfig(SchemaModel):
    """Frozen requirements checked before any Test QA generation."""

    retrieval_run: Path
    qa_subdir: NonEmptyStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    methods: list[NonEmptyStr] = Field(min_length=1)
    expected_prompt_version: NonEmptyStr
    expected_evidence_mode: NonEmptyStr
    max_answerable_refusal_rate: Score = 0.25
    min_answerable_fact_coverage: Score = 0.90
    min_token_f1: Score = 0.25
    min_contains_answer: Score = 0.10
    min_abstention_accuracy: Score = 0.50


class LongMemEvalQAGateResult(SchemaModel):
    """Auditable pass/fail result with every individual check exposed."""

    status: NonEmptyStr
    retrieval_run: NonEmptyStr
    qa_dir: NonEmptyStr
    prompt_version: NonEmptyStr
    evidence_mode: NonEmptyStr
    methods: dict[str, JsonValue]
    required: dict[str, JsonValue]
    checks: dict[str, bool]
    test_labels_used: bool = False


def evaluate_longmemeval_qa_gate(
    config: LongMemEvalQAGateConfig,
) -> LongMemEvalQAGateResult:
    """Check one completed Dev QA run without making model calls."""

    retrieval_run = config.retrieval_run.expanduser().resolve()
    retrieval_manifest_path = retrieval_run / "manifest.json"
    retrieval_manifest = _read_json(retrieval_manifest_path)
    qa_dir = retrieval_run / config.qa_subdir
    qa_manifest_path = qa_dir / "manifest.json"
    qa_manifest = _read_json(qa_manifest_path)
    signature = _mapping(qa_manifest.get("signature"))
    protocol = _mapping(signature.get("protocol"))
    split = _mapping(retrieval_manifest.get("split"))
    expected_count = _as_int(retrieval_manifest.get("sample_count"))

    checks: dict[str, bool] = {
        "retrieval_run_completed": retrieval_manifest.get("status") == "completed",
        "qa_run_completed": qa_manifest.get("status") == "completed",
        "dev_split_only": split.get("name") == "dev",
        "retrieval_manifest_matches": (
            qa_manifest.get("retrieval_manifest_sha256")
            == _sha256(retrieval_manifest_path)
        ),
        "qa_subdir_matches": signature.get("qa_subdir") == config.qa_subdir,
        "qa_methods_match": _string_list(signature.get("methods"))
        == list(config.methods),
        "prompt_version_matches": (
            protocol.get("prompt_version") == config.expected_prompt_version
        ),
        "evidence_mode_matches": (
            protocol.get("evidence_mode") == config.expected_evidence_mode
        ),
        "gold_answers_hidden_from_reader": (
            protocol.get("gold_answers_visible_to_reader") is False
        ),
        "test_labels_not_used": retrieval_manifest.get("test_labels_used") is False,
    }
    method_results: dict[str, JsonValue] = {}
    observed_readers: set[tuple[str, str]] = set()
    all_complete = True
    all_protocol = True
    all_refusal = True
    all_fact_coverage = True
    all_token_f1 = True
    all_contains = True
    all_abstention = True
    for method in config.methods:
        summary = _read_json(qa_dir / f"{method}.summary.json")
        records = _read_qa_records(qa_dir / f"{method}.jsonl")
        metrics = _mapping(summary.get("metrics"))
        processed = _as_int(summary.get("processed_questions"))
        answerable_refusal_rate = _as_float(
            summary.get("answerable_refusal_rate")
        )
        fact_coverage = _as_float(summary.get("answerable_fact_coverage_rate"))
        token_f1 = _as_float(metrics.get("token_f1"))
        contains_answer = _as_float(metrics.get("contains_answer"))
        abstention_accuracy = _as_float(metrics.get("abstention_accuracy"))
        complete = processed == expected_count == len(records) and expected_count > 0
        protocol_matches = bool(records) and all(
            record.reader_prompt_version == config.expected_prompt_version
            and record.reader_evidence_mode == config.expected_evidence_mode
            for record in records
        )
        refusal_pass = answerable_refusal_rate <= config.max_answerable_refusal_rate
        fact_pass = fact_coverage >= config.min_answerable_fact_coverage
        token_f1_pass = token_f1 >= config.min_token_f1
        contains_pass = contains_answer >= config.min_contains_answer
        abstention_pass = abstention_accuracy >= config.min_abstention_accuracy
        observed_readers.update(
            (record.reader_provider, record.reader_model) for record in records
        )
        all_complete = all_complete and complete
        all_protocol = all_protocol and protocol_matches
        all_refusal = all_refusal and refusal_pass
        all_fact_coverage = all_fact_coverage and fact_pass
        all_token_f1 = all_token_f1 and token_f1_pass
        all_contains = all_contains and contains_pass
        all_abstention = all_abstention and abstention_pass
        method_results[method] = {
            "processed_questions": processed,
            "answerable_questions": _as_int(summary.get("answerable_questions")),
            "abstention_questions": _as_int(summary.get("abstention_questions")),
            "answerable_refusal_rate": answerable_refusal_rate,
            "answerable_fact_coverage_rate": fact_coverage,
            "normalized_exact_match": _as_float(
                metrics.get("normalized_exact_match")
            ),
            "token_f1": token_f1,
            "contains_answer": contains_answer,
            "abstention_accuracy": abstention_accuracy,
            "checks": {
                "complete_question_coverage": complete,
                "record_protocol_matches": protocol_matches,
                "answerable_refusal_rate": refusal_pass,
                "answerable_fact_coverage": fact_pass,
                "token_f1": token_f1_pass,
                "contains_answer": contains_pass,
                "abstention_accuracy": abstention_pass,
            },
        }
    checks.update(
        {
            "complete_question_coverage": all_complete,
            "record_protocol_matches": all_protocol,
            "single_shared_reader": len(observed_readers) == 1,
            "answerable_refusal_rate": all_refusal,
            "answerable_fact_coverage": all_fact_coverage,
            "token_f1": all_token_f1,
            "contains_answer": all_contains,
            "abstention_accuracy": all_abstention,
        }
    )
    passed = all(checks.values())
    return LongMemEvalQAGateResult(
        status="passed" if passed else "failed",
        retrieval_run=str(retrieval_run),
        qa_dir=str(qa_dir),
        prompt_version=config.expected_prompt_version,
        evidence_mode=config.expected_evidence_mode,
        methods=method_results,
        required={
            "max_answerable_refusal_rate": config.max_answerable_refusal_rate,
            "min_answerable_fact_coverage": config.min_answerable_fact_coverage,
            "min_token_f1": config.min_token_f1,
            "min_contains_answer": config.min_contains_answer,
            "min_abstention_accuracy": config.min_abstention_accuracy,
        },
        checks=checks,
        test_labels_used=False,
    )


def write_longmemeval_qa_gate_result(
    result: LongMemEvalQAGateResult,
    path: str | Path,
) -> Path:
    """Write a stable gate receipt or failure report."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _read_qa_records(path: Path) -> list[QASampleRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        return [
            QASampleRecord.model_validate_json(line)
            for line in stream
            if line.strip()
        ]


def _read_json(path: Path) -> dict[str, JsonValue]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _mapping(value: object) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _string_list(value: JsonValue | None) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return []
    return [item for item in value if isinstance(item, str)]


def _as_int(value: JsonValue | None) -> int:
    return int(value) if isinstance(value, int | float | str) else 0


def _as_float(value: JsonValue | None) -> float:
    return float(value) if isinstance(value, int | float | str) else 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
