"""Official-prompt-compatible LongMemEval QA judging over local vLLM.

The prompt templates and binary decision rule mirror the MIT-licensed
LongMemEval evaluator at:
https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py

Using a local judge preserves within-paper fairness, but only a run made with
the upstream pinned GPT-4o judge is directly comparable with published
LongMemEval QA accuracy. The artifact schema makes that distinction explicit.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, cast

from pydantic import Field, JsonValue, field_validator

from vmp_memos.llm import ChatMessage, LLMGenerationConfig, LLMResponse
from vmp_memos.longmemeval.loader import load_longmemeval
from vmp_memos.longmemeval.qa_runner import QASampleRecord
from vmp_memos.longmemeval.schema import LongMemEvalSample
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
    Score,
)

LOGGER = logging.getLogger(__name__)

LONGMEMEVAL_OFFICIAL_JUDGE_PROMPT_VERSION: Literal[
    "longmemeval_official_qa_judge_v1"
] = "longmemeval_official_qa_judge_v1"
LONGMEMEVAL_OFFICIAL_EVALUATOR_URL = (
    "https://github.com/xiaowu0162/LongMemEval/blob/main/"
    "src/evaluation/evaluate_qa.py"
)
LOCAL_JUDGE_SCORE_KIND: Literal[
    "official_prompt_local_vllm_judge"
] = "official_prompt_local_vllm_judge"
JudgeParseStatus = Literal["yes", "no", "ambiguous", "missing"]

_STANDARD_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
    "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_TEMPORAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for the "
    "number of days.\nIf the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct.\n\nQuestion: {}"
    "\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
    "correct? Answer yes or no only."
)
_UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}"
    "\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
    "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer "
    "yes or no only."
)
_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information is "
    "not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes "
    "or no only."
)


class ChatClient(Protocol):
    """Minimal client contract shared by the real and test judges."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse: ...


class LongMemEvalOfficialJudgeConfig(SchemaModel):
    """Immutable inputs for one official-prompt-compatible judge run."""

    qa_run: Path
    reference_data: Path
    methods: list[NonEmptyStr] = Field(default_factory=list)
    output_subdir: NonEmptyStr = "official_judge_local_vllm_v1"
    limit: NonNegativeInt | None = None
    resume: bool = False
    judge_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("output_subdir")
    @classmethod
    def validate_output_subdir(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("output_subdir must be one safe directory name")
        return value


class OfficialAutoEvalLabel(SchemaModel):
    """Upstream-compatible binary judge label with explicit provenance."""

    model: NonEmptyStr
    label: bool


class OfficialJudgeRecord(SchemaModel):
    """One prediction judged with the official LongMemEval prompt family."""

    question_id: NonEmptyStr
    question_type: NonEmptyStr
    method: NonEmptyStr
    hypothesis: str
    autoeval_label: OfficialAutoEvalLabel
    score_kind: Literal["official_prompt_local_vllm_judge"] = LOCAL_JUDGE_SCORE_KIND
    judge_provider: NonEmptyStr
    judge_model: NonEmptyStr
    judge_response: str
    parse_status: JudgeParseStatus
    prompt_version: Literal[
        "longmemeval_official_qa_judge_v1"
    ] = LONGMEMEVAL_OFFICIAL_JUDGE_PROMPT_VERSION
    prompt_sha256: NonEmptyStr
    judge_latency_ms: NonNegativeFloat = 0.0
    judge_input_tokens: NonNegativeInt = 0
    judge_output_tokens: NonNegativeInt = 0
    judge_usage: dict[str, JsonValue] = Field(default_factory=dict)


class OfficialJudgeTypeSummary(SchemaModel):
    """Binary accuracy for one official LongMemEval question type."""

    questions: NonNegativeInt
    correct: NonNegativeInt
    accuracy: Score


class OfficialJudgeMethodSummary(SchemaModel):
    """Aggregate local-judge accuracy and cost for one memory method."""

    method: NonEmptyStr
    score_kind: Literal["official_prompt_local_vllm_judge"] = LOCAL_JUDGE_SCORE_KIND
    questions: NonNegativeInt
    correct: NonNegativeInt
    accuracy: Score
    task_averaged_accuracy: Score
    abstention_questions: NonNegativeInt
    abstention_correct: NonNegativeInt
    abstention_accuracy: Score
    by_question_type: dict[str, OfficialJudgeTypeSummary] = Field(default_factory=dict)
    parse_fallbacks: NonNegativeInt = 0
    parse_fallback_rate: Score = 0.0
    mean_judge_latency_ms: NonNegativeFloat = 0.0
    mean_judge_input_tokens: NonNegativeFloat = 0.0
    mean_judge_output_tokens: NonNegativeFloat = 0.0


class OfficialJudgeRunResult(SchemaModel):
    """Paths and summaries emitted by one completed judge run."""

    qa_run: Path
    judge_dir: Path
    manifest_path: Path
    summaries: dict[str, OfficialJudgeMethodSummary]


def build_official_qa_judge_prompt(
    question_type: str,
    question: str,
    answer: str | list[str],
    hypothesis: str,
    *,
    is_abstention: bool,
) -> str:
    """Build the task-specific prompt used by LongMemEval's official evaluator."""

    if is_abstention:
        return _ABSTENTION_TEMPLATE.format(question, answer, hypothesis)
    if question_type in {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    }:
        template = _STANDARD_TEMPLATE
    elif question_type == "temporal-reasoning":
        template = _TEMPORAL_TEMPLATE
    elif question_type == "knowledge-update":
        template = _UPDATE_TEMPLATE
    elif question_type == "single-session-preference":
        template = _PREFERENCE_TEMPLATE
    else:
        raise ValueError(f"Unsupported LongMemEval question type: {question_type}")
    return template.format(question, answer, hypothesis)


def parse_official_qa_judge_response(
    response: str,
) -> tuple[bool, JudgeParseStatus]:
    """Apply the upstream `yes in response` rule and audit malformed responses."""

    lowered = response.casefold()
    has_yes = "yes" in lowered
    has_no = "no" in lowered
    if has_yes and not has_no:
        status: JudgeParseStatus = "yes"
    elif has_no and not has_yes:
        status = "no"
    elif has_yes and has_no:
        status = "ambiguous"
    else:
        status = "missing"
    return has_yes, status


def run_longmemeval_official_judge(
    config: LongMemEvalOfficialJudgeConfig,
    *,
    client: ChatClient,
    generation: LLMGenerationConfig,
) -> OfficialJudgeRunResult:
    """Judge a completed shared-reader QA run with resumable local vLLM calls."""

    qa_run = config.qa_run.expanduser().resolve()
    qa_manifest_path = qa_run / "manifest.json"
    qa_manifest = _read_json_object(qa_manifest_path)
    _validate_qa_manifest(qa_manifest, qa_run=qa_run)
    methods = _resolve_methods(qa_manifest, config.methods)
    references = {
        sample.question_id: sample
        for sample in load_longmemeval(config.reference_data.expanduser().resolve())
    }
    qa_records_by_method = {
        method: _load_qa_records(qa_run / f"{method}.jsonl", limit=config.limit)
        for method in methods
    }
    _validate_method_coverage(qa_records_by_method, references=references)

    judge_dir = qa_run / config.output_subdir
    judge_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = judge_dir / "manifest.json"
    signature = _judge_signature(
        config,
        methods=methods,
        qa_manifest_path=qa_manifest_path,
        generation=generation,
    )
    manifest = _prepare_manifest(manifest_path, signature=signature, resume=config.resume)
    wall_started = perf_counter()
    summaries: dict[str, OfficialJudgeMethodSummary] = {}
    observed_judges: set[tuple[str, str]] = set()
    try:
        for method in methods:
            method_started = perf_counter()
            source_records = qa_records_by_method[method]
            output_path = judge_dir / f"{method}.jsonl"
            existing = _load_judge_records(output_path) if config.resume else []
            existing_ids = {record.question_id for record in existing}
            if len(existing_ids) != len(existing):
                raise ValueError(f"Duplicate question_id in judge output: {output_path}")
            source_ids = {record.question_id for record in source_records}
            if not existing_ids <= source_ids:
                raise ValueError(f"Judge resume output contains unknown questions: {output_path}")
            observed_judges.update(
                (record.judge_provider, record.judge_model) for record in existing
            )
            _validate_one_judge(observed_judges)
            pending = sum(record.question_id not in existing_ids for record in source_records)
            LOGGER.info(
                "Official-prompt judge %s started: total=%d existing=%d pending=%d",
                method,
                len(source_records),
                len(existing),
                pending,
            )
            completed = 0
            with output_path.open("a", encoding="utf-8", newline="\n") as stream:
                for source_record in source_records:
                    if source_record.question_id in existing_ids:
                        continue
                    judged = _judge_one(
                        source_record,
                        reference=references[source_record.question_id],
                        client=client,
                        generation=generation,
                    )
                    observed_judges.add((judged.judge_provider, judged.judge_model))
                    _validate_one_judge(observed_judges)
                    stream.write(judged.model_dump_json())
                    stream.write("\n")
                    stream.flush()
                    existing.append(judged)
                    existing_ids.add(judged.question_id)
                    completed += 1
                    if completed == 1 or completed % 10 == 0 or completed == pending:
                        LOGGER.info(
                            "Official-prompt judge %s progress %d/%d: question_id=%s "
                            "label=%s latency=%.1fms elapsed=%.1fs",
                            method,
                            completed,
                            pending,
                            judged.question_id,
                            judged.autoeval_label.label,
                            judged.judge_latency_ms,
                            perf_counter() - method_started,
                        )
            ordered = _order_judge_records(existing, source_records)
            summary = summarize_official_judge_method(method, ordered)
            summaries[method] = summary
            _write_json(
                judge_dir / f"{method}.summary.json",
                summary.model_dump(mode="json"),
            )
            LOGGER.info(
                "Official-prompt judge %s completed in %.1fs: accuracy=%.4f",
                method,
                perf_counter() - method_started,
                summary.accuracy,
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
        judge_dir / "summary.json",
        {
            "score_kind": LOCAL_JUDGE_SCORE_KIND,
            "directly_comparable_to_published_gpt4o_scores": False,
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
            "observed_judge": (
                {
                    "provider": next(iter(observed_judges))[0],
                    "model": next(iter(observed_judges))[1],
                }
                if observed_judges
                else None
            ),
        }
    )
    _write_json(manifest_path, manifest)
    return OfficialJudgeRunResult(
        qa_run=qa_run,
        judge_dir=judge_dir,
        manifest_path=manifest_path,
        summaries=summaries,
    )


def summarize_official_judge_method(
    method: str,
    records: list[OfficialJudgeRecord],
) -> OfficialJudgeMethodSummary:
    """Aggregate official-prompt binary labels without external dependencies."""

    by_type: dict[str, list[OfficialJudgeRecord]] = defaultdict(list)
    for record in records:
        by_type[record.question_type].append(record)
    type_summaries = {
        question_type: OfficialJudgeTypeSummary(
            questions=len(group),
            correct=sum(record.autoeval_label.label for record in group),
            accuracy=_mean([float(record.autoeval_label.label) for record in group]),
        )
        for question_type, group in sorted(by_type.items())
    }
    abstention_records = [
        record for record in records if record.question_id.endswith("_abs")
    ]
    parse_fallbacks = sum(
        record.parse_status in {"ambiguous", "missing"} for record in records
    )
    return OfficialJudgeMethodSummary(
        method=method,
        questions=len(records),
        correct=sum(record.autoeval_label.label for record in records),
        accuracy=_mean([float(record.autoeval_label.label) for record in records]),
        task_averaged_accuracy=_mean(
            [float(summary.accuracy) for summary in type_summaries.values()]
        ),
        abstention_questions=len(abstention_records),
        abstention_correct=sum(
            record.autoeval_label.label for record in abstention_records
        ),
        abstention_accuracy=_mean(
            [float(record.autoeval_label.label) for record in abstention_records]
        ),
        by_question_type=type_summaries,
        parse_fallbacks=parse_fallbacks,
        parse_fallback_rate=_mean(
            [
                float(record.parse_status in {"ambiguous", "missing"})
                for record in records
            ]
        ),
        mean_judge_latency_ms=_mean(
            [float(record.judge_latency_ms) for record in records]
        ),
        mean_judge_input_tokens=_mean(
            [float(record.judge_input_tokens) for record in records]
        ),
        mean_judge_output_tokens=_mean(
            [float(record.judge_output_tokens) for record in records]
        ),
    )


def _judge_one(
    source_record: QASampleRecord,
    *,
    reference: LongMemEvalSample,
    client: ChatClient,
    generation: LLMGenerationConfig,
) -> OfficialJudgeRecord:
    if source_record.question != reference.question:
        raise ValueError(f"QA/reference question mismatch: {source_record.question_id}")
    if source_record.question_type != reference.question_type:
        raise ValueError(f"QA/reference question type mismatch: {source_record.question_id}")
    prompt = build_official_qa_judge_prompt(
        reference.question_type,
        reference.question,
        reference.answer,
        source_record.prediction,
        is_abstention=reference.question_id.endswith("_abs"),
    )
    started_at = perf_counter()
    response = client.chat(
        [ChatMessage(role="user", content=prompt)],
        generation=generation,
    )
    latency_ms = (perf_counter() - started_at) * 1000.0
    label, parse_status = parse_official_qa_judge_response(response.text)
    return OfficialJudgeRecord(
        question_id=reference.question_id,
        question_type=reference.question_type,
        method=source_record.method,
        hypothesis=source_record.prediction,
        autoeval_label=OfficialAutoEvalLabel(model=response.model, label=label),
        judge_provider=response.provider,
        judge_model=response.model,
        judge_response=response.text,
        parse_status=parse_status,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        judge_latency_ms=latency_ms,
        judge_input_tokens=_usage_int(response.usage, "prompt_tokens"),
        judge_output_tokens=_usage_int(response.usage, "completion_tokens"),
        judge_usage=response.usage,
    )


def _validate_qa_manifest(manifest: dict[str, object], *, qa_run: Path) -> None:
    if manifest.get("status") != "completed":
        raise ValueError(f"QA run is not completed: {qa_run}")
    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("QA manifest is missing signature")
    protocol = signature.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("QA manifest is missing protocol provenance")
    if protocol.get("gold_answers_visible_to_reader") is not False:
        raise ValueError("Refusing to judge a QA run that exposed gold answers to the reader")


def _resolve_methods(
    manifest: dict[str, object],
    requested: list[str],
) -> list[str]:
    signature = manifest.get("signature")
    methods = signature.get("methods") if isinstance(signature, dict) else None
    if not isinstance(methods, list) or not all(isinstance(item, str) for item in methods):
        raise ValueError("QA manifest signature is missing methods")
    available = list(dict.fromkeys(methods))
    selected = list(dict.fromkeys(requested)) if requested else available
    unknown = [method for method in selected if method not in available]
    if unknown:
        raise ValueError(f"Requested methods are absent from QA manifest: {unknown}")
    return selected


def _validate_method_coverage(
    records_by_method: dict[str, list[QASampleRecord]],
    *,
    references: dict[str, LongMemEvalSample],
) -> None:
    expected_ids: list[str] | None = None
    for method, records in records_by_method.items():
        if any(record.method != method for record in records):
            raise ValueError(f"QA record method does not match its artifact: {method}")
        ids = [record.question_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate question_id in QA method: {method}")
        missing = [question_id for question_id in ids if question_id not in references]
        if missing:
            raise ValueError(f"QA method contains unknown reference questions: {missing[:3]}")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("QA methods do not have identical ordered question coverage")


def _judge_signature(
    config: LongMemEvalOfficialJudgeConfig,
    *,
    methods: list[str],
    qa_manifest_path: Path,
    generation: LLMGenerationConfig,
) -> dict[str, JsonValue]:
    reference_path = config.reference_data.expanduser().resolve()
    return {
        "qa_run": str(config.qa_run.expanduser().resolve()),
        "qa_manifest_sha256": _sha256(qa_manifest_path),
        "reference_data": str(reference_path),
        "reference_data_sha256": _sha256(reference_path),
        "methods": cast(JsonValue, methods),
        "limit": config.limit,
        "output_subdir": config.output_subdir,
        "score_kind": LOCAL_JUDGE_SCORE_KIND,
        "directly_comparable_to_published_gpt4o_scores": False,
        "protocol": {
            "prompt_version": LONGMEMEVAL_OFFICIAL_JUDGE_PROMPT_VERSION,
            "source": LONGMEMEVAL_OFFICIAL_EVALUATOR_URL,
            "binary_rule": "yes_substring",
            "gold_answers_visible_to_reader": False,
            "gold_answers_visible_to_judge": True,
        },
        "generation": generation.model_dump(mode="json"),
        "judge": dict(config.judge_metadata),
    }


def _prepare_manifest(
    path: Path,
    *,
    signature: dict[str, JsonValue],
    resume: bool,
) -> dict[str, object]:
    if path.exists():
        existing = _read_json_object(path)
        if not resume:
            raise FileExistsError(f"Judge run already exists: {path.parent}. Use --resume.")
        if existing.get("signature") != signature:
            raise ValueError("Judge resume signature differs from the existing run")
        existing.update(
            {
                "status": "running",
                "finished_at": None,
                "error_type": None,
                "error": None,
            }
        )
        _write_json(path, existing)
        return existing
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "running",
        "signature": signature,
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "error_type": None,
        "error": None,
    }
    _write_json(path, manifest)
    return manifest


def _load_qa_records(path: Path, *, limit: int | None) -> list[QASampleRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    records = [
        QASampleRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records if limit is None else records[:limit]


def _load_judge_records(path: Path) -> list[OfficialJudgeRecord]:
    if not path.exists():
        return []
    return [
        OfficialJudgeRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _order_judge_records(
    records: list[OfficialJudgeRecord],
    source_records: list[QASampleRecord],
) -> list[OfficialJudgeRecord]:
    by_id = {record.question_id: record for record in records}
    return [by_id[source.question_id] for source in source_records]


def _validate_one_judge(observed: set[tuple[str, str]]) -> None:
    if len(observed) > 1:
        raise ValueError(f"A judge run must use one shared provider/model, observed={observed}")


def _usage_int(usage: dict[str, JsonValue], key: str) -> int:
    value = usage.get(key, 0)
    return int(value) if isinstance(value, int | float) else 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
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
