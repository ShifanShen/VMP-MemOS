"""Synthetic tests for resumable shared-vLLM LongMemEval reranking."""

from __future__ import annotations

import json
from typing import Any

import pytest

from vmp_memos.evaluation import compute_retrieval_metrics
from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.llm import (
    LONGMEMEVAL_ROLE_AWARE_EXCERPT_VERSION,
    LONGMEMEVAL_V6_ATOMIC_FACT_SELECTOR_PROMPT_VERSION,
    LONGMEMEVAL_V6_SET_COVERAGE_BOUNDARY_VERSION,
    LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION,
    LONGMEMEVAL_V552_PAIRWISE_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_V552_PAIRWISE_SELECTOR_PROMPT_VERSION,
    LLMResponse,
    LongMemEvalEvidenceReranker,
    LongMemEvalRerankerConfig,
)
from vmp_memos.longmemeval.rerank_runner import (
    LongMemEvalRerankRunConfig,
    _load_records,
    run_longmemeval_rerank,
)
from vmp_memos.longmemeval.retrieval_runner import (
    RetrievalSampleRecord,
    summarize_method,
)
from vmp_memos.longmemeval.tables import export_retrieval_tables


class CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[Any], *, generation: Any = None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            provider="vllm",
            model="Qwen/Qwen2.5-7B-Instruct",
            text=json.dumps(
                {
                    "evidence_needs": ["buried evidence"],
                    "selected_session_ids": ["s6", "s1", "s2", "s3", "s4"],
                    "ranked_session_ids": ["s6", "s1", "s2", "s3", "s4", "s5"],
                }
            ),
            usage={"prompt_tokens": 80, "completion_tokens": 16},
        )


class BoundaryCountingClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[Any], *, generation: Any = None) -> LLMResponse:
        self.calls += 1
        if self.calls % 2 == 1:
            payload = {
                "evidence_needs": ["buried evidence"],
                "selected_session_ids": ["s6", "s1", "s2", "s3", "s4"],
                "ranked_session_ids": ["s6", "s1", "s2", "s3", "s4", "s5"],
            }
        else:
            payload = {
                "evidence_needs": ["buried evidence"],
                "selected_boundary_session_ids": ["s4", "s6"],
                "decision": "replace_one",
                "confidence": "high",
            }
        return LLMResponse(
            provider="vllm",
            model="Qwen/Qwen2.5-7B-Instruct",
            text=json.dumps(payload),
            usage={"prompt_tokens": 40, "completion_tokens": 12},
        )


class RejectingPairwiseClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[Any], *, generation: Any = None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            provider="vllm",
            model="Qwen/Qwen2.5-7B-Instruct",
            text=json.dumps(
                {
                    "decision": "reject",
                    "evidence_needs": ["N1: buried evidence"],
                    "supports_needs": [],
                    "challenger_spans": [],
                    "displaced_slot": None,
                    "adds_missing_evidence": False,
                    "displaced_slot_redundant": False,
                    "confidence": "high",
                }
            ),
            usage={"prompt_tokens": 120, "completion_tokens": 20},
        )


class EmptyAtomicFactClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[Any], *, generation: Any = None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            provider="vllm",
            model="Qwen/Qwen2.5-7B-Instruct",
            text=json.dumps({"candidate_relevant": False, "facts": []}),
            usage={"prompt_tokens": 90, "completion_tokens": 8},
        )


def test_question_id_filter_selects_exact_smoke_cases(tmp_path) -> None:
    path = tmp_path / "retrieval.jsonl"
    first = _source_record().model_copy(update={"question_id": "q-first"})
    target = _source_record().model_copy(update={"question_id": "q-target"})
    path.write_text(
        first.model_dump_json() + "\n" + target.model_dump_json() + "\n",
        encoding="utf-8",
    )

    records = _load_records(path, question_ids=["q-target"])

    assert [record.question_id for record in records] == ["q-target"]
    with pytest.raises(ValueError, match="absent"):
        _load_records(path, question_ids=["missing"])


def test_rerank_runner_writes_replayable_records_and_resumes(tmp_path) -> None:
    source_run = tmp_path / "outputs" / "runs" / "source"
    source_run.mkdir(parents=True)
    manifest_path = source_run / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "completed",
                "run_id": "source",
                "dataset": "longmemeval-cleaned",
                "data_sha256": "data",
                "sample_count": 1,
                "split": {"name": "dev", "split_id": "split"},
            }
        ),
        encoding="utf-8",
    )
    source_record = _source_record()
    for method in ("vmp_tuned", "vmp_hierarchical"):
        method_dir = source_run / method
        method_dir.mkdir()
        method_record = source_record.model_copy(update={"method": method})
        (method_dir / "retrieval.jsonl").write_text(
            method_record.model_dump_json() + "\n",
            encoding="utf-8",
        )
        summary = summarize_method(method, [method_record])
        (method_dir / "summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    client = CountingClient()
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(candidate_count=6, ranked_output_count=6),
    )
    config = LongMemEvalRerankRunConfig(
        source_run=source_run,
        methods=["vmp_tuned", "vmp_hierarchical"],
        output_dir=tmp_path / "outputs",
    )
    result = run_longmemeval_rerank(
        config,
        reranker=reranker,
        run_id="reranked",
    )

    assert client.calls == 2
    v52 = result.summaries["vmp_hierarchical__vllm_rerank"]
    assert v52.metrics["recall_all@5"] == 1.0
    assert v52.parse_fallbacks == 0
    assert v52.recovered_questions == 1
    assert v52.candidate_oracle_recoverable_questions == 1
    assert v52.reranker_provider == "vllm"
    assert v52.candidate_planner_version == "identity_v1"
    assert v52.candidate_planner_applied_questions == 0
    assert v52.candidate_planner_identity_questions == 1
    record = _read_jsonl(result.run_dir / "vmp_hierarchical__vllm_rerank" / "retrieval.jsonl")[0]
    assert record["retrieved_session_ids"][:5] == [
        "s1",
        "s2",
        "s3",
        "s4",
        "s6",
    ]
    assert record["rerank_metadata"]["test_labels_used"] is False
    assert record["rerank_metadata"]["candidate_plan"]["planner_version"] == "identity_v1"
    assert record["rerank_metadata"]["candidate_plan"]["applied"] is False
    assert record["rerank_metadata"]["transition_vs_source"] == "recovered"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["fairness"]["shared_across_methods"] is True
    assert manifest["fairness"]["shared_candidate_planner"] is True
    assert manifest["fairness"]["candidate_planner_uses_gold_labels"] is False
    assert manifest["test_labels_used"] is False
    tables = export_retrieval_tables(result.run_dir, output_dir=tmp_path / "tables")
    assert tables["table1_retrieval_overall_csv"].exists()

    interrupted_path = result.run_dir / "vmp_hierarchical__vllm_rerank" / "retrieval.jsonl"
    with interrupted_path.open("a", encoding="utf-8") as stream:
        stream.write('{"interrupted":')
    interrupted_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    interrupted_manifest.update(
        {
            "status": "failed",
            "error_type": "ConnectionError",
            "error": "temporary vLLM outage",
            "finished_at": "2024-01-01T00:00:00+00:00",
            "wall_duration_seconds": 12.0,
        }
    )
    result.manifest_path.write_text(
        json.dumps(interrupted_manifest),
        encoding="utf-8",
    )
    resumed = run_longmemeval_rerank(
        config.model_copy(update={"resume": True}),
        reranker=reranker,
        run_id="reranked",
    )
    assert client.calls == 2
    assert resumed.summaries["vmp_hierarchical__vllm_rerank"].processed_questions == 1
    assert '{"interrupted":' not in interrupted_path.read_text(encoding="utf-8")
    resumed_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert resumed_manifest["status"] == "completed"
    assert "error_type" not in resumed_manifest
    assert "error" not in resumed_manifest
    assert resumed_manifest["previous_attempts"] == [
        {
            "status": "failed",
            "finished_at": "2024-01-01T00:00:00+00:00",
            "wall_duration_seconds": 12.0,
            "error_type": "ConnectionError",
            "error": "temporary vLLM outage",
        }
    ]


def test_v53_runner_audits_shared_boundary_verification(tmp_path) -> None:
    source_run = tmp_path / "outputs" / "runs" / "source"
    source_run.mkdir(parents=True)
    (source_run / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "completed",
                "run_id": "source",
                "dataset": "longmemeval-cleaned",
                "data_sha256": "data",
                "sample_count": 1,
                "split": {"name": "dev", "split_id": "split"},
            }
        ),
        encoding="utf-8",
    )
    method_dir = source_run / "vmp_hierarchical"
    method_dir.mkdir()
    source_record = _source_record()
    (method_dir / "retrieval.jsonl").write_text(
        source_record.model_dump_json() + "\n",
        encoding="utf-8",
    )

    client = BoundaryCountingClient()
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_count=6,
            protected_top_n=3,
            ranked_output_count=6,
            boundary_verification=True,
        ),
    )
    result = run_longmemeval_rerank(
        LongMemEvalRerankRunConfig(
            source_run=source_run,
            methods=["vmp_hierarchical"],
            output_dir=tmp_path / "outputs",
        ),
        reranker=reranker,
        run_id="v53",
    )

    assert client.calls == 2
    summary = result.summaries["vmp_hierarchical__vllm_boundary"]
    assert summary.metrics["recall_all@5"] == 1.0
    assert summary.boundary_verification is True
    assert summary.boundary_calls == 1
    assert summary.boundary_replacements_accepted == 1
    assert summary.boundary_parse_fallbacks == 0
    record = _read_jsonl(result.run_dir / "vmp_hierarchical__vllm_boundary" / "retrieval.jsonl")[0]
    assert record["retrieved_session_ids"][:5] == ["s1", "s2", "s3", "s4", "s6"]
    assert record["rerank_metadata"]["boundary"]["confidence"] == "high"
    assert record["adapter_stats"]["rerank_calls"] == 2
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["fairness"]["two_stage_boundary_verification"] is True
    resumed = run_longmemeval_rerank(
        LongMemEvalRerankRunConfig(
            source_run=source_run,
            methods=["vmp_hierarchical"],
            output_dir=tmp_path / "outputs",
            resume=True,
        ),
        reranker=reranker,
        run_id="v53",
    )
    assert client.calls == 2
    assert resumed.summaries["vmp_hierarchical__vllm_boundary"].processed_questions == 1


def test_v552_runner_counts_five_selector_calls_without_extra_boundary_call(
    tmp_path,
) -> None:
    source_run = tmp_path / "outputs" / "runs" / "source-v552"
    method_dir = source_run / "vmp_hierarchical"
    method_dir.mkdir(parents=True)
    (source_run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "source-v552",
                "sample_count": 1,
                "split": {"name": "dev", "split_id": "split"},
            }
        ),
        encoding="utf-8",
    )
    source_record = _source_record(10)
    (method_dir / "retrieval.jsonl").write_text(
        source_record.model_dump_json() + "\n",
        encoding="utf-8",
    )
    client = RejectingPairwiseClient()
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            prompt_version=LONGMEMEVAL_V552_PAIRWISE_SELECTOR_PROMPT_VERSION,
            boundary_prompt_version=LONGMEMEVAL_V552_PAIRWISE_BOUNDARY_PROMPT_VERSION,
            candidate_excerpt_version=LONGMEMEVAL_ROLE_AWARE_EXCERPT_VERSION,
            candidate_planner_version=(
                LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION
            ),
            candidate_count=10,
            protected_top_n=3,
            ranked_output_count=10,
            boundary_verification=True,
        ),
    )

    result = run_longmemeval_rerank(
        LongMemEvalRerankRunConfig(
            source_run=source_run,
            methods=["vmp_hierarchical"],
            output_dir=tmp_path / "outputs",
            require_full_candidate_count=True,
        ),
        reranker=reranker,
        run_id="v552",
    )

    assert client.calls == 5
    summary = result.summaries["vmp_hierarchical__vllm_boundary"]
    assert summary.selector_calls == 5
    assert summary.selector_call_fallbacks == 0
    assert summary.boundary_calls == 0
    record = _read_jsonl(
        result.run_dir
        / "vmp_hierarchical__vllm_boundary"
        / "retrieval.jsonl"
    )[0]
    assert record["adapter_stats"]["rerank_calls"] == 5
    assert record["rerank_metadata"]["selector_call_count"] == 5
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["fairness"]["anonymous_pairwise_challenger_protocol"] is True
    assert manifest["fairness"]["integrated_pairwise_boundary_verification"] is True
    assert manifest["fairness"]["two_stage_boundary_verification"] is False


def test_v6_runner_audits_ten_fact_calls_and_deterministic_coverage(tmp_path) -> None:
    source_run = tmp_path / "outputs" / "runs" / "source-v6"
    method_dir = source_run / "vmp_hierarchical"
    method_dir.mkdir(parents=True)
    (source_run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "source-v6",
                "sample_count": 1,
                "split": {"name": "dev", "split_id": "split"},
            }
        ),
        encoding="utf-8",
    )
    (method_dir / "retrieval.jsonl").write_text(
        _source_record(10).model_dump_json() + "\n",
        encoding="utf-8",
    )
    client = EmptyAtomicFactClient()
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            prompt_version=LONGMEMEVAL_V6_ATOMIC_FACT_SELECTOR_PROMPT_VERSION,
            boundary_prompt_version=LONGMEMEVAL_V6_SET_COVERAGE_BOUNDARY_VERSION,
            candidate_excerpt_version=LONGMEMEVAL_ROLE_AWARE_EXCERPT_VERSION,
            candidate_planner_version=(
                LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION
            ),
            candidate_count=10,
            protected_top_n=3,
            ranked_output_count=10,
            boundary_verification=True,
        ),
    )

    result = run_longmemeval_rerank(
        LongMemEvalRerankRunConfig(
            source_run=source_run,
            methods=["vmp_hierarchical"],
            output_dir=tmp_path / "outputs",
            require_full_candidate_count=True,
        ),
        reranker=reranker,
        run_id="v6",
    )

    assert client.calls == 10
    summary = result.summaries["vmp_hierarchical__vllm_boundary"]
    assert summary.selector_calls == 10
    assert summary.boundary_calls == 0
    assert summary.evidence_operator_counts == {"list": 1}
    record = _read_jsonl(
        result.run_dir / "vmp_hierarchical__vllm_boundary" / "retrieval.jsonl"
    )[0]
    assert record["rerank_metadata"]["question_evidence_plan"]["operator"] == "list"
    assert record["rerank_metadata"]["coverage_selection"] is not None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["fairness"]["structured_atomic_fact_protocol"] is True
    assert manifest["fairness"]["deterministic_set_coverage"] is True
    assert manifest["fairness"]["two_stage_boundary_verification"] is False


def test_rerank_preflight_rejects_truncated_duplicate_session_pool(tmp_path) -> None:
    source_run = tmp_path / "outputs" / "runs" / "source"
    method_dir = source_run / "vmp_hierarchical"
    method_dir.mkdir(parents=True)
    (source_run / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "completed",
                "run_id": "source",
                "dataset": "longmemeval-cleaned",
                "data_sha256": "data",
                "sample_count": 1,
                "split": {"name": "dev", "split_id": "split"},
            }
        ),
        encoding="utf-8",
    )
    source_record = _source_record()
    truncated = source_record.model_copy(
        update={"retrieved_memories": source_record.retrieved_memories[:6]}
    )
    (method_dir / "retrieval.jsonl").write_text(
        truncated.model_dump_json() + "\n",
        encoding="utf-8",
    )
    client = CountingClient()
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(candidate_count=6, ranked_output_count=6),
    )

    with pytest.raises(
        ValueError,
        match=r"requires 6 unique sessions.*q1=5",
    ):
        run_longmemeval_rerank(
            LongMemEvalRerankRunConfig(
                source_run=source_run,
                methods=["vmp_hierarchical"],
                output_dir=tmp_path / "outputs",
                require_full_candidate_count=True,
            ),
            reranker=reranker,
            run_id="strict-depth",
        )

    assert client.calls == 0


def _source_record(count: int = 6) -> RetrievalSampleRecord:
    unique_memories = [
        RetrievedMemory(
            memory_id=f"s{index}",
            source_session_id=f"s{index}",
            content=f"user: Evidence from session {index}.",
            score=1.0 / index,
            token_count=8,
        )
        for index in range(1, count + 1)
    ]
    duplicate_first_session = unique_memories[0].model_copy(
        update={"memory_id": "s1-duplicate-chunk"}
    )
    memories = [
        unique_memories[0],
        duplicate_first_session,
        *unique_memories[1:],
    ]
    session_ids = [memory.source_session_id for memory in unique_memories]
    assert all(session_id is not None for session_id in session_ids)
    ranked_ids = [str(session_id) for session_id in session_ids]
    return RetrievalSampleRecord(
        question_id="q1",
        question_type="multi-session",
        question="Which buried evidence is required?",
        answer="answer",
        question_date="2024-02-01",
        method="vmp_hierarchical",
        is_abstention=False,
        gold_session_ids=["s6"],
        retrieved_session_ids=ranked_ids,
        retrieved_memories=memories,
        metrics=compute_retrieval_metrics(ranked_ids, ["s6"]),
        retrieved_tokens=sum(memory.token_count for memory in memories),
        adapter_stats={"total_retrieval_latency_ms": 5.0},
    )


def _read_jsonl(path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
