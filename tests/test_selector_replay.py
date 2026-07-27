"""Tests for exact selector replay used by VMP-v5.3.1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.llm import (
    ChatMessage,
    LLMGenerationConfig,
    LLMResponse,
    LongMemEvalEvidenceReranker,
    LongMemEvalRerankerConfig,
    SelectorReplayClient,
    build_longmemeval_rerank_prompt,
    load_selector_replay_cache,
    validate_selector_replay_source,
)
from vmp_memos.llm.reranker import (
    LONGMEMEVAL_BOUNDARY_SYSTEM_PROMPT,
    LONGMEMEVAL_RERANK_PROMPT_VERSION,
    LONGMEMEVAL_RERANK_SYSTEM_PROMPT,
)
from vmp_memos.longmemeval.rerank_runner import (
    LongMemEvalRerankRunConfig,
    run_longmemeval_rerank,
)


class DelegateClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            provider="vllm",
            model="Qwen/Qwen2.5-7B-Instruct",
            text='{"selected_slots":["B1","B2"],"confidence":"high"}',
        )


def test_selector_replay_uses_cache_for_selector_and_vllm_for_boundary(
    tmp_path: Path,
) -> None:
    selector_prompt = "exact selector prompt"
    source_run, selector_run = _write_replay_fixture(tmp_path, selector_prompt)
    cache = load_selector_replay_cache(
        selector_run,
        source_run=source_run,
        methods=["vmp_hierarchical"],
        expected_model="Qwen/Qwen2.5-7B-Instruct",
    )
    delegate = DelegateClient()
    client = SelectorReplayClient(delegate, cache)

    replayed = client.chat(
        [
            ChatMessage(role="system", content=LONGMEMEVAL_RERANK_SYSTEM_PROMPT),
            ChatMessage(role="user", content=selector_prompt),
        ]
    )
    live = client.chat(
        [
            ChatMessage(role="system", content=LONGMEMEVAL_BOUNDARY_SYSTEM_PROMPT),
            ChatMessage(role="user", content="symbolic boundary prompt"),
        ]
    )

    assert replayed.text == '{"selected_session_ids":["s6"]}'
    assert live.text == '{"selected_slots":["B1","B2"],"confidence":"high"}'
    assert client.selector_replay_hits == 1
    assert client.boundary_live_calls == 1
    assert delegate.calls == 1


def test_selector_replay_fails_on_prompt_hash_mismatch(tmp_path: Path) -> None:
    source_run, selector_run = _write_replay_fixture(tmp_path, "original prompt")
    cache = load_selector_replay_cache(
        selector_run,
        source_run=source_run,
        methods=["vmp_hierarchical"],
    )
    client = SelectorReplayClient(DelegateClient(), cache)

    with pytest.raises(ValueError, match="no sample context"):
        client.chat(
            [
                ChatMessage(role="system", content=LONGMEMEVAL_RERANK_SYSTEM_PROMPT),
                ChatMessage(role="user", content="changed prompt"),
            ]
        )


def test_selector_replay_uses_sample_identity_when_serialized_prompt_differs(
    tmp_path: Path,
) -> None:
    source_run, selector_run = _write_replay_fixture(tmp_path, "original prompt")
    cache = load_selector_replay_cache(
        selector_run,
        source_run=source_run,
        methods=["vmp_hierarchical"],
    )
    client = SelectorReplayClient(DelegateClient(), cache)
    client.set_selector_replay_context(
        source_method="vmp_hierarchical",
        question_id="q1",
        question="What is the evidence?",
        question_date=None,
        candidate_session_ids=["s1"],
    )

    replayed = client.chat(
        [
            ChatMessage(role="system", content=LONGMEMEVAL_RERANK_SYSTEM_PROMPT),
            ChatMessage(role="user", content="equivalent prompt serialized differently"),
        ]
    )

    assert replayed.text == '{"selected_session_ids":["s6"]}'
    assert replayed.raw_response["prompt_sha256_match"] is False
    assert client.selector_prompt_mismatches == 1


def test_selector_replay_rejects_different_candidate_manifest(tmp_path: Path) -> None:
    source_run, selector_run = _write_replay_fixture(tmp_path, "prompt")
    (source_run / "manifest.json").write_text(
        json.dumps({"status": "completed", "sample_count": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate manifest"):
        load_selector_replay_cache(
            selector_run,
            source_run=source_run,
            methods=["vmp_hierarchical"],
        )


def test_runner_replays_by_sample_identity_when_prompt_hash_differs(
    tmp_path: Path,
) -> None:
    source_run, selector_run = _write_replay_fixture(
        tmp_path,
        "saved prompt serialization",
    )
    memory = RetrievedMemory(
        memory_id="s1",
        source_session_id="s1",
        content="user: exact evidence",
        score=1.0,
        token_count=4,
    )
    method_dir = source_run / "vmp_hierarchical"
    method_dir.mkdir()
    (method_dir / "retrieval.jsonl").write_text(
        json.dumps(_source_record_payload(memory)) + "\n",
        encoding="utf-8",
    )
    cache = load_selector_replay_cache(
        selector_run,
        source_run=source_run,
        methods=["vmp_hierarchical"],
    )
    client = SelectorReplayClient(DelegateClient(), cache)
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_count=1,
            output_top_k=1,
            protected_top_n=0,
            ranked_output_count=1,
        ),
    )

    result = run_longmemeval_rerank(
        LongMemEvalRerankRunConfig(
            source_run=source_run,
            methods=["vmp_hierarchical"],
            output_dir=tmp_path / "output",
        ),
        reranker=reranker,
        run_id="identity-replay",
    )

    assert result.summaries["vmp_hierarchical__vllm_rerank"].processed_questions == 1
    assert client.selector_replay_hits == 1
    assert client.selector_prompt_mismatches == 1


def test_selector_replay_preflight_checks_every_prompt_before_live_calls(
    tmp_path: Path,
) -> None:
    config = LongMemEvalRerankerConfig(
        candidate_count=1,
        output_top_k=1,
        protected_top_n=0,
        ranked_output_count=1,
    )
    memory = RetrievedMemory(
        memory_id="s1",
        source_session_id="s1",
        content="user: exact evidence",
        score=1.0,
        token_count=4,
    )
    prompt = build_longmemeval_rerank_prompt(
        question="What is the evidence?",
        question_date=None,
        candidates=[memory],
        config=config,
    )
    source_run, selector_run = _write_replay_fixture(tmp_path, prompt)
    method_dir = source_run / "vmp_hierarchical"
    method_dir.mkdir()
    (method_dir / "retrieval.jsonl").write_text(
        json.dumps(_source_record_payload(memory)) + "\n",
        encoding="utf-8",
    )
    cache = load_selector_replay_cache(
        selector_run,
        source_run=source_run,
        methods=["vmp_hierarchical"],
    )

    preflight = validate_selector_replay_source(
        cache,
        source_run=source_run,
        methods=["vmp_hierarchical"],
        config=config,
    )

    assert preflight.records_checked == 1
    assert preflight.exact_prompt_matches == 1
    assert preflight.prompt_mismatches == 0


def _write_replay_fixture(
    tmp_path: Path,
    selector_prompt: str,
) -> tuple[Path, Path]:
    source_run = tmp_path / "candidate"
    source_run.mkdir()
    source_manifest_path = source_run / "manifest.json"
    source_manifest_path.write_text(
        json.dumps({"status": "completed", "sample_count": 1}),
        encoding="utf-8",
    )
    source_manifest_sha256 = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()

    selector_run = tmp_path / "selector"
    method_dir = selector_run / "vmp_hierarchical__vllm_boundary"
    method_dir.mkdir(parents=True)
    (selector_run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "source_retrieval_manifest_sha256": source_manifest_sha256,
            }
        ),
        encoding="utf-8",
    )
    prompt_sha256 = hashlib.sha256(selector_prompt.encode("utf-8")).hexdigest()
    memory = RetrievedMemory(
        memory_id="s1",
        source_session_id="s1",
        content="user: exact evidence",
        score=1.0,
        token_count=4,
    )
    record = {
        "question_id": "q1",
        "question": "What is the evidence?",
        "question_date": None,
        "retrieved_memories": [memory.model_dump(mode="json")],
        "rerank_metadata": {
            "source_method": "vmp_hierarchical",
            "prompt_version": LONGMEMEVAL_RERANK_PROMPT_VERSION,
            "prompt_sha256": prompt_sha256,
            "provider": "vllm",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            "response_text": '{"selected_session_ids":["s6"]}',
        }
    }
    (method_dir / "retrieval.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    return source_run, selector_run


def _source_record_payload(memory: RetrievedMemory) -> dict[str, object]:
    return {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "What is the evidence?",
        "answer": "evidence",
        "question_date": None,
        "method": "vmp_hierarchical",
        "is_abstention": False,
        "gold_session_ids": ["s1"],
        "retrieved_session_ids": ["s1"],
        "retrieved_memories": [memory.model_dump(mode="json")],
        "retrieved_tokens": memory.token_count,
    }
