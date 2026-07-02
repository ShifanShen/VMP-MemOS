"""Verify an official memory adapter against the configured local models."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from vmp_memos.frameworks import FrameworkRuntimeConfig, adapter_for_name
from vmp_memos.longmemeval import LongMemEvalSample, sample_to_session_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework",
        choices=("mem0", "langmem", "graphiti", "letta"),
        required=True,
    )
    parser.add_argument(
        "--vllm-base-url",
        default=os.getenv("VMP_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--vllm-model",
        default=os.getenv("VMP_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument(
        "--official-llm-max-tokens",
        type=int,
        default=int(os.getenv("VMP_OFFICIAL_LLM_MAX_TOKENS", "512")),
    )
    parser.add_argument(
        "--official-llm-temperature",
        type=float,
        default=float(os.getenv("VMP_OFFICIAL_LLM_TEMPERATURE", "0.0")),
    )
    parser.add_argument(
        "--graphiti-neo4j-uri",
        default=os.getenv("VMP_GRAPHITI_NEO4J_URI", "bolt://127.0.0.1:7687"),
    )
    parser.add_argument(
        "--graphiti-neo4j-user",
        default=os.getenv("VMP_GRAPHITI_NEO4J_USER", "neo4j"),
    )
    parser.add_argument(
        "--graphiti-allow-destructive-reset",
        action="store_true",
        help="Acknowledge use of a dedicated Neo4j instance that this smoke clears.",
    )
    parser.add_argument(
        "--letta-base-url",
        default=os.getenv("VMP_LETTA_BASE_URL", "http://127.0.0.1:8283"),
    )
    parser.add_argument(
        "--letta-embedding-base-url",
        default=os.getenv(
            "VMP_LETTA_EMBEDDING_BASE_URL",
            "http://127.0.0.1:8001/v1",
        ),
    )
    parser.add_argument(
        "--letta-context-window",
        type=int,
        default=int(os.getenv("VMP_LETTA_CONTEXT_WINDOW", "16384")),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("outputs/longmemeval/audit/smoke_workspace"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/longmemeval/audit"),
    )
    args = parser.parse_args()
    if args.framework == "graphiti":
        if not args.graphiti_allow_destructive_reset:
            parser.error(
                "Graphiti smoke requires --graphiti-allow-destructive-reset"
            )
        if not os.getenv("VMP_GRAPHITI_NEO4J_PASSWORD"):
            parser.error("Graphiti smoke requires VMP_GRAPHITI_NEO4J_PASSWORD")

    runtime = FrameworkRuntimeConfig(
        vllm_base_url=args.vllm_base_url,
        llm_model=args.vllm_model,
        vllm_api_key=os.getenv("VMP_LLM_API_KEY") or None,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        embedding_device=args.embedding_device,
        official_memory_infer=True,
        official_llm_max_tokens=args.official_llm_max_tokens,
        official_llm_temperature=args.official_llm_temperature,
        graphiti_neo4j_uri=args.graphiti_neo4j_uri,
        graphiti_neo4j_user=args.graphiti_neo4j_user,
        graphiti_neo4j_password=(
            os.getenv("VMP_GRAPHITI_NEO4J_PASSWORD") or None
        ),
        graphiti_allow_destructive_reset=args.graphiti_allow_destructive_reset,
        letta_base_url=args.letta_base_url,
        letta_api_key=os.getenv("VMP_LETTA_API_KEY") or None,
        letta_server_version=os.getenv("VMP_LETTA_SERVER_VERSION", "0.16.8"),
        letta_embedding_base_url=args.letta_embedding_base_url,
        letta_context_window=args.letta_context_window,
    )
    output_path = args.output_dir / f"{args.framework}_smoke.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework": args.framework,
        "framework_version": version(_distribution_for(args.framework)),
        "vllm_base_url": runtime.vllm_base_url,
        "llm_model": runtime.llm_model,
        "embedding_model": runtime.embedding_model,
        "embedding_dimension": runtime.embedding_dimension,
        "official_llm_max_tokens": runtime.official_llm_max_tokens,
        "official_llm_temperature": runtime.official_llm_temperature,
        "graphiti_neo4j_uri": (
            runtime.graphiti_neo4j_uri
            if args.framework == "graphiti"
            else None
        ),
        "server_version": (
            runtime.letta_server_version
            if args.framework == "letta"
            else None
        ),
        "started_at": datetime.now(UTC).isoformat(),
    }
    adapter = adapter_for_name(args.framework, runtime=runtime)
    try:
        sample = LongMemEvalSample.model_validate(_smoke_sample())
        adapter.reset(args.workspace / "question_1")
        for events in sample_to_session_events(sample):
            adapter.ingest_session(events)
        adapter.finalize_ingestion()
        evidence = adapter.retrieve(
            sample.question,
            top_k=5,
            question_date=sample.question_date,
            metadata={
                "question_id": sample.question_id,
                "question_type": sample.question_type,
            },
        )
        if not evidence:
            raise RuntimeError("official adapter returned no evidence")
        if not all(item.source_session_id for item in evidence):
            raise RuntimeError("official adapter did not export source-session provenance")
        payload.update(
            {
                "status": "passed",
                "finished_at": datetime.now(UTC).isoformat(),
                "evidence_count": len(evidence),
                "top_source_session_id": evidence[0].source_session_id,
                "adapter_stats": adapter.stats(),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(output_path, payload)
        raise
    finally:
        adapter.close()
    _write_json(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _distribution_for(framework: str) -> str:
    return {
        "mem0": "mem0ai",
        "langmem": "langmem",
        "graphiti": "graphiti-core",
        "letta": "letta-client",
    }[framework]


def _smoke_sample() -> dict:
    return {
        "question_id": "official_framework_smoke",
        "question_type": "knowledge_update",
        "question": "What activity does Alex now prefer?",
        "answer": "swimming",
        "question_date": "2024-02-01",
        "haystack_session_ids": ["s_old", "s_new"],
        "haystack_dates": ["2024-01-01", "2024-01-20"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Alex used to prefer hiking."},
                {"role": "assistant", "content": "I will remember that preference."},
            ],
            [
                {
                    "role": "user",
                    "content": "Alex now prefers swimming instead of hiking.",
                },
                {"role": "assistant", "content": "I updated Alex's preference."},
            ],
        ],
        "answer_session_ids": ["s_new"],
        "has_answer": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
