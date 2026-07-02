"""Run reproducible LongMemEval session-level retrieval experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vmp_memos.embeddings import (
    CachedEmbedder,
    SentenceTransformerEmbedder,
    SQLiteEmbeddingCache,
)
from vmp_memos.frameworks import FrameworkRuntimeConfig
from vmp_memos.longmemeval import LongMemEvalRunConfig
from vmp_memos.longmemeval.retrieval_runner import run_longmemeval_retrieval

DEFAULT_METHODS = (
    "empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--top-k", type=int, default=5, help="Evidence depth used by QA.")
    parser.add_argument(
        "--retrieval-depth",
        type=int,
        default=10,
        help="Depth saved and evaluated; must be >= --top-k for Recall@10.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/longmemeval"))
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split", choices=("dev", "test"), default=None)
    parser.add_argument("--vmp-tuned-model", type=Path, default=None)
    parser.add_argument(
        "--ingestion-granularity",
        choices=("session", "turn"),
        default="session",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--embedding-cache-db",
        type=Path,
        default=None,
        help="Optional persistent SQLite cache for repeated text embeddings.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument(
        "--vllm-base-url",
        default=os.getenv("VMP_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--vllm-model",
        default=os.getenv("VMP_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument(
        "--official-memory-infer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use each official framework's native LLM memory extraction.",
    )
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
        help="URI of a dedicated Neo4j instance. Graphiti clears it per question.",
    )
    parser.add_argument(
        "--graphiti-neo4j-user",
        default=os.getenv("VMP_GRAPHITI_NEO4J_USER", "neo4j"),
    )
    parser.add_argument(
        "--graphiti-allow-destructive-reset",
        action="store_true",
        help="Acknowledge that the dedicated Graphiti Neo4j instance may be cleared.",
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
        "--no-embeddings",
        action="store_true",
        help="Use lexical fallbacks for a dependency-light smoke run only.",
    )
    parser.add_argument(
        "--include-abstention-retrieval-metrics",
        action="store_true",
        help="Only useful for datasets that provide gold evidence for abstention cases.",
    )
    args = parser.parse_args()

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    normalized_methods = {
        method.casefold().replace("-", "_") for method in methods
    }
    official_methods = {
        "mem0",
        "mem0_official",
        "langmem",
        "langmem_official",
        "graphiti",
        "graphiti_official",
        "letta",
        "letta_official",
    }
    if args.no_embeddings and normalized_methods & official_methods:
        parser.error("--no-embeddings cannot be used with official memory adapters")
    if normalized_methods & {"graphiti", "graphiti_official"}:
        if not args.graphiti_allow_destructive_reset:
            parser.error(
                "Graphiti requires --graphiti-allow-destructive-reset and a "
                "dedicated Neo4j instance"
            )
        if not os.getenv("VMP_GRAPHITI_NEO4J_PASSWORD"):
            parser.error("Graphiti requires VMP_GRAPHITI_NEO4J_PASSWORD")
    embedder = None
    if not args.no_embeddings and _needs_embeddings(methods):
        base_embedder = SentenceTransformerEmbedder(
            args.embedding_model,
            device=args.embedding_device,
            cache_folder=args.embedding_cache_dir,
            batch_size=args.embedding_batch_size,
        )
        embedder = (
            CachedEmbedder(
                base_embedder,
                SQLiteEmbeddingCache(args.embedding_cache_db),
            )
            if args.embedding_cache_db is not None
            else base_embedder
        )

    config = LongMemEvalRunConfig(
        data_path=args.data,
        methods=methods,
        top_k=args.top_k,
        retrieval_depth=args.retrieval_depth,
        limit=args.limit,
        output_dir=args.output_dir,
        ingestion_granularity=args.ingestion_granularity,
        skip_abstention_for_retrieval=not args.include_abstention_retrieval_metrics,
        split_manifest_path=args.split_manifest,
        split_name=args.split,
        vmp_tuned_model_path=args.vmp_tuned_model,
        metadata={
            "embedding_model": None if args.no_embeddings else args.embedding_model,
            "embedding_device": None if args.no_embeddings else args.embedding_device,
            "embedding_cache_db": (
                str(args.embedding_cache_db)
                if args.embedding_cache_db is not None
                else None
            ),
            "lexical_smoke_only": args.no_embeddings,
        },
    )
    framework_runtime = FrameworkRuntimeConfig(
        vllm_base_url=args.vllm_base_url,
        llm_model=args.vllm_model,
        vllm_api_key=os.getenv("VMP_LLM_API_KEY") or None,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        embedding_device=args.embedding_device,
        official_memory_infer=args.official_memory_infer,
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
    result = run_longmemeval_retrieval(
        config,
        embedder=embedder,
        framework_runtime=framework_runtime,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "manifest": str(result.manifest_path),
                "methods": {
                    name: summary.model_dump(mode="json")
                    for name, summary in result.summaries.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _needs_embeddings(methods: list[str]) -> bool:
    vector_methods = {
        "naive_vector",
        "naive_vector_rag",
        "vector_rag",
        "vector_recency",
        "vector_importance",
        "vmp_rule",
        "vmp_tuned",
        "langmem",
        "langmem_official",
        "graphiti",
        "graphiti_official",
    }
    normalized = {method.casefold().replace("-", "_") for method in methods}
    uses_vmp_tuned = any(
        method == "vmp_full" or method.startswith("vmp_tuned")
        for method in normalized
    )
    return uses_vmp_tuned or bool(normalized & vector_methods)


if __name__ == "__main__":
    raise SystemExit(main())
