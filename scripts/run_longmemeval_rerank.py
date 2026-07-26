#!/usr/bin/env python3
"""Rerank saved LongMemEval candidates with one shared local-vLLM policy."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from vmp_memos.llm import (
    LLMGenerationConfig,
    LongMemEvalEvidenceReranker,
    LongMemEvalRerankerConfig,
    VLLMClient,
    VLLMClientConfig,
)
from vmp_memos.longmemeval.rerank_runner import (
    LongMemEvalRerankRunConfig,
    run_longmemeval_rerank,
)

LOGGER = logging.getLogger("vmp_memos.run_longmemeval_rerank")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/longmemeval"),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("VMP_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VMP_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VMP_LLM_API_KEY") or None,
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--output-top-k", type=int, default=5)
    parser.add_argument("--protected-top-n", type=int, default=4)
    parser.add_argument("--ranked-output-count", type=int, default=10)
    parser.add_argument("--max-candidate-chars", type=int, default=1200)
    parser.add_argument("--max-excerpt-turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    if not methods:
        parser.error("--methods must contain at least one method")
    generation = LLMGenerationConfig(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    client = VLLMClient(
        VLLMClientConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            generation=generation,
        )
    )
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_count=args.candidate_count,
            output_top_k=args.output_top_k,
            protected_top_n=args.protected_top_n,
            ranked_output_count=args.ranked_output_count,
            max_candidate_chars=args.max_candidate_chars,
            max_excerpt_turns=args.max_excerpt_turns,
            generation=generation,
        ),
    )
    LOGGER.info(
        "Starting shared rerank: source=%s run_id=%s methods=%s "
        "model=%s candidates=%d protect=%d/%d resume=%s",
        args.source_run,
        args.run_id,
        ",".join(methods),
        args.model,
        args.candidate_count,
        args.protected_top_n,
        args.output_top_k,
        args.resume,
    )
    result = run_longmemeval_rerank(
        LongMemEvalRerankRunConfig(
            source_run=args.source_run,
            methods=methods,
            output_dir=args.output_dir,
            resume=args.resume,
            limit=args.limit,
            metadata={
                "paper_version": "VMP-v5.2",
                "shared_across_frameworks": True,
                "test_labels_used": False,
            },
        ),
        reranker=reranker,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "manifest": str(result.manifest_path),
                "methods": {
                    method: summary.model_dump(mode="json")
                    for method, summary in result.summaries.items()
                },
                "test_labels_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
