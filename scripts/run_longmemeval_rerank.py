#!/usr/bin/env python3
"""Rerank saved LongMemEval candidates with one shared local-vLLM policy."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from pydantic import JsonValue

from vmp_memos.llm import (
    LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_IDENTITY_CANDIDATE_PLANNER_VERSION,
    LONGMEMEVAL_RERANK_PROMPT_VERSION,
    LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_PROMPT_VERSION,
    LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION,
    LONGMEMEVAL_V551_COMPLETE_CHALLENGER_SELECTOR_PROMPT_VERSION,
    LLMGenerationConfig,
    LongMemEvalEvidenceReranker,
    LongMemEvalRerankerConfig,
    SelectorReplayClient,
    VLLMClient,
    VLLMClientConfig,
    load_selector_replay_cache,
    validate_selector_replay_source,
)
from vmp_memos.longmemeval.rerank_runner import (
    LongMemEvalRerankRunConfig,
    run_longmemeval_rerank,
)

LOGGER = logging.getLogger("vmp_memos.run_longmemeval_rerank")


def _paper_version(
    *,
    selector_prompt_version: str,
    boundary_prompt_version: str,
    boundary_verification: bool,
) -> str:
    if (
        selector_prompt_version
        == LONGMEMEVAL_V551_COMPLETE_CHALLENGER_SELECTOR_PROMPT_VERSION
        and boundary_prompt_version
        == LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION
    ):
        return "VMP-v5.5.1"
    if (
        selector_prompt_version == LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION
        and boundary_prompt_version
        == LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION
    ):
        return "VMP-v5.5"
    if (
        selector_prompt_version == LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_PROMPT_VERSION
        and boundary_prompt_version
        == LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION
    ):
        return "VMP-v5.4"
    if boundary_prompt_version == LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION:
        return "VMP-v5.3.2"
    if boundary_prompt_version == LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION:
        return "VMP-v5.3.1"
    return "VMP-v5.3" if boundary_verification else "VMP-v5.2"


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
    parser.add_argument(
        "--candidate-planner-version",
        default=LONGMEMEVAL_IDENTITY_CANDIDATE_PLANNER_VERSION,
    )
    parser.add_argument("--candidate-planner-rrf-k", type=int, default=60)
    parser.add_argument(
        "--candidate-planner-hierarchical-weight",
        type=float,
        default=0.8,
    )
    parser.add_argument("--output-top-k", type=int, default=5)
    parser.add_argument("--protected-top-n", type=int, default=4)
    parser.add_argument("--ranked-output-count", type=int, default=10)
    parser.add_argument("--max-candidate-chars", type=int, default=1200)
    parser.add_argument("--max-excerpt-turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--prompt-version",
        default=LONGMEMEVAL_RERANK_PROMPT_VERSION,
    )
    parser.add_argument("--boundary-verification", action="store_true")
    parser.add_argument(
        "--boundary-prompt-version",
        default=LONGMEMEVAL_BOUNDARY_PROMPT_VERSION,
        help="Boundary protocol version; defaults to the legacy V5.3 protocol.",
    )
    parser.add_argument("--boundary-protected-top-n", type=int, default=3)
    parser.add_argument("--boundary-max-promotions", type=int, default=2)
    parser.add_argument("--boundary-max-tokens", type=int, default=256)
    parser.add_argument("--boundary-min-confidence", default="high")
    parser.add_argument(
        "--selector-replay-run",
        type=Path,
        default=None,
        help="Replay exact first-stage selector responses from a completed rerank run.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--require-full-candidate-count",
        action="store_true",
        help=(
            "Fail before any LLM call unless every sample has candidate-count "
            "unique sessions after deduplication."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    if not methods:
        parser.error("--methods must contain at least one method")
    if args.selector_replay_run is not None and not args.boundary_verification:
        parser.error("--selector-replay-run requires --boundary-verification")
    generation = LLMGenerationConfig(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    boundary_generation = LLMGenerationConfig(
        max_tokens=args.boundary_max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    live_client = VLLMClient(
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
    reranker_config = LongMemEvalRerankerConfig(
        prompt_version=args.prompt_version,
        candidate_planner_version=args.candidate_planner_version,
        candidate_planner_rrf_k=args.candidate_planner_rrf_k,
        candidate_planner_hierarchical_weight=(
            args.candidate_planner_hierarchical_weight
        ),
        candidate_count=args.candidate_count,
        output_top_k=args.output_top_k,
        protected_top_n=args.protected_top_n,
        ranked_output_count=args.ranked_output_count,
        max_candidate_chars=args.max_candidate_chars,
        max_excerpt_turns=args.max_excerpt_turns,
        generation=generation,
        boundary_verification=args.boundary_verification,
        boundary_prompt_version=args.boundary_prompt_version,
        boundary_protected_top_n=args.boundary_protected_top_n,
        boundary_max_promotions=args.boundary_max_promotions,
        boundary_min_confidence=args.boundary_min_confidence,
        boundary_generation=boundary_generation,
    )
    replay_client: SelectorReplayClient | None = None
    replay_metadata: dict[str, JsonValue] = {}
    client: VLLMClient | SelectorReplayClient
    if args.selector_replay_run is not None:
        replay_cache = load_selector_replay_cache(
            args.selector_replay_run,
            source_run=args.source_run,
            methods=methods,
            expected_model=args.model,
        )
        replay_client = SelectorReplayClient(live_client, replay_cache)
        client = replay_client
        replay_preflight = validate_selector_replay_source(
            replay_cache,
            source_run=args.source_run,
            methods=methods,
            config=reranker_config,
            limit=args.limit,
        )
        LOGGER.info(
            "Selector replay preflight passed: records=%d samples_checked=%d "
            "exact_prompt_matches=%d prompt_mismatches=%d",
            replay_cache.record_count,
            replay_preflight.records_checked,
            replay_preflight.exact_prompt_matches,
            replay_preflight.prompt_mismatches,
        )
        replay_metadata = {
            "selector_replay": True,
            "selector_replay_run": str(replay_cache.selector_run),
            "selector_replay_manifest_sha256": replay_cache.selector_manifest_sha256,
            "selector_replay_record_count": replay_cache.record_count,
            "selector_replay_exact_prompt_matches": (replay_preflight.exact_prompt_matches),
            "selector_replay_prompt_mismatches": replay_preflight.prompt_mismatches,
        }
    else:
        client = live_client
    reranker = LongMemEvalEvidenceReranker(
        client,
        reranker_config,
    )
    LOGGER.info(
        "Starting shared rerank: source=%s run_id=%s methods=%s "
        "model=%s candidates=%d planner=%s protect=%d/%d boundary=%s "
        "selector_replay=%s resume=%s",
        args.source_run,
        args.run_id,
        ",".join(methods),
        args.model,
        args.candidate_count,
        args.candidate_planner_version,
        args.protected_top_n,
        args.output_top_k,
        args.boundary_verification,
        args.selector_replay_run,
        args.resume,
    )
    result = run_longmemeval_rerank(
        LongMemEvalRerankRunConfig(
            source_run=args.source_run,
            methods=methods,
            output_dir=args.output_dir,
            resume=args.resume,
            limit=args.limit,
            require_full_candidate_count=args.require_full_candidate_count,
            metadata={
                "paper_version": _paper_version(
                    selector_prompt_version=args.prompt_version,
                    boundary_prompt_version=args.boundary_prompt_version,
                    boundary_verification=args.boundary_verification,
                ),
                "shared_across_frameworks": True,
                "test_labels_used": False,
                **replay_metadata,
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
                "selector_replay": (
                    {
                        "source_run": str(replay_client.cache.selector_run),
                        "cached_records": replay_client.cache.record_count,
                        "cache_hits": replay_client.selector_replay_hits,
                        "runtime_prompt_mismatches": (replay_client.selector_prompt_mismatches),
                        "live_boundary_calls": replay_client.boundary_live_calls,
                    }
                    if replay_client is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
