"""Answer LongMemEval retrieval runs with one shared local vLLM reader."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vmp_memos.llm import (
    LONGMEMEVAL_FACT_READER_PROMPT_VERSION,
    LONGMEMEVAL_HYBRID_READER_PROMPT_VERSION,
    LONGMEMEVAL_LEGACY_READER_PROMPT_VERSION,
    LLMGenerationConfig,
    LongMemEvalReader,
    LongMemEvalReaderConfig,
    VLLMClient,
    load_vllm_config,
)
from vmp_memos.longmemeval.qa_runner import (
    LongMemEvalQARunConfig,
    run_longmemeval_qa,
)

LOGGER = logging.getLogger("vmp_memos.run_longmemeval_qa")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--reader", choices=("vllm",), default="vllm")
    parser.add_argument("--config", type=Path, default=Path("configs/llm.yaml"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--prompt-version",
        choices=(
            LONGMEMEVAL_LEGACY_READER_PROMPT_VERSION,
            LONGMEMEVAL_FACT_READER_PROMPT_VERSION,
            LONGMEMEVAL_HYBRID_READER_PROMPT_VERSION,
        ),
        default=LONGMEMEVAL_LEGACY_READER_PROMPT_VERSION,
    )
    parser.add_argument(
        "--evidence-mode",
        choices=(
            "full_sessions",
            "reranker_facts",
            "reranker_facts_with_query_windows",
        ),
        default="full_sessions",
    )
    parser.add_argument("--qa-subdir", default="qa")
    parser.add_argument(
        "--protocol-selection-split",
        choices=("dev",),
        default=None,
        help="Declare the split used to select the frozen reader protocol.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client_config = load_vllm_config(args.config)
    client_updates = {}
    if args.base_url:
        client_updates["base_url"] = args.base_url
    if args.model:
        client_updates["model"] = args.model
    if args.api_key:
        client_updates["api_key"] = args.api_key
    generation = LLMGenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    client_updates["generation"] = generation
    client_config = client_config.model_copy(update=client_updates)
    LOGGER.info(
        "Starting QA run: retrieval_run=%s model=%s base_url=%s resume=%s",
        args.retrieval_run,
        client_config.model,
        client_config.base_url,
        args.resume,
    )

    client = VLLMClient(client_config)
    served_models = client.ensure_ready()
    LOGGER.info("vLLM preflight passed: served_models=%s", ",".join(served_models))
    reader = LongMemEvalReader(
        client,
        LongMemEvalReaderConfig(
            top_k=args.top_k,
            prompt_version=args.prompt_version,
            evidence_mode=args.evidence_mode,
            generation=generation,
        ),
    )
    methods = (
        [method.strip() for method in args.methods.split(",") if method.strip()]
        if args.methods
        else []
    )
    run_config = LongMemEvalQARunConfig(
        retrieval_run=args.retrieval_run,
        methods=methods,
        top_k=args.top_k,
        limit=args.limit,
        resume=args.resume,
        qa_subdir=args.qa_subdir,
        reader_metadata={
            "provider": "vllm",
            "base_url": client_config.base_url,
            "model": client_config.model,
            "prompt_version": args.prompt_version,
            "evidence_mode": args.evidence_mode,
            "gold_answers_visible_to_reader": False,
            "protocol_selection_split": args.protocol_selection_split,
        },
    )
    result = run_longmemeval_qa(run_config, reader=reader)
    print(
        json.dumps(
            {
                "retrieval_run": str(result.retrieval_run),
                "qa_dir": str(result.qa_dir),
                "manifest": str(result.manifest_path),
                "prompt_version": args.prompt_version,
                "evidence_mode": args.evidence_mode,
                "methods": {
                    method: summary.model_dump(mode="json")
                    for method, summary in result.summaries.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
