"""Judge a completed LongMemEval QA run with the official prompt and local vLLM."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vmp_memos.llm import LLMGenerationConfig, VLLMClient, load_vllm_config
from vmp_memos.longmemeval.official_qa import (
    LongMemEvalOfficialJudgeConfig,
    run_longmemeval_official_judge,
)

LOGGER = logging.getLogger("vmp_memos.run_longmemeval_official_judge")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-run", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--output-subdir", default="official_judge_local_vllm_v1")
    parser.add_argument("--config", type=Path, default=Path("configs/llm.yaml"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client_config = load_vllm_config(args.config)
    updates = {}
    if args.base_url:
        updates["base_url"] = args.base_url
    if args.model:
        updates["model"] = args.model
    if args.api_key:
        updates["api_key"] = args.api_key
    generation = LLMGenerationConfig(
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
    )
    updates["generation"] = generation
    client_config = client_config.model_copy(update=updates)
    client = VLLMClient(client_config)
    served_models = client.ensure_ready()
    LOGGER.info(
        "Local judge preflight passed: model=%s served_models=%s",
        client_config.model,
        ",".join(served_models),
    )
    methods = (
        [method.strip() for method in args.methods.split(",") if method.strip()]
        if args.methods
        else []
    )
    result = run_longmemeval_official_judge(
        LongMemEvalOfficialJudgeConfig(
            qa_run=args.qa_run,
            reference_data=args.reference_data,
            methods=methods,
            output_subdir=args.output_subdir,
            limit=args.limit,
            resume=args.resume,
            judge_metadata={
                "provider": "vllm",
                "base_url": client_config.base_url,
                "model": client_config.model,
            },
        ),
        client=client,
        generation=generation,
    )
    print(
        json.dumps(
            {
                "qa_run": str(result.qa_run),
                "judge_dir": str(result.judge_dir),
                "manifest": str(result.manifest_path),
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
