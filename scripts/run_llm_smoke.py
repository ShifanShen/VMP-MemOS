#!/usr/bin/env python3
"""Call a running vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from vmp_memos.extraction import LLMMemoryExtractor, LLMMemoryExtractorConfig
from vmp_memos.llm import ChatMessage, LLMGenerationConfig, VLLMClient, load_vllm_config
from vmp_memos.schemas import Event

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "llm.yaml",
        help="Flat vLLM client config YAML.",
    )
    parser.add_argument("--base-url", default=None, help="Override OpenAI base URL.")
    parser.add_argument("--model", default=None, help="Override served model name.")
    parser.add_argument("--api-key", default=None, help="Override bearer API key.")
    parser.add_argument(
        "--prompt",
        default="Summarize VMP-MemOS in one short sentence.",
        help="Prompt for chat smoke mode.",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--extract-memory",
        action="store_true",
        help="Extract MemoryCandidate JSON from the prompt instead of plain chat.",
    )
    parser.add_argument("--session-id", default="llm_smoke")
    parser.add_argument("--scope", default="global")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    config = load_vllm_config(args.config)
    updates = {}
    if args.base_url:
        updates["base_url"] = args.base_url
    if args.model:
        updates["model"] = args.model
    if args.api_key:
        updates["api_key"] = args.api_key
    if updates:
        config = config.model_copy(update=updates)

    generation = config.generation
    generation_updates = {}
    if args.max_tokens is not None:
        generation_updates["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        generation_updates["temperature"] = args.temperature
    if generation_updates:
        generation = generation.model_copy(update=generation_updates)
        config = config.model_copy(update={"generation": generation})

    client = VLLMClient(config)
    if args.extract_memory:
        event = Event(
            session_id=args.session_id,
            event_type="user_message",
            content=args.prompt,
        )
        extractor = LLMMemoryExtractor(
            client,
            LLMMemoryExtractorConfig(default_scope=args.scope),
        )
        candidates = extractor.extract(event)
        for candidate in candidates:
            print(candidate.model_dump_json())
        if not candidates:
            print("No memory candidates extracted.")
        return 0

    response = client.chat(
        [
            ChatMessage(role="system", content="You are a concise assistant."),
            ChatMessage(role="user", content=args.prompt),
        ],
        generation=generation,
    )
    print(response.text)
    if response.usage:
        print(f"\nusage={response.usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
