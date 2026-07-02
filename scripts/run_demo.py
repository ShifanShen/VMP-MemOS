#!/usr/bin/env python3
"""Run the smallest end-to-end demo supported by the current implementation phase."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from vmp_memos.backends import FileMemoryBackend, HybridMemoryBackend, VectorMemoryBackend
from vmp_memos.embeddings import EmbeddingDependencyError, SentenceTransformerEmbedder
from vmp_memos.schemas import MemoryItem, MemorySource, PolicyFeatures

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME: Final = "sentence-transformers/all-MiniLM-L6-v2"


def run_file_demo(workspace: Path) -> None:
    """Exercise add, update, archive, and logged lexical retrieval."""

    backend = FileMemoryBackend(workspace)
    preference = MemoryItem(
        type="semantic",
        scope="career/agent-dev",
        content="用户之前考虑转 Java 后端。",
        summary="旧职业方向偏好",
        source=MemorySource(source_type="demo"),
        features=PolicyFeatures(importance=0.88, confidence=0.92, novelty=0.9),
    )
    backend.add(preference, reason="Demo: capture the initial career preference.")
    updated = backend.update(
        preference.id,
        {
            "content": "用户现在主攻 Agent 开发和 LLM 应用开发，不再 all in Java。",
            "summary": "当前职业方向偏 Agent / LLM 应用开发",
        },
        reason="Demo: a newer user statement supersedes the old preference.",
        policy_score=0.82,
        confidence=0.91,
    )

    transient = MemoryItem(
        type="episodic",
        scope="demo",
        content="用于展示归档流程的临时 demo 事件。",
        source=MemorySource(source_type="demo"),
        features=PolicyFeatures(importance=0.1, confidence=1.0, staleness=1.0),
    )
    backend.add(transient, reason="Demo: add a short-lived episodic memory.")
    archived = backend.archive(
        transient.id,
        reason="Demo: the temporary event has completed and is now stale.",
        policy_score=0.9,
    )
    retrieved = backend.search("Agent 开发", top_k=5, filters={"scope": "career/agent-dev"})

    print(f"Workspace: {backend.workspace}")
    print(f"Updated memory: {updated.id} (version {updated.metadata.version})")
    print(f"Archived memory: {archived.id}")
    print(f"Retrieved: {[item.id for item in retrieved]}")
    print(f"Operation log: {backend.operation_log_path}")
    print(f"Retrieval log: {backend.retrieval_log_path}")


def run_vector_demo(
    workspace: Path,
    *,
    model_name: str,
    device: str,
    model_cache_dir: Path | None,
) -> None:
    """Exercise SQLite vector persistence and cosine retrieval."""

    embedder = SentenceTransformerEmbedder(
        model_name=model_name,
        device=device,
        cache_folder=model_cache_dir,
    )
    backend = VectorMemoryBackend(workspace, embedder=embedder)
    memories = [
        MemoryItem(
            type="semantic",
            scope="career/agent-dev",
            content="用户现在主攻 Agent 开发、LLM 应用开发和长期记忆系统。",
            summary="当前职业方向偏 Agent / LLM 应用开发",
            source=MemorySource(source_type="demo"),
            features=PolicyFeatures(importance=0.9, confidence=0.92, novelty=0.88),
        ),
        MemoryItem(
            type="semantic",
            scope="career/java",
            content="用户过去考虑过 Java 后端，但这不是当前主线。",
            summary="过往 Java 后端方向",
            source=MemorySource(source_type="demo"),
            features=PolicyFeatures(importance=0.55, confidence=0.88, staleness=0.65),
        ),
        MemoryItem(
            type="procedural",
            scope="project/vmp-memos",
            content="FileMemoryBackend 使用 Markdown frontmatter 保存可读记忆文件。",
            summary="File backend 的持久化方式",
            source=MemorySource(source_type="demo"),
            features=PolicyFeatures(importance=0.7, confidence=0.95),
        ),
    ]

    stored_ids: list[str] = []
    for item in memories:
        stored = backend.add(item, reason="Demo: seed vector-search memory.")
        stored_ids.append(stored.id)

    retrieved = backend.search("Agent 长期记忆开发", top_k=3)

    print(f"Workspace: {backend.workspace}")
    print(f"Vector DB: {backend.db_path}")
    print(f"Seeded memories: {stored_ids}")
    print(f"Retrieved: {[(item.id, item.summary) for item in retrieved]}")
    print(f"Operation log: {backend.operation_log_path}")
    print(f"Retrieval log: {backend.retrieval_log_path}")


def run_hybrid_demo(
    workspace: Path,
    *,
    model_name: str,
    device: str,
    model_cache_dir: Path | None,
) -> None:
    """Exercise readable file storage plus vector-ranked retrieval."""

    embedder = SentenceTransformerEmbedder(
        model_name=model_name,
        device=device,
        cache_folder=model_cache_dir,
    )
    backend = HybridMemoryBackend(workspace, embedder=embedder)
    preference = MemoryItem(
        type="semantic",
        scope="career/agent-dev",
        content="用户之前考虑转 Java 后端。",
        summary="旧职业方向偏好",
        source=MemorySource(source_type="demo"),
        features=PolicyFeatures(importance=0.88, confidence=0.92, novelty=0.9),
    )
    stored = backend.add(preference, reason="Demo: write readable and vector-indexed memory.")
    updated = backend.update(
        stored.id,
        {
            "content": "用户现在主攻 Agent 开发和 LLM 应用开发，不再 all in Java。",
            "summary": "当前职业方向偏 Agent / LLM 应用开发",
        },
        reason="Demo: update source-of-truth file storage and vector index.",
        policy_score=0.82,
        confidence=0.91,
    )

    procedural = MemoryItem(
        type="procedural",
        scope="project/vmp-memos",
        content="HybridMemoryBackend 使用 Markdown 作为 source of truth，并使用向量库排序召回。",
        summary="Hybrid backend 的组合方式",
        source=MemorySource(source_type="demo"),
        features=PolicyFeatures(importance=0.72, confidence=0.95),
    )
    backend.add(procedural, reason="Demo: add a procedural project memory.")
    retrieved = backend.search("Agent 长期记忆开发", top_k=3)

    print(f"Workspace: {backend.workspace}")
    print(f"Updated memory: {updated.id} (version {updated.metadata.version})")
    print(f"Retrieved: {[(item.id, item.summary) for item in retrieved]}")
    print(f"Operation log: {backend.operation_log_path}")
    print(f"Retrieval log: {backend.retrieval_log_path}")


def parse_args() -> argparse.Namespace:
    """Parse demo arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("file", "vector", "hybrid"),
        default="file",
        help="Backend to demonstrate.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT / "memory_workspace",
        help="Workspace path (default: <project>/memory_workspace).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformers model for --backend vector.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="SentenceTransformers device for --backend vector, e.g. auto/cuda/cpu.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=None,
        help="Optional local model cache directory for SentenceTransformers.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    try:
        if args.backend == "file":
            run_file_demo(workspace)
        elif args.backend == "vector":
            model_cache_dir = (
                args.model_cache_dir.expanduser().resolve()
                if args.model_cache_dir is not None
                else None
            )
            run_vector_demo(
                workspace,
                model_name=args.model_name,
                device=args.device,
                model_cache_dir=model_cache_dir,
            )
        elif args.backend == "hybrid":
            model_cache_dir = (
                args.model_cache_dir.expanduser().resolve()
                if args.model_cache_dir is not None
                else None
            )
            run_hybrid_demo(
                workspace,
                model_name=args.model_name,
                device=args.device,
                model_cache_dir=model_cache_dir,
            )
    except EmbeddingDependencyError as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
