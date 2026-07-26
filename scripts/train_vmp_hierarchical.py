#!/usr/bin/env python3
"""Tune VMP-v5 hierarchical fusion on LongMemEval Dev and freeze it."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vmp_memos.embeddings import (
    CachedEmbedder,
    SentenceTransformerEmbedder,
    SQLiteEmbeddingCache,
)
from vmp_memos.longmemeval.hierarchical_tuning import (
    train_vmp_hierarchical,
)

LOGGER = logging.getLogger("vmp_memos.train_vmp_hierarchical")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("outputs/longmemeval/models/vmp_v43_seed42.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/longmemeval/models/vmp_v5_seed42.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/longmemeval/models/vmp_v5_seed42_search.json"),
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument("--embedding-cache-db", type=Path, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--grid-step", type=float, default=0.2)
    parser.add_argument("--turn-pooling", default="1,2,3")
    parser.add_argument("--retrieval-depth", type=int, default=10)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--stability-folds", type=int, default=5)
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Lexical smoke test only; invalid for paper results.",
    )
    args = parser.parse_args()
    turn_pooling = tuple(
        int(value.strip())
        for value in args.turn_pooling.split(",")
        if value.strip()
    )

    embedder = None
    if not args.no_embeddings:
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
    LOGGER.info(
        "Starting VMP-v5 Dev tuning: data=%s base=%s grid_step=%.3f "
        "turn_pooling=%s device=%s",
        args.data,
        args.base_model,
        args.grid_step,
        turn_pooling,
        args.embedding_device,
    )
    try:
        result = train_vmp_hierarchical(
            args.data,
            args.split_manifest,
            args.base_model,
            embedder=embedder,
            grid_step=args.grid_step,
            turn_pooling_options=turn_pooling,
            retrieval_depth=args.retrieval_depth,
            qa_top_k=args.qa_top_k,
            token_budget=args.token_budget,
            stability_folds=args.stability_folds,
        )
    finally:
        if embedder is not None:
            embedder.release()

    model_path = result.model.save(args.output)
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": str(model_path),
                "search_report": str(report_path),
                "schema_version": result.model.schema_version,
                "model_type": result.model.model_type,
                "split_id": result.model.split_id,
                "split_assignment_sha256": (
                    result.model.split_assignment_sha256
                ),
                "split_manifest_file_sha256_matches_base": (
                    result.model.metadata.get(
                        "split_manifest_file_sha256_matches_base"
                    )
                ),
                "training_split": result.model.training_split,
                "best_objective": result.model.best_objective,
                "dev_metrics": result.model.dev_metrics,
                "fusion": {
                    "session_semantic_weight": (
                        result.model.session_semantic_weight
                    ),
                    "turn_semantic_weight": (
                        result.model.turn_semantic_weight
                    ),
                    "turn_lexical_weight": result.model.turn_lexical_weight,
                    "turn_pooling_top_n": result.model.turn_pooling_top_n,
                },
                "delta_vs_session_only": result.model.metadata.get(
                    "dev_recall_all_at_5_delta_vs_session_only"
                ),
                "delta_vs_v43": result.model.metadata.get(
                    "dev_recall_all_at_5_delta_vs_v43"
                ),
                "dev_oracle_ceiling_metrics": result.model.metadata.get(
                    "dev_oracle_ceiling_metrics"
                ),
                "trials": result.trials_evaluated,
                "test_labels_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
