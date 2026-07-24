#!/usr/bin/env python3
"""Tune VMP-v4 retrieval on LongMemEval dev and freeze a test-safe model."""

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
from vmp_memos.longmemeval.tuning import train_vmp_tuned

LOGGER = logging.getLogger("vmp_memos.train_vmp_tuned")


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
        "--output",
        type=Path,
        default=Path("outputs/longmemeval/models/vmp_v4_seed42.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/longmemeval/models/vmp_v4_seed42_search.json"),
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--embedding-cache-db",
        type=Path,
        default=None,
        help="Optional persistent SQLite cache shared with retrieval runs.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=1)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--tuning-seed", type=int, default=2025)
    parser.add_argument("--retrieval-depth", type=int, default=10)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument(
        "--stability-folds",
        type=int,
        default=5,
        help="Deterministic Dev folds used to penalize unstable trial results.",
    )
    parser.add_argument(
        "--min-required-recall-all-at-5",
        type=float,
        default=0.90,
        help="Absolute Dev gate used when selecting among robust trials.",
    )
    parser.add_argument("--min-required-delta-vs-dense", type=float, default=0.02)
    parser.add_argument(
        "--min-required-macro-delta-vs-dense", type=float, default=0.0
    )
    parser.add_argument(
        "--min-required-worst-type-delta-vs-dense", type=float, default=-0.03
    )
    parser.add_argument(
        "--max-allowed-fold-recall-stddev", type=float, default=0.20
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Dependency-light pipeline smoke only; not valid for paper results.",
    )
    args = parser.parse_args()

    LOGGER.info(
        "Starting tuning: data=%s split=%s trials=%d device=%s",
        args.data,
        args.split_manifest,
        args.trials,
        args.embedding_device,
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
            "Embedding configured: model=%s batch_size=%d cache_db=%s",
            args.embedding_model,
            args.embedding_batch_size,
            args.embedding_cache_db or "disabled",
        )
    try:
        result = train_vmp_tuned(
            args.data,
            args.split_manifest,
            embedder=embedder,
            trials=args.trials,
            tuning_seed=args.tuning_seed,
            retrieval_depth=args.retrieval_depth,
            qa_top_k=args.qa_top_k,
            token_budget=args.token_budget,
            stability_folds=args.stability_folds,
            min_required_recall_all_at_5=args.min_required_recall_all_at_5,
            min_required_delta_vs_dense=args.min_required_delta_vs_dense,
            min_required_macro_delta_vs_dense=(
                args.min_required_macro_delta_vs_dense
            ),
            min_required_worst_type_delta_vs_dense=(
                args.min_required_worst_type_delta_vs_dense
            ),
            max_allowed_fold_recall_stddev=(
                args.max_allowed_fold_recall_stddev
            ),
        )
    finally:
        if embedder is not None:
            embedder.release()

    LOGGER.info(
        "Tuning finished: examples=%d skipped=%d best_objective=%.6f",
        result.candidate_examples,
        result.skipped_examples,
        result.model.best_objective,
    )
    model_path = result.model.save(args.output)
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model": str(model_path),
                "search_report": str(report_path),
                "split_id": result.model.split_id,
                "training_split": result.model.training_split,
                "best_objective": result.model.best_objective,
                "dev_metrics": result.model.dev_metrics,
                "semantic_anchor_weight": result.model.semantic_anchor_weight,
                "lexical_anchor_weight": result.model.lexical_anchor_weight,
                "policy_adjustment_limit": result.model.policy_adjustment_limit,
                "protected_dense_count": result.model.protected_dense_count,
                "promotion_margin": result.model.promotion_margin,
                "dev_delta_vs_dense": result.model.metadata.get(
                    "dev_recall_all_at_5_delta_vs_dense"
                ),
                "selected_trial": result.model.metadata.get("selected_trial"),
                "max_recall_trial": result.model.metadata.get("max_recall_trial"),
                "max_dev_recall_all_at_5_seen": result.model.metadata.get(
                    "max_dev_recall_all_at_5_seen"
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
