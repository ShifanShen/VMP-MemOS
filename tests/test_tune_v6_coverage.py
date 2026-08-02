"""Tests for Dev-only offline VMP-v6 coverage tuning."""

from __future__ import annotations

import json

from scripts.tune_v6_coverage import tune_v6_coverage


def test_tuner_replays_saved_fact_profiles_without_llm_calls(tmp_path) -> None:
    run = tmp_path / "v6"
    (run / "manifest.json").parent.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "split": {"name": "dev"},
                "fairness": {
                    "structured_atomic_fact_protocol": True,
                    "deterministic_set_coverage": True,
                },
            }
        ),
        encoding="utf-8",
    )
    profiles = []
    for rank in range(1, 11):
        facts = []
        if rank == 7:
            facts = [
                {
                    "fact_id": "C07:F01",
                    "entity": "museum visit",
                    "relation": "occurred_at",
                    "value": "visit",
                    "temporal_anchor": "2024-01-01",
                    "supports_needs": ["N1"],
                    "evidence_spans": ["X:S01"],
                    "confidence": "high",
                }
            ]
        profiles.append(
            {
                "candidate_label": f"C{rank:02d}",
                "session_id": f"s{rank}",
                "rank": rank,
                "candidate_relevant": bool(facts),
                "lexical_overlap": 1.0 if facts else 0.0,
                "facts": facts,
            }
        )
    record = {
        "question_id": "q1",
        "question_type": "temporal-reasoning",
        "gold_session_ids": ["s7"],
        "evaluation_skipped": False,
        "rerank_metadata": {
            "question_evidence_plan": {
                "operator": "temporal",
                "evidence_needs": ["N1: relevant event"],
                "query_terms": ["museum"],
                "temporal_coverage_required": True,
            },
            "selector_evidence_selections": profiles,
            "candidate_label_session_ids": {
                f"C{rank:02d}": f"s{rank}" for rank in range(1, 11)
            },
            "output_top_k": 5,
            "protected_top_n": 3,
            "source_recall_all@5": 0.0,
        },
    }
    for method in (
        "vmp_hierarchical__vllm_boundary",
        "vmp_tuned__vllm_boundary",
    ):
        method_dir = run / method
        method_dir.mkdir()
        (method_dir / "retrieval.jsonl").write_text(
            json.dumps(record) + "\n",
            encoding="utf-8",
        )

    report = tune_v6_coverage(
        run,
        vmp_method="vmp_hierarchical__vllm_boundary",
        baseline_method="vmp_tuned__vllm_boundary",
        trials=2,
    )

    assert report["trials"] == 2
    assert report["best"]["vmp"]["recall_all@5"] == 1.0
    assert report["best"]["vmp"]["recovered_questions"] == 1
    assert report["dev_labels_used_for_weight_selection"] is True
    assert report["test_labels_used"] is False
