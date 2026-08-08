#!/usr/bin/env python3
"""Tune VMP-v6 set-coverage weights on saved Dev fact profiles only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.llm import (
    CandidateEvidenceProfile,
    QuestionEvidencePlan,
    build_candidate_evidence_profile,
    candidate_evidence_spans,
    candidate_excerpt,
    select_evidence_coverage,
)

_WEIGHT_CHOICES = {
    "min_gain": (0.0, 0.1, 0.25, 0.5, 0.75),
    "need_weight": (0.0, 1.5, 3.0, 5.0),
    "relevance_weight": (0.5, 1.5, 3.0),
    "diversity_weight": (0.0, 0.5, 1.25, 2.5),
    "temporal_weight": (0.0, 0.5, 1.25, 2.5),
    "rank_weight": (0.02, 0.08, 0.2, 0.4),
}
_DEFAULT_WEIGHTS = {
    "min_gain": 0.25,
    "need_weight": 3.0,
    "relevance_weight": 1.5,
    "diversity_weight": 1.25,
    "temporal_weight": 1.25,
    "rank_weight": 0.08,
}


def tune_v6_coverage(
    run_dir: Path,
    *,
    vmp_method: str,
    baseline_method: str,
    trials: int = 512,
    seed: int = 2026,
    min_recall: float = 0.93,
    min_delta_vs_raw: float = 0.025,
    min_delta_vs_baseline: float = 0.03,
    min_macro_delta_vs_raw: float = 0.0,
    max_type_regression_vs_raw: float = 0.03,
    max_regressions: int = 0,
    min_recoveries: int = 3,
    reparse_raw_responses: bool = False,
) -> dict[str, Any]:
    """Search shared weights without making new embedding or LLM calls."""

    if trials < 1:
        raise ValueError("trials must be positive")
    resolved_run = run_dir.expanduser().resolve()
    manifest_path = resolved_run / "manifest.json"
    manifest = _read_object(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("VMP-v6 fact run must be completed")
    split = manifest.get("split")
    split_name = split.get("name") if isinstance(split, dict) else split
    if split_name != "dev":
        raise ValueError("VMP-v6 coverage tuning is restricted to Dev")
    fairness = manifest.get("fairness")
    if not (
        isinstance(fairness, dict)
        and fairness.get("structured_atomic_fact_protocol") is True
        and fairness.get("deterministic_set_coverage") is True
    ):
        raise ValueError("run does not contain an audited VMP-v6 fact protocol")
    records_by_method = {
        vmp_method: _read_jsonl(resolved_run / vmp_method / "retrieval.jsonl"),
        baseline_method: _read_jsonl(
            resolved_run / baseline_method / "retrieval.jsonl"
        ),
    }
    _validate_records(records_by_method)
    reparse_settings = (
        _raw_response_reparse_settings(manifest) if reparse_raw_responses else None
    )
    if reparse_settings is not None:
        records_by_method = {
            method: [
                _record_with_reparsed_profiles(record, reparse_settings)
                for record in records
            ]
            for method, records in records_by_method.items()
        }
    weight_candidates = _sample_weights(trials=trials, seed=seed)
    reports: list[dict[str, Any]] = []
    for trial_index, weights in enumerate(weight_candidates, start=1):
        method_metrics = {
            method: _evaluate_weights(records, weights)
            for method, records in records_by_method.items()
        }
        vmp = method_metrics[vmp_method]
        baseline = method_metrics[baseline_method]
        delta_baseline = vmp["recall_all@5"] - baseline["recall_all@5"]
        feasible = (
            vmp["recall_all@5"] >= min_recall
            and vmp["delta_vs_raw"] >= min_delta_vs_raw
            and delta_baseline >= min_delta_vs_baseline
            and vmp["macro_delta_vs_raw"] >= min_macro_delta_vs_raw
            and vmp["max_type_regression_vs_raw"] <= max_type_regression_vs_raw
            and vmp["regressed_questions"] <= max_regressions
            and vmp["recovered_questions"] >= min_recoveries
        )
        objective = (
            vmp["recall_all@5"]
            + 0.25 * delta_baseline
            + 0.10 * vmp["macro_recall_all@5"]
            - 1.0
            * vmp["regressed_questions"]
            / max(1, vmp["evaluated_questions"])
            - 0.0001 * vmp["promotions"]
        )
        reports.append(
            {
                "trial": trial_index,
                "weights": weights,
                "vmp": vmp,
                "baseline": baseline,
                "delta_vs_shared_baseline": delta_baseline,
                "gate_feasible": feasible,
                "objective": objective,
            }
        )
    reports.sort(key=_trial_sort_key, reverse=True)
    best = reports[0]
    return {
        "schema_version": "1.0",
        "source_run": str(resolved_run),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "split": "dev",
        "vmp_method": vmp_method,
        "baseline_method": baseline_method,
        "trials": len(reports),
        "seed": seed,
        "best": best,
        "top_trials": reports[:20],
        "required": {
            "min_recall_all@5": min_recall,
            "min_delta_vs_raw": min_delta_vs_raw,
            "min_delta_vs_shared_baseline": min_delta_vs_baseline,
            "min_macro_delta_vs_raw": min_macro_delta_vs_raw,
            "max_type_regression_vs_raw": max_type_regression_vs_raw,
            "max_regressed_questions": max_regressions,
            "min_recovered_questions": min_recoveries,
        },
        "dev_labels_used_for_weight_selection": True,
        "test_labels_used": False,
        "requires_fresh_frozen_weight_validation_run": True,
        "raw_responses_reparsed": reparse_raw_responses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--vmp-method",
        default="vmp_hierarchical__vllm_boundary",
    )
    parser.add_argument(
        "--baseline-method",
        default="vmp_tuned__vllm_boundary",
    )
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reparse-raw-responses",
        action="store_true",
        help=(
            "Rebuild profiles from the saved per-candidate response text using "
            "the recorded excerpt protocol; makes parser fixes replayable without vLLM."
        ),
    )
    args = parser.parse_args()
    report = tune_v6_coverage(
        args.run,
        vmp_method=args.vmp_method,
        baseline_method=args.baseline_method,
        trials=args.trials,
        seed=args.seed,
        reparse_raw_responses=args.reparse_raw_responses,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _sample_weights(*, trials: int, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    observed = {_weight_key(_DEFAULT_WEIGHTS)}
    candidates = [dict(_DEFAULT_WEIGHTS)]
    max_unique = 1
    for values in _WEIGHT_CHOICES.values():
        max_unique *= len(values)
    target = min(trials, max_unique)
    while len(candidates) < target:
        candidate = {
            name: float(rng.choice(values))
            for name, values in _WEIGHT_CHOICES.items()
        }
        key = _weight_key(candidate)
        if key not in observed:
            observed.add(key)
            candidates.append(candidate)
    return candidates


def _weight_key(weights: dict[str, float]) -> tuple[float, ...]:
    return tuple(weights[name] for name in _WEIGHT_CHOICES)


def _evaluate_weights(
    records: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    evaluated = 0
    successes = 0
    recoveries = 0
    regressions = 0
    promotions = 0
    raw_successes = 0
    by_type: dict[str, list[bool]] = defaultdict(list)
    raw_by_type: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        metadata = _mapping(record.get("rerank_metadata"))
        plan = QuestionEvidencePlan.model_validate(
            _mapping(metadata.get("question_evidence_plan"))
        )
        profiles = [
            CandidateEvidenceProfile.model_validate(value)
            for value in _list(metadata.get("selector_evidence_selections"))
        ]
        if not profiles:
            raise ValueError("VMP-v6 record is missing candidate evidence profiles")
        output_top_k = _integer(metadata.get("output_top_k"), default=5)
        protected_top_n = _integer(metadata.get("protected_top_n"), default=3)
        selection = select_evidence_coverage(
            plan,
            profiles,
            output_top_k=output_top_k,
            protected_top_n=protected_top_n,
            min_gain=weights["min_gain"],
            need_weight=weights["need_weight"],
            relevance_weight=weights["relevance_weight"],
            diversity_weight=weights["diversity_weight"],
            temporal_weight=weights["temporal_weight"],
            rank_weight=weights["rank_weight"],
        )
        promotions += len(selection.promoted_candidate_labels)
        label_map = _mapping(metadata.get("candidate_label_session_ids"))
        selected_ids = {
            session_id
            for label in selection.selected_candidate_labels
            if isinstance((session_id := label_map.get(label)), str)
        }
        gold = set(_strings(record.get("gold_session_ids")))
        if record.get("evaluation_skipped") is True or not gold:
            continue
        evaluated += 1
        success = len(gold) <= output_top_k and gold.issubset(selected_ids)
        raw_success = metadata.get("source_recall_all@5") == 1.0
        successes += int(success)
        raw_successes += int(raw_success)
        recoveries += int(success and not raw_success)
        regressions += int(raw_success and not success)
        question_type = str(record.get("question_type") or "unknown")
        by_type[question_type].append(success)
        raw_by_type[question_type].append(raw_success)
    recall = successes / evaluated if evaluated else 0.0
    raw_recall = raw_successes / evaluated if evaluated else 0.0
    type_recalls = {
        name: sum(values) / len(values) for name, values in sorted(by_type.items())
    }
    raw_type_recalls = {
        name: sum(values) / len(values)
        for name, values in sorted(raw_by_type.items())
    }
    macro = sum(type_recalls.values()) / len(type_recalls) if type_recalls else 0.0
    raw_macro = (
        sum(raw_type_recalls.values()) / len(raw_type_recalls)
        if raw_type_recalls
        else 0.0
    )
    max_type_regression = max(
        (
            raw_type_recalls.get(name, 0.0) - recall_value
            for name, recall_value in type_recalls.items()
        ),
        default=0.0,
    )
    return {
        "evaluated_questions": evaluated,
        "recall_all@5": recall,
        "raw_recall_all@5": raw_recall,
        "delta_vs_raw": recall - raw_recall,
        "macro_recall_all@5": macro,
        "raw_macro_recall_all@5": raw_macro,
        "macro_delta_vs_raw": macro - raw_macro,
        "max_type_regression_vs_raw": max_type_regression,
        "by_question_type": type_recalls,
        "raw_by_question_type": raw_type_recalls,
        "recovered_questions": recoveries,
        "regressed_questions": regressions,
        "promotions": promotions,
    }


def _trial_sort_key(trial: dict[str, Any]) -> tuple[Any, ...]:
    vmp = _mapping(trial.get("vmp"))
    weights = _mapping(trial.get("weights"))
    return (
        trial.get("gate_feasible") is True,
        _integer(vmp.get("regressed_questions")) == 0,
        _number(vmp.get("recall_all@5")),
        _number(trial.get("delta_vs_shared_baseline")),
        _number(vmp.get("macro_recall_all@5")),
        _integer(vmp.get("recovered_questions")),
        -_integer(vmp.get("promotions")),
        -sum(_number(value) for value in weights.values()),
    )


def _validate_records(records_by_method: dict[str, list[dict[str, Any]]]) -> None:
    question_sets = [
        {str(record.get("question_id")) for record in records}
        for records in records_by_method.values()
    ]
    if not question_sets or not question_sets[0]:
        raise ValueError("no rerank records found")
    if any(observed != question_sets[0] for observed in question_sets[1:]):
        raise ValueError("VMP and baseline rerank records cover different questions")


def _raw_response_reparse_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    signature = _mapping(manifest.get("signature"))
    reranker = _mapping(signature.get("reranker"))
    return {
        "max_candidate_chars": _integer(
            reranker.get("max_candidate_chars"), default=1200
        ),
        "max_excerpt_turns": _integer(
            reranker.get("max_excerpt_turns"), default=4
        ),
        "candidate_excerpt_version": str(
            reranker.get("candidate_excerpt_version") or "role_aware_fact_v2"
        ),
    }


def _record_with_reparsed_profiles(
    record: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild grounded profiles while preserving the frozen model responses."""

    metadata = dict(_mapping(record.get("rerank_metadata")))
    plan = QuestionEvidencePlan.model_validate(
        _mapping(metadata.get("question_evidence_plan"))
    )
    question = str(record.get("question") or "")
    if not question:
        raise ValueError("raw response replay requires the saved question")
    label_map = _mapping(metadata.get("candidate_label_session_ids"))
    memories = [
        RetrievedMemory.model_validate(value)
        for value in _list(record.get("retrieved_memories"))
        if isinstance(value, dict)
    ]
    memory_by_session = {
        str(memory.source_session_id or memory.memory_id): memory
        for memory in memories
    }
    response_entries = _atomic_response_entries(metadata.get("response_text"))
    profiles: list[dict[str, Any]] = []
    for entry in response_entries:
        label = str(entry.get("candidate") or "")
        if not label.startswith("C") or not label[1:].isdigit():
            raise ValueError("saved atomic response has an invalid candidate label")
        session_id = label_map.get(label)
        if not isinstance(session_id, str) or session_id not in memory_by_session:
            raise ValueError(f"saved atomic response cannot resolve {label}")
        memory = memory_by_session[session_id]
        payload = _parse_saved_response_object(entry.get("response"))
        max_chars = _integer(settings.get("max_candidate_chars"), default=1200)
        max_turns = _integer(settings.get("max_excerpt_turns"), default=4)
        excerpt_version = str(settings.get("candidate_excerpt_version"))
        excerpt = candidate_excerpt(
            question,
            memory.content,
            max_chars=max_chars,
            max_turns=max_turns,
            excerpt_version=excerpt_version,
        )
        spans = candidate_evidence_spans(
            question,
            memory.content,
            max_chars=max_chars,
            max_turns=max_turns,
            excerpt_version=excerpt_version,
        )
        profile = build_candidate_evidence_profile(
            payload,
            candidate_label=label,
            session_id=session_id,
            rank=int(label[1:]),
            plan=plan,
            allowed_span_ids={
                f"X:S{index:02d}" for index in range(1, len(spans) + 1)
            },
            excerpt=excerpt,
        )
        profiles.append(profile.model_dump(mode="json"))
    expected_count = _integer(metadata.get("candidate_count_observed"), default=10)
    if len(profiles) != expected_count:
        raise ValueError(
            "saved atomic response count differs from the recorded candidate count"
        )
    metadata["selector_evidence_selections"] = profiles
    return {**record, "rerank_metadata": metadata}


def _atomic_response_entries(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("raw response replay requires rerank_metadata.response_text")
    parsed = json.loads(value)
    entries = [item for item in _list(parsed) if isinstance(item, dict)]
    if not entries:
        raise ValueError("saved atomic response text contains no candidate responses")
    return entries


def _parse_saved_response_object(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        payload
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        if isinstance((payload := json.loads(line)), dict)
    ]


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _strings(value: object) -> list[str]:
    return [item for item in _list(value) if isinstance(item, str)]


def _integer(value: object, *, default: int = 0) -> int:
    return int(value) if isinstance(value, int | float) else default


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
