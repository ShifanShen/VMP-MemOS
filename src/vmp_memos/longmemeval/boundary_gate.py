"""Strict Dev-only quality gate for the VMP-v5.3 boundary verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from vmp_memos.llm import (
    LONGMEMEVAL_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_RERANK_PROMPT_VERSION,
    LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_PROMPT_VERSION,
    LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION,
    LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION,
    LONGMEMEVAL_V551_COMPLETE_CHALLENGER_SELECTOR_PROMPT_VERSION,
)
from vmp_memos.longmemeval.rerank_runner import reranked_method_name


def evaluate_v53_gate(
    candidate_run: str | Path,
    rerank_run: str | Path,
    *,
    vmp_method: str = "vmp_hierarchical",
    baseline_method: str = "vmp_tuned",
    min_recall_all_at_5: float = 0.93,
    min_delta_vs_raw_v5: float = 0.025,
    min_delta_vs_shared_v43: float = 0.03,
    min_macro_delta_vs_raw_v5: float = 0.0,
    max_type_regression_vs_raw_v5: float = 0.03,
    max_parse_fallback_rate: float = 0.02,
    max_boundary_fallback_rate: float = 0.02,
    max_regressed_questions: int = 0,
    min_recovered_questions: int = 3,
    min_candidate_count: int = 30,
    expected_selector_prompt_version: str = LONGMEMEVAL_RERANK_PROMPT_VERSION,
    expected_boundary_prompt_version: str = LONGMEMEVAL_BOUNDARY_PROMPT_VERSION,
    expected_candidate_planner_version: str | None = None,
) -> dict[str, JsonValue]:
    """Evaluate the two-stage policy without reading Test labels or dataset rows."""

    candidate_dir = Path(candidate_run).expanduser().resolve()
    rerank_dir = Path(rerank_run).expanduser().resolve()
    candidate_manifest_path = candidate_dir / "manifest.json"
    rerank_manifest_path = rerank_dir / "manifest.json"
    candidate_manifest = _read_json(candidate_manifest_path)
    rerank_manifest = _read_json(rerank_manifest_path)
    vmp_method = _normalize_method(vmp_method)
    baseline_method = _normalize_method(baseline_method)
    vmp_reranked_method = reranked_method_name(
        vmp_method,
        boundary_verification=True,
    )
    baseline_reranked_method = reranked_method_name(
        baseline_method,
        boundary_verification=True,
    )
    raw_v5 = _read_json(candidate_dir / vmp_method / "summary.json")
    raw_v43 = _read_json(candidate_dir / baseline_method / "summary.json")
    reranked_v5 = _read_json(rerank_dir / vmp_reranked_method / "summary.json")
    reranked_v43 = _read_json(rerank_dir / baseline_reranked_method / "summary.json")

    v53_recall = _metric(reranked_v5, "recall_all@5")
    raw_v5_recall = _metric(raw_v5, "recall_all@5")
    shared_v43_recall = _metric(reranked_v43, "recall_all@5")
    raw_v43_recall = _metric(raw_v43, "recall_all@5")
    v53_macro = _type_macro(reranked_v5)
    raw_v5_macro = _type_macro(raw_v5)
    shared_v43_macro = _type_macro(reranked_v43)
    max_type_regression = _max_type_regression(raw_v5, reranked_v5)
    delta_raw_v5 = v53_recall - raw_v5_recall
    delta_shared_v43 = v53_recall - shared_v43_recall
    delta_raw_v43 = v53_recall - raw_v43_recall
    macro_delta_raw_v5 = v53_macro - raw_v5_macro
    macro_delta_shared_v43 = v53_macro - shared_v43_macro

    fallback_rate = _number(reranked_v5.get("parse_fallback_rate"), default=1.0)
    baseline_fallback_rate = _number(
        reranked_v43.get("parse_fallback_rate"),
        default=1.0,
    )
    boundary_fallback_rate = _number(
        reranked_v5.get("boundary_parse_fallback_rate"),
        default=1.0,
    )
    baseline_boundary_fallback_rate = _number(
        reranked_v43.get("boundary_parse_fallback_rate"),
        default=1.0,
    )
    recovered_questions = _integer(reranked_v5.get("recovered_questions"))
    regressed_questions = _integer(reranked_v5.get("regressed_questions"))
    candidate_count = _integer(reranked_v5.get("candidate_count"))
    baseline_candidate_count = _integer(reranked_v43.get("candidate_count"))
    observed_candidate_count = _integer(reranked_v5.get("min_observed_candidate_count"))
    baseline_observed_candidate_count = _integer(reranked_v43.get("min_observed_candidate_count"))
    split_name = _split_name(candidate_manifest)
    rerank_split_name = _split_name(rerank_manifest)
    candidate_manifest_sha256 = _sha256(candidate_manifest_path)
    rerank_source_sha = rerank_manifest.get("source_retrieval_manifest_sha256")
    provider = reranked_v5.get("reranker_provider")
    baseline_provider = reranked_v43.get("reranker_provider")
    model = reranked_v5.get("reranker_model")
    baseline_model = reranked_v43.get("reranker_model")
    selector_prompt = reranked_v5.get("prompt_version")
    baseline_selector_prompt = reranked_v43.get("prompt_version")
    boundary_prompt = reranked_v5.get("boundary_prompt_version")
    baseline_boundary_prompt = reranked_v43.get("boundary_prompt_version")
    candidate_planner = reranked_v5.get("candidate_planner_version")
    baseline_candidate_planner = reranked_v43.get("candidate_planner_version")
    candidate_planner_applied_questions = _integer(
        reranked_v5.get("candidate_planner_applied_questions")
    )
    baseline_candidate_planner_identity_questions = _integer(
        reranked_v43.get("candidate_planner_identity_questions")
    )
    candidate_config = candidate_manifest.get("config")
    dev_selection_marked = (
        isinstance(candidate_config, dict)
        and candidate_config.get("allow_dev_model_selection") is True
    )
    rerank_fairness = rerank_manifest.get("fairness")
    two_stage_marked = (
        isinstance(rerank_fairness, dict)
        and rerank_fairness.get("two_stage_boundary_verification") is True
    )
    symbolic_span_expected = (
        expected_selector_prompt_version
        in {
            LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_PROMPT_VERSION,
            LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION,
            LONGMEMEVAL_V551_COMPLETE_CHALLENGER_SELECTOR_PROMPT_VERSION,
        }
        or expected_boundary_prompt_version
        == LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION
    )
    symbolic_span_marked = (
        isinstance(rerank_fairness, dict)
        and rerank_fairness.get("symbolic_selector_labels") is True
        and rerank_fairness.get("selector_evidence_span_binding") is True
        and rerank_fairness.get("boundary_evidence_span_binding") is True
    )
    sample_count = _integer(candidate_manifest.get("sample_count"))
    summaries = (raw_v5, raw_v43, reranked_v5, reranked_v43)
    processed_counts = [_integer(summary.get("processed_questions")) for summary in summaries]
    evaluated_counts = [_integer(summary.get("evaluated_questions")) for summary in summaries]
    candidate_planner_contract = True
    if expected_candidate_planner_version is not None:
        candidate_planner_contract = (
            isinstance(rerank_fairness, dict)
            and rerank_fairness.get("shared_candidate_planner") is True
            and rerank_fairness.get("candidate_planner_uses_gold_labels") is False
            and rerank_fairness.get("candidate_planner_version")
            == expected_candidate_planner_version
            and candidate_planner == expected_candidate_planner_version
            and baseline_candidate_planner == expected_candidate_planner_version
        )
        if (
            expected_candidate_planner_version
            == LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION
        ):
            candidate_planner_contract = (
                candidate_planner_contract
                and candidate_planner_applied_questions == sample_count
                and baseline_candidate_planner_identity_questions == sample_count
            )

    checks = {
        "candidate_run_completed": candidate_manifest.get("status") == "completed",
        "rerank_run_completed": rerank_manifest.get("status") == "completed",
        "dev_split_only": split_name == "dev" and rerank_split_name == "dev",
        "dev_selection_explicitly_marked": dev_selection_marked,
        "complete_question_coverage": (
            sample_count > 0
            and all(count == sample_count for count in processed_counts)
            and len(set(evaluated_counts)) == 1
            and evaluated_counts[0] > 0
        ),
        "source_manifest_matches": rerank_source_sha == candidate_manifest_sha256,
        "test_labels_not_used": rerank_manifest.get("test_labels_used") is False,
        "shared_local_vllm": (
            provider == "vllm"
            and baseline_provider == provider
            and isinstance(model, str)
            and bool(model)
            and baseline_model == model
        ),
        "shared_two_stage_prompts": (
            two_stage_marked
            and reranked_v5.get("boundary_verification") is True
            and reranked_v43.get("boundary_verification") is True
            and selector_prompt == expected_selector_prompt_version
            and baseline_selector_prompt == selector_prompt
            and boundary_prompt == expected_boundary_prompt_version
            and baseline_boundary_prompt == boundary_prompt
        ),
        "shared_symbolic_span_protocol": (
            symbolic_span_marked if symbolic_span_expected else True
        ),
        "label_free_candidate_planner": candidate_planner_contract,
        "candidate_depth": (
            candidate_count >= min_candidate_count
            and baseline_candidate_count >= min_candidate_count
            and observed_candidate_count >= min_candidate_count
            and baseline_observed_candidate_count >= min_candidate_count
        ),
        "parse_fallback_rate": (
            fallback_rate <= max_parse_fallback_rate
            and baseline_fallback_rate <= max_parse_fallback_rate
        ),
        "boundary_fallback_rate": (
            boundary_fallback_rate <= max_boundary_fallback_rate
            and baseline_boundary_fallback_rate <= max_boundary_fallback_rate
        ),
        "min_recall_all_at_5": v53_recall >= min_recall_all_at_5,
        "delta_vs_raw_v5": delta_raw_v5 >= min_delta_vs_raw_v5,
        "delta_vs_shared_v43": delta_shared_v43 >= min_delta_vs_shared_v43,
        "macro_non_regression_vs_raw_v5": (macro_delta_raw_v5 >= min_macro_delta_vs_raw_v5),
        "per_type_guard_vs_raw_v5": (max_type_regression <= max_type_regression_vs_raw_v5),
        "zero_or_bounded_regression": regressed_questions <= max_regressed_questions,
        "minimum_recoveries": recovered_questions >= min_recovered_questions,
    }
    passed = all(checks.values())
    return {
        "schema_version": "1.1",
        "status": "passed" if passed else "failed",
        "candidate_run": str(candidate_dir),
        "rerank_run": str(rerank_dir),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "rerank_manifest_sha256": _sha256(rerank_manifest_path),
        "split": split_name,
        "sample_count": sample_count,
        "processed_question_counts": cast(JsonValue, processed_counts),
        "evaluated_question_counts": cast(JsonValue, evaluated_counts),
        "vmp_method": vmp_method,
        "baseline_method": baseline_method,
        "vmp_reranked_method": vmp_reranked_method,
        "baseline_reranked_method": baseline_reranked_method,
        "metrics": cast(
            JsonValue,
            {
                "v53_recall_all@5": v53_recall,
                "raw_v5_recall_all@5": raw_v5_recall,
                "shared_reranked_v43_recall_all@5": shared_v43_recall,
                "raw_v43_recall_all@5": raw_v43_recall,
                "delta_vs_raw_v5": delta_raw_v5,
                "delta_vs_shared_reranked_v43": delta_shared_v43,
                "delta_vs_raw_v43": delta_raw_v43,
                "v53_macro_type_recall_all@5": v53_macro,
                "raw_v5_macro_type_recall_all@5": raw_v5_macro,
                "shared_v43_macro_type_recall_all@5": shared_v43_macro,
                "macro_delta_vs_raw_v5": macro_delta_raw_v5,
                "macro_delta_vs_shared_v43": macro_delta_shared_v43,
                "max_type_regression_vs_raw_v5": max_type_regression,
            },
        ),
        "reranker": cast(
            JsonValue,
            {
                "provider": provider,
                "model": model,
                "selector_prompt_version": selector_prompt,
                "expected_selector_prompt_version": expected_selector_prompt_version,
                "boundary_prompt_version": boundary_prompt,
                "expected_boundary_prompt_version": expected_boundary_prompt_version,
                "candidate_count": candidate_count,
                "candidate_planner_version": candidate_planner,
                "expected_candidate_planner_version": (
                    expected_candidate_planner_version
                ),
                "candidate_planner_applied_questions": (
                    candidate_planner_applied_questions
                ),
                "baseline_candidate_planner_version": baseline_candidate_planner,
                "baseline_candidate_planner_identity_questions": (
                    baseline_candidate_planner_identity_questions
                ),
                "min_observed_candidate_count": observed_candidate_count,
                "baseline_min_observed_candidate_count": (baseline_observed_candidate_count),
                "selector_protected_top_n": reranked_v5.get("protected_top_n"),
                "boundary_protected_top_n": reranked_v5.get("boundary_protected_top_n"),
                "output_top_k": reranked_v5.get("output_top_k"),
                "parse_fallback_rate": fallback_rate,
                "boundary_parse_fallback_rate": boundary_fallback_rate,
                "recovered_questions": recovered_questions,
                "regressed_questions": regressed_questions,
                "boundary_calls": reranked_v5.get("boundary_calls"),
                "boundary_replacements_accepted": reranked_v5.get("boundary_replacements_accepted"),
                "boundary_atomic_support_failures": reranked_v5.get(
                    "boundary_atomic_support_failures"
                ),
                "boundary_grounded_promotions_accepted": reranked_v5.get(
                    "boundary_grounded_promotions_accepted"
                ),
                "baseline_boundary_grounded_promotions_accepted": reranked_v43.get(
                    "boundary_grounded_promotions_accepted"
                ),
                "selector_prompt_mismatch_count": reranked_v5.get("selector_prompt_mismatch_count"),
                "invalid_candidate_label_count": reranked_v5.get(
                    "invalid_candidate_label_count"
                ),
                "selector_span_binding_failure_count": reranked_v5.get(
                    "selector_span_binding_failure_count"
                ),
                "selector_grounded_promotion_count": reranked_v5.get(
                    "selector_grounded_promotion_count"
                ),
                "shared_across_methods": True,
            },
        ),
        "required": cast(
            JsonValue,
            {
                "min_recall_all@5": min_recall_all_at_5,
                "min_delta_vs_raw_v5": min_delta_vs_raw_v5,
                "min_delta_vs_shared_v43": min_delta_vs_shared_v43,
                "min_macro_delta_vs_raw_v5": min_macro_delta_vs_raw_v5,
                "max_type_regression_vs_raw_v5": max_type_regression_vs_raw_v5,
                "max_parse_fallback_rate": max_parse_fallback_rate,
                "max_boundary_fallback_rate": max_boundary_fallback_rate,
                "max_regressed_questions": max_regressed_questions,
                "min_recovered_questions": min_recovered_questions,
                "min_candidate_count": min_candidate_count,
                "expected_candidate_planner_version": (
                    expected_candidate_planner_version
                ),
            },
        ),
        "checks": cast(JsonValue, checks),
        "test_labels_used": False,
    }


def _metric(summary: dict[str, JsonValue], name: str) -> float:
    metrics = summary.get("metrics")
    return _number(metrics.get(name)) if isinstance(metrics, dict) else 0.0


def _type_values(summary: dict[str, JsonValue]) -> dict[str, float]:
    by_type = summary.get("by_question_type")
    if not isinstance(by_type, dict):
        return {}
    return {
        str(name): _number(metrics.get("recall_all@5"))
        for name, metrics in by_type.items()
        if isinstance(metrics, dict)
    }


def _type_macro(summary: dict[str, JsonValue]) -> float:
    values = list(_type_values(summary).values())
    return sum(values) / len(values) if values else 0.0


def _max_type_regression(
    reference: dict[str, JsonValue],
    candidate: dict[str, JsonValue],
) -> float:
    reference_values = _type_values(reference)
    candidate_values = _type_values(candidate)
    shared = set(reference_values).intersection(candidate_values)
    return max(
        (reference_values[name] - candidate_values[name] for name in shared),
        default=0.0,
    )


def _split_name(manifest: dict[str, JsonValue]) -> str | None:
    split = manifest.get("split")
    if not isinstance(split, dict):
        return None
    name = split.get("name")
    return name if isinstance(name, str) else None


def _number(value: object, *, default: float = 0.0) -> float:
    return float(value) if isinstance(value, int | float) else default


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _read_json(path: Path) -> dict[str, JsonValue]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_method(method: str) -> str:
    return method.strip().casefold().replace("-", "_")
