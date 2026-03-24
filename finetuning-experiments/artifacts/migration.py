from __future__ import annotations

from copy import deepcopy
from typing import Any

from artifacts.models import CURRENT_ARTIFACT_VERSION
from artifacts.summaries import build_run_summary


def _repair_generation_scores(generation_scores: dict[str, Any]) -> None:
    if not isinstance(generation_scores, dict):
        return

    deterministic = generation_scores.get("deterministic_rubric") or {}
    llm_judge = generation_scores.get("llm_judge") or {}
    deterministic_score = deterministic.get("score")
    deterministic_grade = deterministic.get("grade")
    llm_score = llm_judge.get("score")
    llm_grade = llm_judge.get("overall_grade")

    if "llm_judge_score" not in generation_scores:
        generation_scores["llm_judge_score"] = llm_score
    if "llm_judge_grade" not in generation_scores:
        generation_scores["llm_judge_grade"] = llm_grade

    # Repair older artifacts where judge_score was overwritten with the LLM-judge score.
    if deterministic_score is not None and llm_score is not None and generation_scores.get("judge_score") == llm_score:
        generation_scores.setdefault("judge_score_legacy_overwritten", llm_score)
        generation_scores["judge_score"] = deterministic_score
        generation_scores["judge_grade"] = deterministic_grade

    deterministic_applicable = (deterministic.get("applicable") if isinstance(deterministic, dict) else None)
    if deterministic_applicable is None:
        deterministic_applicable = generation_scores.get("deterministic_rubric_applicable")
    if deterministic_applicable is False:
        generation_scores["answer_quality_score"] = None
        generation_scores["answer_quality_grade"] = None
        generation_scores["judge_score"] = None
        generation_scores["judge_grade"] = None
        generation_scores["required_fact_recall"] = None
        generation_scores["faithfulness_to_gold_passage"] = None
        generation_scores["exact_pass"] = None

    if generation_scores.get("required_fact_total") == 0:
        generation_scores["required_fact_recall"] = None
    if generation_scores.get("forbidden_fact_total") == 0:
        generation_scores["forbidden_fact_violations"] = None


def migrate_run_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = deepcopy(payload)
    artifact.setdefault("artifact_type", "run")
    artifact.setdefault("artifact_version", CURRENT_ARTIFACT_VERSION)
    artifact.setdefault("notes", "")
    artifact.setdefault("change_note", "")
    artifact.setdefault("cache", {})
    artifact.setdefault("ingestion_summary", {})
    artifact.setdefault("environment", {})
    artifact.setdefault("source_mapping_summary", {})
    artifact.setdefault("retrieval_summary", {})
    artifact.setdefault("generation_summary", {})
    artifact.setdefault("api_summary", {})
    artifact.setdefault("normalized_metrics", {})
    artifact.setdefault("per_case_results", [])
    for case in artifact["per_case_results"]:
        if isinstance(case, dict):
            case.setdefault("endpoint_envelope", {})
            _repair_generation_scores(case.get("generation_scores") or {})
    artifact["cache"].setdefault("run_fingerprint", None)
    artifact["cache"].setdefault("ingestion_fingerprint", None)
    artifact["cache"].setdefault("ingestion_cache", {})
    artifact["cache"].setdefault("run_registry_status", None)
    artifact["api_summary"].setdefault("endpoint_summaries", {})
    artifact["api_summary"].setdefault("failure_taxonomy", {})
    artifact["api_summary"].setdefault("load_test_metadata", {})
    artifact["api_summary"].setdefault("benchmark_case_api", {})

    if artifact.get("per_case_results"):
        from scoring.aggregation import summarize_results
        from scoring.normalization import normalize_run_metrics

        recomputed = summarize_results(artifact.get("per_case_results") or [])
        artifact["retrieval_summary"] = recomputed.get("retrieval_summary") or artifact.get("retrieval_summary") or {}
        artifact["generation_summary"] = recomputed.get("generation_summary") or artifact.get("generation_summary") or {}
        artifact["api_summary"] = {
            **(artifact.get("api_summary") or {}),
            **(recomputed.get("api_summary") or {}),
        }
        artifact["api_summary"].setdefault("endpoint_summaries", {})
        artifact["api_summary"].setdefault("failure_taxonomy", {})
        artifact["api_summary"].setdefault("load_test_metadata", {})
        artifact["api_summary"].setdefault("benchmark_case_api", {})
        artifact["normalized_metrics"] = normalize_run_metrics(
            artifact.get("retrieval_summary"),
            artifact.get("generation_summary"),
            artifact.get("api_summary"),
        )

    artifact["artifact_version"] = CURRENT_ARTIFACT_VERSION
    return artifact


def migrate_summary_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    summary = deepcopy(payload)
    summary.setdefault("artifact_type", "run_summary")
    summary.setdefault("artifact_version", CURRENT_ARTIFACT_VERSION)
    summary.setdefault("api_summary", {})
    summary.setdefault("environment", {})
    summary.setdefault("telemetry_summary", {})
    summary["artifact_version"] = CURRENT_ARTIFACT_VERSION
    return summary


def ensure_summary_for_run_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    return build_run_summary(migrate_run_artifact(payload))
