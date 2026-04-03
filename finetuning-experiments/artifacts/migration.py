from __future__ import annotations

from copy import deepcopy
from typing import Any

from adapters.guidance_payloads import normalize_guidance_record
from artifacts.compatibility import backfill_generation_summary_fields, build_config_overview, build_telemetry_summary
from artifacts.models import CURRENT_ARTIFACT_VERSION
from artifacts.summaries import build_run_summary
from scoring.aggregation import summarize_results
from scoring.generation import finalize_generation_score_fields
from scoring.retrieval import score_retrieval




def _backfill_case_generation(case: dict[str, Any]) -> None:
    generation_scores = dict(case.get("generation_scores") or {})
    if not generation_scores:
        return
    case["generation_scores"] = finalize_generation_score_fields(generation_scores)

def _backfill_case_retrieval(case: dict[str, Any]) -> None:
    raw_record = case.get("raw_endpoint_result") or {}
    if not isinstance(raw_record, dict) or not raw_record:
        return
    normalized = normalize_guidance_record(raw_record)
    enriched_chunks = list(normalized.get("rag") or [])
    if enriched_chunks:
        case["retrieved_chunks"] = enriched_chunks

    retrieval_scores = dict(case.get("retrieval_scores") or {})
    needs_rebuild = any(
        key not in retrieval_scores
        for key in (
            "retrieved_overlap_score_available_rate",
            "retrieved_semantic_score_available_rate",
            "retrieved_average_ranking_score",
            "retrieved_ranking_score_available_rate",
        )
    )
    if not needs_rebuild:
        return

    source_mapping = {
        "source_list": dict(case.get("source_list") or {}),
        "matches": list(case.get("source_match_candidates") or []),
        "metadata": {
            "applicable": retrieval_scores.get("applicable", True),
            "skipped_reason": retrieval_scores.get("skipped_reason"),
        },
    }
    rebuilt = score_retrieval(source_mapping, enriched_chunks)
    rebuilt.update({key: value for key, value in retrieval_scores.items() if key not in rebuilt})
    case["retrieval_scores"] = rebuilt


def migrate_run_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = deepcopy(payload)
    artifact.setdefault("artifact_type", "run")
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
    artifact["config_overview"] = build_config_overview(artifact)
    artifact["telemetry_summary"] = build_telemetry_summary(artifact)
    artifact.setdefault("case_count", len(artifact.get("per_case_results") or []))
    for case in artifact["per_case_results"]:
        if isinstance(case, dict):
            case.setdefault("endpoint_envelope", {})
            case.setdefault("retrieval_scores", {})
            case.setdefault("generation_scores", {})
            case.setdefault("timings", {})
            case.setdefault("telemetry", {})
            _backfill_case_generation(case)
            _backfill_case_retrieval(case)
    recomputed = summarize_results(artifact["per_case_results"]) if artifact.get("per_case_results") else {}
    artifact["retrieval_summary"] = {**(artifact.get("retrieval_summary") or {}), **(recomputed.get("retrieval_summary") or {})}
    artifact["generation_summary"] = backfill_generation_summary_fields({**(artifact.get("generation_summary") or {}), **(recomputed.get("generation_summary") or {})})
    artifact["api_summary"] = {**(artifact.get("api_summary") or {}), **(recomputed.get("api_summary") or {})}
    artifact["telemetry_summary"] = build_telemetry_summary(artifact)
    artifact["cache"].setdefault("run_fingerprint", None)
    artifact["cache"].setdefault("ingestion_fingerprint", None)
    artifact["cache"].setdefault("ingestion_cache", {})
    artifact["cache"].setdefault("run_registry_status", None)
    artifact["api_summary"].setdefault("endpoint_summaries", {})
    artifact["api_summary"].setdefault("failure_taxonomy", {})
    artifact["api_summary"].setdefault("load_test_metadata", {})
    artifact["api_summary"].setdefault("benchmark_case_api", {})
    artifact["artifact_version"] = CURRENT_ARTIFACT_VERSION
    return artifact


def migrate_summary_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    summary = deepcopy(payload)
    summary.setdefault("artifact_type", "run_summary")
    summary.setdefault("api_summary", {})
    summary.setdefault("environment", {})
    summary["generation_summary"] = backfill_generation_summary_fields(summary.get("generation_summary") or {})
    summary["telemetry_summary"] = build_telemetry_summary(summary)
    summary["config_overview"] = build_config_overview(summary)
    summary["artifact_version"] = CURRENT_ARTIFACT_VERSION
    return summary


def ensure_summary_for_run_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    return build_run_summary(migrate_run_artifact(payload))
