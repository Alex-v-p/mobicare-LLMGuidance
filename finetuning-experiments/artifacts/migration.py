from __future__ import annotations

from typing import Any

from artifacts.models import CURRENT_ARTIFACT_VERSION
from scoring.normalization import normalize_run_metrics
from telemetry.stage_recorder import extract_guidance_telemetry, extract_ingestion_telemetry


def _legacy_source_list_from_matches(matches: list[dict[str, Any]] | None) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "direct_evidence": [],
        "partial_direct_evidence": [],
        "supporting": [],
        "tangential": [],
        "irrelevant": [],
    }
    for match in matches or []:
        item = dict(match)
        item.setdefault("label", "direct_evidence")
        buckets[item["label"]].append(item)
    return buckets


def migrate_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated.setdefault("artifact_type", "run")
    migrated.setdefault("artifact_version", "1.0")
    migrated.setdefault("notes", "")
    migrated.setdefault("change_note", "")
    migrated.setdefault("config", {})
    migrated.setdefault("cache", {})
    migrated.setdefault("ingestion_summary", {})
    migrated.setdefault("source_mapping_summary", {})
    migrated.setdefault("retrieval_summary", {})
    migrated.setdefault("generation_summary", {})
    migrated.setdefault("api_summary", {})
    migrated.setdefault("per_case_results", [])

    if migrated.get("ingestion_summary") and not migrated["ingestion_summary"].get("telemetry"):
        migrated["ingestion_summary"]["telemetry"] = extract_ingestion_telemetry(
            migrated["ingestion_summary"].get("raw_endpoint_result") or migrated["ingestion_summary"]
        )

    source_summary = migrated.get("source_mapping_summary") or {}
    assignments = source_summary.get("case_chunk_assignments") or []
    if assignments and not source_summary.get("label_totals"):
        source_summary["label_totals"] = {
            label: sum(len((item.get("source_list") or _legacy_source_list_from_matches(item.get("matches"))).get(label) or []) for item in assignments)
            for label in ("direct_evidence", "partial_direct_evidence", "supporting", "tangential", "irrelevant")
        }
        migrated["source_mapping_summary"] = source_summary

    for item in migrated.get("per_case_results") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("telemetry"):
            item["telemetry"] = extract_guidance_telemetry(item.get("raw_endpoint_result") or {})
        if not item.get("source_list"):
            item["source_list"] = _legacy_source_list_from_matches(item.get("source_match_candidates"))

    if not migrated.get("normalized_metrics"):
        migrated["normalized_metrics"] = normalize_run_metrics(
            migrated.get("retrieval_summary"),
            migrated.get("generation_summary"),
            migrated.get("api_summary"),
        )

    migrated["artifact_version"] = CURRENT_ARTIFACT_VERSION
    return migrated
