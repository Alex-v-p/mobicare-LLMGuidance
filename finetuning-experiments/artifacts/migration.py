from __future__ import annotations

from typing import Any

from artifacts.models import CURRENT_ARTIFACT_VERSION
from scoring.normalization import normalize_run_metrics
from telemetry.stage_recorder import extract_guidance_telemetry, extract_ingestion_telemetry



def migrate_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated.setdefault("artifact_type", "run")
    migrated.setdefault("artifact_version", "1.0")
    migrated.setdefault("notes", "")
    migrated.setdefault("change_note", "")
    migrated.setdefault("config", {})
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

    for item in migrated.get("per_case_results") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("telemetry"):
            item["telemetry"] = extract_guidance_telemetry(item.get("raw_endpoint_result") or {})

    if not migrated.get("normalized_metrics"):
        migrated["normalized_metrics"] = normalize_run_metrics(
            migrated.get("retrieval_summary"),
            migrated.get("generation_summary"),
            migrated.get("api_summary"),
        )

    migrated["artifact_version"] = CURRENT_ARTIFACT_VERSION
    return migrated
