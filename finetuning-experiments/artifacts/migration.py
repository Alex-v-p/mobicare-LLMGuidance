from __future__ import annotations

from typing import Any

from artifacts.models import CURRENT_ARTIFACT_VERSION
from scoring.normalization import normalize_run_metrics



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

    if not migrated.get("normalized_metrics"):
        migrated["normalized_metrics"] = normalize_run_metrics(
            migrated.get("retrieval_summary"),
            migrated.get("generation_summary"),
            migrated.get("api_summary"),
        )

    migrated["artifact_version"] = CURRENT_ARTIFACT_VERSION
    return migrated
