from __future__ import annotations

from copy import deepcopy
from typing import Any

from artifacts.models import CURRENT_ARTIFACT_VERSION
from artifacts.summaries import build_run_summary



def migrate_run_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = deepcopy(payload)
    artifact.setdefault("artifact_type", "run")
    artifact.setdefault("artifact_version", CURRENT_ARTIFACT_VERSION)
    artifact.setdefault("notes", "")
    artifact.setdefault("change_note", "")
    artifact.setdefault("cache", {})
    artifact.setdefault("ingestion_summary", {})
    artifact.setdefault("source_mapping_summary", {})
    artifact.setdefault("retrieval_summary", {})
    artifact.setdefault("generation_summary", {})
    artifact.setdefault("api_summary", {})
    artifact.setdefault("normalized_metrics", {})
    artifact.setdefault("per_case_results", [])
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
    summary.setdefault("artifact_version", CURRENT_ARTIFACT_VERSION)
    summary.setdefault("api_summary", {})
    summary.setdefault("telemetry_summary", {})
    summary["artifact_version"] = CURRENT_ARTIFACT_VERSION
    return summary



def ensure_summary_for_run_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    return build_run_summary(migrate_run_artifact(payload))
