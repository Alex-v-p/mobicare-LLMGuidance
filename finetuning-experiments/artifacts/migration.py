from __future__ import annotations

from copy import deepcopy
from typing import Any

from artifacts.compatibility import build_config_overview, build_telemetry_summary
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
    artifact.setdefault("environment", {})
    artifact.setdefault("source_mapping_summary", {})
    artifact.setdefault("retrieval_summary", {})
    artifact.setdefault("generation_summary", {})
    artifact.setdefault("api_summary", {})
    artifact.setdefault("normalized_metrics", {})
    artifact.setdefault("per_case_results", [])
    artifact.setdefault("config_overview", build_config_overview(artifact))
    artifact.setdefault("telemetry_summary", build_telemetry_summary(artifact))
    artifact.setdefault("case_count", len(artifact.get("per_case_results") or []))
    for case in artifact["per_case_results"]:
        if isinstance(case, dict):
            case.setdefault("endpoint_envelope", {})
            case.setdefault("retrieval_scores", {})
            case.setdefault("generation_scores", {})
            case.setdefault("timings", {})
            case.setdefault("telemetry", {})
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
    summary.setdefault("environment", {})
    summary.setdefault("telemetry_summary", build_telemetry_summary(summary))
    summary.setdefault("config_overview", build_config_overview(summary))
    summary["artifact_version"] = CURRENT_ARTIFACT_VERSION
    return summary



def ensure_summary_for_run_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    return build_run_summary(migrate_run_artifact(payload))
