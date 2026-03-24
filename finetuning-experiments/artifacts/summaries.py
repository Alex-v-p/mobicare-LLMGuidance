from __future__ import annotations

from pathlib import Path
from typing import Any

from artifacts.models import CURRENT_ARTIFACT_VERSION, RunSummaryArtifact
from utils.json import write_json



def build_run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config") or {}
    api_summary = payload.get("api_summary") or {}
    primary_api = api_summary.get("endpoint_summaries", {}).get("guidance_endpoint") or api_summary
    summary = RunSummaryArtifact(
        artifact_type="run_summary",
        artifact_version=CURRENT_ARTIFACT_VERSION,
        run_id=payload["run_id"],
        label=payload.get("label", ""),
        datetime=payload.get("datetime", ""),
        dataset_version=payload.get("dataset_version"),
        documents_version=payload.get("documents_version"),
        notes=payload.get("notes", ""),
        change_note=payload.get("change_note", ""),
        case_count=len(payload.get("per_case_results") or []),
        config_overview={
            "ingestion": (config.get("ingestion") or {}),
            "inference": (config.get("inference") or {}),
            "source_mapping": (config.get("source_mapping") or {}),
            "execution": {
                "api_test": ((config.get("execution") or {}).get("api_test") or {}),
                "environment": ((config.get("execution") or {}).get("environment") or {}),
            },
        },
        cache={
            "run_fingerprint": (payload.get("cache") or {}).get("run_fingerprint"),
            "ingestion_fingerprint": (payload.get("cache") or {}).get("ingestion_fingerprint"),
            "ingestion_cache_status": ((payload.get("cache") or {}).get("ingestion_cache") or {}).get("status"),
            "run_registry_status": (payload.get("cache") or {}).get("run_registry_status"),
        },
        environment={
            "git": ((payload.get("environment") or {}).get("git") or {}),
            "models": ((payload.get("environment") or {}).get("models") or {}),
            "runtime": ((payload.get("environment") or {}).get("runtime") or {}),
            "hardware_note": ((payload.get("environment") or {}).get("hardware_note") or ""),
            "container_versions": ((payload.get("environment") or {}).get("container_versions") or {}),
            "minio": ((payload.get("environment") or {}).get("minio") or {}),
        },
        normalized_metrics=payload.get("normalized_metrics") or {},
        retrieval_summary=payload.get("retrieval_summary") or {},
        generation_summary=payload.get("generation_summary") or {},
        api_summary={
            **api_summary,
            "endpoint_summaries": api_summary.get("endpoint_summaries") or {},
            "failure_taxonomy": api_summary.get("failure_taxonomy") or {},
            "load_test_metadata": api_summary.get("load_test_metadata") or {},
        },
        ingestion_summary={
            **(payload.get("ingestion_summary") or {}),
            "ingestion_metrics": ((payload.get("ingestion_summary") or {}).get("ingestion_metrics") or {}),
        },
        source_mapping_summary={
            "mapping_label": (payload.get("source_mapping_summary") or {}).get("mapping_label"),
            "strategy": (payload.get("source_mapping_summary") or {}).get("strategy"),
            "case_count": len((payload.get("source_mapping_summary") or {}).get("case_chunk_assignments") or []),
            "label_totals": (payload.get("source_mapping_summary") or {}).get("label_totals") or {},
        },
        telemetry_summary={
            "success_only": primary_api.get("success_only") or {},
            "queue_delay": primary_api.get("queue_delay") or {},
            "execution_duration": primary_api.get("execution_duration") or {},
            "stage_latency_summary": primary_api.get("stage_latency_summary") or {},
            "endpoint_summaries": api_summary.get("endpoint_summaries") or {},
            "failure_taxonomy": api_summary.get("failure_taxonomy") or {},
        },
    )
    return summary.to_dict()



def write_run_summary(output_dir: str | Path, run_id: str, payload: dict[str, Any]) -> Path:
    return write_json(Path(output_dir) / f"{run_id}.summary.json", build_run_summary(payload))
