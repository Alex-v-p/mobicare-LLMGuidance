from __future__ import annotations

from pathlib import Path
from typing import Any

from artifacts.models import CURRENT_ARTIFACT_VERSION, RunSummaryArtifact
from utils.json import write_json



def build_run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config") or {}
    api_summary = payload.get("api_summary") or {}
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
        },
        cache={
            "run_fingerprint": (payload.get("cache") or {}).get("run_fingerprint"),
            "ingestion_fingerprint": (payload.get("cache") or {}).get("ingestion_fingerprint"),
            "ingestion_cache_status": ((payload.get("cache") or {}).get("ingestion_cache") or {}).get("status"),
            "run_registry_status": (payload.get("cache") or {}).get("run_registry_status"),
        },
        normalized_metrics=payload.get("normalized_metrics") or {},
        retrieval_summary=payload.get("retrieval_summary") or {},
        generation_summary=payload.get("generation_summary") or {},
        api_summary=api_summary,
        ingestion_summary=payload.get("ingestion_summary") or {},
        source_mapping_summary={
            "mapping_label": (payload.get("source_mapping_summary") or {}).get("mapping_label"),
            "strategy": (payload.get("source_mapping_summary") or {}).get("strategy"),
            "case_count": len((payload.get("source_mapping_summary") or {}).get("case_chunk_assignments") or []),
            "label_totals": (payload.get("source_mapping_summary") or {}).get("label_totals") or {},
        },
        telemetry_summary={
            "queue_delay": api_summary.get("queue_delay") or {},
            "execution_duration": api_summary.get("execution_duration") or {},
            "stage_latency_summary": api_summary.get("stage_latency_summary") or {},
        },
    )
    return summary.to_dict()



def write_run_summary(output_dir: str | Path, run_id: str, payload: dict[str, Any]) -> Path:
    return write_json(Path(output_dir) / f"{run_id}.summary.json", build_run_summary(payload))
