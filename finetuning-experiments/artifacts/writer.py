from __future__ import annotations

from pathlib import Path
from typing import Any

from artifacts.summaries import write_run_summary
from utils.json import write_json


def _write_source_map_artifact(root: Path, run_id: str, payload: dict[str, Any]) -> Path:
    source_summary = payload.get("source_mapping_summary") or {}
    return write_json(
        root / f"{run_id}.source_maps.json",
        {
            "artifact_type": "source_maps",
            "artifact_version": payload.get("artifact_version"),
            "run_id": run_id,
            "label": payload.get("label"),
            "datetime": payload.get("datetime"),
            "dataset_version": payload.get("dataset_version"),
            "documents_version": payload.get("documents_version"),
            "source_mapping_summary": source_summary,
            "case_source_maps": source_summary.get("case_chunk_assignments") or [],
        },
    )


def write_run_artifact(output_dir: str, run_id: str, payload: dict[str, Any]) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = write_json(root / f"{run_id}.json", payload)
    write_run_summary(root, run_id, payload)
    _write_source_map_artifact(root, run_id, payload)
    return path
