from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import BenchmarkRunConfig, ExecutionConfig, InferenceConfig, IngestionConfig, SourceMappingConfig


def load_run_config(path: str | Path) -> BenchmarkRunConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkRunConfig(
        label=raw["label"],
        dataset_path=raw["dataset_path"],
        documents_version=raw.get("documents_version", "docs_v1"),
        dataset_version=raw.get("dataset_version"),
        notes=raw.get("notes", ""),
        change_note=raw.get("change_note", ""),
        ingestion=IngestionConfig(**raw.get("ingestion", {})),
        source_mapping=SourceMappingConfig(**raw.get("source_mapping", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
        execution=ExecutionConfig(**raw.get("execution", {})),
    )
