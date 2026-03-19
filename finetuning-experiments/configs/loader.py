from __future__ import annotations

from pathlib import Path

from utils.json import read_json

from .schema import APITestConfig, BenchmarkRunConfig, ExecutionConfig, InferenceConfig, IngestionConfig, SourceMappingConfig
from .validator import validate_run_config



def build_run_config(raw: dict) -> BenchmarkRunConfig:
    execution_raw = dict(raw.get("execution", {}))
    api_test_raw = dict(execution_raw.pop("api_test", {}))
    config = BenchmarkRunConfig(
        label=raw["label"],
        dataset_path=raw["dataset_path"],
        documents_version=raw.get("documents_version", "docs_v1"),
        dataset_version=raw.get("dataset_version"),
        notes=raw.get("notes", ""),
        change_note=raw.get("change_note", ""),
        ingestion=IngestionConfig(**raw.get("ingestion", {})),
        source_mapping=SourceMappingConfig(**raw.get("source_mapping", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
        execution=ExecutionConfig(api_test=APITestConfig(**api_test_raw), **execution_raw),
    )
    validate_run_config(config)
    return config


def load_run_config(path: str | Path) -> BenchmarkRunConfig:
    return build_run_config(read_json(path))
