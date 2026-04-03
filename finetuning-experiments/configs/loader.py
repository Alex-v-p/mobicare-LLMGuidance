from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path

from utils.json import read_json

from .schema import APITestConfig, BenchmarkRunConfig, DeterministicRubricConfig, EnvironmentCaptureConfig, EvaluationConfig, ExecutionConfig, InferenceConfig, IngestionConfig, LLMJudgeConfig, SourceMappingConfig
from .validator import validate_run_config



def _filter_dataclass_kwargs(cls, raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {}
    if not is_dataclass(cls):
        return dict(raw)
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in dict(raw).items() if k in allowed}




def build_run_config(raw: dict) -> BenchmarkRunConfig:
    execution_raw = dict(raw.get("execution", {}))
    api_test_raw = dict(execution_raw.pop("api_test", {}))
    environment_raw = dict(execution_raw.pop("environment", {}))
    evaluation_raw = dict(raw.get("evaluation", {}))
    rubric_raw = dict(evaluation_raw.get("deterministic_rubric", {}))
    llm_judge_raw = dict(evaluation_raw.get("llm_judge", {}))
    config = BenchmarkRunConfig(
        label=raw["label"],
        dataset_path=raw["dataset_path"],
        documents_version=raw.get("documents_version", "docs_v1"),
        dataset_version=raw.get("dataset_version"),
        notes=raw.get("notes", ""),
        change_note=raw.get("change_note", ""),
        ingestion=IngestionConfig(**_filter_dataclass_kwargs(IngestionConfig, raw.get("ingestion", {}))),
        source_mapping=SourceMappingConfig(**_filter_dataclass_kwargs(SourceMappingConfig, raw.get("source_mapping", {}))),
        inference=InferenceConfig(**_filter_dataclass_kwargs(InferenceConfig, raw.get("inference", {}))),
        evaluation=EvaluationConfig(
            deterministic_rubric=DeterministicRubricConfig(**_filter_dataclass_kwargs(DeterministicRubricConfig, rubric_raw)),
            llm_judge=LLMJudgeConfig(**_filter_dataclass_kwargs(LLMJudgeConfig, llm_judge_raw)),
        ),
        execution=ExecutionConfig(api_test=APITestConfig(**_filter_dataclass_kwargs(APITestConfig, api_test_raw)), environment=EnvironmentCaptureConfig(**_filter_dataclass_kwargs(EnvironmentCaptureConfig, environment_raw)), **_filter_dataclass_kwargs(ExecutionConfig, execution_raw)),
    )
    if config.evaluation.llm_judge.enabled and not str(config.evaluation.llm_judge.model or '').strip():
        config.evaluation.llm_judge.model = config.inference.llm_model
    validate_run_config(config)
    return config


def load_run_config(path: str | Path) -> BenchmarkRunConfig:
    return build_run_config(read_json(path))
