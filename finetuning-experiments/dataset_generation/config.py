from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass(slots=True)
class ExtractionConfig:
    min_words: int = 40
    max_words: int = 140
    skip_first_pages: int = 4
    stop_at_references: bool = True


@dataclass(slots=True)
class GenerationConfig:
    dataset_version: str = "benchmark_v1"
    dataset_id: str = "benchmark_v1"
    dataset_size: int = 120
    seed: int = 42
    provider: str = "ollama"
    api_base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    timeout_seconds: int = 180
    temperature: float = 0.2
    prompt_version: str = "dataset_generation_v4"
    case_kind_mix: dict[str, float] = field(default_factory=lambda: {"answerable": 0.8, "unanswerable": 0.2})
    question_mix: dict[str, float] = field(default_factory=lambda: {
        "factual": 0.4,
        "clinical-scenario": 0.3,
        "paraphrased-factual": 0.2,
        "slightly-indirect": 0.1,
    })
    concurrency: int = 1
    retries: int = 4
    continue_on_error: bool = True
    resume_from_jsonl: bool = True
    ollama_options: dict[str, Any] = field(default_factory=lambda: {
        "num_predict": 400,
        "num_ctx": 2048,
        "temperature": 0.2,
    })


@dataclass(slots=True)
class DatasetGenerationSettings:
    input_path: str
    output_path: str
    output_jsonl_path: str | None
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)



def load_settings(config_path: str | Path, input_path: str, output_path: str, output_jsonl_path: str | None = None) -> DatasetGenerationSettings:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
    extraction = ExtractionConfig(**raw.get("extraction", {}))
    generation = GenerationConfig(**raw.get("generation", {}))
    return DatasetGenerationSettings(
        input_path=input_path,
        output_path=output_path,
        output_jsonl_path=output_jsonl_path,
        extraction=extraction,
        generation=generation,
    )
