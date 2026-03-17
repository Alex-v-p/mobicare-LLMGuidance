from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MIX = {
    "factual": 0.40,
    "clinical_scenario": 0.30,
    "paraphrased_factual": 0.20,
    "slightly_indirect": 0.10,
}


@dataclass(slots=True)
class PdfExtractionConfig:
    min_words: int = 35
    max_words: int = 180
    min_unique_words: int = 10
    target_block_words: int = 85
    word_window_overlap: int = 30
    merge_small_paragraphs: bool = True
    skip_reference_sections: bool = True
    skip_initial_pages: int = 5
    reference_page_markers: list[str] = field(
        default_factory=lambda: [
            "references",
            "bibliography",
            "to access the supplementary materials",
            "downloaded from",
        ]
    )
    reject_passage_markers: list[str] = field(
        default_factory=lambda: [
            "all rights reserved",
            "downloaded from",
            "permissions",
            "copyright",
            "doi:",
        ]
    )


@dataclass(slots=True)
class GenerationConfig:
    dataset_size: int = 100
    mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    temperature: float = 0.2
    seed: int = 42
    retries: int = 3
    provider: str = "ollama"
    model: str | None = None
    api_base_url: str | None = None
    api_key: str | None = None
    system_prompt: str = (
        "You generate benchmark cases for a medical RAG system. "
        "Only use the information in the supplied passage. "
        "Do not add outside medical knowledge. Return strict JSON only."
    )


@dataclass(slots=True)
class DatasetGenerationSettings:
    input_path: str | None = None
    output_path: str | None = None
    extraction: PdfExtractionConfig = field(default_factory=PdfExtractionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetGenerationSettings":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        extraction = PdfExtractionConfig(**raw.get("extraction", {}))
        generation = GenerationConfig(**raw.get("generation", raw))
        return cls(
            input_path=raw.get("input_path"),
            output_path=raw.get("output_path"),
            extraction=extraction,
            generation=generation,
        )
