from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class IngestionConfig:
    cleaning_strategy: str = "deep"
    cleaning_params: dict[str, Any] = field(default_factory=dict)
    chunking_strategy: str = "naive"
    chunking_params: dict[str, Any] = field(default_factory=lambda: {"chunk_size": 300, "chunk_overlap": 100})
    embedding_model: str | None = None
    delete_collection_first: bool = True


@dataclass(slots=True)
class SourceMappingConfig:
    max_matches: int = 5
    page_window: int = 2
    page_offset_candidates: list[int] = field(default_factory=lambda: [0, -1, -2, 1, -3, 2, 3])
    semantic_fallback_enabled: bool = False
    include_chunk_pairs: bool = True


@dataclass(slots=True)
class InferenceConfig:
    top_k: int = 3
    temperature: float = 0.0
    max_tokens: int = 256
    retrieval_mode: str = "hybrid"
    hybrid_dense_weight: float = 0.65
    hybrid_sparse_weight: float = 0.35
    use_graph_augmentation: bool = True
    graph_max_extra_nodes: int = 2
    enable_query_rewriting: bool = False
    enable_response_verification: bool = False
    enable_regeneration: bool = False
    max_regeneration_attempts: int = 1
    llm_model: str | None = None
    embedding_model: str | None = None
    prompt_engineering_label: str = "default"
    use_example_response: bool = False


@dataclass(slots=True)
class ExecutionConfig:
    gateway_url: str = "http://localhost:8000"
    qdrant_url: str = "http://localhost:6333"
    collection: str = "guidance_chunks"
    batch_size: int = 256
    poll_interval_seconds: float = 2.0
    max_wait_seconds: float = 1800.0
    max_cases: int | None = None
    warmup_cases: int = 0
    include_unanswerable: bool = True
    output_dir: str = "./artifacts/runs"


@dataclass(slots=True)
class BenchmarkRunConfig:
    label: str
    dataset_path: str
    documents_version: str = "docs_v1"
    dataset_version: str | None = None
    notes: str = ""
    change_note: str = ""
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    source_mapping: SourceMappingConfig = field(default_factory=SourceMappingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
