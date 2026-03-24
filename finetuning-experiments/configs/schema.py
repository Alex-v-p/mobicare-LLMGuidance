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
    llm_second_pass_enabled: bool = False
    max_soft_candidates: int = 12


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
    pipeline_variant: str = "standard"


@dataclass(slots=True)
class DeterministicRubricConfig:
    enabled: bool = True
    required_fact_weight: float = 0.35
    reference_alignment_weight: float = 0.15
    gold_alignment_weight: float = 0.10
    context_alignment_weight: float = 0.15
    groundedness_weight: float = 0.15
    verification_weight: float = 0.10


@dataclass(slots=True)
class LLMJudgeConfig:
    enabled: bool = False
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 384
    timeout_seconds: int = 120
    fail_open: bool = True
    include_retrieved_context: bool = True


@dataclass(slots=True)
class EvaluationConfig:
    deterministic_rubric: DeterministicRubricConfig = field(default_factory=DeterministicRubricConfig)
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)


@dataclass(slots=True)
class APITestConfig:
    enabled: bool = False
    include_guidance_endpoint: bool = True
    include_ingestion_endpoint: bool = False
    warmup_requests: int = 1
    guidance_repeat_count: int = 5
    ingestion_repeat_count: int = 1
    percentile_policy: str = "success_only"
    outlier_policy: str = "keep_all"
    ingestion_delete_collection_each_run: bool = False
    metrics_endpoint_path: str | None = None
    sample_case_ids: list[str] = field(default_factory=list)




@dataclass(slots=True)
class EnvironmentCaptureConfig:
    capture_enabled: bool = True
    hardware_note: str = ""
    docker_compose_path: str = "../docker-compose.yml"
    container_names: list[str] = field(default_factory=list)
    include_minio: bool = False
    minio_url: str | None = None
    minio_bucket: str | None = None


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
    api_test: APITestConfig = field(default_factory=APITestConfig)
    environment: EnvironmentCaptureConfig = field(default_factory=EnvironmentCaptureConfig)


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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
