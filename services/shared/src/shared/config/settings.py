from __future__ import annotations

from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_port: int = 8000
    inference_port: int = 8001

    inference_url: str = "http://inference:8001"
    inference_timeout_s: float = 60.0
    healthcheck_timeout_s: float = 2.0

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "guidance_chunks"

    minio_endpoint: str = "http://minio:9000"
    minio_secure: bool = False
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_documents_bucket: str = "guidance-documents"
    minio_documents_prefix: str = ""
    minio_results_bucket: str = "guidance-job-results"
    minio_job_retention_days: int = 7

    clinical_config_source: str = "packaged"
    clinical_config_bucket: str = "guidance-config"
    clinical_config_prefix: str = "clinical"
    clinical_config_cache_seconds: int = 60
    clinical_marker_ranges_object_name: str = "marker_ranges.json"
    clinical_drug_dosing_catalog_object_name: str = "drug_dosing_catalog.json"

    document_upload_max_bytes: int = 50 * 1024 * 1024
    document_allowed_extensions_csv: str = ""
    document_allowed_content_types_csv: str = ""

    redis_url: str = "redis://redis:6379/0"
    redis_job_queue: str = "guidance_jobs"
    redis_ingestion_job_queue: str = "ingestion_jobs"
    job_ttl_seconds: int = 7 * 24 * 60 * 60
    job_lease_seconds: int = 60
    job_heartbeat_interval_seconds: int = 20

    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:0.5b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_timeout_s: float = 120.0
    ollama_embedding_batch_size: int = 8
    ollama_embedding_fallback_concurrency: int = 2

    callback_timeout_s: int = 10
    callback_max_attempts: int = 3

    auth_validation_url: str = "http://auth-provider.local/validate"
    auth_validation_timeout_s: float = 10.0
    jwt_secret_key: str = "change-me"
    jwt_access_token_exp_minutes: int = 60
    jwt_issuer: str = "mobicare-llm-api"
    jwt_audience: str = "mobicare-gateway"

    retrieval_top_k: int = 3
    jobs_dir: str = "/data/jobs"
    worker_id: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def minio_client_endpoint(self) -> str:
        return self.minio_endpoint.replace("http://", "").replace("https://", "")

    @property
    def document_allowed_extensions(self) -> set[str]:
        return {value.strip().lower().lstrip(".") for value in self.document_allowed_extensions_csv.split(",") if value.strip()}

    @property
    def document_allowed_content_types(self) -> set[str]:
        return {value.strip().lower() for value in self.document_allowed_content_types_csv.split(",") if value.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
