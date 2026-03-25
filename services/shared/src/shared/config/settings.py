from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Literal

from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["dev", "prod"] = "dev"

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
    clinical_config_versions_prefix: str = "_versions"

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
    internal_service_token: str = ""

    retrieval_top_k: int = 3
    jobs_dir: str = "/data/jobs"
    worker_id: str | None = None

    production_guidance_top_k: int = 3
    production_guidance_temperature: float = 0.2
    production_guidance_max_tokens: int = 256
    production_guidance_use_graph_augmentation: bool = True
    production_guidance_pipeline_variant: Literal["standard", "drug_dosing"] = "standard"
    production_guidance_enable_response_verification: bool = True
    production_guidance_enable_unknown_fallback: bool = True

    production_ingestion_cleaning_strategy: Literal["none", "basic", "deep", "medical_guideline_deep"] = "deep"
    production_ingestion_chunking_strategy: Literal["naive", "page_indexed", "late"] = "naive"
    production_ingestion_chunk_size: int = 300
    production_ingestion_chunk_overlap: int = 100

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

    @property
    def is_production(self) -> bool:
        return self.app_env == "prod"

    @property
    def require_public_auth(self) -> bool:
        return self.is_production

    @property
    def enable_internal_service_auth(self) -> bool:
        return self.is_production

    @property
    def expose_debug_metadata(self) -> bool:
        return not self.is_production

    @property
    def allow_runtime_option_overrides(self) -> bool:
        return not self.is_production

    @property
    def allow_ingestion_collection_delete(self) -> bool:
        return not self.is_production

    @property
    def expose_api_docs(self) -> bool:
        return not self.is_production

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if not self.is_production:
            return self

        errors: list[str] = []
        if not self.internal_service_token.strip():
            errors.append("INTERNAL_SERVICE_TOKEN must be set when APP_ENV=prod.")
        if self.jwt_secret_key.strip() in {"", "change-me", "replace-this-in-real-environments"}:
            errors.append("JWT_SECRET_KEY must be changed from its development default when APP_ENV=prod.")
        if errors:
            raise ValueError(" ".join(errors))
        return self


def _resolve_settings() -> Settings:
    app_env = os.environ.get("APP_ENV")
    is_pytest = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules

    if is_pytest and app_env is None:
        return Settings(_env_file=None, app_env="dev")

    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _resolve_settings()
