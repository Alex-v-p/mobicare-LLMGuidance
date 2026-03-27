from __future__ import annotations

import os
import sys
from typing import Literal, TypeVar

from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SettingsT = TypeVar("SettingsT", bound="SharedServiceSettings")


class SharedServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["dev", "prod"] = "dev"

    healthcheck_timeout_s: float = 2.0
    qdrant_url: str = "http://qdrant:6333"
    redis_url: str = "redis://redis:6379/0"
    ollama_url: str = "http://ollama:11434"

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

    internal_service_token: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def minio_client_endpoint(self) -> str:
        return self.minio_endpoint.replace("http://", "").replace("https://", "")

    @property
    def is_production(self) -> bool:
        return self.app_env == "prod"

    @property
    def enable_internal_service_auth(self) -> bool:
        return self.is_production

    @property
    def expose_api_docs(self) -> bool:
        return not self.is_production

    @model_validator(mode="after")
    def validate_internal_service_security(self) -> "SharedServiceSettings":
        if self.is_production and not self.internal_service_token.strip():
            raise ValueError("INTERNAL_SERVICE_TOKEN must be set when APP_ENV=prod.")
        return self


def resolve_runtime_settings(settings_cls: type[SettingsT]) -> SettingsT:
    app_env = os.environ.get("APP_ENV")
    is_pytest = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules

    if is_pytest and app_env is None:
        return settings_cls(_env_file=None, app_env="dev")

    return settings_cls()
