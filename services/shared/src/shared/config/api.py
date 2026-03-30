from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator

from shared.config.base import SharedServiceSettings, resolve_runtime_settings


class ApiSettings(SharedServiceSettings):
    api_port: int = 8000

    inference_url: str = "http://inference:8001"
    inference_timeout_s: float = 60.0

    auth_validation_url: str = "http://auth-provider.local/validate"
    auth_validation_timeout_s: float = 10.0
    jwt_secret_key: str = "change-me"
    jwt_access_token_exp_minutes: int = 60
    jwt_issuer: str = "mobicare-llm-api"
    jwt_audience: str = "mobicare-gateway"

    document_upload_max_bytes: int = 50 * 1024 * 1024
    document_allowed_extensions_csv: str = ""
    document_allowed_content_types_csv: str = ""

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
    ingestion_collection_delete_enabled: bool = True

    @property
    def document_allowed_extensions(self) -> set[str]:
        return {value.strip().lower().lstrip(".") for value in self.document_allowed_extensions_csv.split(",") if value.strip()}

    @property
    def document_allowed_content_types(self) -> set[str]:
        return {value.strip().lower() for value in self.document_allowed_content_types_csv.split(",") if value.strip()}

    @property
    def require_public_auth(self) -> bool:
        return self.is_production

    @property
    def expose_debug_metadata(self) -> bool:
        return not self.is_production

    @property
    def allow_runtime_option_overrides(self) -> bool:
        return not self.is_production

    @property
    def allow_ingestion_collection_delete(self) -> bool:
        return self.ingestion_collection_delete_enabled

    @model_validator(mode="after")
    def validate_api_production_security(self) -> "ApiSettings":
        if self.is_production and self.jwt_secret_key.strip() in {"", "change-me", "replace-this-in-real-environments"}:
            raise ValueError("JWT_SECRET_KEY must be changed from its development default when APP_ENV=prod.")
        return self


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    return resolve_runtime_settings(ApiSettings)
