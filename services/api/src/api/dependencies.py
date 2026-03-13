from __future__ import annotations

from functools import lru_cache

from minio import Minio

from api.application.services.document_service import DocumentService
from api.application.services.guidance_service import GuidanceService
from api.application.services.health_service import HealthService
from api.application.services.ingestion_service import IngestionService
from api.clients.inference_client import InferenceClient
from api.infrastructure.minio import create_minio_client
from api.repositories.document_repository import DocumentRepository
from shared.config import Settings, get_settings


@lru_cache(maxsize=1)
def get_api_settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    return create_minio_client(get_api_settings())


@lru_cache(maxsize=1)
def get_document_repository() -> DocumentRepository:
    settings = get_api_settings()
    return DocumentRepository(
        client=get_minio_client(),
        documents_bucket=settings.minio_documents_bucket,
        documents_prefix=settings.minio_documents_prefix,
    )


@lru_cache(maxsize=1)
def get_inference_client() -> InferenceClient:
    settings = get_api_settings()
    return InferenceClient(
        base_url=settings.inference_url,
        timeout_s=settings.inference_timeout_s,
        settings=settings,
    )


def get_document_service() -> DocumentService:
    return DocumentService(repository=get_document_repository())


def get_guidance_service() -> GuidanceService:
    return GuidanceService(inference_client=get_inference_client())


def get_ingestion_service() -> IngestionService:
    return IngestionService(inference_client=get_inference_client())


def get_health_service() -> HealthService:
    settings = get_api_settings()
    return HealthService(timeout_s=settings.healthcheck_timeout_s, settings=settings)
