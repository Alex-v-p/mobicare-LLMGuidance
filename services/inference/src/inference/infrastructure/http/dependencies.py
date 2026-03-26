from __future__ import annotations

from inference.application.services.guidance_service import GuidanceJobService, GuidanceRequestService
from inference.application.services.ingestion_service import IngestionJobService, IngestionRequestService
from inference.composition.common import (
    get_document_store,
    get_guidance_job_result_store,
    get_guidance_job_store,
    get_guidance_pipeline,
    get_ingestion_job_result_store,
    get_ingestion_job_store,
    get_ingestion_service,
    get_inference_settings,
    get_vector_store,
)


def get_guidance_request_service() -> GuidanceRequestService:
    return GuidanceRequestService(pipeline=get_guidance_pipeline())



def get_guidance_job_service() -> GuidanceJobService:
    return GuidanceJobService(
        store_factory=get_guidance_job_store,
        result_store=get_guidance_job_result_store(),
    )



def get_ingestion_request_service() -> IngestionRequestService:
    return IngestionRequestService(
        ingestion_service=get_ingestion_service(),
        vector_store=get_vector_store(),
    )



def get_ingestion_job_service() -> IngestionJobService:
    return IngestionJobService(
        store_factory=get_ingestion_job_store,
        result_store=get_ingestion_job_result_store(),
    )


__all__ = [
    "get_document_store",
    "get_guidance_job_result_store",
    "get_guidance_job_service",
    "get_guidance_job_store",
    "get_guidance_pipeline",
    "get_guidance_request_service",
    "get_ingestion_job_result_store",
    "get_ingestion_job_service",
    "get_ingestion_job_store",
    "get_ingestion_request_service",
    "get_ingestion_service",
    "get_inference_settings",
    "get_vector_store",
]
