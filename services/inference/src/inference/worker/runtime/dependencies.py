from __future__ import annotations

from inference.composition import common
from shared.config import get_worker_settings


def get_document_store():
    return common.get_document_store()



def get_guidance_job_store():
    return common.get_guidance_job_store()



def get_guidance_pipeline():
    return common.get_guidance_pipeline()



def get_guidance_job_result_store():
    return common.get_guidance_job_result_store()



def get_ingestion_job_store():
    return common.get_ingestion_job_store()



def get_ingestion_service():
    return common.get_ingestion_service()



def get_ingestion_job_result_store():
    return common.get_ingestion_job_result_store()


__all__ = [
    "get_document_store",
    "get_guidance_job_result_store",
    "get_guidance_job_store",
    "get_guidance_pipeline",
    "get_ingestion_job_result_store",
    "get_ingestion_job_store",
    "get_ingestion_service",
    "get_worker_settings",
]
