from __future__ import annotations

import json

from inference.storage.minio_documents import MinioDocumentStore
from inference.storage.minio_guidance_job_results import MinioGuidanceJobResultStore
from inference.storage.minio_ingestion_job_results import MinioIngestionJobResultStore
from shared.bootstrap import bootstrap_minio_resources
from shared.config import get_inference_settings


def bootstrap_inference_minio() -> dict[str, object]:
    settings = get_inference_settings()
    document_store = MinioDocumentStore(settings)
    bootstrap_report = bootstrap_minio_resources(settings=settings, client=document_store.client)

    guidance_results = MinioGuidanceJobResultStore(settings)
    guidance_results.ensure_bucket()

    ingestion_results = MinioIngestionJobResultStore(settings)
    ingestion_results.ensure_bucket()

    return {
        "buckets_created": bootstrap_report.buckets_created,
        "clinical_configs_seeded": bootstrap_report.clinical_configs_seeded,
        "results_bucket": settings.minio_results_bucket,
        "documents_bucket": settings.minio_documents_bucket,
        "clinical_config_bucket": settings.clinical_config_bucket,
    }


def main() -> None:
    print(json.dumps(bootstrap_inference_minio(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
