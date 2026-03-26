from __future__ import annotations

import inference.worker.handlers.guidance_handler as guidance_handler
import inference.worker.handlers.ingestion_handler as ingestion_handler
import inference.worker.runtime.main as worker_main


def test_worker_main_uses_worker_runtime_dependencies() -> None:
    assert worker_main.get_document_store.__module__ == "inference.worker.runtime.dependencies"
    assert worker_main.get_guidance_job_result_store.__module__ == "inference.worker.runtime.dependencies"
    assert worker_main.get_ingestion_job_result_store.__module__ == "inference.worker.runtime.dependencies"


def test_worker_handlers_use_worker_runtime_dependencies() -> None:
    assert guidance_handler.get_guidance_job_store.__module__ == "inference.worker.runtime.dependencies"
    assert guidance_handler.get_guidance_pipeline.__module__ == "inference.worker.runtime.dependencies"
    assert guidance_handler.get_guidance_job_result_store.__module__ == "inference.worker.runtime.dependencies"

    assert ingestion_handler.get_ingestion_job_store.__module__ == "inference.worker.runtime.dependencies"
    assert ingestion_handler.get_ingestion_service.__module__ == "inference.worker.runtime.dependencies"
    assert ingestion_handler.get_ingestion_job_result_store.__module__ == "inference.worker.runtime.dependencies"
