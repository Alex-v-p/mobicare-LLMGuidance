from __future__ import annotations

import os

from shared.clients.http import create_async_client
from shared.contracts.inference import InferenceRequest, InferenceResponse, JobAcceptedResponse, JobRecord
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionResponse


class InferenceClient:
    def __init__(self, base_url: str | None = None, timeout_s: float | None = None) -> None:
        self._base_url = (base_url or os.getenv("INFERENCE_URL", "http://inference:8001")).rstrip("/")
        self._timeout_s = timeout_s if timeout_s is not None else float(os.getenv("INFERENCE_TIMEOUT_S", "60"))

    async def generate(self, payload: InferenceRequest) -> InferenceResponse:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/generate", json=payload.model_dump(mode="json"))
            response.raise_for_status()
            return InferenceResponse.model_validate(response.json())

    async def ingest(self, payload: IngestDocumentsRequest | None = None) -> IngestionResponse:
        body = (payload or IngestDocumentsRequest()).model_dump(mode="json")
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/ingest", json=body)
            response.raise_for_status()
            return IngestionResponse.model_validate(response.json())

    async def submit_job(self, payload: InferenceRequest) -> JobAcceptedResponse:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/jobs", json=payload.model_dump(mode="json"))
            response.raise_for_status()
            return JobAcceptedResponse.model_validate(response.json())

    async def get_job_status(self, job_id: str) -> JobRecord:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.get(f"{self._base_url}/jobs/{job_id}")
            response.raise_for_status()
            return JobRecord.model_validate(response.json())
