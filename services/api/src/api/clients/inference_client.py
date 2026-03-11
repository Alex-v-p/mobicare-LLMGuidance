from __future__ import annotations

import os

import httpx

from shared.clients.http import create_async_client
from shared.contracts.inference import (
    InferenceRequest,
    InferenceResponse,
    JobAcceptedResponse,
    JobRecord,
)
from shared.contracts.ingestion import (
    IngestDocumentsRequest,
    IngestionJobAcceptedResponse,
    IngestionJobRecord,
)


class InferenceClientError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _extract_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Inference service request failed"

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail

    return response.text or "Inference service request failed"


class InferenceClient:
    def __init__(self, base_url: str | None = None, timeout_s: float | None = None) -> None:
        self._base_url = (
            base_url or os.getenv("INFERENCE_URL", "http://inference:8001")
        ).rstrip("/")
        self._timeout_s = timeout_s if timeout_s is not None else float(
            os.getenv("INFERENCE_TIMEOUT_S", "60")
        )

    async def generate_guidance(self, payload: InferenceRequest) -> InferenceResponse:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.post(
                f"{self._base_url}/guidance/generate",
                json=payload.model_dump(mode="json"),
            )
            if response.is_error:
                raise InferenceClientError(
                    response.status_code,
                    _extract_detail(response),
                )
            return InferenceResponse.model_validate(response.json())

    async def submit_guidance_job(
        self,
        payload: InferenceRequest,
    ) -> JobAcceptedResponse:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.post(
                f"{self._base_url}/guidance/jobs",
                json=payload.model_dump(mode="json"),
            )
            if response.is_error:
                raise InferenceClientError(
                    response.status_code,
                    _extract_detail(response),
                )
            return JobAcceptedResponse.model_validate(response.json())

    async def get_guidance_job_status(self, job_id: str) -> JobRecord:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.get(f"{self._base_url}/guidance/jobs/{job_id}")
            if response.is_error:
                raise InferenceClientError(
                    response.status_code,
                    _extract_detail(response),
                )
            return JobRecord.model_validate(response.json())

    async def submit_ingestion_job(
        self,
        payload: IngestDocumentsRequest,
    ) -> IngestionJobAcceptedResponse:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.post(
                f"{self._base_url}/ingestion/jobs",
                json=payload.model_dump(mode="json"),
            )
            if response.is_error:
                raise InferenceClientError(
                    response.status_code,
                    _extract_detail(response),
                )
            return IngestionJobAcceptedResponse.model_validate(response.json())

    async def get_ingestion_job_status(self, job_id: str) -> IngestionJobRecord:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await client.get(f"{self._base_url}/ingestion/jobs/{job_id}")
            if response.is_error:
                raise InferenceClientError(
                    response.status_code,
                    _extract_detail(response),
                )
            return IngestionJobRecord.model_validate(response.json())
