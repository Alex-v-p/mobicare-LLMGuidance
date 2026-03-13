from __future__ import annotations

from typing import Any

import httpx

from shared.config import Settings, get_settings
from shared.observability import REQUEST_ID_HEADER

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
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def _extract_error_payload(response: httpx.Response) -> tuple[str, str, dict[str, Any]]:
    fallback_message = response.text or "Inference service request failed"
    try:
        payload = response.json()
    except ValueError:
        return "INFERENCE_REQUEST_FAILED", fallback_message, {}

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            details = error.get("details")
            if isinstance(code, str) and isinstance(message, str):
                return code, message, details if isinstance(details, dict) else {}

        detail = payload.get("detail")
        if isinstance(detail, str):
            return "INFERENCE_REQUEST_FAILED", detail, {}

    return "INFERENCE_REQUEST_FAILED", fallback_message, {}


class InferenceClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._base_url = (base_url or self._settings.inference_url).rstrip("/")
        self._timeout_s = timeout_s if timeout_s is not None else self._settings.inference_timeout_s

    async def generate_guidance(self, payload: InferenceRequest) -> InferenceResponse:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await self._request(
                client,
                method="POST",
                path="/guidance/generate",
                json=payload.model_dump(mode="json"),
            )
            return InferenceResponse.model_validate(response.json())

    async def submit_guidance_job(
        self,
        payload: InferenceRequest,
    ) -> JobAcceptedResponse:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await self._request(
                client,
                method="POST",
                path="/guidance/jobs",
                json=payload.model_dump(mode="json"),
            )
            return JobAcceptedResponse.model_validate(response.json())

    async def get_guidance_job_status(self, job_id: str) -> JobRecord:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await self._request(
                client,
                method="GET",
                path=f"/guidance/jobs/{job_id}",
            )
            return JobRecord.model_validate(response.json())

    async def submit_ingestion_job(
        self,
        payload: IngestDocumentsRequest,
    ) -> IngestionJobAcceptedResponse:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await self._request(
                client,
                method="POST",
                path="/ingestion/jobs",
                json=payload.model_dump(mode="json"),
            )
            return IngestionJobAcceptedResponse.model_validate(response.json())

    async def get_ingestion_job_status(self, job_id: str) -> IngestionJobRecord:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            response = await self._request(
                client,
                method="GET",
                path=f"/ingestion/jobs/{job_id}",
            )
            return IngestionJobRecord.model_validate(response.json())

    async def _request(
        self,
        client: httpx.AsyncClient,
        *,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        try:
            response = await client.request(method=method, url=url, json=json)
        except httpx.TimeoutException as exc:
            raise InferenceClientError(
                status_code=503,
                code="INFERENCE_TIMEOUT",
                message="The inference service timed out.",
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceClientError(
                status_code=503,
                code="INFERENCE_SERVICE_UNAVAILABLE",
                message="The inference service is unavailable.",
                details={"reason": str(exc)},
            ) from exc

        if response.is_error:
            code, message, details = _extract_error_payload(response)
            upstream_request_id = response.headers.get(REQUEST_ID_HEADER)
            if upstream_request_id:
                details = {**details, "upstream_request_id": upstream_request_id}
            raise InferenceClientError(
                status_code=response.status_code,
                code=code,
                message=message,
                details=details,
            )
        return response
