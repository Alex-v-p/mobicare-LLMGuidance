from __future__ import annotations

from typing import Any, Protocol

from shared.contracts.inference import InferenceRequest, InferenceResponse, JobAcceptedResponse, JobRecord
from shared.contracts.ingestion import (
    IngestDocumentsRequest,
    IngestionCollectionDeleteResponse,
    IngestionJobAcceptedResponse,
    IngestionJobRecord,
)


class InferenceGatewayError(RuntimeError):
    def __init__(self, *, status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


class InferenceGateway(Protocol):
    async def generate_guidance(self, payload: InferenceRequest) -> InferenceResponse: ...

    async def submit_guidance_job(self, payload: InferenceRequest) -> JobAcceptedResponse: ...

    async def get_guidance_job_status(self, job_id: str) -> JobRecord: ...

    async def submit_ingestion_job(self, payload: IngestDocumentsRequest) -> IngestionJobAcceptedResponse: ...

    async def get_ingestion_job_status(self, job_id: str) -> IngestionJobRecord: ...

    async def delete_ingestion_collection(self) -> IngestionCollectionDeleteResponse: ...


__all__ = ["InferenceGateway", "InferenceGatewayError"]
