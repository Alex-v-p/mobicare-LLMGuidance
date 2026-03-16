from __future__ import annotations

from api.clients.inference_client import InferenceClient, InferenceClientError
from api.errors import AppError, ServiceUnavailableError
from shared.contracts.ingestion import (
    IngestDocumentsRequest,
    IngestionCollectionDeleteResponse,
    IngestionJobAcceptedResponse,
    IngestionJobRecord,
)


class IngestionService:
    def __init__(self, inference_client: InferenceClient) -> None:
        self._inference_client = inference_client

    async def submit_job(
        self,
        payload: IngestDocumentsRequest,
    ) -> IngestionJobAcceptedResponse:
        try:
            return await self._inference_client.submit_ingestion_job(payload)
        except InferenceClientError as exc:
            raise self._map_inference_error(exc) from exc

    async def get_job_status(self, job_id: str) -> IngestionJobRecord:
        try:
            return await self._inference_client.get_ingestion_job_status(job_id)
        except InferenceClientError as exc:
            raise self._map_inference_error(exc) from exc

    async def delete_collection(self) -> IngestionCollectionDeleteResponse:
        try:
            return await self._inference_client.delete_ingestion_collection()
        except InferenceClientError as exc:
            raise self._map_inference_error(exc) from exc

    def _map_inference_error(self, exc: InferenceClientError) -> AppError:
        if exc.status_code >= 500:
            return ServiceUnavailableError(
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
        return AppError(
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            details=exc.details,
        )
