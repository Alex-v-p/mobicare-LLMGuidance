from __future__ import annotations

from api.clients.inference_client import InferenceClient
from shared.contracts.ingestion import IngestionJobAcceptedResponse, IngestionJobRecord


class IngestionService:
    def __init__(self, inference_client: InferenceClient | None = None) -> None:
        self._inference_client = inference_client or InferenceClient()

    async def submit_job(self) -> IngestionJobAcceptedResponse:
        return await self._inference_client.submit_ingestion_job()

    async def get_job_status(self, job_id: str) -> IngestionJobRecord:
        return await self._inference_client.get_ingestion_job_status(job_id)
