from __future__ import annotations

from api.application.error_mapping import map_inference_client_error
from api.application.ports import InferenceGateway, InferenceGatewayError
from api.errors import NotFoundError
from shared.config import Settings, get_settings
from shared.contracts.error_codes import ErrorCode
from shared.contracts.ingestion import (
    ApiIngestionJobStatus,
    IngestDocumentsRequest,
    IngestionCollectionDeleteResponse,
    IngestionJobAcceptedResponse,
    IngestionJobRecord,
    IngestionOptions,
)


class IngestionService:
    def __init__(self, inference_client: InferenceGateway, settings: Settings | None = None) -> None:
        self._inference_client = inference_client
        self._settings = settings or get_settings()

    async def submit_job(
        self,
        payload: IngestDocumentsRequest,
    ) -> IngestionJobAcceptedResponse:
        try:
            return await self._inference_client.submit_ingestion_job(self._apply_request_policy(payload))
        except InferenceGatewayError as exc:
            raise map_inference_client_error(exc) from exc

    async def get_job_status(self, job_id: str) -> ApiIngestionJobStatus:
        try:
            record = await self._inference_client.get_ingestion_job_status(job_id)
        except InferenceGatewayError as exc:
            raise map_inference_client_error(exc) from exc
        return self._to_api_job_status(record)

    async def delete_collection(self) -> IngestionCollectionDeleteResponse:
        if not self._settings.allow_ingestion_collection_delete:
            raise NotFoundError(
                code=ErrorCode.NOT_FOUND,
                message="This endpoint is not available in the current environment.",
            )
        try:
            return await self._inference_client.delete_ingestion_collection()
        except InferenceGatewayError as exc:
            raise map_inference_client_error(exc) from exc

    def _apply_request_policy(self, payload: IngestDocumentsRequest) -> IngestDocumentsRequest:
        if self._settings.allow_runtime_option_overrides:
            return payload

        safe_options = IngestionOptions(
            cleaning_strategy=self._settings.production_ingestion_cleaning_strategy,
            chunking_strategy=self._settings.production_ingestion_chunking_strategy,
            chunking_params={
                "chunk_size": self._settings.production_ingestion_chunk_size,
                "chunk_overlap": self._settings.production_ingestion_chunk_overlap,
            },
        )
        return payload.model_copy(update={"options": safe_options})

    def _to_api_job_status(self, record: IngestionJobRecord) -> ApiIngestionJobStatus:
        if self._settings.expose_debug_metadata:
            return self._to_verbose_api_job_status(record)
        return self._to_production_api_job_status(record)

    def _to_verbose_api_job_status(self, record: IngestionJobRecord) -> ApiIngestionJobStatus:
        if record.result is None:
            return ApiIngestionJobStatus(
                job_id=record.job_id,
                status=record.status,
                error=record.error,
                created_at=record.created_at,
                started_at=record.started_at,
                completed_at=record.completed_at,
                updated_at=record.updated_at,
            )

        return ApiIngestionJobStatus(
            job_id=record.job_id,
            status=record.status,
            documents_found=record.result.documents_found,
            chunks_created=record.result.chunks_created,
            vectors_upserted=record.result.vectors_upserted,
            collection=record.result.collection,
            error=record.error,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            updated_at=record.updated_at,
        )

    def _to_production_api_job_status(self, record: IngestionJobRecord) -> ApiIngestionJobStatus:
        if record.result is None:
            return ApiIngestionJobStatus(job_id=record.job_id, status=record.status, error=record.error)

        return ApiIngestionJobStatus(
            job_id=record.job_id,
            status=record.status,
            documents_found=record.result.documents_found,
            chunks_created=record.result.chunks_created,
            vectors_upserted=record.result.vectors_upserted,
            error=record.error,
        )

