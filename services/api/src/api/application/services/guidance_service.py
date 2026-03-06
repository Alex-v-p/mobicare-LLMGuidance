from __future__ import annotations

from api.clients.inference_client import InferenceClient
from shared.contracts.inference import (
    ApiGuidanceJobStatus,
    ApiGuidanceResponse,
    GuidanceRequest,
    InferenceRequest,
    JobAcceptedResponse,
    JobRecord,
)


class GuidanceService:
    def __init__(self, inference_client: InferenceClient | None = None) -> None:
        self._inference_client = inference_client or InferenceClient()

    def _to_inference_request(self, request: GuidanceRequest) -> InferenceRequest:
        return InferenceRequest(
            request_id=request.request_id,
            question=request.question,
            patient_variables=request.patient.values,
            options=request.options,
        )

    async def generate(self, request: GuidanceRequest) -> ApiGuidanceResponse:
        inference_response = await self._inference_client.generate(self._to_inference_request(request))
        return ApiGuidanceResponse(
            request_id=inference_response.request_id,
            status=inference_response.status,
            answer=inference_response.answer,
            model=inference_response.model,
            rag=inference_response.retrieved_context,
            used_variables=inference_response.used_variables,
            warnings=inference_response.warnings,
            metadata=inference_response.metadata,
        )

    async def submit_job(self, request: GuidanceRequest) -> JobAcceptedResponse:
        return await self._inference_client.submit_job(self._to_inference_request(request))

    async def get_job_status(self, request_id: str) -> ApiGuidanceJobStatus:
        record = await self._inference_client.get_job_status(request_id)
        return self._to_api_job_status(record)

    def _to_api_job_status(self, record: JobRecord) -> ApiGuidanceJobStatus:
        if record.result is None:
            return ApiGuidanceJobStatus(
                request_id=record.request_id,
                status=record.status,
                error=record.error,
                result_object_key=record.result_object_key,
                callback_attempts=record.callback_attempts,
                callback_last_status=record.callback_last_status,
                callback_last_error=record.callback_last_error,
                created_at=record.created_at,
                started_at=record.started_at,
                completed_at=record.completed_at,
                updated_at=record.updated_at,
            )

        return ApiGuidanceJobStatus(
            request_id=record.request_id,
            status=record.status,
            answer=record.result.answer,
            model=record.result.model,
            rag=record.result.retrieved_context,
            used_variables=record.result.used_variables,
            warnings=record.result.warnings,
            metadata=record.result.metadata,
            error=record.error,
            result_object_key=record.result_object_key,
            callback_attempts=record.callback_attempts,
            callback_last_status=record.callback_last_status,
            callback_last_error=record.callback_last_error,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            updated_at=record.updated_at,
        )
