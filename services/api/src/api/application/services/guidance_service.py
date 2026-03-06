from __future__ import annotations

from api.clients.inference_client import InferenceClient
from shared.contracts.inference import ApiGuidanceResponse, GuidanceRequest, InferenceRequest


class GuidanceService:
    def __init__(self, inference_client: InferenceClient | None = None) -> None:
        self._inference_client = inference_client or InferenceClient()

    async def generate(self, request: GuidanceRequest) -> ApiGuidanceResponse:
        inference_request = InferenceRequest(
            request_id=request.request_id,
            question=request.question,
            patient_variables=request.patient.values,
            options=request.options,
        )
        inference_response = await self._inference_client.generate(inference_request)
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
