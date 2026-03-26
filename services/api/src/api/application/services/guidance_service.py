from __future__ import annotations

from api.clients.inference_client import InferenceClient, InferenceClientError
from api.errors import AppError, ServiceUnavailableError
from shared.config import Settings, get_settings
from shared.contracts.inference import (
    ApiGuidanceJobStatus,
    GenerationOptions,
    GuidanceRequest,
    InferenceRequest,
    JobAcceptedResponse,
    JobRecord,
)


class GuidanceService:
    def __init__(self, inference_client: InferenceClient, settings: Settings | None = None) -> None:
        self._inference_client = inference_client
        self._settings = settings or get_settings()

    def _to_inference_request(self, request: GuidanceRequest) -> InferenceRequest:
        request = self._apply_request_policy(request)
        return InferenceRequest(
            request_id=request.request_id,
            question=request.question,
            patient_variables=request.patient.values,
            options=request.options,
        )

    async def submit_job(self, request: GuidanceRequest) -> JobAcceptedResponse:
        try:
            return await self._inference_client.submit_guidance_job(
                self._to_inference_request(request)
            )
        except InferenceClientError as exc:
            raise self._map_inference_error(exc) from exc

    async def get_job_status(self, job_id: str) -> ApiGuidanceJobStatus:
        try:
            record = await self._inference_client.get_guidance_job_status(job_id)
        except InferenceClientError as exc:
            raise self._map_inference_error(exc) from exc
        return self._to_api_job_status(record)

    def _apply_request_policy(self, request: GuidanceRequest) -> GuidanceRequest:
        if self._settings.allow_runtime_option_overrides:
            return request

        safe_options = GenerationOptions(
            top_k=min(max(request.options.top_k, 1), self._settings.production_guidance_top_k),
            temperature=min(max(request.options.temperature, 0.0), self._settings.production_guidance_temperature),
            max_tokens=min(max(request.options.max_tokens, 1), self._settings.production_guidance_max_tokens),
            use_graph_augmentation=self._settings.production_guidance_use_graph_augmentation,
            enable_response_verification=self._settings.production_guidance_enable_response_verification,
            enable_unknown_fallback=self._settings.production_guidance_enable_unknown_fallback,
            pipeline_variant=self._settings.production_guidance_pipeline_variant,
        )
        return request.model_copy(update={"options": safe_options})

    def _to_api_job_status(self, record: JobRecord) -> ApiGuidanceJobStatus:
        if self._settings.expose_debug_metadata:
            return self._to_verbose_api_job_status(record)
        return self._to_production_api_job_status(record)

    def _to_verbose_api_job_status(self, record: JobRecord) -> ApiGuidanceJobStatus:
        if record.result is None:
            return ApiGuidanceJobStatus(
                job_id=record.job_id,
                request_id=record.request_id,
                status=record.status,
                error=record.error,
                result_object_key=record.result_object_key,
                callback_attempts=record.callback_attempts,
                callback_last_status=record.callback_last_status,
                callback_last_error=record.callback_last_error,
                worker_id=record.worker_id,
                lease_expires_at=record.lease_expires_at,
                created_at=record.created_at,
                started_at=record.started_at,
                completed_at=record.completed_at,
                updated_at=record.updated_at,
            )

        return ApiGuidanceJobStatus(
            job_id=record.job_id,
            request_id=record.request_id,
            status=record.status,
            answer=record.result.answer,
            model=record.result.model,
            rag=record.result.retrieved_context,
            used_variables=record.result.used_variables,
            warnings=record.result.warnings,
            metadata=record.result.metadata,
            verification=record.result.verification,
            error=record.error,
            result_object_key=record.result_object_key,
            callback_attempts=record.callback_attempts,
            callback_last_status=record.callback_last_status,
            callback_last_error=record.callback_last_error,
            worker_id=record.worker_id,
            lease_expires_at=record.lease_expires_at,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            updated_at=record.updated_at,
        )

    def _to_production_api_job_status(self, record: JobRecord) -> ApiGuidanceJobStatus:
        if record.result is None:
            return ApiGuidanceJobStatus(
                job_id=record.job_id,
                request_id=record.request_id,
                status=record.status,
                error=record.error,
            )

        return ApiGuidanceJobStatus(
            job_id=record.job_id,
            request_id=record.request_id,
            status=record.status,
            answer=record.result.answer,
            model=record.result.model,
            rag=record.result.retrieved_context,
            used_variables=record.result.used_variables,
            warnings=record.result.warnings,
            verification=record.result.verification,
            error=record.error,
        )

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
