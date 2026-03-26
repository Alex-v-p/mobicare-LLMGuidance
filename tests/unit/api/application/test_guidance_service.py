from __future__ import annotations

import pytest

from api.application.services.guidance_service import GuidanceService
from api.clients.inference_client import InferenceClientError
from api.errors import AppError, ServiceUnavailableError
from shared.config import Settings
from shared.contracts.inference import ApiGuidanceJobStatus, GuidanceRequest, InferenceRequest, InferenceResponse, JobAcceptedResponse, JobRecord


class StubInferenceClient:
    def __init__(self, *, accepted: JobAcceptedResponse | None = None, record: JobRecord | None = None, error: Exception | None = None):
        self.accepted = accepted
        self.record = record
        self.error = error
        self.last_request: InferenceRequest | None = None

    async def submit_guidance_job(self, request: InferenceRequest) -> JobAcceptedResponse:
        self.last_request = request
        if self.error:
            raise self.error
        assert self.accepted is not None
        return self.accepted

    async def get_guidance_job_status(self, job_id: str) -> JobRecord:
        if self.error:
            raise self.error
        assert self.record is not None
        return self.record


@pytest.mark.asyncio
async def test_submit_job_maps_request_and_returns_accepted(guidance_request: GuidanceRequest):
    accepted = JobAcceptedResponse(job_id="job-1", request_id=guidance_request.request_id, status_url="http://api/jobs/job-1")
    client = StubInferenceClient(accepted=accepted)
    service = GuidanceService(inference_client=client)

    result = await service.submit_job(guidance_request)

    assert result == accepted
    assert client.last_request is not None
    assert client.last_request.request_id == guidance_request.request_id
    assert client.last_request.question == guidance_request.question
    assert client.last_request.patient_variables == guidance_request.patient.values
    assert client.last_request.options == guidance_request.options


@pytest.mark.asyncio
async def test_prod_submit_job_uses_safe_guidance_defaults(guidance_request: GuidanceRequest):
    prod_request = guidance_request.model_copy(
        update={
            "options": guidance_request.options.model_copy(
                update={
                    "top_k": 10,
                    "temperature": 0.8,
                    "max_tokens": 1024,
                    "use_example_response": True,
                    "enable_query_rewriting": True,
                    "enable_regeneration": True,
                    "callback_url": "https://example.com/callback",
                    "llm_model": "custom-llm",
                    "embedding_model": "custom-embed",
                    "pipeline_variant": "drug_dosing",
                }
            )
        },
        deep=True,
    )
    accepted = JobAcceptedResponse(job_id="job-1", request_id=guidance_request.request_id, status_url="http://api/jobs/job-1")
    client = StubInferenceClient(accepted=accepted)
    service = GuidanceService(
        inference_client=client,
        settings=Settings(app_env="prod", jwt_secret_key="secret", internal_service_token="token"),
    )

    await service.submit_job(prod_request)

    assert client.last_request is not None
    assert client.last_request.options.top_k == 3
    assert client.last_request.options.temperature == 0.2
    assert client.last_request.options.max_tokens == 256
    assert client.last_request.options.use_example_response is False
    assert client.last_request.options.enable_query_rewriting is False
    assert client.last_request.options.enable_regeneration is False
    assert client.last_request.options.callback_url is None
    assert client.last_request.options.llm_model is None
    assert client.last_request.options.embedding_model is None
    assert client.last_request.options.pipeline_variant == "standard"


@pytest.mark.asyncio
async def test_get_job_status_maps_completed_record(guidance_request: GuidanceRequest, inference_response: InferenceResponse):
    record = JobRecord(request_id=guidance_request.request_id, status="completed", request=InferenceRequest(request_id=guidance_request.request_id, question=guidance_request.question), result=inference_response)
    service = GuidanceService(inference_client=StubInferenceClient(record=record))

    result = await service.get_job_status(record.job_id)

    assert isinstance(result, ApiGuidanceJobStatus)
    assert result.answer == inference_response.answer
    assert result.model == inference_response.model
    assert result.rag == inference_response.retrieved_context
    assert result.used_variables == inference_response.used_variables


@pytest.mark.asyncio
async def test_prod_status_hides_debug_metadata(guidance_request: GuidanceRequest, inference_response: InferenceResponse):
    record = JobRecord(
        request_id=guidance_request.request_id,
        status="completed",
        request=InferenceRequest(request_id=guidance_request.request_id, question=guidance_request.question),
        result=inference_response,
        result_object_key="secret-key",
        callback_attempts=2,
        worker_id="worker-1",
    )
    service = GuidanceService(
        inference_client=StubInferenceClient(record=record),
        settings=Settings(app_env="prod", jwt_secret_key="secret", internal_service_token="token"),
    )

    result = await service.get_job_status(record.job_id)

    assert result.metadata == {}
    assert result.result_object_key is None
    assert result.callback_attempts == 0
    assert result.worker_id is None


@pytest.mark.asyncio
async def test_submit_job_maps_upstream_errors(guidance_request: GuidanceRequest):
    error = InferenceClientError(status_code=503, code="UPSTREAM_DOWN", message="nope", details={"reason": "boom"})
    service = GuidanceService(inference_client=StubInferenceClient(error=error))

    with pytest.raises(ServiceUnavailableError) as exc:
        await service.submit_job(guidance_request)

    assert exc.value.code == "UPSTREAM_DOWN"
    assert exc.value.details == {"reason": "boom"}


@pytest.mark.asyncio
async def test_get_job_status_maps_non_5xx_to_app_error():
    error = InferenceClientError(status_code=404, code="NOT_FOUND", message="missing", details={"job_id": "x"})
    service = GuidanceService(inference_client=StubInferenceClient(error=error))

    with pytest.raises(AppError) as exc:
        await service.get_job_status("job-1")

    assert exc.value.status_code == 404
    assert exc.value.code == "NOT_FOUND"



def test_map_inference_client_error_promotes_5xx_to_service_unavailable():
    from api.application.services.inference_error_mapping import map_inference_client_error

    exc = InferenceClientError(status_code=503, code="UPSTREAM_DOWN", message="down", details={"reason": "boom"})

    mapped = map_inference_client_error(exc)

    assert isinstance(mapped, ServiceUnavailableError)
    assert mapped.code == "UPSTREAM_DOWN"
