from __future__ import annotations

import pytest

from api.application.services.ingestion_service import IngestionService
from api.clients.inference_client import InferenceClientError
from api.errors import AppError, NotFoundError, ServiceUnavailableError
from shared.config import Settings
from shared.contracts.ingestion import ApiIngestionJobStatus, IngestDocumentsRequest, IngestionCollectionDeleteResponse, IngestionJobAcceptedResponse, IngestionJobRecord


class StubInferenceClient:
    def __init__(self, *, job_record: IngestionJobRecord | None = None, accepted: IngestionJobAcceptedResponse | None = None, delete_response: IngestionCollectionDeleteResponse | None = None, error: Exception | None = None):
        self.job_record = job_record
        self.accepted = accepted
        self.delete_response = delete_response
        self.error = error
        self.submitted_payload: IngestDocumentsRequest | None = None

    async def submit_ingestion_job(self, payload: IngestDocumentsRequest) -> IngestionJobAcceptedResponse:
        self.submitted_payload = payload
        if self.error:
            raise self.error
        assert self.accepted is not None
        return self.accepted

    async def get_ingestion_job_status(self, job_id: str) -> IngestionJobRecord:
        if self.error:
            raise self.error
        assert self.job_record is not None
        return self.job_record

    async def delete_ingestion_collection(self) -> IngestionCollectionDeleteResponse:
        if self.error:
            raise self.error
        assert self.delete_response is not None
        return self.delete_response


@pytest.mark.asyncio
async def test_submit_job_delegates_payload():
    payload = IngestDocumentsRequest()
    accepted = IngestionJobAcceptedResponse(job_id="ing-1", status_url="http://api/ingestion/jobs/ing-1")
    client = StubInferenceClient(accepted=accepted)
    service = IngestionService(inference_client=client)

    result = await service.submit_job(payload)

    assert result == accepted
    assert client.submitted_payload == payload


@pytest.mark.asyncio
async def test_get_job_status_returns_api_status():
    record = IngestionJobRecord(status="running")
    service = IngestionService(inference_client=StubInferenceClient(job_record=record))

    result = await service.get_job_status(record.job_id)

    assert isinstance(result, ApiIngestionJobStatus)
    assert result.job_id == record.job_id
    assert result.status == record.status


@pytest.mark.asyncio
async def test_prod_submit_job_uses_safe_ingestion_defaults():
    payload = IngestDocumentsRequest(
        options={
            "cleaning_strategy": "none",
            "chunking_strategy": "late",
            "chunking_params": {"chunk_size": 1024, "chunk_overlap": 512},
            "embedding_model": "custom-model",
        }
    )
    accepted = IngestionJobAcceptedResponse(job_id="ing-1", status_url="http://api/ingestion/jobs/ing-1")
    client = StubInferenceClient(accepted=accepted)
    service = IngestionService(
        inference_client=client,
        settings=Settings(app_env="prod", jwt_secret_key="secret", internal_service_token="token"),
    )

    await service.submit_job(payload)

    assert client.submitted_payload is not None
    assert client.submitted_payload.options.cleaning_strategy == "deep"
    assert client.submitted_payload.options.chunking_strategy == "naive"
    assert client.submitted_payload.options.chunking_params == {"chunk_size": 300, "chunk_overlap": 100}
    assert client.submitted_payload.options.embedding_model is None


@pytest.mark.asyncio
async def test_prod_delete_collection_is_disabled():
    service = IngestionService(
        inference_client=StubInferenceClient(delete_response=IngestionCollectionDeleteResponse(collection="guidance_chunks", existed=True)),
        settings=Settings(app_env="prod", jwt_secret_key="secret", internal_service_token="token"),
    )

    with pytest.raises(NotFoundError):
        await service.delete_collection()


@pytest.mark.asyncio
async def test_delete_collection_maps_upstream_5xx_to_service_unavailable():
    error = InferenceClientError(status_code=500, code="BROKEN", message="upstream bad")
    service = IngestionService(inference_client=StubInferenceClient(error=error))

    with pytest.raises(ServiceUnavailableError) as exc:
        await service.delete_collection()

    assert exc.value.code == "BROKEN"


@pytest.mark.asyncio
async def test_get_job_status_maps_non_5xx_to_app_error():
    error = InferenceClientError(status_code=400, code="BAD_REQUEST", message="bad payload")
    service = IngestionService(inference_client=StubInferenceClient(error=error))

    with pytest.raises(AppError) as exc:
        await service.get_job_status("ing-1")

    assert exc.value.status_code == 400
    assert exc.value.code == "BAD_REQUEST"
