from __future__ import annotations

import importlib.util
import sys
import types


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# Lightweight import stubs for optional infrastructure clients not installed in the test environment.
# Important: only install stubs when the real dependency is unavailable, so integration tests can use
# the actual Docker-backed libraries when they are installed in the environment.
if not _module_available("minio"):
    minio_mod = types.ModuleType("minio")

    class _Minio: ...

    minio_mod.Minio = _Minio
    sys.modules["minio"] = minio_mod

    error_mod = types.ModuleType("minio.error")

    class S3Error(Exception):
        def __init__(self, *args, code: str = "", **kwargs):
            super().__init__(*args)
            self.code = code

    error_mod.S3Error = S3Error
    sys.modules["minio.error"] = error_mod

    datatypes_mod = types.ModuleType("minio.datatypes")

    class Object: ...

    datatypes_mod.Object = Object
    sys.modules["minio.datatypes"] = datatypes_mod

    helpers_mod = types.ModuleType("minio.helpers")

    class ObjectWriteResult: ...

    helpers_mod.ObjectWriteResult = ObjectWriteResult
    sys.modules["minio.helpers"] = helpers_mod

    common_mod = types.ModuleType("minio.commonconfig")
    common_mod.ENABLED = "Enabled"

    class Filter:
        def __init__(self, prefix=None):
            self.prefix = prefix

    common_mod.Filter = Filter
    sys.modules["minio.commonconfig"] = common_mod

    life_mod = types.ModuleType("minio.lifecycleconfig")

    class Expiration:
        def __init__(self, days=None):
            self.days = days

    class Rule:
        def __init__(self, status, rule_filter=None, rule_id=None, expiration=None):
            self.status = status
            self.rule_filter = rule_filter
            self.rule_id = rule_id
            self.expiration = expiration

    class LifecycleConfig:
        def __init__(self, rules):
            self.rules = rules

    life_mod.Expiration = Expiration
    life_mod.Rule = Rule
    life_mod.LifecycleConfig = LifecycleConfig
    sys.modules["minio.lifecycleconfig"] = life_mod

if not _module_available("redis"):
    redis_pkg = types.ModuleType("redis")
    asyncio_mod = types.ModuleType("redis.asyncio")

    class RedisError(Exception):
        pass

    class AuthenticationError(RedisError):
        pass

    class _RedisClient:
        @classmethod
        def from_url(cls, *args, **kwargs):
            return cls()

        async def ping(self):
            return True

        async def aclose(self):
            return None

        async def set(self, *args, **kwargs):
            return True

        async def rpush(self, *args, **kwargs):
            return 1

        async def get(self, *args, **kwargs):
            return None

        async def blpop(self, *args, **kwargs):
            return None

        async def scan_iter(self, *args, **kwargs):
            if False:
                yield None
            return

    asyncio_mod.Redis = _RedisClient
    asyncio_mod.RedisError = RedisError
    asyncio_mod.AuthenticationError = AuthenticationError
    asyncio_mod.from_url = _RedisClient.from_url
    redis_pkg.asyncio = asyncio_mod
    sys.modules["redis"] = redis_pkg
    sys.modules["redis.asyncio"] = asyncio_mod

if not _module_available("qdrant_client"):
    qdrant_mod = types.ModuleType("qdrant_client")

    class QdrantClient:
        def __init__(self, *args, **kwargs):
            pass

    qdrant_mod.QdrantClient = QdrantClient
    sys.modules["qdrant_client"] = qdrant_mod
    http_models = types.ModuleType("qdrant_client.http.models")
    for name in [
        "Distance",
        "VectorParams",
        "PointStruct",
        "Filter",
        "FieldCondition",
        "MatchValue",
        "SparseVector",
        "NamedVector",
        "NamedSparseVector",
        "SparseIndexParams",
        "Modifier",
        "SearchRequest",
        "FusionQuery",
        "Prefetch",
        "Fusion",
    ]:
        http_models.__dict__[name] = type(name, (), {})
    sys.modules["qdrant_client.http.models"] = http_models
    models_mod = types.ModuleType("qdrant_client.models")

    class Distance:
        COSINE = "cosine"

    class VectorParams:
        def __init__(self, size=None, distance=None):
            self.size = size
            self.distance = distance

    class PointStruct:
        def __init__(self, id=None, vector=None, payload=None):
            self.id = id
            self.vector = vector
            self.payload = payload

    models_mod.Distance = Distance
    models_mod.VectorParams = VectorParams
    models_mod.PointStruct = PointStruct
    sys.modules["qdrant_client.models"] = models_mod

import io
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from shared.contracts.documents import DocumentDeleteResponse, DocumentMetadata
from shared.contracts.inference import GuidanceRequest, InferenceResponse, JobRecord, RetrievedContext
from shared.contracts.ingestion import IngestionResponse


class StubDocumentService:
    def __init__(self) -> None:
        self.documents = [
            DocumentMetadata(
                object_name="guidelines/esc-heart-failure.pdf",
                title="esc-heart-failure.pdf",
                bucket="guidance-documents",
                prefix="guidelines",
                size_bytes=123,
                extension="pdf",
                content_type="application/pdf",
                etag="etag-1",
                last_modified=datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
            )
        ]
        self.blob_content = b"sample document"

    def list_metadata(self, *, offset: int = 0, limit: int = 100):
        docs = self.documents[offset : offset + limit]
        from shared.contracts.documents import DocumentMetadataListResponse

        return DocumentMetadataListResponse(
            documents=docs,
            count=len(docs),
            total_count=len(self.documents),
            offset=offset,
            limit=limit,
            has_more=False,
        )

    def get_document(self, object_name: str):
        from api.infrastructure.repositories.documents.models import DocumentBlob

        return DocumentBlob(
            object_name=object_name,
            bucket="guidance-documents",
            content=self.blob_content,
            content_type="application/pdf",
            etag="etag-1",
            last_modified=datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
            size_bytes=len(self.blob_content),
        )

    def upload_document(self, **kwargs):
        from shared.contracts.documents import DocumentUploadResponse

        return DocumentUploadResponse(document=self.documents[0])

    def delete_document(self, object_name: str):
        return DocumentDeleteResponse(object_name=object_name, bucket="guidance-documents")


@pytest.fixture
def guidance_request() -> GuidanceRequest:
    return GuidanceRequest(question="What should I prescribe?", patient={"values": {"age": 72, "egfr": 55}})


@pytest.fixture
def inference_response(guidance_request: GuidanceRequest) -> InferenceResponse:
    return InferenceResponse(
        request_id=guidance_request.request_id,
        status="ok",
        model="qwen2.5:0.5b",
        answer="Use guideline-directed medical therapy.",
        retrieved_context=[
            RetrievedContext(
                source_id="doc-1",
                title="ESC Heart Failure",
                snippet="Initiate ACE inhibitor.",
                chunk_id="chunk-1",
                page_number=2,
            )
        ],
        used_variables={"age": 72, "egfr": 55},
        warnings=[],
        metadata={"retrieval_mode": "hybrid"},
    )


@pytest.fixture
def completed_job_record(guidance_request: GuidanceRequest, inference_response: InferenceResponse) -> JobRecord:
    return JobRecord(
        request_id=guidance_request.request_id,
        status="completed",
        request=inference_response.model_copy(update={"question": guidance_request.question}).model_construct(),
    )


@pytest.fixture
def ingestion_response() -> IngestionResponse:
    return IngestionResponse(
        documents_bucket="guidance-documents",
        documents_prefix="guidelines",
        documents_found=1,
        chunks_created=3,
        vectors_upserted=3,
        collection="guidance_chunks",
        cleaning_strategy="deep",
        chunking_strategy="naive",
        cleaning_params={},
        chunking_params={"chunk_size": 300, "chunk_overlap": 100},
        embedding_model="nomic-embed-text",
    )


@pytest.fixture
def api_app():
    from api.main import create_app as create_api_app
    return create_api_app(bootstrap_minio_on_startup=False)


@pytest.fixture
def api_client(api_app):
    with TestClient(api_app) as client:
        yield client


@pytest.fixture
def stub_document_service() -> StubDocumentService:
    return StubDocumentService()


@pytest.fixture
def upload_file():
    return ("guideline.pdf", io.BytesIO(b"pdf-bytes"), "application/pdf")
