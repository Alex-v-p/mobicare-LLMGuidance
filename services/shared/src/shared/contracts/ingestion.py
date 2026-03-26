from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from shared.ids import new_ingestion_job_id


IngestionJobState = Literal["queued", "running", "completed", "failed", "not_found"]
CleaningStrategy = Literal["none", "basic", "deep", "medical_guideline_deep"]
ChunkingStrategy = Literal["naive", "page_indexed", "late"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionOptions(BaseModel):
    cleaning_strategy: CleaningStrategy = "deep"
    cleaning_params: dict[str, Any] = Field(default_factory=dict)
    chunking_strategy: ChunkingStrategy = "naive"
    chunking_params: dict[str, Any] = Field(
        default_factory=lambda: {"chunk_size": 300, "chunk_overlap": 100}
    )
    embedding_model: Optional[str] = None


class IngestDocumentsRequest(BaseModel):
    """Request to ingest configured guidance documents.

    Infra details such as bucket and collection names are configured internally.
    """

    options: IngestionOptions = Field(default_factory=IngestionOptions)


class IngestionResponse(BaseModel):
    status: str = "ok"
    documents_bucket: str
    documents_prefix: str = ""
    documents_found: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0
    collection: str
    cleaning_strategy: CleaningStrategy = "deep"
    chunking_strategy: ChunkingStrategy = "naive"
    cleaning_params: dict[str, Any] = Field(default_factory=dict)
    chunking_params: dict[str, Any] = Field(default_factory=dict)
    embedding_model: Optional[str] = None


class IngestionJobAcceptedResponse(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
    status_url: str


class IngestionJobRecord(BaseModel):
    job_id: str = Field(default_factory=new_ingestion_job_id)
    status: IngestionJobState
    request: IngestDocumentsRequest = Field(default_factory=IngestDocumentsRequest)
    result: Optional[IngestionResponse] = None
    error: Optional[str] = None
    result_object_key: Optional[str] = None
    worker_id: Optional[str] = None
    lease_expires_at: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str = Field(default_factory=utc_now_iso)


class ApiIngestionJobStatus(BaseModel):
    job_id: str
    status: IngestionJobState
    documents_found: int | None = None
    chunks_created: int | None = None
    vectors_upserted: int | None = None
    collection: str | None = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None


class IngestionCollectionDeleteResponse(BaseModel):
    status: str = "deleted"
    collection: str
    existed: bool
