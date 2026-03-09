from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


IngestionJobState = Literal["queued", "running", "completed", "failed", "not_found"]


def new_ingestion_job_id() -> str:
    return f"ingest_job_{uuid4()}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestDocumentsRequest(BaseModel):
    """Request to ingest configured guidance documents.

    Infra details such as bucket and collection names are configured internally.
    """

    pass


class IngestionResponse(BaseModel):
    status: str = "ok"
    documents_bucket: str
    documents_prefix: str = ""
    documents_found: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0
    collection: str


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
