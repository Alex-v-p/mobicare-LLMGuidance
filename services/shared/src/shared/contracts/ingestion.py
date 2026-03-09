from __future__ import annotations

from pydantic import BaseModel


class IngestDocumentsRequest(BaseModel):
    """Public/internal request to ingest configured guidance documents.

    Infra details such as bucket name and collection name are configured via env,
    not supplied by clients.
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
