from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    object_name: str
    title: str
    bucket: str
    prefix: str = ""
    size_bytes: int
    extension: str | None = None
    content_type: str | None = None
    etag: str | None = None
    last_modified: datetime | None = None


class DocumentMetadataListResponse(BaseModel):
    documents: list[DocumentMetadata]
    count: int
    total_count: int
    offset: int = 0
    limit: int
    has_more: bool = False


class DocumentUploadResponse(BaseModel):
    document: DocumentMetadata
    status: str = "uploaded"


class DocumentDeleteResponse(BaseModel):
    object_name: str
    bucket: str
    status: str = "deleted"
