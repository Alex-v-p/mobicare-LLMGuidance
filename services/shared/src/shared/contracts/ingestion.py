from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    recreate_collection: bool = False
    bucket: Optional[str] = None
    prefix: str = ""


class IngestionResponse(BaseModel):
    status: str = "ok"
    bucket: str
    prefix: str = ""
    documents_found: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0
    collection: str
