from __future__ import annotations

from fastapi import APIRouter

from inference.indexing.ingestion_service import IngestionService
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionResponse

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(_: IngestDocumentsRequest | None = None) -> IngestionResponse:
    service = IngestionService()
    return await service.ingest()
