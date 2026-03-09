from __future__ import annotations

from fastapi import APIRouter

from inference.indexing.ingestion_service import IngestionService
from shared.contracts.ingestion import IngestionRequest, IngestionResponse

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(request: IngestionRequest) -> IngestionResponse:
    service = IngestionService()
    return await service.ingest(
        recreate_collection=request.recreate_collection,
        bucket=request.bucket,
        prefix=request.prefix,
    )
