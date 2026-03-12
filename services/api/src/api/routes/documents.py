from __future__ import annotations

from email.utils import format_datetime
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api.application.services.document_service import DocumentService
from api.repositories.document_repository import DocumentNotFoundError, DocumentRepositoryError
from shared.contracts.documents import DocumentMetadataListResponse

router = APIRouter(tags=["documents"])


def get_document_service() -> DocumentService:
    return DocumentService()


@router.get("/documents", response_model=DocumentMetadataListResponse)
async def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> DocumentMetadataListResponse:
    try:
        return service.list_metadata()
    except DocumentRepositoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/documents/{object_name:path}")
async def get_document(
    object_name: str,
    service: DocumentService = Depends(get_document_service),
) -> Response:
    try:
        document = service.get_document(object_name)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentRepositoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    headers = {
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(document.object_name)}",
        "Content-Length": str(document.size_bytes),
        "X-Document-Bucket": document.bucket,
        "X-Document-Object-Name": document.object_name,
    }
    if document.etag:
        headers["ETag"] = document.etag
    if document.last_modified is not None:
        headers["Last-Modified"] = format_datetime(document.last_modified, usegmt=True)

    return Response(content=document.content, media_type=document.content_type, headers=headers)
