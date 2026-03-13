from __future__ import annotations

import os
from email.utils import format_datetime
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response

from api.application.services.document_service import DocumentService
from api.dependencies import get_document_service
from api.repositories.document_repository import DocumentNotFoundError, DocumentRepositoryError
from shared.contracts.documents import (
    DocumentDeleteResponse,
    DocumentMetadataListResponse,
    DocumentUploadResponse,
)

router = APIRouter(tags=["documents"])


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


@router.post("/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    size_bytes = _get_upload_size(file)

    try:
        return service.upload_document(
            filename=file.filename,
            content_stream=file.file,
            size_bytes=size_bytes,
            content_type=file.content_type,
            object_name=file.filename,
        )
    except DocumentRepositoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()


@router.delete("/documents/{object_name:path}", response_model=DocumentDeleteResponse)
async def delete_document(
    object_name: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentDeleteResponse:
    try:
        return service.delete_document(object_name)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentRepositoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _get_upload_size(file: UploadFile) -> int:
    stream = file.file
    current_position = stream.tell()
    stream.seek(0, os.SEEK_END)
    size_bytes = stream.tell()
    stream.seek(current_position)
    return size_bytes
