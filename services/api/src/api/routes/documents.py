from __future__ import annotations

import os
from email.utils import format_datetime
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import Response

from api.application.services.document_service import DocumentService
from api.dependencies import get_document_service
from api.errors import BadRequestError, ConflictError, NotFoundError, ServiceUnavailableError
from api.repositories.document_repository import DocumentNotFoundError, DocumentRepositoryError
from api.repositories.documents import DocumentAlreadyExistsError, DocumentStorageUnavailableError, InvalidDocumentError
from shared.config import get_settings
from shared.contracts.documents import (
    DocumentDeleteResponse,
    DocumentMetadataListResponse,
    DocumentUploadResponse,
)

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentMetadataListResponse)
async def list_documents(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    service: DocumentService = Depends(get_document_service),
) -> DocumentMetadataListResponse:
    try:
        return service.list_metadata(offset=offset, limit=limit)
    except DocumentStorageUnavailableError as exc:
        raise ServiceUnavailableError(
            code=exc.code,
            message="Document storage is currently unavailable.",
        ) from exc
    except DocumentRepositoryError as exc:
        raise ServiceUnavailableError(
            code=exc.code,
            message=exc.message,
        ) from exc


@router.get("/documents/{object_name:path}")
async def get_document(
    object_name: str,
    service: DocumentService = Depends(get_document_service),
) -> Response:
    try:
        document = service.get_document(object_name)
    except DocumentNotFoundError as exc:
        raise NotFoundError(
            code=exc.code,
            message="The requested document was not found.",
            details={"object_name": object_name},
        ) from exc
    except DocumentStorageUnavailableError as exc:
        raise ServiceUnavailableError(
            code=exc.code,
            message="Document storage is currently unavailable.",
            details={"object_name": object_name},
        ) from exc
    except DocumentRepositoryError as exc:
        raise ServiceUnavailableError(
            code=exc.code,
            message=exc.message,
            details={"object_name": object_name},
        ) from exc

    headers = {
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(document.object_name)}",
        "Content-Length": str(document.size_bytes),
    }
    if get_settings().expose_debug_metadata:
        headers["X-Document-Bucket"] = document.bucket
        headers["X-Document-Object-Name"] = document.object_name
    if document.etag:
        headers["ETag"] = document.etag
    if document.last_modified is not None:
        headers["Last-Modified"] = format_datetime(document.last_modified, usegmt=True)

    return Response(content=document.content, media_type=document.content_type, headers=headers)


@router.post("/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    overwrite: bool = Query(True),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    settings = get_settings()
    filename = (file.filename or "").strip()
    if not filename:
        raise BadRequestError(
            code="DOCUMENT_UPLOAD_INVALID",
            message="Uploaded file must include a filename.",
        )

    size_bytes = _get_upload_size(file)
    _validate_upload(filename=filename, size_bytes=size_bytes, content_type=file.content_type, settings=settings)

    try:
        return service.upload_document(
            filename=filename,
            content_stream=file.file,
            size_bytes=size_bytes,
            content_type=file.content_type,
            object_name=filename,
            overwrite=overwrite,
        )
    except InvalidDocumentError as exc:
        raise BadRequestError(
            code=exc.code,
            message=exc.message,
            details={"filename": filename},
        ) from exc
    except DocumentAlreadyExistsError as exc:
        raise ConflictError(
            code=exc.code,
            message=exc.message,
            details={"filename": filename},
        ) from exc
    except DocumentStorageUnavailableError as exc:
        raise ServiceUnavailableError(
            code=exc.code,
            message="Document storage is currently unavailable.",
            details={"filename": filename},
        ) from exc
    except DocumentRepositoryError as exc:
        raise ServiceUnavailableError(
            code=exc.code,
            message=exc.message,
            details={"filename": filename},
        ) from exc
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
        raise NotFoundError(
            code=exc.code,
            message="The requested document was not found.",
            details={"object_name": object_name},
        ) from exc
    except DocumentStorageUnavailableError as exc:
        raise ServiceUnavailableError(
            code=exc.code,
            message="Document storage is currently unavailable.",
            details={"object_name": object_name},
        ) from exc
    except DocumentRepositoryError as exc:
        raise ServiceUnavailableError(
            code=exc.code,
            message=exc.message,
            details={"object_name": object_name},
        ) from exc


def _get_upload_size(file: UploadFile) -> int:
    stream = file.file
    current_position = stream.tell()
    stream.seek(0, os.SEEK_END)
    size_bytes = stream.tell()
    stream.seek(current_position)
    return size_bytes


def _validate_upload(*, filename: str, size_bytes: int, content_type: str | None, settings) -> None:
    if size_bytes > settings.document_upload_max_bytes:
        raise BadRequestError(
            code="DOCUMENT_UPLOAD_INVALID",
            message=f"Uploaded file exceeds the maximum allowed size of {settings.document_upload_max_bytes} bytes.",
            details={"filename": filename, "size_bytes": size_bytes},
        )

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if settings.document_allowed_extensions and extension not in settings.document_allowed_extensions:
        raise BadRequestError(
            code="DOCUMENT_UPLOAD_INVALID",
            message="Uploaded file type is not allowed.",
            details={"filename": filename, "extension": extension},
        )

    normalized_content_type = (content_type or "").lower()
    if settings.document_allowed_content_types and normalized_content_type not in settings.document_allowed_content_types:
        raise BadRequestError(
            code="DOCUMENT_UPLOAD_INVALID",
            message="Uploaded content type is not allowed.",
            details={"filename": filename, "content_type": content_type},
        )
