from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

from api.application.error_mapping import map_document_error
from api.application.ports import (
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
    DocumentRepositoryError,
    DocumentRepositoryPort,
    DocumentStorageUnavailableError,
    InvalidDocumentError,
)
from shared.contracts.documents import DocumentDeleteResponse, DocumentMetadataListResponse, DocumentUploadResponse

if TYPE_CHECKING:
    from api.infrastructure.repositories.documents.models import DocumentBlob


class DocumentService:
    def __init__(self, repository: DocumentRepositoryPort) -> None:
        self._repository = repository

    def list_metadata(self, *, offset: int = 0, limit: int = 100) -> DocumentMetadataListResponse:
        try:
            documents, total_count = self._repository.list_documents(offset=offset, limit=limit)
        except (DocumentStorageUnavailableError, DocumentRepositoryError) as exc:
            raise map_document_error(exc) from exc
        return DocumentMetadataListResponse(
            documents=documents,
            count=len(documents),
            total_count=total_count,
            offset=offset,
            limit=limit,
            has_more=offset + len(documents) < total_count,
        )

    def get_document(self, object_name: str) -> DocumentBlob:
        try:
            return self._repository.get_document(object_name)
        except (DocumentNotFoundError, DocumentStorageUnavailableError, DocumentRepositoryError) as exc:
            raise map_document_error(exc, object_name=object_name) from exc

    def upload_document(
        self,
        *,
        filename: str,
        content_stream: BinaryIO,
        size_bytes: int,
        content_type: str | None = None,
        object_name: str | None = None,
        overwrite: bool = True,
    ) -> DocumentUploadResponse:
        try:
            document = self._repository.upload_document(
                filename=filename,
                content_stream=content_stream,
                size_bytes=size_bytes,
                content_type=content_type,
                object_name=object_name,
                overwrite=overwrite,
            )
        except (InvalidDocumentError, DocumentAlreadyExistsError, DocumentStorageUnavailableError, DocumentRepositoryError) as exc:
            raise map_document_error(exc, filename=filename) from exc
        return DocumentUploadResponse(document=document)

    def delete_document(self, object_name: str) -> DocumentDeleteResponse:
        try:
            return self._repository.delete_document(object_name)
        except (DocumentNotFoundError, DocumentStorageUnavailableError, DocumentRepositoryError) as exc:
            raise map_document_error(exc, object_name=object_name) from exc
