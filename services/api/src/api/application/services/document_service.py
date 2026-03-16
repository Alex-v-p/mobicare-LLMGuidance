from __future__ import annotations

from typing import BinaryIO

from api.repositories.document_repository import DocumentBlob, DocumentRepository
from shared.contracts.documents import DocumentDeleteResponse, DocumentMetadataListResponse, DocumentUploadResponse


class DocumentService:
    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    def list_metadata(self, *, offset: int = 0, limit: int = 100) -> DocumentMetadataListResponse:
        documents, total_count = self._repository.list_documents(offset=offset, limit=limit)
        return DocumentMetadataListResponse(
            documents=documents,
            count=len(documents),
            total_count=total_count,
            offset=offset,
            limit=limit,
            has_more=offset + len(documents) < total_count,
        )

    def get_document(self, object_name: str) -> DocumentBlob:
        return self._repository.get_document(object_name)

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
        document = self._repository.upload_document(
            filename=filename,
            content_stream=content_stream,
            size_bytes=size_bytes,
            content_type=content_type,
            object_name=object_name,
            overwrite=overwrite,
        )
        return DocumentUploadResponse(document=document)

    def delete_document(self, object_name: str) -> DocumentDeleteResponse:
        return self._repository.delete_document(object_name)
