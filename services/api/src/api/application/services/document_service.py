from __future__ import annotations

from typing import BinaryIO

from api.repositories.document_repository import DocumentBlob, DocumentRepository
from shared.contracts.documents import DocumentDeleteResponse, DocumentMetadataListResponse, DocumentUploadResponse


class DocumentService:
    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    def list_metadata(self) -> DocumentMetadataListResponse:
        documents = self._repository.list_documents()
        return DocumentMetadataListResponse(documents=documents, count=len(documents))

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
    ) -> DocumentUploadResponse:
        document = self._repository.upload_document(
            filename=filename,
            content_stream=content_stream,
            size_bytes=size_bytes,
            content_type=content_type,
            object_name=object_name,
        )
        return DocumentUploadResponse(document=document)

    def delete_document(self, object_name: str) -> DocumentDeleteResponse:
        return self._repository.delete_document(object_name)
