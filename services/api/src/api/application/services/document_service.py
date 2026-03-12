from __future__ import annotations

from api.repositories.document_repository import DocumentBlob, DocumentRepository
from shared.contracts.documents import DocumentMetadataListResponse


class DocumentService:
    def __init__(self, repository: DocumentRepository | None = None) -> None:
        self._repository = repository or DocumentRepository()

    def list_metadata(self) -> DocumentMetadataListResponse:
        documents = self._repository.list_documents()
        return DocumentMetadataListResponse(documents=documents, count=len(documents))

    def get_document(self, object_name: str) -> DocumentBlob:
        return self._repository.get_document(object_name)
