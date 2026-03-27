from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO, Protocol

from shared.contracts.documents import DocumentDeleteResponse, DocumentMetadata

if TYPE_CHECKING:
    from api.infrastructure.repositories.documents.models import DocumentBlob


class DocumentRepositoryError(RuntimeError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DocumentNotFoundError(DocumentRepositoryError):
    pass


class DocumentStorageUnavailableError(DocumentRepositoryError):
    pass


class InvalidDocumentError(DocumentRepositoryError):
    pass


class DocumentAlreadyExistsError(DocumentRepositoryError):
    pass


class DocumentRepositoryPort(Protocol):
    def list_documents(self, *, offset: int = 0, limit: int = 100) -> tuple[list[DocumentMetadata], int]: ...

    def get_document(self, object_name: str) -> DocumentBlob: ...

    def upload_document(
        self,
        *,
        filename: str,
        content_stream: BinaryIO,
        size_bytes: int,
        content_type: str | None = None,
        object_name: str | None = None,
        overwrite: bool = True,
    ) -> DocumentMetadata: ...

    def delete_document(self, object_name: str) -> DocumentDeleteResponse: ...


__all__ = [
    "DocumentAlreadyExistsError",
    "DocumentNotFoundError",
    "DocumentRepositoryError",
    "DocumentRepositoryPort",
    "DocumentStorageUnavailableError",
    "InvalidDocumentError",
]
