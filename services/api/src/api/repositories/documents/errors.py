from __future__ import annotations

from minio.error import S3Error


class DocumentRepositoryError(RuntimeError):
    def __init__(self, message: str, *, code: str = "DOCUMENT_REPOSITORY_ERROR") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DocumentNotFoundError(DocumentRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="DOCUMENT_NOT_FOUND")


class DocumentStorageUnavailableError(DocumentRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="DOCUMENT_STORAGE_UNAVAILABLE")


class InvalidDocumentError(DocumentRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="DOCUMENT_UPLOAD_INVALID")


def map_storage_error(exc: S3Error, object_name: str) -> DocumentRepositoryError:
    if getattr(exc, "code", "") in {"NoSuchKey", "NoSuchObject", "NoSuchVersion", "ResourceNotFound"}:
        return DocumentNotFoundError(f"Document '{object_name}' was not found")
    return DocumentStorageUnavailableError("Document storage request failed")
