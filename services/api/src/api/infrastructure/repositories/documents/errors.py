from __future__ import annotations

from minio.error import S3Error
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, NewConnectionError

from api.application.ports.document_repository import (
    DocumentAlreadyExistsError as DocumentAlreadyExistsPortError,
    DocumentNotFoundError as DocumentNotFoundPortError,
    DocumentRepositoryError as DocumentRepositoryPortError,
    DocumentStorageUnavailableError as DocumentStorageUnavailablePortError,
    InvalidDocumentError as InvalidDocumentPortError,
)
from shared.contracts.error_codes import ErrorCode


class DocumentRepositoryError(DocumentRepositoryPortError):
    def __init__(self, message: str, *, code: str = ErrorCode.DOCUMENT_REPOSITORY_ERROR) -> None:
        super().__init__(message, code=code)


class DocumentNotFoundError(DocumentNotFoundPortError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.DOCUMENT_NOT_FOUND)


class DocumentStorageUnavailableError(DocumentStorageUnavailablePortError):
    def __init__(self, message: str, *, code: str = ErrorCode.DOCUMENT_STORAGE_UNAVAILABLE) -> None:
        super().__init__(message, code=code)


class InvalidDocumentError(InvalidDocumentPortError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.DOCUMENT_UPLOAD_INVALID)


class DocumentAlreadyExistsError(DocumentAlreadyExistsPortError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CONFLICT)


def map_storage_error(exc: Exception, object_name: str) -> DocumentRepositoryError:
    if isinstance(exc, S3Error):
        code = getattr(exc, "code", "")
        if code in {"NoSuchKey", "NoSuchObject", "NoSuchVersion", "ResourceNotFound"}:
            return DocumentNotFoundError(f"Document '{object_name}' was not found")
        if code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
            return DocumentStorageUnavailableError("Document storage authentication failed", code=ErrorCode.DOCUMENT_STORAGE_AUTH_FAILED)
        if code in {"NoSuchBucket"}:
            return DocumentStorageUnavailableError("Documents bucket is missing", code=ErrorCode.DOCUMENT_BUCKET_MISSING)
        return DocumentStorageUnavailableError("Document storage request failed")
    if isinstance(exc, (ConnectTimeoutError, TimeoutError)):
        return DocumentStorageUnavailableError("Document storage timed out", code=ErrorCode.DOCUMENT_STORAGE_TIMEOUT)
    if isinstance(exc, (MaxRetryError, NewConnectionError, ConnectionError)):
        return DocumentStorageUnavailableError("Could not reach document storage")
    return DocumentStorageUnavailableError("Document storage request failed")
