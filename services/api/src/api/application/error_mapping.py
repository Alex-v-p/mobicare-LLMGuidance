from __future__ import annotations

from api.errors import AppError, BadRequestError, ConflictError, NotFoundError, ServiceUnavailableError
from api.infrastructure.clients.inference_client import InferenceClientError
from api.infrastructure.repositories.clinical_config import (
    ClinicalConfigAlreadyExistsError,
    ClinicalConfigNotFoundError,
    ClinicalConfigOptimisticLockError,
    ClinicalConfigRepositoryError,
    ClinicalConfigVersionNotFoundError,
    InvalidClinicalConfigError,
    UnknownClinicalConfigError,
)
from api.infrastructure.repositories.document_repository import DocumentNotFoundError, DocumentRepositoryError
from api.infrastructure.repositories.documents import (
    DocumentAlreadyExistsError,
    DocumentStorageUnavailableError,
    InvalidDocumentError,
)

_DOCUMENT_NOT_FOUND_MESSAGE = "The requested document was not found."
_DOCUMENT_STORAGE_UNAVAILABLE_MESSAGE = "Document storage is currently unavailable."


def _context_details(**kwargs: str | None) -> dict[str, str]:
    return {key: value for key, value in kwargs.items() if value is not None}


def map_inference_client_error(exc: InferenceClientError) -> AppError:
    if exc.status_code >= 500:
        return ServiceUnavailableError(
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )
    return AppError(
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
    )


def map_document_error(
    exc: Exception,
    *,
    object_name: str | None = None,
    filename: str | None = None,
) -> AppError:
    details = _context_details(object_name=object_name, filename=filename)
    if isinstance(exc, InvalidDocumentError):
        return BadRequestError(code=exc.code, message=exc.message, details=details)
    if isinstance(exc, DocumentAlreadyExistsError):
        return ConflictError(code=exc.code, message=exc.message, details=details)
    if isinstance(exc, DocumentNotFoundError):
        return NotFoundError(code=exc.code, message=_DOCUMENT_NOT_FOUND_MESSAGE, details=details)
    if isinstance(exc, DocumentStorageUnavailableError):
        return ServiceUnavailableError(
            code=exc.code,
            message=_DOCUMENT_STORAGE_UNAVAILABLE_MESSAGE,
            details=details,
        )
    if isinstance(exc, DocumentRepositoryError):
        return ServiceUnavailableError(code=exc.code, message=exc.message, details=details)
    raise TypeError(f"Unsupported document error type: {type(exc).__name__}")



def map_clinical_config_error(exc: Exception, *, config_name: str | None = None) -> AppError:
    details = _context_details(config_name=config_name)
    if isinstance(exc, InvalidClinicalConfigError):
        return BadRequestError(code=exc.code, message=exc.message, details=details)
    if isinstance(exc, (ClinicalConfigAlreadyExistsError, ClinicalConfigOptimisticLockError)):
        return ConflictError(code=exc.code, message=exc.message, details=details)
    if isinstance(exc, (ClinicalConfigNotFoundError, ClinicalConfigVersionNotFoundError, UnknownClinicalConfigError)):
        return NotFoundError(code=exc.code, message=exc.message, details=details)
    if isinstance(exc, ClinicalConfigRepositoryError):
        return ServiceUnavailableError(code=exc.code, message=exc.message, details=details)
    raise TypeError(f"Unsupported clinical config error type: {type(exc).__name__}")


__all__ = [
    "map_clinical_config_error",
    "map_document_error",
    "map_inference_client_error",
]
