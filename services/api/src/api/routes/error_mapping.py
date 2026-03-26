from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from api.errors import BadRequestError, ConflictError, NotFoundError, ServiceUnavailableError
from api.repositories.clinical_config_repository import (
    ClinicalConfigAlreadyExistsError,
    ClinicalConfigNotFoundError,
    ClinicalConfigOptimisticLockError,
    ClinicalConfigRepositoryError,
    ClinicalConfigVersionNotFoundError,
    InvalidClinicalConfigError,
    UnknownClinicalConfigError,
)
from api.repositories.document_repository import DocumentNotFoundError, DocumentRepositoryError
from api.repositories.documents import DocumentAlreadyExistsError, DocumentStorageUnavailableError, InvalidDocumentError

_DOCUMENT_NOT_FOUND_MESSAGE = "The requested document was not found."
_DOCUMENT_STORAGE_UNAVAILABLE_MESSAGE = "Document storage is currently unavailable."


@contextmanager
def clinical_config_list_errors() -> Iterator[None]:
    try:
        yield
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message) from exc


@contextmanager
def clinical_config_read_errors(*, config_name: str) -> Iterator[None]:
    try:
        yield
    except (ClinicalConfigNotFoundError, UnknownClinicalConfigError) as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@contextmanager
def clinical_config_version_errors(*, config_name: str) -> Iterator[None]:
    try:
        yield
    except (ClinicalConfigNotFoundError, ClinicalConfigVersionNotFoundError, UnknownClinicalConfigError) as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@contextmanager
def clinical_config_create_errors(*, config_name: str) -> Iterator[None]:
    try:
        yield
    except InvalidClinicalConfigError as exc:
        raise BadRequestError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except (ClinicalConfigAlreadyExistsError, ClinicalConfigOptimisticLockError) as exc:
        raise ConflictError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except UnknownClinicalConfigError as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@contextmanager
def clinical_config_update_errors(*, config_name: str) -> Iterator[None]:
    try:
        yield
    except InvalidClinicalConfigError as exc:
        raise BadRequestError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigOptimisticLockError as exc:
        raise ConflictError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except UnknownClinicalConfigError as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@contextmanager
def clinical_config_rollback_errors(*, config_name: str) -> Iterator[None]:
    try:
        yield
    except (ClinicalConfigNotFoundError, ClinicalConfigVersionNotFoundError, UnknownClinicalConfigError) as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigOptimisticLockError as exc:
        raise ConflictError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@contextmanager
def clinical_config_delete_errors(*, config_name: str) -> Iterator[None]:
    try:
        yield
    except (ClinicalConfigNotFoundError, UnknownClinicalConfigError) as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigOptimisticLockError as exc:
        raise ConflictError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@contextmanager
def document_list_errors() -> Iterator[None]:
    try:
        yield
    except DocumentStorageUnavailableError as exc:
        raise ServiceUnavailableError(code=exc.code, message=_DOCUMENT_STORAGE_UNAVAILABLE_MESSAGE) from exc
    except DocumentRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message) from exc


@contextmanager
def document_lookup_errors(*, object_name: str) -> Iterator[None]:
    try:
        yield
    except DocumentNotFoundError as exc:
        raise NotFoundError(code=exc.code, message=_DOCUMENT_NOT_FOUND_MESSAGE, details={"object_name": object_name}) from exc
    except DocumentStorageUnavailableError as exc:
        raise ServiceUnavailableError(
            code=exc.code,
            message=_DOCUMENT_STORAGE_UNAVAILABLE_MESSAGE,
            details={"object_name": object_name},
        ) from exc
    except DocumentRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"object_name": object_name}) from exc


@contextmanager
def document_upload_errors(*, filename: str) -> Iterator[None]:
    try:
        yield
    except InvalidDocumentError as exc:
        raise BadRequestError(code=exc.code, message=exc.message, details={"filename": filename}) from exc
    except DocumentAlreadyExistsError as exc:
        raise ConflictError(code=exc.code, message=exc.message, details={"filename": filename}) from exc
    except DocumentStorageUnavailableError as exc:
        raise ServiceUnavailableError(code=exc.code, message=_DOCUMENT_STORAGE_UNAVAILABLE_MESSAGE, details={"filename": filename}) from exc
    except DocumentRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"filename": filename}) from exc


@contextmanager
def document_delete_errors(*, object_name: str) -> Iterator[None]:
    with document_lookup_errors(object_name=object_name):
        yield


__all__ = [
    "clinical_config_create_errors",
    "clinical_config_delete_errors",
    "clinical_config_list_errors",
    "clinical_config_read_errors",
    "clinical_config_rollback_errors",
    "clinical_config_update_errors",
    "clinical_config_version_errors",
    "document_delete_errors",
    "document_list_errors",
    "document_lookup_errors",
    "document_upload_errors",
]
