from __future__ import annotations

from minio.error import S3Error
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, NewConnectionError

from shared.contracts.error_codes import ErrorCode


class ClinicalConfigRepositoryError(RuntimeError):
    def __init__(self, message: str, *, code: str = ErrorCode.CLINICAL_CONFIG_STORAGE_UNAVAILABLE) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ClinicalConfigNotFoundError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_NOT_FOUND)


class ClinicalConfigVersionNotFoundError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_VERSION_NOT_FOUND)


class ClinicalConfigAlreadyExistsError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_CONFLICT)


class ClinicalConfigOptimisticLockError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_OPTIMISTIC_LOCK_FAILED)


class InvalidClinicalConfigError(ClinicalConfigRepositoryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CLINICAL_CONFIG_INVALID)


class UnknownClinicalConfigError(ClinicalConfigNotFoundError):
    pass


def map_storage_error(exc: Exception, object_name: str) -> ClinicalConfigRepositoryError:
    if isinstance(exc, S3Error):
        code = getattr(exc, "code", "")
        if code in {"NoSuchKey", "NoSuchObject", "NoSuchVersion", "ResourceNotFound", "NoSuchBucket"}:
            if "/_versions/" in object_name:
                return ClinicalConfigVersionNotFoundError(f"Clinical config version '{object_name}' was not found")
            return ClinicalConfigNotFoundError(f"Clinical config '{object_name}' was not found")
        if code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
            return ClinicalConfigRepositoryError(
                "Clinical config storage authentication failed",
                code=ErrorCode.DEPENDENCY_AUTH_FAILED,
            )
        return ClinicalConfigRepositoryError("Clinical config storage request failed")
    if isinstance(exc, (ConnectTimeoutError, TimeoutError)):
        return ClinicalConfigRepositoryError("Clinical config storage timed out", code=ErrorCode.DEPENDENCY_TIMEOUT)
    if isinstance(exc, (MaxRetryError, NewConnectionError, ConnectionError)):
        return ClinicalConfigRepositoryError(
            "Could not reach clinical config storage",
            code=ErrorCode.DEPENDENCY_UNAVAILABLE,
        )
    return ClinicalConfigRepositoryError("Clinical config storage request failed")


__all__ = [
    "ClinicalConfigAlreadyExistsError",
    "ClinicalConfigNotFoundError",
    "ClinicalConfigOptimisticLockError",
    "ClinicalConfigRepositoryError",
    "ClinicalConfigVersionNotFoundError",
    "InvalidClinicalConfigError",
    "UnknownClinicalConfigError",
    "map_storage_error",
]
