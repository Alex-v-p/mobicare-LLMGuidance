from __future__ import annotations

from typing import Any

from shared.contracts.error_codes import ErrorCode


class AppError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class BadRequestError(AppError):
    def __init__(self, *, code: str = ErrorCode.BAD_REQUEST, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=400, details=details)


class NotFoundError(AppError):
    def __init__(self, *, code: str = ErrorCode.NOT_FOUND, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=404, details=details)


class ConflictError(AppError):
    def __init__(self, *, code: str = ErrorCode.CONFLICT, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=409, details=details)


class UnauthorizedError(AppError):
    def __init__(self, *, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=401, details=details)


class ServiceUnavailableError(AppError):
    def __init__(self, *, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=503, details=details)
