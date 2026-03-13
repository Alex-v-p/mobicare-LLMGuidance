from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from minio.error import S3Error

from inference.retrieval.common import RetrievalCollectionNotReadyError
from shared.contracts.error_codes import ErrorCode
from shared.contracts.errors import ErrorBody, ErrorResponse
from shared.observability import REQUEST_ID_HEADER, get_logger, get_metrics_registry, get_or_create_request_id

logger = get_logger(__name__, service="inference")
metrics = get_metrics_registry()


class InferenceHttpError(Exception):
    def __init__(self, *, code: str, message: str, status_code: int, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class BadRequestError(InferenceHttpError):
    def __init__(self, message: str, *, code: str = ErrorCode.BAD_REQUEST, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=400, details=details)


class NotFoundError(InferenceHttpError):
    def __init__(self, message: str, *, code: str = ErrorCode.NOT_FOUND, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=404, details=details)


class ConflictError(InferenceHttpError):
    def __init__(self, message: str, *, code: str = ErrorCode.CONFLICT, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=409, details=details)


class DependencyUnavailableError(InferenceHttpError):
    def __init__(self, message: str, *, code: str = ErrorCode.DEPENDENCY_UNAVAILABLE, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=503, details=details)


KnownErrorMapper = Callable[[Exception], InferenceHttpError | None]


def build_error_response(request: Request, *, status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> JSONResponse:
    request_id = get_or_create_request_id(request)
    metrics.inc("app_exceptions_total", labels={"service": "inference", "code": str(code), "status_code": str(status_code)})
    payload = ErrorResponse(error=ErrorBody(code=str(code), message=message, request_id=request_id, details=details or {}))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"), headers={REQUEST_ID_HEADER: request_id})


def map_known_exception(exc: Exception) -> InferenceHttpError | None:
    if isinstance(exc, InferenceHttpError):
        return exc
    if isinstance(exc, FileExistsError):
        return ConflictError(str(exc), code=ErrorCode.JOB_ALREADY_EXISTS)
    if isinstance(exc, FileNotFoundError):
        return NotFoundError(str(exc), code=ErrorCode.JOB_NOT_FOUND)
    if isinstance(exc, RetrievalCollectionNotReadyError):
        return ConflictError(str(exc), code=ErrorCode.RETRIEVAL_COLLECTION_NOT_READY)
    if isinstance(exc, ValueError):
        return BadRequestError(str(exc), code=ErrorCode.BAD_REQUEST)
    if isinstance(exc, httpx.TimeoutException):
        return DependencyUnavailableError("A required dependency timed out.", code=ErrorCode.DEPENDENCY_TIMEOUT, details={"reason": type(exc).__name__})
    if isinstance(exc, redis.AuthenticationError):
        return DependencyUnavailableError("A required dependency rejected authentication.", code=ErrorCode.DEPENDENCY_AUTH_FAILED, details={"reason": type(exc).__name__})
    if isinstance(exc, httpx.ConnectError):
        return DependencyUnavailableError("A required dependency could not be reached.", code=ErrorCode.DEPENDENCY_UNAVAILABLE, details={"reason": type(exc).__name__})
    if isinstance(exc, (httpx.HTTPError, redis.RedisError, S3Error)):
        return DependencyUnavailableError("A required dependency is unavailable.", code=ErrorCode.DEPENDENCY_UNAVAILABLE, details={"reason": type(exc).__name__})
    return None


def register_exception_handlers(app: FastAPI, mapper: KnownErrorMapper = map_known_exception) -> None:
    @app.exception_handler(InferenceHttpError)
    async def handle_inference_http_error(request: Request, exc: InferenceHttpError) -> JSONResponse:
        logger.info("inference_error", extra={"event": "inference_error", "request_id": get_or_create_request_id(request), "method": request.method, "path": request.url.path, "status_code": exc.status_code, "error_code": exc.code})
        return build_error_response(request, status_code=exc.status_code, code=exc.code, message=exc.message, details=exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [{"field": ".".join(str(part) for part in err.get("loc", []) if part != "body"), "message": err.get("msg", "Invalid value."), "type": err.get("type", "validation_error")} for err in exc.errors()]
        return build_error_response(request, status_code=422, code=ErrorCode.VALIDATION_ERROR, message="Request validation failed.", details={"fields": fields})

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        mapped = mapper(exc)
        if mapped is not None:
            return await handle_inference_http_error(request, mapped)
        request_id = get_or_create_request_id(request)
        logger.exception("unhandled_inference_exception", extra={"event": "unhandled_exception", "request_id": request_id, "method": request.method, "path": request.url.path, "status_code": 500, "error_code": ErrorCode.INTERNAL_SERVER_ERROR}, exc_info=exc)
        return build_error_response(request, status_code=500, code=ErrorCode.INTERNAL_SERVER_ERROR, message="An unexpected error occurred.")
