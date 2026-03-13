from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.errors import AppError
from shared.contracts.error_codes import ErrorCode
from shared.contracts.errors import ErrorBody, ErrorResponse
from shared.observability import REQUEST_ID_HEADER, get_logger, get_metrics_registry, get_or_create_request_id

logger = get_logger(__name__, service="api")
metrics = get_metrics_registry()


def _normalize_validation_loc(loc: Sequence[Any]) -> str:
    return ".".join(str(part) for part in loc if part != "body")


def build_error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> JSONResponse:
    request_id = get_or_create_request_id(request)
    metrics.inc(
        "app_exceptions_total",
        labels={"service": "api", "code": str(code), "status_code": str(status_code)},
    )
    payload = ErrorResponse(
        error=ErrorBody(
            code=str(code),
            message=message,
            request_id=request_id,
            details=dict(details or {}),
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: request_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.info(
            "application_error",
            extra={
                "event": "application_error",
                "request_id": get_or_create_request_id(request),
                "method": request.method,
                "path": request.url.path,
                "status_code": exc.status_code,
                "error_code": exc.code,
            },
        )
        return build_error_response(request, status_code=exc.status_code, code=exc.code, message=exc.message, details=exc.details)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        extra_details = {} if isinstance(exc.detail, str) else {"detail": exc.detail}
        return build_error_response(request, status_code=exc.status_code, code=ErrorCode.HTTP_ERROR, message=detail, details=extra_details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {
                "field": _normalize_validation_loc(err.get("loc", [])),
                "message": err.get("msg", "Invalid value."),
                "type": err.get("type", "validation_error"),
            }
            for err in exc.errors()
        ]
        return build_error_response(request, status_code=422, code=ErrorCode.VALIDATION_ERROR, message="Request validation failed.", details={"fields": fields})

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        request_id = get_or_create_request_id(request)
        logger.exception(
            "unhandled_exception",
            extra={
                "event": "unhandled_exception",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "error_code": ErrorCode.INTERNAL_SERVER_ERROR,
            },
            exc_info=exc,
        )
        return build_error_response(request, status_code=500, code=ErrorCode.INTERNAL_SERVER_ERROR, message="An unexpected error occurred.")
