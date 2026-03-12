from __future__ import annotations

from collections.abc import Callable

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from minio.error import S3Error

from inference.retrieval.dense import RetrievalCollectionNotReadyError as DenseRetrievalCollectionNotReadyError
from inference.retrieval.hybrid import RetrievalCollectionNotReadyError as HybridRetrievalCollectionNotReadyError
from inference.storage.qdrant_store import MissingCollectionError


class InferenceHttpError(Exception):
    status_code = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class BadRequestError(InferenceHttpError):
    status_code = 400


class NotFoundError(InferenceHttpError):
    status_code = 404


class ConflictError(InferenceHttpError):
    status_code = 409


class DependencyUnavailableError(InferenceHttpError):
    status_code = 503


KnownErrorMapper = Callable[[Exception], InferenceHttpError | None]


def map_known_exception(exc: Exception) -> InferenceHttpError | None:
    if isinstance(exc, InferenceHttpError):
        return exc
    if isinstance(exc, FileExistsError):
        return ConflictError(str(exc))
    if isinstance(
        exc,
        (
            DenseRetrievalCollectionNotReadyError,
            HybridRetrievalCollectionNotReadyError,
            MissingCollectionError,
        ),
    ):
        return ConflictError(str(exc))
    if isinstance(exc, ValueError):
        return BadRequestError(str(exc))
    if isinstance(exc, (httpx.HTTPError, redis.RedisError, S3Error)):
        return DependencyUnavailableError(str(exc))
    return None


def register_exception_handlers(app: FastAPI, mapper: KnownErrorMapper = map_known_exception) -> None:
    @app.exception_handler(InferenceHttpError)
    async def handle_inference_http_error(
        _request: Request,
        exc: InferenceHttpError,
    ) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        mapped = mapper(exc)
        if mapped is not None:
            return await handle_inference_http_error(request, mapped)
        raise exc
