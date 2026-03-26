from __future__ import annotations

from api.clients.inference_client import InferenceClientError
from api.errors import AppError, ServiceUnavailableError


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


__all__ = ["map_inference_client_error"]
