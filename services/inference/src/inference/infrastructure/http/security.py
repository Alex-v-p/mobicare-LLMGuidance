from __future__ import annotations

from fastapi import Header

from inference.infrastructure.http.exceptions import InferenceHttpError
from shared.config import get_inference_settings
from shared.contracts.error_codes import ErrorCode
from shared.security import INTERNAL_SERVICE_TOKEN_HEADER


def require_internal_service_request(
    internal_service_token: str | None = Header(default=None, alias=INTERNAL_SERVICE_TOKEN_HEADER),
) -> None:
    settings = get_inference_settings()
    if not settings.enable_internal_service_auth:
        return
    if internal_service_token != settings.internal_service_token:
        raise InferenceHttpError(
            code=ErrorCode.AUTH_TOKEN_INVALID,
            message="A valid internal service token is required.",
            status_code=401,
        )
