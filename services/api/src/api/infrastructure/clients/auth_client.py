from __future__ import annotations

from typing import Any

import httpx

from shared.clients.http import create_async_client
from shared.config import Settings, get_settings
from shared.contracts.error_codes import ErrorCode


class AuthClientError(RuntimeError):
    def __init__(self, *, status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def _classify_transport_error(exc: httpx.HTTPError) -> tuple[str, str]:
    if isinstance(exc, httpx.TimeoutException):
        return ErrorCode.AUTH_VALIDATION_UNAVAILABLE, "The authentication provider timed out."
    if isinstance(exc, httpx.ConnectError):
        return ErrorCode.AUTH_VALIDATION_UNAVAILABLE, "The authentication provider could not be reached."
    if isinstance(exc, httpx.NetworkError):
        return ErrorCode.AUTH_VALIDATION_UNAVAILABLE, "The authentication provider is unavailable."
    return ErrorCode.AUTH_VALIDATION_UNAVAILABLE, "The authentication provider request failed."


def _extract_boolean(payload: Any) -> bool | None:
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, dict):
        for key in ("valid", "authenticated", "is_valid", "success"):
            value = payload.get(key)
            if isinstance(value, bool):
                return value
    if isinstance(payload, str):
        normalized = payload.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


class AuthClient:
    def __init__(self, validation_url: str | None = None, timeout_s: float | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._validation_url = validation_url or self._settings.auth_validation_url
        self._timeout_s = timeout_s if timeout_s is not None else self._settings.auth_validation_timeout_s

    async def validate_credentials(self, *, email: str, password: str) -> bool:
        async with create_async_client(timeout_s=self._timeout_s) as client:
            try:
                response = await client.post(self._validation_url, json={"email": email, "password": password})
            except httpx.HTTPError as exc:
                code, message = _classify_transport_error(exc)
                raise AuthClientError(status_code=503, code=code, message=message, details={"reason": type(exc).__name__}) from exc

        if response.status_code in {401, 403}:
            return False
        if response.is_error:
            raise AuthClientError(
                status_code=503,
                code=ErrorCode.AUTH_VALIDATION_UNAVAILABLE,
                message="The authentication provider returned an error.",
                details={"status_code": response.status_code},
            )

        try:
            payload = response.json()
        except ValueError:
            payload = response.text

        is_valid = _extract_boolean(payload)
        if is_valid is None:
            raise AuthClientError(
                status_code=503,
                code=ErrorCode.AUTH_BAD_RESPONSE,
                message="The authentication provider returned an unexpected response.",
                details={"status_code": response.status_code},
            )
        return is_valid
