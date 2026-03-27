from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from shared.config import ApiSettings, get_api_settings
from shared.contracts.auth import AuthenticatedUser
from shared.contracts.error_codes import ErrorCode


class JwtValidationError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _urlsafe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _urlsafe_b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _json_dumps(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def create_access_token(*, email: str, settings: ApiSettings | None = None) -> tuple[str, int]:
    resolved = settings or get_api_settings()
    now = int(time.time())
    expires_in = resolved.jwt_access_token_exp_minutes * 60
    payload = {
        "sub": email,
        "email": email,
        "iss": resolved.jwt_issuer,
        "aud": resolved.jwt_audience,
        "iat": now,
        "exp": now + expires_in,
    }
    header = {"alg": "HS256", "typ": "JWT"}
    encoded_header = _urlsafe_b64encode(_json_dumps(header))
    encoded_payload = _urlsafe_b64encode(_json_dumps(payload))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    signature = hmac.new(resolved.jwt_secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    token = f"{encoded_header}.{encoded_payload}.{_urlsafe_b64encode(signature)}"
    return token, expires_in


def decode_access_token(token: str, settings: ApiSettings | None = None) -> AuthenticatedUser:
    resolved = settings or get_api_settings()
    parts = token.split(".")
    if len(parts) != 3:
        raise JwtValidationError(code=ErrorCode.AUTH_TOKEN_INVALID, message="Token format is invalid.")

    encoded_header, encoded_payload, encoded_signature = parts
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    expected_signature = hmac.new(resolved.jwt_secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()

    try:
        provided_signature = _urlsafe_b64decode(encoded_signature)
        payload = json.loads(_urlsafe_b64decode(encoded_payload))
    except (ValueError, json.JSONDecodeError) as exc:
        raise JwtValidationError(code=ErrorCode.AUTH_TOKEN_INVALID, message="Token payload is invalid.") from exc

    if not hmac.compare_digest(provided_signature, expected_signature):
        raise JwtValidationError(code=ErrorCode.AUTH_TOKEN_INVALID, message="Token signature is invalid.")

    if not isinstance(payload, dict):
        raise JwtValidationError(code=ErrorCode.AUTH_TOKEN_INVALID, message="Token payload is invalid.")

    if payload.get("iss") != resolved.jwt_issuer or payload.get("aud") != resolved.jwt_audience:
        raise JwtValidationError(code=ErrorCode.AUTH_TOKEN_INVALID, message="Token issuer or audience is invalid.")

    exp = payload.get("exp")
    if not isinstance(exp, int):
        raise JwtValidationError(code=ErrorCode.AUTH_TOKEN_INVALID, message="Token expiry is invalid.")
    if exp <= int(time.time()):
        raise JwtValidationError(code=ErrorCode.AUTH_TOKEN_EXPIRED, message="Token has expired.")

    try:
        return AuthenticatedUser.model_validate(payload)
    except Exception as exc:
        raise JwtValidationError(code=ErrorCode.AUTH_TOKEN_INVALID, message="Token claims are invalid.") from exc
