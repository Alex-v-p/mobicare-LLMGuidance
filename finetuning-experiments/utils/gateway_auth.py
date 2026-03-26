from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class GatewayAuthContext:
    base_url: str
    auth_mode: str = "none"
    auth_token: str | None = None
    auth_email: str | None = None
    auth_password: str | None = None
    jwt_secret: str | None = None
    jwt_issuer: str = "mobicare-llm-api"
    jwt_audience: str = "mobicare-gateway"
    jwt_exp_minutes: int = 60
    verify_ssl: bool = True
    ca_bundle_path: str | None = None
    timeout_seconds: int = 30


def _urlsafe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _json_dumps(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def create_local_access_token(*, email: str, secret_key: str, issuer: str, audience: str, exp_minutes: int = 60) -> tuple[str, int]:
    now = int(time.time())
    expires_in = int(exp_minutes) * 60
    payload = {
        "sub": email,
        "email": email,
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + expires_in,
    }
    header = {"alg": "HS256", "typ": "JWT"}
    encoded_header = _urlsafe_b64encode(_json_dumps(header))
    encoded_payload = _urlsafe_b64encode(_json_dumps(payload))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    signature = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    token = f"{encoded_header}.{encoded_payload}.{_urlsafe_b64encode(signature)}"
    return token, expires_in


def resolve_ssl_verify(verify_ssl: bool = True, ca_bundle_path: str | None = None) -> bool | str:
    bundle = (ca_bundle_path or "").strip()
    if bundle:
        return bundle
    return bool(verify_ssl)


def login_for_access_token(*, base_url: str, email: str, password: str, timeout_seconds: int = 30, verify_ssl: bool | str = True) -> tuple[str, int | None]:
    response = requests.post(
        f"{base_url.rstrip('/')}/auth/token",
        json={"email": email, "password": password},
        timeout=timeout_seconds,
        verify=verify_ssl,
    )
    response.raise_for_status()
    payload = response.json()
    token = str(payload.get("access_token") or "").strip()
    if not token:
        raise ValueError("Gateway auth response did not include access_token.")
    expires_in = payload.get("expires_in")
    return token, int(expires_in) if isinstance(expires_in, int) else None


def resolve_gateway_auth_token(context: GatewayAuthContext) -> str | None:
    explicit_token = (context.auth_token or "").strip()
    if explicit_token:
        return explicit_token

    auth_mode = (context.auth_mode or "none").strip().lower()
    verify_ssl = resolve_ssl_verify(context.verify_ssl, context.ca_bundle_path)

    if auth_mode in {"", "none"}:
        return None

    if auth_mode in {"bearer", "bearer_token", "token"}:
        return explicit_token or None

    if auth_mode in {"local_jwt", "generate_local_jwt", "jwt"}:
        email = (context.auth_email or "").strip()
        secret = (context.jwt_secret or "").strip()
        if not email:
            raise ValueError("execution.gateway_auth_email is required when gateway_auth_mode=local_jwt.")
        if not secret:
            raise ValueError("execution.gateway_jwt_secret is required when gateway_auth_mode=local_jwt.")
        token, _ = create_local_access_token(
            email=email,
            secret_key=secret,
            issuer=context.jwt_issuer,
            audience=context.jwt_audience,
            exp_minutes=context.jwt_exp_minutes,
        )
        return token

    if auth_mode in {"gateway_login", "login"}:
        email = (context.auth_email or "").strip()
        password = context.auth_password or ""
        if not email:
            raise ValueError("execution.gateway_auth_email is required when gateway_auth_mode=gateway_login.")
        if not password:
            raise ValueError("execution.gateway_auth_password is required when gateway_auth_mode=gateway_login.")
        token, _ = login_for_access_token(
            base_url=context.base_url,
            email=email,
            password=password,
            timeout_seconds=context.timeout_seconds,
            verify_ssl=verify_ssl,
        )
        return token

    raise ValueError(f"Unsupported gateway_auth_mode: {context.auth_mode}")
