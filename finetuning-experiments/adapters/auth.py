from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

import requests


def _urlsafe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _json_dumps(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def build_test_access_token(
    *,
    email: str,
    secret_key: str,
    issuer: str,
    audience: str,
    exp_minutes: int = 60,
) -> str:
    now = int(time.time())
    expires_in = max(exp_minutes, 1) * 60
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
    return f"{encoded_header}.{encoded_payload}.{_urlsafe_b64encode(signature)}"


def request_gateway_access_token(
    *,
    base_url: str,
    email: str,
    password: str,
    timeout_seconds: int = 30,
) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/auth/token",
        json={"email": email, "password": password},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Gateway /auth/token did not return an access_token.")
    return token
