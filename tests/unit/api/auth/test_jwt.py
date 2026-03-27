from __future__ import annotations

import base64
import json

import pytest

from api.auth.jwt import JwtValidationError, _urlsafe_b64encode, create_access_token, decode_access_token
from shared.config import ApiSettings


def test_create_and_decode_access_token_round_trip():
    settings = ApiSettings(jwt_secret_key="super-secret", jwt_issuer="issuer", jwt_audience="aud", jwt_access_token_exp_minutes=5)

    token, expires_in = create_access_token(email="user@example.com", settings=settings)
    user = decode_access_token(token, settings=settings)

    assert expires_in == 300
    assert user.email == "user@example.com"
    assert user.sub == "user@example.com"


def test_decode_access_token_rejects_invalid_format():
    with pytest.raises(JwtValidationError) as exc:
        decode_access_token("not.a.jwt.with.too.many.parts", settings=ApiSettings(jwt_secret_key="x"))

    assert exc.value.code == "AUTH_TOKEN_INVALID"


def test_decode_access_token_rejects_expired_token():
    settings = ApiSettings(jwt_secret_key="secret", jwt_issuer="issuer", jwt_audience="aud")
    header = _urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload = _urlsafe_b64encode(json.dumps({"sub": "user@example.com", "email": "user@example.com", "iss": "issuer", "aud": "aud", "iat": 1, "exp": 1}, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    import hashlib, hmac
    signature = hmac.new(settings.jwt_secret_key.encode("utf-8"), f"{header}.{payload}".encode("utf-8"), hashlib.sha256).digest()
    token = f"{header}.{payload}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('utf-8')}"

    with pytest.raises(JwtValidationError) as exc:
        decode_access_token(token, settings=settings)

    assert exc.value.code == "AUTH_TOKEN_EXPIRED"
