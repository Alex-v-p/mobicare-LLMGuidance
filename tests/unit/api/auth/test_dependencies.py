from __future__ import annotations

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from api.auth.dependencies import get_current_user
from api.auth.jwt import JwtValidationError
from api.errors import UnauthorizedError


def test_get_current_user_requires_bearer_token():
    with pytest.raises(UnauthorizedError) as exc:
        get_current_user(None)

    assert exc.value.code == "AUTH_TOKEN_INVALID"


def test_get_current_user_maps_jwt_errors(monkeypatch: pytest.MonkeyPatch):
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad-token")
    monkeypatch.setattr("api.auth.dependencies.decode_access_token", lambda token: (_ for _ in ()).throw(JwtValidationError(code="AUTH_TOKEN_EXPIRED", message="expired")))

    with pytest.raises(UnauthorizedError) as exc:
        get_current_user(credentials)

    assert exc.value.code == "AUTH_TOKEN_EXPIRED"
