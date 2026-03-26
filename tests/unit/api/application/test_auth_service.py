from __future__ import annotations

import pytest

from api.application.services.auth_service import AuthService
from api.infrastructure.clients.auth_client import AuthClientError
from api.errors import ServiceUnavailableError, UnauthorizedError
from shared.contracts.auth import LoginRequest
from shared.config import ApiSettings


class StubAuthClient:
    def __init__(self, *, is_valid: bool | None = None, error: Exception | None = None) -> None:
        self.is_valid = is_valid
        self.error = error
        self.calls: list[dict[str, str]] = []

    async def validate_credentials(self, *, email: str, password: str) -> bool:
        self.calls.append({"email": email, "password": password})
        if self.error:
            raise self.error
        assert self.is_valid is not None
        return self.is_valid


@pytest.mark.asyncio
async def test_login_returns_token_response(monkeypatch: pytest.MonkeyPatch):
    request = LoginRequest(email="user@example.com", password="secret")
    service = AuthService(auth_client=StubAuthClient(is_valid=True), settings=ApiSettings(jwt_secret_key="secret"))

    monkeypatch.setattr("api.application.services.auth_service.create_access_token", lambda *, email, settings: (f"token-for:{email}", 3600))

    response = await service.login(request)

    assert response.access_token == "token-for:user@example.com"
    assert response.expires_in == 3600


@pytest.mark.asyncio
async def test_login_raises_unauthorized_on_invalid_credentials():
    service = AuthService(auth_client=StubAuthClient(is_valid=False))

    with pytest.raises(UnauthorizedError) as exc:
        await service.login(LoginRequest(email="user@example.com", password="bad"))

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_login_maps_auth_client_errors_to_service_unavailable():
    error = AuthClientError(status_code=503, code="AUTH_DOWN", message="down", details={"reason": "timeout"})
    service = AuthService(auth_client=StubAuthClient(error=error))

    with pytest.raises(ServiceUnavailableError) as exc:
        await service.login(LoginRequest(email="user@example.com", password="secret"))

    assert exc.value.code == "AUTH_DOWN"
    assert exc.value.details == {"reason": "timeout"}
