from __future__ import annotations

from fastapi.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.dependencies import get_auth_service
from api.main import create_app
from shared.contracts.auth import AuthenticatedUser, LoginRequest, TokenResponse


class StubAuthService:
    async def login(self, request: LoginRequest) -> TokenResponse:
        return TokenResponse(access_token=f"token-for:{request.email}", expires_in=3600)


def test_auth_token_route_returns_token():
    app = create_app(bootstrap_minio_on_startup=False)
    app.dependency_overrides[get_auth_service] = lambda: StubAuthService()

    with TestClient(app) as client:
        response = client.post("/auth/token", json={"email": "user@example.com", "password": "secret"})

    assert response.status_code == 200
    assert response.json()["access_token"] == "token-for:user@example.com"


def test_protected_route_returns_current_user():
    app = create_app(bootstrap_minio_on_startup=False)
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(sub="user@example.com", email="user@example.com", iss="issuer", aud="aud", iat=1, exp=9999999999)

    with TestClient(app) as client:
        response = client.get("/auth/example-protected", headers={"Authorization": "Bearer ignored"})

    assert response.status_code == 200
    assert response.json()["user"]["email"] == "user@example.com"
