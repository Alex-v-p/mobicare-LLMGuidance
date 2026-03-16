from __future__ import annotations

from api.auth import create_access_token
from api.clients.auth_client import AuthClient, AuthClientError
from api.errors import ServiceUnavailableError, UnauthorizedError
from shared.config import Settings, get_settings
from shared.contracts.auth import LoginRequest, TokenResponse
from shared.contracts.error_codes import ErrorCode


class AuthService:
    def __init__(self, auth_client: AuthClient, settings: Settings | None = None) -> None:
        self._auth_client = auth_client
        self._settings = settings or get_settings()

    async def login(self, request: LoginRequest) -> TokenResponse:
        try:
            is_valid = await self._auth_client.validate_credentials(email=request.email, password=request.password)
        except AuthClientError as exc:
            raise ServiceUnavailableError(code=exc.code, message=exc.message, details=exc.details) from exc

        if not is_valid:
            raise UnauthorizedError(
                code=ErrorCode.AUTH_INVALID_CREDENTIALS,
                message="Invalid email or password.",
            )

        token, expires_in = create_access_token(email=request.email, settings=self._settings)
        return TokenResponse(access_token=token, expires_in=expires_in)
