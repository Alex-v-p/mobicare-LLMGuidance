from __future__ import annotations

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth import JwtValidationError, decode_access_token
from api.errors import UnauthorizedError
from shared.contracts.auth import AuthenticatedUser
from shared.contracts.error_codes import ErrorCode

security = HTTPBearer(auto_error=False)


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise UnauthorizedError(code=ErrorCode.AUTH_TOKEN_INVALID, message="Bearer token is required.")

    try:
        return decode_access_token(credentials.credentials)
    except JwtValidationError as exc:
        raise UnauthorizedError(code=exc.code, message=exc.message) from exc
