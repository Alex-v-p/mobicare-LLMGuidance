from __future__ import annotations

from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AuthenticatedUser(BaseModel):
    sub: str
    email: str
    iss: str
    aud: str
    exp: int
    iat: int


class ProtectedExampleResponse(BaseModel):
    message: str
    user: AuthenticatedUser
