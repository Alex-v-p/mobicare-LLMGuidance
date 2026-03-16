from __future__ import annotations

from fastapi import APIRouter, Depends

from api.application.services.auth_service import AuthService
from api.auth.dependencies import get_current_user
from api.dependencies import get_auth_service
from shared.contracts.auth import AuthenticatedUser, LoginRequest, ProtectedExampleResponse, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
async def create_token(
    request: LoginRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    return await service.login(request)


@router.get("/example-protected", response_model=ProtectedExampleResponse)
def example_protected_endpoint(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> ProtectedExampleResponse:
    return ProtectedExampleResponse(
        message="JWT authentication is working for this example endpoint.",
        user=current_user,
    )
