from __future__ import annotations

from fastapi import APIRouter

from api.application.services.guidance_service import GuidanceService
from shared.contracts.inference import ApiGuidanceResponse, GuidanceRequest

router = APIRouter(tags=["guidance"])


@router.post("/guidance", response_model=ApiGuidanceResponse)
async def generate_guidance(request: GuidanceRequest) -> ApiGuidanceResponse:
    return await GuidanceService().generate(request)
