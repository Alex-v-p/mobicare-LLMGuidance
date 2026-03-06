from __future__ import annotations

from fastapi import APIRouter

from inference.pipeline.generate_guidance import GuidancePipeline
from shared.contracts.inference import InferenceRequest, InferenceResponse

router = APIRouter(tags=["guidance"])


@router.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest) -> InferenceResponse:
    return await GuidancePipeline().run(request)
