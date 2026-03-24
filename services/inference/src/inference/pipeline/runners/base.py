from __future__ import annotations

from typing import Protocol

from shared.contracts.inference import InferenceRequest, InferenceResponse


class PipelineRunner(Protocol):
    async def run(self, request: InferenceRequest) -> InferenceResponse: ...
