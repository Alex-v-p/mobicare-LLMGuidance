from __future__ import annotations

from inference.pipeline.runners import PipelineRunnerRegistry
from shared.contracts.inference import InferenceRequest, InferenceResponse


class GuidancePipeline:
    def __init__(self, runner_registry: PipelineRunnerRegistry) -> None:
        self._runner_registry = runner_registry

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        runner = self._runner_registry.resolve(request.options.pipeline_variant)
        return await runner.run(request)
