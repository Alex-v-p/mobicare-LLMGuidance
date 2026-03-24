from __future__ import annotations

from inference.pipeline.runners.base import PipelineRunner


class PipelineRunnerRegistry:
    def __init__(self, standard_runner: PipelineRunner) -> None:
        self._standard_runner = standard_runner

    def resolve(self, pipeline_name: str | None = None) -> PipelineRunner:
        if pipeline_name in {None, "", "standard"}:
            return self._standard_runner
        raise ValueError(f"Unknown pipeline runner: {pipeline_name}")
