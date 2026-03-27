from __future__ import annotations

from inference.application.pipelines.runners.base import PipelineRunner


class PipelineRunnerRegistry:
    def __init__(self, standard_runner: PipelineRunner, drug_dosing_runner: PipelineRunner) -> None:
        self._standard_runner = standard_runner
        self._drug_dosing_runner = drug_dosing_runner

    def resolve(self, pipeline_name: str | None = None) -> PipelineRunner:
        if pipeline_name in {None, "", "standard"}:
            return self._standard_runner
        if pipeline_name == "drug_dosing":
            return self._drug_dosing_runner
        raise ValueError(f"Unknown pipeline runner: {pipeline_name}")
