from __future__ import annotations

from inference.pipeline.support.drug_dosing import (
    build_drug_dosing_payload,
    render_drug_dosing_answer,
    summarize_drug_dosing_warnings,
)
from shared.contracts.inference import InferenceRequest, InferenceResponse, VerificationResult


class DrugDosingPipelineRunner:
    async def run(self, request: InferenceRequest) -> InferenceResponse:
        payload = build_drug_dosing_payload(request.patient_variables)
        return InferenceResponse(
            request_id=request.request_id,
            status="ok",
            model="drug-dosing-rule-engine-v1",
            answer=render_drug_dosing_answer(payload),
            retrieved_context=[],
            used_variables=request.patient_variables,
            warnings=summarize_drug_dosing_warnings(request.patient_variables),
            metadata={
                "pipeline_runner": "drug_dosing",
                "drug_dosing_mode": "deterministic_guideline_rules",
                "guideline_basis": "ESC 2021 supplementary tables 2-7",
                "drug_dosing_payload": payload,
            },
            verification=VerificationResult(
                verdict="pass",
                issues=["deterministic drug dosing pipeline"],
                confidence="medium",
            ),
        )
