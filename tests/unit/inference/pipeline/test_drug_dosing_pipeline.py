import json

import pytest

from dataclasses import dataclass

from inference.pipeline.generate_guidance import GuidancePipeline
from shared.contracts.inference import GenerationOptions, InferenceRequest, OllamaGenerateResponse


@dataclass
class FakeDenseRetriever:
    items: list

    class _Emb:
        model = "fake-embed"

    _embedding_client = _Emb()

    async def retrieve(self, query: str, limit: int | None = None, embedding_model: str | None = None):
        return []


@dataclass
class FakeHybridRetriever:
    items: list

    async def retrieve(self, **kwargs):
        from inference.retrieval.hybrid import HybridRetrievalResult

        return HybridRetrievalResult(items=[], metadata={"retrieval_mode": "hybrid"})


class FakeOllamaClient:
    model = "fake-llm"

    def with_model(self, model: str | None):
        return self

    async def generate(self, prompt: str, temperature: float, max_tokens: int):
        return OllamaGenerateResponse(model=self.model, response="unused")


@pytest.mark.asyncio
async def test_guidance_pipeline_routes_to_drug_dosing_runner_and_returns_structured_json():
    pipeline = GuidancePipeline(retriever=FakeDenseRetriever([]), hybrid_retriever=FakeHybridRetriever([]), ollama_client=FakeOllamaClient())

    response = await pipeline.run(
        InferenceRequest(
            request_id="drug-dose-1",
            question="",
            patient_variables={
                "ef": 30,
                "potassium": 4.6,
                "egfr": 52,
                "creatinine": 1.4,
                "sbp": 108,
                "heartrate": 72,
                "dosebb_prev": 1.25,
                "rasdose_prev": 2.5,
                "dosespiro_prev": 25,
                "loop_dose_prev": 40,
                "congestion_present": True,
                "switch_to_arni": True,
            },
            options=GenerationOptions(pipeline_variant="drug_dosing"),
        )
    )

    assert response.metadata["pipeline_runner"] == "drug_dosing"
    payload = json.loads(response.answer)
    assert payload["mode"] == "drug_dosing"
    assert payload["recommendations"]["mra"]["action"] in {"increase", "maintain"}
    assert payload["recommendations"]["beta_blocker"]["action"] == "increase"
    assert payload["recommendations"]["arni"]["action"] in {"start", "switch"}
    assert payload["recommendations"]["loop_diuretic"]["action"] in {"increase", "maintain"}


@pytest.mark.asyncio
async def test_drug_dosing_pipeline_blocks_mra_and_arni_when_safety_thresholds_crossed():
    pipeline = GuidancePipeline(retriever=FakeDenseRetriever([]), hybrid_retriever=FakeHybridRetriever([]), ollama_client=FakeOllamaClient())

    response = await pipeline.run(
        InferenceRequest(
            request_id="drug-dose-2",
            question="",
            patient_variables={
                "ef": 25,
                "potassium": 5.8,
                "egfr": 24,
                "creatinine": 3.0,
                "sbp": 86,
                "heartrate": 48,
                "dosespiro_prev": 25,
                "arnidose_prev": "49/51 mg",
                "dosebb_prev": 2.5,
            },
            options=GenerationOptions(pipeline_variant="drug_dosing"),
        )
    )

    payload = json.loads(response.answer)
    assert payload["recommendations"]["mra"]["action"] in {"reduce", "stop"}
    assert payload["recommendations"]["arni"]["action"] == "stop"
    assert payload["recommendations"]["beta_blocker"]["action"] in {"reduce", "hold", "avoid_start"}


@pytest.mark.asyncio
async def test_drug_dosing_pipeline_uses_default_agents_when_agent_not_supplied():
    pipeline = GuidancePipeline(retriever=FakeDenseRetriever([]), hybrid_retriever=FakeHybridRetriever([]), ollama_client=FakeOllamaClient())

    response = await pipeline.run(
        InferenceRequest(
            request_id="drug-dose-3",
            question="",
            patient_variables={
                "ef": 28,
                "potassium": 4.2,
                "egfr": 65,
                "creatinine": 1.2,
                "sbp": 118,
                "heartrate": 76,
            },
            options=GenerationOptions(pipeline_variant="drug_dosing"),
        )
    )

    payload = json.loads(response.answer)
    assert payload["recommendations"]["beta_blocker"]["drug"] == "bisoprolol"
    assert payload["recommendations"]["ras"]["drug"] == "enalapril"
    assert payload["recommendations"]["sglt2"]["drug"] == "dapagliflozin"
