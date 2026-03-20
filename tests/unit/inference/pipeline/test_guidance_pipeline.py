from dataclasses import dataclass

import pytest

from inference.pipeline.generate_guidance import GuidancePipeline
from shared.contracts.inference import GenerationOptions, InferenceRequest, OllamaGenerateResponse, RetrievedContext


@dataclass
class FakeDenseRetriever:
    items: list[RetrievedContext]

    class _Emb:
        model = "fake-embed"

    _embedding_client = _Emb()

    async def retrieve(self, query: str, limit: int | None = None, embedding_model: str | None = None):
        return self.items[: limit or len(self.items)]


@dataclass
class FakeHybridRetriever:
    items: list[RetrievedContext]

    async def retrieve(self, **kwargs):
        from inference.retrieval.hybrid import HybridRetrievalResult

        return HybridRetrievalResult(
            items=self.items[: kwargs.get("limit", len(self.items))],
            metadata={"retrieval_mode": "hybrid"},
        )


class FakeOllamaClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.model = "fake-llm"

    def with_model(self, model: str | None):
        return self

    async def generate(self, prompt: str, temperature: float, max_tokens: int):
        text = self.responses.pop(0)
        return OllamaGenerateResponse(model=self.model, response=text)


@pytest.mark.asyncio
async def test_guidance_pipeline_infers_task_from_patient_data_only_and_normalizes_source_talk():
    retrieved = [
        RetrievedContext(
            source_id="doc-1",
            title="Heart failure guideline",
            snippet="If creatinine or potassium rises excessively, review nephrotoxic drugs and consider adjusting diuretics when congestion is absent.",
            chunk_id="c1",
        )
    ]
    pipeline = GuidancePipeline(
        retriever=FakeDenseRetriever(retrieved),
        hybrid_retriever=FakeHybridRetriever(retrieved),
        ollama_client=FakeOllamaClient([
            "1. Direct answer\nThe PDF says the best document recommends reviewing nephrotoxic drugs when creatinine and potassium are elevated.\n\n"
            "2. Rationale\nMonitor renal function and potassium after changes.\n\n"
            "3. Caution\nI don't know the full treatment plan because symptoms, medications, and baseline values are missing.\n\n"
            "4. General advice\nReview the patient values with prior results.",
        ]),
    )

    response = await pipeline.run(
        InferenceRequest(
            request_id="req-1",
            question="",
            patient_variables={"gender": "male", "creatinine": 1.8, "potassium": 5.6},
            options=GenerationOptions(use_retrieval=True, retrieval_mode="dense", top_k=1),
        )
    )

    assert response.metadata["effective_question"]
    assert "patient-data-driven task was inferred" in " ".join(response.warnings)
    assert response.retrieved_context
    assert "pdf" not in response.answer.lower()
    assert "best document" not in response.answer.lower()
    assert "Direct answer" in response.answer
    assert "General advice" in response.answer
    assert response.verification is not None


@pytest.mark.asyncio
async def test_guidance_pipeline_falls_back_to_deterministic_answer_on_contradiction():
    retrieved = [
        RetrievedContext(
            source_id="doc-1",
            title="Heart failure guideline",
            snippet="Renal function should be monitored with creatinine and electrolytes. Review nephrotoxic drugs when creatinine rises.",
            chunk_id="c1",
        )
    ]
    pipeline = GuidancePipeline(
        retriever=FakeDenseRetriever(retrieved),
        hybrid_retriever=FakeHybridRetriever(retrieved),
        ollama_client=FakeOllamaClient([
            "1. Direct answer\nPrevent hypokalemia with an MRA.\n\n2. Rationale\nPotassium is high.\n\n3. Caution\nNone.\n\n4. General advice\nNone."
        ]),
    )

    response = await pipeline.run(
        InferenceRequest(
            request_id="req-2",
            question="",
            patient_variables={"gender": "male", "creatinine": 1.8, "potassium": 5.6, "sodium": 132},
            options=GenerationOptions(use_retrieval=True, retrieval_mode="dense", top_k=1),
        )
    )

    assert "hypokalemia" not in response.answer.lower()
    assert "1. Direct answer" in response.answer
    assert "2. Rationale" in response.answer
    assert "3. Caution" in response.answer
    assert "4. General advice" in response.answer
