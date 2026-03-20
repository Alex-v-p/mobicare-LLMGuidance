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
            "1. Main answer\nThe PDF says the best document recommends reviewing nephrotoxic drugs when creatinine and potassium are elevated.\n\n"
            "2. General guidance\nMonitor renal function and potassium after changes.\n\n"
            "3. Uncertainty and missing data\nI don't know the full treatment plan because symptoms, medications, and baseline values are missing.",
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
    assert "Main answer" in response.answer
    assert response.verification is not None
