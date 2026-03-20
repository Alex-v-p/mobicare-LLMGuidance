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
        return HybridRetrievalResult(items=self.items[: kwargs.get('limit', len(self.items))], metadata={"retrieval_mode": "hybrid"})


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
async def test_guidance_pipeline_infers_task_from_patient_data_only():
    retrieved = [
        RetrievedContext(
            source_id="doc-1",
            title="Heart failure guideline",
            snippet="Reduced ejection fraction should be assessed and managed carefully.",
            chunk_id="c1",
        )
    ]
    pipeline = GuidancePipeline(
        retriever=FakeDenseRetriever(retrieved),
        hybrid_retriever=FakeHybridRetriever(retrieved),
        ollama_client=FakeOllamaClient([
            "1. placeholder",  # not used when query rewriting disabled? safe extra
            "Evidence-based recommendation\nUse the heart failure guidance in context.\n\nDocument-grounded general guidance\nMonitor symptoms based on the guideline excerpt.\n\nUncertainty and missing data\nI don't know the full treatment recommendation because the retrieved evidence is limited.",
        ]),
    )

    response = await pipeline.run(
        InferenceRequest(
            request_id="req-1",
            question="",
            patient_variables={"ef": 28, "nt_pro_bnp": 1200},
            options=GenerationOptions(use_retrieval=True, retrieval_mode="dense", top_k=1),
        )
    )

    assert response.metadata["effective_question"]
    assert "patient-data-driven task was inferred" in " ".join(response.warnings)
    assert response.retrieved_context
