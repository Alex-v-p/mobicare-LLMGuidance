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
        query_lower = query.lower()
        if "anemia" in query_lower or "iron" in query_lower:
            return [item for item in self.items if "ferritin" in item.snippet.lower() or "hemoglobin" in item.snippet.lower()][: limit or len(self.items)]
        if "inflammation" in query_lower or "crp" in query_lower:
            return [item for item in self.items if "crp" in item.snippet.lower()][: limit or len(self.items)]
        if "renal" in query_lower or "creatinine" in query_lower or "potassium" in query_lower:
            return [item for item in self.items if "creatinine" in item.snippet.lower() or "potassium" in item.snippet.lower()][: limit or len(self.items)]
        return self.items[: limit or len(self.items)]


@dataclass
class FakeHybridRetriever:
    items: list[RetrievedContext]

    async def retrieve(self, **kwargs):
        from inference.retrieval.hybrid import HybridRetrievalResult

        query_lower = kwargs.get("query", "").lower()
        if "anemia" in query_lower or "iron" in query_lower:
            filtered = [item for item in self.items if "ferritin" in item.snippet.lower() or "hemoglobin" in item.snippet.lower()]
        elif "inflammation" in query_lower or "crp" in query_lower:
            filtered = [item for item in self.items if "crp" in item.snippet.lower()]
        elif "renal" in query_lower or "creatinine" in query_lower or "potassium" in query_lower:
            filtered = [item for item in self.items if "creatinine" in item.snippet.lower() or "potassium" in item.snippet.lower()]
        else:
            filtered = self.items

        return HybridRetrievalResult(
            items=filtered[: kwargs.get("limit", len(filtered))],
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
async def test_guidance_pipeline_infers_task_from_patient_data_only_and_replaces_bad_small_model_output_with_structured_answer():
    retrieved = [
        RetrievedContext(
            source_id="doc-1",
            title="Heart failure guideline",
            snippet="If creatinine or potassium rises excessively, review nephrotoxic drugs and consider adjusting diuretics when congestion is absent.",
            chunk_id="c1",
        ),
        RetrievedContext(
            source_id="doc-2",
            title="Iron deficiency note",
            snippet="Low hemoglobin with low ferritin is compatible with iron deficiency and warrants follow-up testing and clinical assessment.",
            chunk_id="c2",
        ),
        RetrievedContext(
            source_id="doc-3",
            title="Inflammation note",
            snippet="An elevated CRP supports ongoing inflammation but does not identify the cause on its own.",
            chunk_id="c3",
        ),
    ]
    pipeline = GuidancePipeline(
        retriever=FakeDenseRetriever(retrieved),
        hybrid_retriever=FakeHybridRetriever(retrieved),
        ollama_client=FakeOllamaClient([
            "1. Main answer\nThe PDF says aliskiren should be adjusted and this is the best document.\n\n"
            "2. General guidance\nMonitor renal function and potassium after changes.\n\n"
            "3. Uncertainty and missing data\nI don't know the full treatment plan because symptoms, medications, and baseline values are missing.",
        ]),
    )

    response = await pipeline.run(
        InferenceRequest(
            request_id="req-1",
            question="",
            patient_variables={
                "gender": "male",
                "hemoglobin": 10.2,
                "ferritin": 8,
                "creatinine": 1.8,
                "potassium": 5.6,
                "crp": 18,
            },
            options=GenerationOptions(use_retrieval=True, retrieval_mode="dense", top_k=3),
        )
    )

    assert response.metadata["effective_question"]
    assert "patient-data-driven task was inferred" in " ".join(response.warnings)
    assert response.retrieved_context
    assert "pdf" not in response.answer.lower()
    assert "best document" not in response.answer.lower()
    assert "aliskiren" not in response.answer.lower()
    assert "1. Direct answer" in response.answer
    assert "2. Rationale" in response.answer
    assert "3. Caution" in response.answer
    assert "4. General advice" in response.answer
    assert "Renal function and potassium" in response.metadata["retrieval_clusters"]
    assert "Anemia and iron status" in response.metadata["retrieval_clusters"]
    assert response.verification is not None
