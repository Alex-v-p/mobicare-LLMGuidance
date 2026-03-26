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

    def resolve_embedding_model(self, requested_embedding_model: str | None = None) -> str:
        return requested_embedding_model or self._embedding_client.model

    async def retrieve(self, query: str, limit: int | None = None, embedding_model: str | None = None):
        return self.items[: limit or len(self.items)]


@dataclass
class FakeHybridRetriever:
    items: list[RetrievedContext]

    def resolve_embedding_model(self, requested_embedding_model: str | None = None) -> str:
        return requested_embedding_model or "fake-embed"

    async def retrieve(self, **kwargs):
        from inference.retrieval.hybrid import HybridRetrievalResult

        return HybridRetrievalResult(
            items=self.items[: kwargs.get("limit", len(self.items))],
            metadata={
                "retrieval_mode": "hybrid",
                "embedding_model": kwargs.get("embedding_model") or "fake-embed",
            },
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




@pytest.mark.asyncio
async def test_guidance_pipeline_returns_minimal_unknown_fallback_for_generic_patient_only_non_answer():
    retrieved = [
        RetrievedContext(
            source_id="doc-1",
            title="Heart failure guideline",
            snippet="Monitor congestion, creatinine, potassium, and diuretic tolerance in heart failure follow-up.",
            chunk_id="c1",
        )
    ]
    pipeline = GuidancePipeline(
        retriever=FakeDenseRetriever(retrieved),
        hybrid_retriever=FakeHybridRetriever(retrieved),
        ollama_client=FakeOllamaClient([
            "1. Direct answer\n- The main value abnormalities point to a clinically relevant pattern that should be interpreted as a whole rather than marker by marker.\n- Use the retrieved guidance to prioritize the most abnormal findings first, while keeping uncertainty explicit where evidence is thin.\n\n2. Rationale\n- The main value abnormalities point to a clinically relevant pattern that should be interpreted as a whole rather than marker by marker.\n\n3. Caution\n- Key missing context that could change the recommendation: current symptoms, medication history, prior laboratory trends.\n\n4. General advice\n- Review these results together with symptoms, medication history, and prior laboratory trends."
        ]),
    )

    response = await pipeline.run(
        InferenceRequest(
            request_id="req-generic-patient-fallback",
            question="",
            patient_variables={"ef": 28, "nt_pro_bnp": 2400, "creatinine": 1.8, "potassium": 5.4},
            options=GenerationOptions(use_retrieval=True, retrieval_mode="dense", top_k=1),
        )
    )

    assert response.answer == "Based on the provided context, I don't know."


@pytest.mark.asyncio
async def test_guidance_pipeline_surfaces_heart_failure_specialty_focus_metadata():
    retrieved = [
        RetrievedContext(
            source_id="doc-1",
            title="Heart failure guideline",
            snippet="In patients with HFrEF and hyperkalaemia, review nephrotoxic drugs and monitor creatinine and electrolytes.",
            chunk_id="c1",
        )
    ]
    pipeline = GuidancePipeline(
        retriever=FakeDenseRetriever(retrieved),
        hybrid_retriever=FakeHybridRetriever(retrieved),
        ollama_client=FakeOllamaClient([
            "1. Direct answer\n- Heart-failure-oriented guidance.\n\n2. Rationale\n- Creatinine and potassium need follow-up.\n\n3. Caution\n- I don't know the full picture.\n\n4. General advice\n- Review symptoms and medications."
        ]),
    )

    response = await pipeline.run(
        InferenceRequest(
            request_id="req-hf-focus",
            question="",
            patient_variables={"ef": 28, "nt_pro_bnp": 2400, "creatinine": 1.7, "potassium": 5.4},
            options=GenerationOptions(use_retrieval=True, retrieval_mode="dense", top_k=1),
        )
    )

    assert response.metadata["specialty_focus"] == "heart_failure"


@pytest.mark.asyncio
async def test_guidance_pipeline_does_not_force_deterministic_fallback_only_because_context_is_partial():
    retrieved = [
        RetrievedContext(
            source_id="doc-1",
            title="Heart failure guideline",
            snippet="Monitor congestion, creatinine, potassium, and diuretic tolerance in heart failure follow-up.",
            chunk_id="c1",
        )
    ]
    pipeline = GuidancePipeline(
        retriever=FakeDenseRetriever(retrieved),
        hybrid_retriever=FakeHybridRetriever(retrieved),
        ollama_client=FakeOllamaClient([
            "1. Direct answer\n- Prioritize near-term reassessment of congestion and cardio-renal tolerance rather than making a treatment-specific change from these labs alone.\n\n"
            "2. Rationale\n- The dominant pattern is heart-failure burden with renal and electrolyte safety concerns.\n\n"
            "3. Caution\n- I don't know the full treatment conclusion because the evidence is partial and the current symptom burden is missing.\n\n"
            "4. General advice\n- Review symptoms, blood pressure tolerance, and recent medication changes together with the abnormal labs."
        ]),
    )

    response = await pipeline.run(
        InferenceRequest(
            request_id="req-no-forced-fallback",
            question="",
            patient_variables={
                "ef": 28,
                "nt_pro_bnp": 2400,
                "creatinine": 1.8,
                "potassium": 5.4,
                "glucose": 155,
                "hba1c": 7.2,
            },
            options=GenerationOptions(use_retrieval=True, retrieval_mode="dense", top_k=1),
        )
    )

    assert "prioritize near-term reassessment" in response.answer.lower()
    assert response.metadata["response_regeneration_attempts"] == 1


@pytest.mark.asyncio
async def test_guidance_pipeline_replaces_generic_non_answer_for_literal_context_question():
    retrieved = [
        RetrievedContext(
            source_id="doc-1",
            title="Supplementary Table 22",
            snippet="Percutaneous mechanical circulatory supports can be characterized by one of four circuit configurations: 1. intra-aortic devices, 2. transvalvular aortic (Impella) 3. left atrium to systemic artery (TandemHeart); 4. right atrium to systemic artery (veno-arterial extracorporeal memb",
            chunk_id="c1",
        )
    ]
    pipeline = GuidancePipeline(
        retriever=FakeDenseRetriever(retrieved),
        hybrid_retriever=FakeHybridRetriever(retrieved),
        ollama_client=FakeOllamaClient([
            "1. Direct answer\n- The main value abnormalities point to a clinically relevant pattern that should be interpreted as a whole rather than marker by marker.\n- Use the retrieved guidance to prioritize the most abnormal findings first, while keeping uncertainty explicit where evidence is thin.\n\n2. Rationale\n- The main value abnormalities point to a clinically relevant pattern that should be interpreted as a whole rather than marker by marker.\n\n3. Caution\n- Key missing context that could change the recommendation: current symptoms, medication history, prior laboratory trends.\n\n4. General advice\n- Review these results together with symptoms, medication history, and prior laboratory trends."
        ]),
    )

    response = await pipeline.run(
        InferenceRequest(
            request_id="req-literal-question",
            question="What are the four circuit configurations for percutaneous mechanical circulatory support as described in Supplementary Table 22?",
            patient_variables={},
            options=GenerationOptions(use_retrieval=True, retrieval_mode="dense", top_k=1),
        )
    )

    lowered = response.answer.lower()
    assert "clinically relevant pattern" not in lowered
    assert "intra-aortic devices" in lowered
    assert "transvalvular aortic" in lowered
    assert "tandemheart" in lowered


@pytest.mark.asyncio
async def test_guidance_pipeline_replaces_generic_non_answer_for_explicit_question_only_prompt():
    retrieved = [
        RetrievedContext(
            source_id="doc-1",
            title="Heart failure escalation",
            snippet="For symptomatic HFrEF, additional options may include sacubitril/valsartan, dapagliflozin, ivabradine, and CRT in eligible patients.",
            chunk_id="c1",
        )
    ]
    pipeline = GuidancePipeline(
        retriever=FakeDenseRetriever(retrieved),
        hybrid_retriever=FakeHybridRetriever(retrieved),
        ollama_client=FakeOllamaClient([
            "1. Direct answer\n- The main value abnormalities point to a clinically relevant pattern that should be interpreted as a whole rather than marker by marker.\n- Use the retrieved guidance to prioritize the most abnormal findings first, while keeping uncertainty explicit where evidence is thin.\n\n2. Rationale\n- The main value abnormalities point to a clinically relevant pattern that should be interpreted as a whole rather than marker by marker.\n\n3. Caution\n- Key missing context that could change the recommendation: current symptoms, medication history, prior laboratory trends.\n\n4. General advice\n- Review these results together with symptoms, medication history, and prior laboratory trends."
        ]),
    )

    response = await pipeline.run(
        InferenceRequest(
            request_id="req-explicit-question",
            question="What escalation of therapy should be considered for symptomatic HFrEF despite ACE inhibitor and beta blocker therapy?",
            patient_variables={},
            options=GenerationOptions(use_retrieval=True, retrieval_mode="dense", top_k=1),
        )
    )

    lowered = response.answer.lower()
    assert "clinically relevant pattern" not in lowered
    assert "hfref" in lowered or "sacubitril" in lowered or "dapagliflozin" in lowered or "crt" in lowered
