from dataclasses import dataclass

import pytest

from inference.pipeline.generate_guidance import GuidancePipeline
from inference.retrieval.hybrid import HybridRetrievalResult
from shared.contracts.inference import GenerationOptions, InferenceRequest, OllamaGenerateResponse, RetrievedContext


MRA_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 4",
    snippet=(
        "WHICH MRA AND WHAT DOSE? Eplerenone: starting dose 25 mg o.d., target dose 50 mg o.d. "
        "Spironolactone: starting dose 25 mg o.d., target dose 50 mg o.d. "
        "If K rises above 5.5 mmol/L or creatinine rises to 2.5 mg/dL/eGFR <30, halve a dose. "
        "If K rises to >6.0 mmol/L or creatinine to 3.5 mg/dL/eGFR <20, stop MRA immediately."
    ),
    chunk_id="chunk-mra",
    page_number=12,
)

BETA_BLOCKER_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 3",
    snippet=(
        "WHICH BETA-BLOCKER AND WHAT DOSE? Bisoprolol: starting dose 1.25 mg o.d., target dose 10 mg o.d. "
        "Double the dose at not less than 2-week intervals. "
        "If persisting signs of congestion, hypotension (SBP <90 mmHg), relieve congestion and achieve euvolaemia before starting a beta-blocker. "
        "If <50 b.p.m. and worsening symptoms, halve the dose of beta-blocker."
    ),
    chunk_id="chunk-bb",
    page_number=11,
)

RAS_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 2",
    snippet=(
        "WHICH ACE-I AND WHAT DOSE? Enalapril: starting dose 2.5 mg b.i.d., target dose 10-20 mg b.i.d. "
        "Double the dose at not less than 2-week intervals. "
        "Significant hyperkalaemia (K >5.0 mmol/L), significant renal dysfunction [creatinine >2.5 mg/dL or eGFR <30], "
        "and SBP <90 mmHg are cautions. An increase in K to <= 5.5 mmol/L is acceptable. "
        "If K rises to >5.5 mmol/L or creatinine to >3.5 mg/dL/eGFR <20, the ACE-I should be stopped."
    ),
    chunk_id="chunk-ras",
    page_number=9,
)

ARNI_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 5",
    snippet=(
        "WHAT DOSE? Sac/Val: starting dose 49/51 mg b.i.d., target dose 97/103 mg b.i.d. "
        "24/26 mg b.i.d. in selected patients. A washout period of at least 36 h after ACE-I therapy is required. "
        "In some patients, one may consider a reduced starting dose (24/26 mg b.i.d.), namely in those with SBP 100-110 mmHg, eGFR 30-60 mL/min/1.73 m2. "
        "Significant hyperkalaemia (K >5.0 mmol/L). eGFR <30 and SBP <90 mmHg are cautions/contraindications. "
        "If K rises to >5.5 mmol/L or eGFR lowers to <30, the ARNI should be stopped."
    ),
    chunk_id="chunk-arni",
    page_number=14,
)

SGLT2_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 6",
    snippet=(
        "WHAT DOSE? Dapagliflozin: starting (and target) dose 10 mg o.d. Empagliflozin: starting (and target) dose 10 mg o.d. "
        "eGFR <20 mL/min/1.73 m2 and SBP <95 mmHg are contraindications/cautions. "
        "Fluid balance needs to be monitored because SGLT2 inhibitors may intensify diuresis."
    ),
    chunk_id="chunk-sglt2",
    page_number=15,
)

LOOP_CONTEXT = RetrievedContext(
    source_id="HF_GuidLine_Christof_ehab368_Suppl.pdf",
    title="Supplementary Table 7",
    snippet=(
        "WHICH DIURETIC AND WHAT DAILY DOSE? Loop diuretics: Furosemide: starting dose 20-40 mg, usual dose 40-240 mg. "
        "Bumetanide: starting dose 0.5-1 mg, usual dose 1-5 mg. Torasemide: starting dose 5-10 mg, usual dose 10-20 mg. "
        "Not indicated if the patient has never had symptoms or signs of congestion. "
        "SBP <90 mmHg and renal dysfunction are cautions. Hypovolaemia/dehydration: consider a diuretic dosage reduction."
    ),
    chunk_id="chunk-loop",
    page_number=17,
)


@dataclass
class FakeDenseRetriever:
    class _Emb:
        model = "fake-embed"

    _embedding_client = _Emb()

    async def retrieve(self, query: str, limit: int | None = None, embedding_model: str | None = None):
        return _contexts_for_query(query)[: limit or 2]


@dataclass
class FakeHybridRetriever:
    async def retrieve(self, **kwargs):
        query = kwargs["query"]
        items = _contexts_for_query(query)[: kwargs.get("limit", 2)]
        return HybridRetrievalResult(
            items=items,
            metadata={
                "retrieval_mode": "hybrid",
                "dense_candidates": len(items),
                "sparse_candidates": len(items),
                "hybrid_dense_weight": kwargs.get("dense_weight", 0.65),
                "hybrid_sparse_weight": kwargs.get("sparse_weight", 0.35),
                "graph_augmented": False,
                "graph_nodes_added": 0,
                "graph_edges_used": [],
            },
        )


class FakeOllamaClient:
    model = "fake-llm"

    def with_model(self, model: str | None):
        return self

    async def generate(self, prompt: str, temperature: float, max_tokens: int):
        return OllamaGenerateResponse(model=self.model, response="unused")


def _contexts_for_query(query: str) -> list[RetrievedContext]:
    lowered = query.lower()
    items: list[RetrievedContext] = []
    if "mra" in lowered or "spironolactone" in lowered or "eplerenone" in lowered:
        items.append(MRA_CONTEXT)
    if "beta-blocker" in lowered or "beta blocker" in lowered or "bisoprolol" in lowered:
        items.append(BETA_BLOCKER_CONTEXT)
    if "ace-i" in lowered or "enalapril" in lowered or "arb" in lowered:
        items.append(RAS_CONTEXT)
    if "sacubitril" in lowered or "arni" in lowered or "sac/val" in lowered:
        items.append(ARNI_CONTEXT)
    if "sglt2" in lowered or "dapagliflozin" in lowered or "empagliflozin" in lowered:
        items.append(SGLT2_CONTEXT)
    if "diuretic" in lowered or "furosemide" in lowered or "congestion" in lowered:
        items.append(LOOP_CONTEXT)
    return items


@pytest.mark.asyncio
async def test_grounded_drug_dosing_pipeline_returns_retrieved_context_and_grounded_answer():
    pipeline = GuidancePipeline(retriever=FakeDenseRetriever(), hybrid_retriever=FakeHybridRetriever(), ollama_client=FakeOllamaClient())

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
                "switch_to_arni": True,
            },
            options=GenerationOptions(pipeline_variant="drug_dosing"),
        )
    )

    assert response.metadata["pipeline_runner"] == "drug_dosing"
    assert response.metadata["drug_dosing_mode"] == "grounded_hybrid_evidence"
    assert response.retrieved_context
    payload = response.metadata["drug_dosing_payload"]
    assert payload["mode"] == "drug_dosing_grounded"
    assert payload["evidence_rows_used"]["arni"]["source_chunk_ids"] == ["chunk-arni"]
    assert payload["recommendations"]["arni"]["grounded"] is True
    assert payload["recommendations"]["beta_blocker"]["action"] == "increase"
    assert "sacubitril/valsartan:" in response.answer
    assert response.verification.verdict == "pass"


@pytest.mark.asyncio
async def test_grounded_drug_dosing_pipeline_blocks_unsafe_recommendations_from_retrieved_thresholds():
    pipeline = GuidancePipeline(retriever=FakeDenseRetriever(), hybrid_retriever=FakeHybridRetriever(), ollama_client=FakeOllamaClient())

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
                "arnidose_prev": "49/51 mg b.i.d.",
                "dosebb_prev": 2.5,
                "rasdose_prev": 10,
            },
            options=GenerationOptions(pipeline_variant="drug_dosing"),
        )
    )

    payload = response.metadata["drug_dosing_payload"]
    assert payload["recommendations"]["mra"]["action"] in {"reduce", "stop"}
    assert payload["recommendations"]["arni"]["action"] == "stop"
    assert payload["recommendations"]["beta_blocker"]["action"] in {"reduce", "avoid_start"}
    assert response.answer == "No grounded drug dose recommendation could be made from the retrieved guideline context."


@pytest.mark.asyncio
async def test_grounded_drug_dosing_pipeline_uses_evidence_backed_default_agents_when_unspecified():
    pipeline = GuidancePipeline(retriever=FakeDenseRetriever(), hybrid_retriever=FakeHybridRetriever(), ollama_client=FakeOllamaClient())

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

    payload = response.metadata["drug_dosing_payload"]
    assert payload["recommendations"]["beta_blocker"]["drug"] == "bisoprolol"
    assert payload["recommendations"]["ras"]["drug"] == "enalapril"
    assert payload["recommendations"]["sglt2"]["drug"] == "dapagliflozin"
    assert payload["recommendations"]["beta_blocker"]["grounded"] is True


@pytest.mark.asyncio
async def test_grounded_drug_dosing_pipeline_includes_evidence_and_short_answer_for_sample_case():
    pipeline = GuidancePipeline(retriever=FakeDenseRetriever(), hybrid_retriever=FakeHybridRetriever(), ollama_client=FakeOllamaClient())

    response = await pipeline.run(
        InferenceRequest(
            request_id="drug-dose-4",
            question="",
            patient_variables={
                "age": 74,
                "gender": "male",
                "bnp": 1050,
                "nt_pro_bnp": 4800,
                "creatinine": 2.0,
                "egfr": 34,
                "urea": 78,
                "potassium": 5.3,
                "sodium": 129,
                "blood_pressure_systolic": 95,
                "blood_pressure_diastolic": 60,
                "heart_rate": 108,
                "weight": 82,
                "nyha": 3,
                "ef": 28,
                "DoseSpiro_prev": 25,
                "DoseBB_prev": 2.5,
                "RASDose_prev": 10,
                "ARNIDose_prev": 0,
                "SGLT2Dose_prev": 0,
                "Loop_dose_prev": 40,
            },
            options=GenerationOptions(pipeline_variant="drug_dosing"),
        )
    )

    lines = [line for line in response.answer.splitlines() if line.strip()]
    assert len(lines) <= 3
    assert response.retrieved_context
    payload = response.metadata["drug_dosing_payload"]
    assert payload["evidence_rows_used"]
    assert any(line.startswith("dapagliflozin:") for line in lines)
    assert all(item["grounded"] for item in payload["selected_recommendations"])
