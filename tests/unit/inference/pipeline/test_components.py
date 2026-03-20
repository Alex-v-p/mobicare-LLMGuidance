from inference.clinical import build_clinical_profile, build_question_from_patient_data
from inference.pipeline.components import ChunkRelevanceRanker, ContextJudge, QueryPlanner
from shared.contracts.inference import GenerationOptions, InferenceRequest, RetrievedContext


def test_build_clinical_profile_flags_abnormal_variables():
    profile = build_clinical_profile({"ef": 32, "potassium": 4.1, "mystery": "x"})

    assert [finding.label for finding in profile.abnormal_variables] == ["Ejection fraction"]
    assert "mystery" in profile.unknown_variables


def test_build_question_from_patient_data_infers_task_without_question():
    profile = build_clinical_profile({"ef": 30, "nt_pro_bnp": 900})

    question = build_question_from_patient_data({"ef": 30, "nt_pro_bnp": 900}, profile)

    assert "treatment" in question.lower()
    assert "Ejection fraction" in question


def test_query_planner_adds_adaptive_queries_for_abnormal_variables():
    planner = QueryPlanner()
    request = InferenceRequest(
        request_id="req-1",
        question="",
        patient_variables={"ef": 28, "nt_pro_bnp": 1200, "age": 72},
        options=GenerationOptions(adaptive_retrieval_enabled=True),
    )

    plan = planner.create_plan(request)

    assert plan.effective_question
    assert len(plan.expanded_queries) >= 2
    assert any("abnormal" in query.lower() for query in plan.expanded_queries)


def test_context_judge_marks_context_as_insufficient_when_overlap_is_missing():
    judge = ContextJudge()
    profile = build_clinical_profile({"ef": 30})
    assessment = judge.assess(
        retrieved_context=[RetrievedContext(source_id="a", title="Nutrition", snippet="General dietary text")],
        retrieval_query="heart failure ejection fraction treatment",
        clinical_profile=profile,
        minimum_results=2,
    )

    assert assessment.sufficient is False
    assert "too_few_context_chunks" in assessment.reasons


def test_chunk_ranker_prefers_chunks_with_clinical_term_overlap():
    ranker = ChunkRelevanceRanker()
    profile = build_clinical_profile({"ef": 30})
    ranked, details = ranker.rank(
        contexts=[
            RetrievedContext(source_id="1", title="Heart failure", snippet="Reduced ejection fraction therapy", chunk_id="a"),
            RetrievedContext(source_id="2", title="Dermatology", snippet="Skin care guidance", chunk_id="b"),
        ],
        retrieval_query="heart failure treatment",
        clinical_profile=profile,
        limit=2,
    )

    assert ranked[0].chunk_id == "a"
    assert details[0]["score"] >= details[1]["score"]
