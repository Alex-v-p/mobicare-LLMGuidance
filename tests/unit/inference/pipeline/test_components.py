from inference.clinical import build_clinical_profile, build_question_from_patient_data
from inference.pipeline.components import ChunkRelevanceRanker, ContextJudge, QueryPlanner, ResponseVerifier
from shared.contracts.inference import GenerationOptions, InferenceRequest, RetrievedContext


def test_build_clinical_profile_uses_gender_specific_reference_ranges():
    male_profile = build_clinical_profile({"gender": "male", "hemoglobin": 12.5})
    female_profile = build_clinical_profile({"gender": "female", "hemoglobin": 12.5})

    assert [finding.label for finding in male_profile.abnormal_variables] == ["Hemoglobin"]
    assert female_profile.abnormal_variables == []
    assert "Gender" in [finding.label for finding in male_profile.informational_variables]


def test_build_clinical_profile_uses_age_specific_reference_ranges():
    younger_profile = build_clinical_profile({"age": 40, "cysc": 1.1})
    older_profile = build_clinical_profile({"age": 65, "cysc": 1.1})

    assert [finding.label for finding in younger_profile.abnormal_variables] == ["Cystatin C"]
    assert older_profile.abnormal_variables == []


def test_build_question_from_patient_data_infers_task_without_question():
    profile = build_clinical_profile({"gender": "male", "ef": 30, "nt_pro_bnp": 900})

    question = build_question_from_patient_data({"gender": "male", "ef": 30, "nt_pro_bnp": 900}, profile)

    assert "treatment" in question.lower()
    assert "Ejection fraction" in question


def test_query_planner_adds_targeted_adaptive_queries_for_abnormal_variables():
    planner = QueryPlanner()
    request = InferenceRequest(
        request_id="req-1",
        question="",
        patient_variables={"gender": "male", "ef": 28, "nt_pro_bnp": 1200, "age": 72},
        options=GenerationOptions(adaptive_retrieval_enabled=True),
    )

    plan = planner.create_plan(request)

    assert plan.effective_question
    assert len(plan.expanded_queries) >= 3
    assert any("clinical management" in query.lower() for query in plan.expanded_queries)
    assert any("cardiac status" in query.lower() for query in plan.expanded_queries)


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


def test_response_verifier_flags_document_mentions():
    verifier = ResponseVerifier(ollama_client=None)  # type: ignore[arg-type]
    result = verifier.heuristic_verify(
        "1. Direct answer\nThe PDF says the best document is helpful.\n\n"
        "2. Rationale\nMore text.\n\n"
        "3. Caution\nI don't know.\n\n"
        "4. General advice\nMore text."
    )

    assert result.verdict == "fail"
    assert any("document-selection" in issue for issue in result.issues)


def test_response_verifier_flags_potassium_contradiction():
    verifier = ResponseVerifier(ollama_client=None)
    profile = build_clinical_profile({"potassium": 5.6, "creatinine": 1.8})
    result = verifier.heuristic_verify(
        "1. Direct answer\nPrevent hypokalemia.\n\n"
        "2. Rationale\nPotassium is high.\n\n"
        "3. Caution\nI don't know.\n\n"
        "4. General advice\nReview medications.",
        patient_variables={"potassium": 5.6, "creatinine": 1.8},
        clinical_profile=profile,
    )

    assert result.verdict == "fail"
    assert any("potassium value" in issue for issue in result.issues)
