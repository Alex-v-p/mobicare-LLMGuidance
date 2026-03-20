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


def test_query_planner_adds_cluster_queries_for_abnormal_variables():
    planner = QueryPlanner()
    request = InferenceRequest(
        request_id="req-1",
        question="",
        patient_variables={"gender": "male", "hemoglobin": 10.2, "ferritin": 8, "creatinine": 1.8, "potassium": 5.6},
        options=GenerationOptions(adaptive_retrieval_enabled=True),
    )

    plan = planner.create_plan(request)

    assert plan.effective_question
    assert len(plan.expanded_queries) >= 4
    assert any("renal function and potassium" in query.lower() for query in plan.expanded_queries)
    assert any("anemia and iron status" in query.lower() for query in plan.expanded_queries)


def test_context_judge_marks_context_as_insufficient_when_cluster_coverage_is_missing():
    judge = ContextJudge()
    profile = build_clinical_profile({"hemoglobin": 10.2, "ferritin": 8, "creatinine": 1.8, "potassium": 5.6})
    planner = QueryPlanner()
    plan = planner.create_plan(
        InferenceRequest(
            request_id="req-1",
            question="",
            patient_variables={"hemoglobin": 10.2, "ferritin": 8, "creatinine": 1.8, "potassium": 5.6},
            options=GenerationOptions(),
        )
    )
    assessment = judge.assess(
        retrieved_context=[RetrievedContext(source_id="a", title="Renal", snippet="Creatinine and potassium management")],
        retrieval_query="anemia and renal follow-up",
        clinical_profile=profile,
        minimum_results=2,
        clusters=plan.clusters,
    )

    assert assessment.sufficient is False
    assert "incomplete_cluster_coverage" in assessment.reasons


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
        clusters=[],
    )

    assert ranked[0].chunk_id == "a"
    assert details[0]["score"] >= details[1]["score"]


def test_response_verifier_flags_document_mentions_and_unsupported_treatment_terms():
    verifier = ResponseVerifier(ollama_client=None)
    result = verifier.heuristic_verify(
        "1. Direct answer\nThe PDF says aliskiren should be adjusted.\n\n"
        "2. Rationale\nMore text.\n\n"
        "3. Caution\nI don't know.\n\n"
        "4. General advice\nMore text.",
        [RetrievedContext(source_id="1", title="HF", snippet="Review nephrotoxic drugs and diuretics")],
    )

    assert result.verdict == "fail"
    assert any("document-selection" in issue for issue in result.issues)
    assert any("unsupported treatment-specific term" in issue for issue in result.issues)
