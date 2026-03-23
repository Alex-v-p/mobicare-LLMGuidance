from inference.clinical import build_clinical_profile, build_question_from_patient_data
from inference.pipeline.answer_support import build_caution_lines, build_deterministic_answer, infer_specialty_focus
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

    assert "next-step priorities" in question.lower()
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
    assert any("heart failure" in query.lower() or "hf severity and congestion" in query.lower() for query in plan.expanded_queries)
    assert "heart-failure-oriented profile" in plan.effective_question.lower()


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


def test_context_judge_allows_secondary_clusters_to_remain_uncovered_when_primary_hf_clusters_are_covered():
    judge = ContextJudge()
    profile = build_clinical_profile(
        {
            "ef": 30,
            "nt_pro_bnp": 2400,
            "creatinine": 1.8,
            "potassium": 5.4,
            "glucose": 155,
            "hba1c": 7.2,
        }
    )
    assessment = judge.assess(
        retrieved_context=[
            RetrievedContext(
                source_id="a",
                title="Heart failure",
                snippet="Heart failure follow-up should monitor congestion, creatinine, potassium, and renal function.",
            ),
            RetrievedContext(
                source_id="b",
                title="Cardio-renal safety",
                snippet="Review diuretic and RAAS-related safety when creatinine or potassium worsen.",
            ),
        ],
        retrieval_query="heart failure guidance congestion creatinine potassium follow-up",
        clinical_profile=profile,
        minimum_results=2,
    )

    assert assessment.sufficient is True
    assert assessment.cluster_coverage["Glycemic and cardiometabolic risk"] == 0


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


def test_infer_specialty_focus_prefers_heart_failure_when_cardiac_variables_dominate():
    profile = build_clinical_profile({"ef": 28, "nt_pro_bnp": 2400, "creatinine": 1.7, "potassium": 5.4})

    focus = infer_specialty_focus({"ef": 28, "nt_pro_bnp": 2400, "creatinine": 1.7, "potassium": 5.4}, profile)

    assert focus.name == "heart_failure"


def test_query_planner_marks_heart_failure_specialty_focus_for_cardiac_cases():
    planner = QueryPlanner()
    request = InferenceRequest(
        request_id="req-hf",
        question="",
        patient_variables={"ef": 28, "nt_pro_bnp": 2400, "creatinine": 1.7, "potassium": 5.4},
        options=GenerationOptions(adaptive_retrieval_enabled=True),
    )

    plan = planner.create_plan(request)

    assert plan.specialty_focus == "heart_failure"
    assert any("heart failure" in query.lower() for query in plan.expanded_queries)


def test_build_caution_lines_uses_proxy_fields_and_aggregates_missing_context():
    profile = build_clinical_profile(
        {
            "ef": 28,
            "nt_pro_bnp": 2400,
            "creatinine": 1.7,
            "potassium": 5.4,
            "bpsyst": 102,
            "bpdiast": 68,
            "edema": 1,
            "dosebb_prev": 1,
        }
    )
    specialty = infer_specialty_focus(
        {
            "ef": 28,
            "nt_pro_bnp": 2400,
            "creatinine": 1.7,
            "potassium": 5.4,
            "bpsyst": 102,
            "bpdiast": 68,
            "edema": 1,
            "dosebb_prev": 1,
        },
        profile,
    )
    context_assessment = type(
        "Assessment",
        (),
        {
            "reasons": ["incomplete_cluster_coverage"],
            "cluster_coverage": {
                "HF severity and congestion": 1,
                "Cardio-renal and electrolyte safety": 1,
                "Glycemic and cardiometabolic risk": 0,
            },
        },
    )()

    caution = build_caution_lines(
        {
            "ef": 28,
            "nt_pro_bnp": 2400,
            "creatinine": 1.7,
            "potassium": 5.4,
            "bpsyst": 102,
            "bpdiast": 68,
            "edema": 1,
            "dosebb_prev": 1,
        },
        context_assessment,
        profile,
        specialty,
    )

    caution_text = " ".join(caution).lower()
    assert "blood pressure" not in caution_text
    assert "volume status" not in caution_text
    assert "medications" not in caution_text
    assert "key missing context" in caution_text


def test_response_verifier_flags_literal_question_answers_that_fall_back_to_generic_clinical_summary():
    verifier = ResponseVerifier(ollama_client=None)
    question = "What are the four circuit configurations for percutaneous mechanical circulatory support as described in Supplementary Table 22?"
    retrieved = [
        RetrievedContext(
            source_id="1",
            title="Supplementary Table 22",
            snippet="Percutaneous mechanical circulatory supports can be characterized by one of four circuit configurations: 1. intra-aortic devices, 2. transvalvular aortic (Impella) 3. left atrium to systemic artery (TandemHeart); 4. right atrium to systemic artery (veno-arterial extracorporeal memb",
            chunk_id="c1",
        )
    ]

    result = verifier.heuristic_verify(
        "1. Direct answer\n- The main value abnormalities point to a clinically relevant pattern that should be interpreted as a whole rather than marker by marker.\n- Use the retrieved guidance to prioritize the most abnormal findings first, while keeping uncertainty explicit where evidence is thin.\n\n2. Rationale\n- The main value abnormalities point to a clinically relevant pattern that should be interpreted as a whole rather than marker by marker.\n\n3. Caution\n- Key missing context that could change the recommendation: current symptoms, medication history, prior laboratory trends.\n\n4. General advice\n- Review these results together with symptoms, medication history, and prior laboratory trends.",
        question=question,
        retrieved_context=retrieved,
    )

    assert result.verdict == "fail"
    assert any("literal question" in issue.lower() for issue in result.issues)


def test_build_deterministic_answer_for_literal_question_extracts_enumerated_items_from_context():
    question = "What are the four circuit configurations for percutaneous mechanical circulatory support as described in Supplementary Table 22?"
    retrieved = [
        RetrievedContext(
            source_id="1",
            title="Supplementary Table 22",
            snippet="Percutaneous mechanical circulatory supports can be characterized by one of four circuit configurations: 1. intra-aortic devices, 2. transvalvular aortic (Impella) 3. left atrium to systemic artery (TandemHeart); 4. right atrium to systemic artery (veno-arterial extracorporeal memb",
            chunk_id="c1",
        )
    ]
    profile = build_clinical_profile({})
    context_assessment = type("Assessment", (), {"sufficient": True})()

    answer = build_deterministic_answer(
        question=question,
        patient_variables={},
        clinical_profile=profile,
        retrieved_context=retrieved,
        context_assessment=context_assessment,
    )

    lowered = answer.lower()
    assert "intra-aortic devices" in lowered
    assert "transvalvular aortic" in lowered
    assert "tandemheart" in lowered
    assert "right atrium to systemic artery" in lowered
