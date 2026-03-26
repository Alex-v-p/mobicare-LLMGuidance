from __future__ import annotations

import re

from inference.clinical import ClinicalProfile, build_clinical_profile, build_question_from_patient_data
from inference.application.ports import ModelSelectableTextGenerationClient, TextGenerationClient
from inference.domain.guidance import detected_clusters, extract_terms, infer_specialty_focus, prioritized_clusters
from inference.application.pipelines.prompts.multistep import build_query_rewrite_prompt
from inference.application.pipelines.steps.contracts import QueryPlan, QueryRewriteResult
from shared.contracts.inference import InferenceRequest

class QueryPlanner:
    def _expand_question_only_queries(self, question: str) -> list[str]:
        lowered = question.lower()
        ignored_terms = {
            "what", "which", "when", "where", "does", "stand", "mean", "means", "described",
            "describe", "context", "supplementary", "table", "tables", "figure", "figures",
        }
        topic_terms = [term for term in extract_terms(question) if term not in ignored_terms]
        acronym_tokens = re.findall(r"\b[A-Z][A-Z0-9/+.-]{1,8}\b", question)
        topic_text = " ".join(topic_terms[:6])

        candidate_queries: list[str] = []
        if topic_text:
            candidate_queries.append(f"{topic_text} guidance")
            candidate_queries.append(f"{topic_text} definition glossary abbreviation")
        for token in acronym_tokens[:2]:
            if topic_text:
                candidate_queries.append(f"{token} abbreviation meaning definition in {topic_text}")
                candidate_queries.append(f"{topic_text} glossary {token}")
            else:
                candidate_queries.append(f"{token} abbreviation meaning definition")
        if any(marker in lowered for marker in {"table", "supplementary table", "figure", "supplementary figure"}):
            focus = acronym_tokens[0] if acronym_tokens else topic_text
            if focus:
                candidate_queries.append(f"{focus} exact wording glossary table definition")
        if any(marker in lowered for marker in {"what are", "which are", "list", "enumerate", "configurations", "types", "categories"}) and topic_text:
            candidate_queries.append(f"{topic_text} categories list configurations")

        deduped: list[str] = []
        for candidate in candidate_queries:
            normalized = candidate.strip()
            if normalized and normalized not in deduped and normalized != question.strip():
                deduped.append(normalized)
        return deduped

    def _build_effective_question(
        self,
        request: InferenceRequest,
        profile: ClinicalProfile,
        specialty_focus,
    ) -> str:
        explicit_question = request.question.strip()
        if explicit_question:
            return explicit_question

        relevant_terms = [finding.label for finding in profile.abnormal_variables[:4]]
        if specialty_focus.is_heart_failure:
            focus = ", ".join(relevant_terms[:3]) or "congestion and cardio-renal safety"
            return (
                "For this heart-failure-oriented profile, what near-term management priorities, safety checks, "
                f"and escalation considerations are most relevant, especially for {focus}?"
            )
        if specialty_focus.name == "renal":
            focus = ", ".join(relevant_terms[:3]) or "renal function and electrolytes"
            return (
                "For this renal-safety-oriented profile, what near-term monitoring priorities and treatment "
                f"safety considerations are most relevant, especially for {focus}?"
            )
        if specialty_focus.name == "metabolic":
            focus = ", ".join(relevant_terms[:3]) or "glucose control"
            return (
                "For this metabolic profile, what follow-up priorities and interpretation points are most relevant, "
                f"especially for {focus}?"
            )
        return build_question_from_patient_data(request.patient_variables, profile)

    def _build_base_query(
        self,
        request: InferenceRequest,
        profile: ClinicalProfile,
        specialty_focus,
    ) -> str:
        explicit_question = request.question.strip()
        if explicit_question:
            return explicit_question

        focus_clusters = prioritized_clusters(profile, specialty_focus, limit=2)
        focus_findings = [finding.label for finding in profile.abnormal_variables[:4]]
        if specialty_focus.is_heart_failure:
            pieces = [
                "heart failure guidance",
                *(cluster.lower() for cluster in focus_clusters),
                *(term.lower() for term in focus_findings[:4]),
                "congestion",
                "cardio-renal safety",
                "follow-up",
            ]
            return " ".join(dict.fromkeys(piece for piece in pieces if piece))
        if specialty_focus.name == "renal":
            pieces = [
                "renal function electrolyte safety guidance",
                *(cluster.lower() for cluster in focus_clusters),
                *(term.lower() for term in focus_findings[:4]),
                "monitoring",
            ]
            return " ".join(dict.fromkeys(piece for piece in pieces if piece))
        if specialty_focus.name == "metabolic":
            pieces = [
                "glycemic cardiometabolic guidance",
                *(cluster.lower() for cluster in focus_clusters),
                *(term.lower() for term in focus_findings[:4]),
                "follow-up",
            ]
            return " ".join(dict.fromkeys(piece for piece in pieces if piece))
        return build_question_from_patient_data(request.patient_variables, profile)

    def create_plan(self, request: InferenceRequest) -> QueryPlan:
        profile = build_clinical_profile(request.patient_variables)
        abnormal_clusters = detected_clusters(profile)
        specialty_focus = infer_specialty_focus(request.patient_variables, profile)
        effective_question = self._build_effective_question(request, profile, specialty_focus)
        base_query = self._build_base_query(request, profile, specialty_focus).strip()

        expanded_queries: list[str] = []
        if request.options.adaptive_retrieval_enabled:
            explicit_question = request.question.strip()
            if explicit_question and not request.patient_variables:
                expanded_queries.extend(self._expand_question_only_queries(explicit_question))

            variable_names = [key.replace("_", " ") for key in sorted(request.patient_variables.keys())]
            abnormal_terms = [finding.label for finding in profile.abnormal_variables]
            if variable_names:
                expanded_queries.append(f"{base_query} Patient variables: {', '.join(variable_names[:6])}.")
            if abnormal_terms:
                expanded_queries.append(
                    f"{base_query} Focus on abnormal or clinically relevant findings: {', '.join(abnormal_terms[:4])}."
                )
            for cluster_name, findings in list(abnormal_clusters.items())[:4]:
                focus = ", ".join(finding.label for finding in findings[:3])
                prefix = "Heart failure" if specialty_focus.is_heart_failure else "Clinical"
                expanded_queries.append(
                    f"{prefix} management or follow-up guidance for {cluster_name} with focus on {focus}."
                )
            if specialty_focus.is_heart_failure:
                focus_terms = ", ".join(finding.label for finding in profile.abnormal_variables[:4])
                expanded_queries.append(
                    f"Heart failure guidance for this patient profile, especially cardio-renal safety, congestion, and GDMT considerations for {focus_terms}."
                )
                if any(cluster in abnormal_clusters for cluster in {"HF severity and congestion", "Cardio-renal and electrolyte safety"}):
                    expanded_queries.append(
                        "Heart failure monitoring and follow-up guidance for congestion, renal function, creatinine, potassium, sodium, and diuretic or RAAS-related safety."
                    )
                if any(cluster in abnormal_clusters for cluster in {"Rhythm and conduction"}):
                    expanded_queries.append(
                        "Heart failure rhythm or conduction guidance including QRS, atrial fibrillation, heart rate, and device-related considerations."
                    )
            for finding in profile.abnormal_variables[:3]:
                expanded_queries.append(
                    f"Clinical management or follow-up guidance for {finding.label} with patient value {finding.value}."
                )

        deduped: list[str] = []
        for candidate in [base_query, *expanded_queries]:
            normalized = candidate.strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)

        return QueryPlan(
            effective_question=effective_question,
            base_query=base_query,
            expanded_queries=deduped,
            clinical_profile=profile,
            clusters=list(abnormal_clusters.keys()),
            specialty_focus=specialty_focus.name,
        )


class QueryRewriter:
    def __init__(self, ollama_client: ModelSelectableTextGenerationClient) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> TextGenerationClient:
        return self._ollama_client.with_model(request.options.llm_model)

    async def rewrite(self, request: InferenceRequest, query: str, specialty_focus: str | None = None) -> QueryRewriteResult:
        if not request.options.enable_query_rewriting:
            return QueryRewriteResult(query=query, rewritten=False)

        prompt = build_query_rewrite_prompt(query, request.patient_variables, type("Specialty", (), {"is_heart_failure": specialty_focus == "heart_failure"})())
        response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=min(96, request.options.max_tokens),
        )
        rewritten_query = response.response.strip()
        match = re.search(r"REWRITTEN_QUERY\s*:\s*(.+)", rewritten_query, flags=re.IGNORECASE | re.DOTALL)
        normalized = (match.group(1) if match else rewritten_query).strip().splitlines()[0].strip()
        if not normalized:
            return QueryRewriteResult(query=query, rewritten=False)
        return QueryRewriteResult(query=normalized, rewritten=normalized != query.strip())
