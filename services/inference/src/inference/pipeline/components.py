from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from inference.clinical import ClinicalProfile, ClinicalFinding, build_clinical_profile, build_question_from_patient_data
from inference.http.clients.ollama_client import OllamaClient
from inference.pipeline.prompts.multistep import (
    DISALLOWED_SOURCE_REFERENCES,
    build_generation_prompt,
    build_query_rewrite_prompt,
    build_verification_prompt,
)
from inference.retrieval.dense import DenseRetriever
from inference.retrieval.hybrid import HybridRetriever, HybridRetrievalResult
from shared.contracts.inference import InferenceRequest, RetrievedContext, VerificationResult


class GuidanceRetriever(Protocol):
    async def retrieve_context(
        self,
        *,
        request: InferenceRequest,
        retrieval_query: str,
    ) -> tuple[list[RetrievedContext], dict[str, object]]: ...


@dataclass(slots=True)
class QueryRewriteResult:
    query: str
    rewritten: bool


@dataclass(slots=True)
class QueryPlan:
    effective_question: str
    base_query: str
    expanded_queries: list[str]
    clinical_profile: ClinicalProfile
    clusters: list[str]


@dataclass(slots=True)
class ContextAssessment:
    sufficient: bool
    confidence: str
    reasons: list[str]
    topical_terms: list[str]
    cluster_coverage: dict[str, int]


class ExampleResponseBuilder:
    def build(self, request: InferenceRequest, retrieved_context: list[RetrievedContext]) -> str:
        question = request.question.strip() or "No question was supplied."
        variable_lines = [f"- {key}: {value}" for key, value in sorted(request.patient_variables.items())]
        context_lines = [f"- {item.title}: {item.snippet}" for item in retrieved_context[:3]]

        parts = [
            "[MOCK RESPONSE]",
            f"Question: {question}",
            "",
            "This is an example guidance response generated without running retrieval or the LLM.",
        ]

        if variable_lines:
            parts.extend(["", "Patient variables:", *variable_lines])

        if context_lines:
            parts.extend(["", "Provided context:", *context_lines])

        parts.extend(
            [
                "",
                "Example guidance:",
                "Use this output for UI testing, API integration checks, or local development on low-spec hardware.",
            ]
        )
        return "\n".join(parts)


class QueryPlanner:
    def create_plan(self, request: InferenceRequest) -> QueryPlan:
        profile = build_clinical_profile(request.patient_variables)
        effective_question = request.question.strip() or build_question_from_patient_data(
            request.patient_variables,
            profile,
        )
        base_query = effective_question.strip()
        abnormal_clusters = _detected_clusters(profile)

        expanded_queries: list[str] = []
        if request.options.adaptive_retrieval_enabled:
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
                expanded_queries.append(
                    f"Clinical management or follow-up guidance for {cluster_name} with focus on {focus}."
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
        )


class QueryRewriter:
    def __init__(self, ollama_client: OllamaClient) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        return self._ollama_client.with_model(request.options.llm_model)

    async def rewrite(self, request: InferenceRequest, query: str) -> QueryRewriteResult:
        if not request.options.enable_query_rewriting:
            return QueryRewriteResult(query=query, rewritten=False)

        prompt = build_query_rewrite_prompt(query, request.patient_variables)
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


class ContextJudge:
    def assess(
        self,
        *,
        retrieved_context: list[RetrievedContext],
        retrieval_query: str,
        clinical_profile: ClinicalProfile,
        minimum_results: int,
    ) -> ContextAssessment:
        query_terms = _extract_terms(retrieval_query)
        profile_terms = {term.lower() for term in clinical_profile.relevant_terms()}
        abnormal_clusters = _detected_clusters(clinical_profile)
        coverage_hits = 0
        cluster_coverage = {cluster: 0 for cluster in abnormal_clusters}
        for item in retrieved_context:
            combined = f"{item.title} {item.snippet}".lower()
            if query_terms & _extract_terms(combined):
                coverage_hits += 1
            elif any(term in combined for term in profile_terms):
                coverage_hits += 1
            for cluster_name, findings in abnormal_clusters.items():
                if _context_matches_findings(combined, findings):
                    cluster_coverage[cluster_name] += 1

        reasons: list[str] = []
        if len(retrieved_context) < minimum_results:
            reasons.append("too_few_context_chunks")
        if coverage_hits == 0:
            reasons.append("no_clear_query_term_overlap")
        if clinical_profile.has_abnormal_variables and not profile_terms:
            reasons.append("no_abnormal_terms_available")
        if cluster_coverage and any(count == 0 for count in cluster_coverage.values()):
            reasons.append("incomplete_cluster_coverage")

        sufficient = not reasons
        confidence = "high" if sufficient and len(retrieved_context) >= max(3, minimum_results) else "medium"
        if reasons:
            confidence = "low" if len(reasons) > 1 else "medium"
        return ContextAssessment(
            sufficient=sufficient,
            confidence=confidence,
            reasons=reasons,
            topical_terms=sorted(profile_terms or query_terms)[:8],
            cluster_coverage=cluster_coverage,
        )


class ChunkRelevanceRanker:
    def rank(
        self,
        *,
        contexts: list[RetrievedContext],
        retrieval_query: str,
        clinical_profile: ClinicalProfile,
        limit: int,
    ) -> tuple[list[RetrievedContext], list[dict[str, Any]]]:
        query_terms = _extract_terms(retrieval_query)
        profile_terms = {term.lower() for term in clinical_profile.relevant_terms()}
        abnormal_clusters = _detected_clusters(clinical_profile)
        scored: list[tuple[float, RetrievedContext, dict[str, Any]]] = []
        for item in contexts:
            combined = f"{item.title} {item.snippet}".lower()
            overlap = len(query_terms & _extract_terms(combined))
            profile_overlap = sum(1 for term in profile_terms if term in combined)
            covered_clusters = [
                cluster_name
                for cluster_name, findings in abnormal_clusters.items()
                if _context_matches_findings(combined, findings)
            ]
            cluster_hits = len(covered_clusters)
            score = float(overlap) + (1.5 * float(profile_overlap)) + float(cluster_hits)
            details = {
                "chunk_id": item.chunk_id,
                "source_id": item.source_id,
                "query_term_overlap": overlap,
                "clinical_term_overlap": profile_overlap,
                "cluster_hits": cluster_hits,
                "clusters": covered_clusters,
                "score": score,
            }
            scored.append((score, item, details))
        ranked = sorted(scored, key=lambda entry: entry[0], reverse=True)
        return [item for _, item, _ in ranked[:limit]], [details for _, _, details in ranked[:limit]]


class RetrievalOrchestrator:
    def __init__(
        self,
        *,
        retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        chunk_ranker: ChunkRelevanceRanker | None = None,
        context_judge: ContextJudge | None = None,
    ) -> None:
        self._retriever = retriever
        self._hybrid_retriever = hybrid_retriever
        self._chunk_ranker = chunk_ranker or ChunkRelevanceRanker()
        self._context_judge = context_judge or ContextJudge()

    def assess_context(
        self,
        *,
        retrieved_context: list[RetrievedContext],
        retrieval_query: str,
        clinical_profile: ClinicalProfile,
        minimum_results: int,
    ) -> ContextAssessment:
        return self._context_judge.assess(
            retrieved_context=retrieved_context,
            retrieval_query=retrieval_query,
            clinical_profile=clinical_profile,
            minimum_results=minimum_results,
        )

    async def retrieve_context(
        self,
        *,
        request: InferenceRequest,
        retrieval_plan: QueryPlan,
    ) -> tuple[list[RetrievedContext], dict[str, object]]:
        retrieved_context: list[RetrievedContext] = list(request.retrieved_context)
        retrieval_metadata: dict[str, object] = {
            "retrieval_mode": request.options.retrieval_mode,
            "use_graph_augmentation": request.options.use_graph_augmentation,
            "retrieval_queries": list(retrieval_plan.expanded_queries),
            "retrieval_clusters": list(retrieval_plan.clusters),
        }
        if not request.options.use_retrieval:
            assessment = self._context_judge.assess(
                retrieved_context=retrieved_context,
                retrieval_query=retrieval_plan.base_query,
                clinical_profile=retrieval_plan.clinical_profile,
                minimum_results=request.options.retrieval_low_context_min_results,
            )
            retrieval_metadata["context_assessment"] = {
                "sufficient": assessment.sufficient,
                "confidence": assessment.confidence,
                "reasons": assessment.reasons,
                "cluster_coverage": assessment.cluster_coverage,
            }
            return retrieved_context, retrieval_metadata

        combined: list[RetrievedContext] = list(retrieved_context)
        per_query_metadata: list[dict[str, Any]] = []
        seen = {_context_key(item) for item in combined}

        for query_index, query in enumerate(retrieval_plan.expanded_queries, start=1):
            query_items, query_metadata = await self._retrieve_single_query(
                request=request,
                retrieval_query=query,
            )
            per_query_metadata.append({"query_index": query_index, "query": query, **query_metadata})
            for item in query_items:
                identity = _context_key(item)
                if identity in seen:
                    continue
                seen.add(identity)
                combined.append(item)
            assessment = self._context_judge.assess(
                retrieved_context=combined,
                retrieval_query=query,
                clinical_profile=retrieval_plan.clinical_profile,
                minimum_results=request.options.retrieval_low_context_min_results,
            )
            if assessment.sufficient or not request.options.adaptive_retrieval_enabled:
                break

        ranked_contexts, ranking_details = self._chunk_ranker.rank(
            contexts=combined,
            retrieval_query=retrieval_plan.base_query,
            clinical_profile=retrieval_plan.clinical_profile,
            limit=max(request.options.top_k, request.options.retrieval_low_context_min_results),
        )
        final_assessment = self._context_judge.assess(
            retrieved_context=ranked_contexts,
            retrieval_query=retrieval_plan.base_query,
            clinical_profile=retrieval_plan.clinical_profile,
            minimum_results=request.options.retrieval_low_context_min_results,
        )
        return ranked_contexts[: request.options.top_k], {
            **retrieval_metadata,
            "retrieval_attempts": len(per_query_metadata),
            "retrieval_attempt_details": per_query_metadata,
            "retrieval_ranking": ranking_details,
            "context_assessment": {
                "sufficient": final_assessment.sufficient,
                "confidence": final_assessment.confidence,
                "reasons": final_assessment.reasons,
                "topical_terms": final_assessment.topical_terms,
                "cluster_coverage": final_assessment.cluster_coverage,
            },
        }

    async def _retrieve_single_query(
        self,
        *,
        request: InferenceRequest,
        retrieval_query: str,
    ) -> tuple[list[RetrievedContext], dict[str, object]]:
        if request.options.retrieval_mode == "dense":
            retrieved_context = await self._retriever.retrieve(
                query=retrieval_query,
                limit=request.options.top_k,
                embedding_model=request.options.embedding_model,
            )
            return retrieved_context, {
                "retrieval_mode": "dense",
                "returned_items": len(retrieved_context),
            }

        hybrid_result: HybridRetrievalResult = await self._hybrid_retriever.retrieve(
            query=retrieval_query,
            limit=request.options.top_k,
            dense_weight=request.options.hybrid_dense_weight,
            sparse_weight=request.options.hybrid_sparse_weight,
            use_graph_augmentation=request.options.use_graph_augmentation,
            graph_max_extra_nodes=request.options.graph_max_extra_nodes,
            embedding_model=request.options.embedding_model,
        )
        return hybrid_result.items, {**hybrid_result.metadata, "returned_items": len(hybrid_result.items)}


class AnswerGenerator:
    def __init__(self, ollama_client: OllamaClient) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        return self._ollama_client.with_model(request.options.llm_model)

    async def generate(
        self,
        *,
        request: InferenceRequest,
        effective_question: str,
        clinical_profile: ClinicalProfile,
        retrieved_context: list[RetrievedContext],
        rewritten_query: str | None,
        verification_feedback: list[str] | None,
        attempt_number: int,
        context_assessment: ContextAssessment,
    ) -> tuple[str, int]:
        prompt = build_generation_prompt(
            question=effective_question,
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context,
            rewritten_query=rewritten_query,
            verification_feedback=verification_feedback,
            attempt_number=attempt_number,
            allow_general_guidance=request.options.enable_general_guidance_section,
            context_assessment=context_assessment,
        )
        llm_response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=request.options.temperature,
            max_tokens=request.options.max_tokens,
        )
        normalized_answer = _normalize_generated_answer(
            llm_response.response,
            retrieved_context=retrieved_context,
            patient_variables=request.patient_variables,
        )
        if _should_force_deterministic_answer(
            answer=normalized_answer,
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            context_assessment=context_assessment,
        ):
            normalized_answer = _build_deterministic_answer(
                question=effective_question,
                patient_variables=request.patient_variables,
                clinical_profile=clinical_profile,
                retrieved_context=retrieved_context,
                context_assessment=context_assessment,
            )
        return normalized_answer, len(prompt)


class ResponseVerifier:
    def __init__(self, ollama_client: OllamaClient | None) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        if self._ollama_client is None:
            raise RuntimeError("Response verification requested without an Ollama client.")
        return self._ollama_client.with_model(request.options.llm_model)

    def heuristic_verify(
        self,
        answer: str,
        *,
        patient_variables: dict[str, Any] | None = None,
        clinical_profile: ClinicalProfile | None = None,
        retrieved_context: list[RetrievedContext] | None = None,
    ) -> VerificationResult:
        issues = _collect_answer_issues(
            answer=answer,
            patient_variables=patient_variables or {},
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context or [],
        )
        return VerificationResult(
            verdict="fail" if issues else "pass",
            issues=issues or ["none"],
            confidence="medium" if issues else "low",
        )

    def parse(self, raw: str) -> VerificationResult | None:
        verdict_match = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", raw, flags=re.IGNORECASE)
        confidence_match = re.search(r"CONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)", raw, flags=re.IGNORECASE)
        issues_match = re.search(r"ISSUES\s*:(.*?)(?:CONFIDENCE\s*:|$)", raw, flags=re.IGNORECASE | re.DOTALL)
        if verdict_match is None:
            return None

        issues: list[str] = []
        if issues_match:
            for line in issues_match.group(1).splitlines():
                cleaned = re.sub(r"^[\s\-•*]+", "", line).strip()
                if cleaned:
                    issues.append(cleaned)
        if not issues:
            issues = ["none"]

        return VerificationResult(
            verdict=verdict_match.group(1).lower(),
            issues=issues,
            confidence=(confidence_match.group(1).lower() if confidence_match else "low"),
        )

    async def verify(
        self,
        *,
        request: InferenceRequest,
        effective_question: str,
        clinical_profile: ClinicalProfile,
        retrieved_context: list[RetrievedContext],
        answer: str,
    ) -> VerificationResult:
        heuristic = self.heuristic_verify(
            answer,
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context,
        )
        if heuristic.verdict == "fail" or not request.options.enable_response_verification or self._ollama_client is None:
            return heuristic

        prompt = build_verification_prompt(
            question=effective_question,
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context,
            answer=answer,
        )
        response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=160,
        )
        parsed = self.parse(response.response)
        if parsed is None:
            return heuristic
        if parsed.verdict == "pass" and heuristic.verdict == "fail":
            return heuristic
        if parsed.verdict == "pass":
            return parsed
        combined = sorted({*heuristic.issues, *parsed.issues})
        return VerificationResult(verdict="fail", issues=combined, confidence="medium")


def _context_key(item: RetrievedContext) -> tuple[str, str | None, str]:
    return (item.source_id, item.chunk_id, item.snippet)


def _extract_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]{3,}", text.lower()) if term not in _STOPWORDS}


_STOPWORDS = {
    "about", "according", "after", "all", "and", "are", "based", "been", "for", "from",
    "guidance", "has", "have", "into", "most", "not", "that", "the", "their", "this", "treatment",
    "what", "with", "would", "patient", "variables", "management", "follow", "relevant", "document",
    "grounded", "available", "data", "should", "could", "regarding", "focus", "clinical",
}


def _normalize_generated_answer(
    answer: str,
    *,
    retrieved_context: list[RetrievedContext],
    patient_variables: dict[str, Any],
) -> str:
    normalized = answer.strip()
    if not normalized:
        return normalized

    replacements = {
        "Evidence-based recommendation": "Direct answer",
        "Main answer": "Direct answer",
        "Document-grounded general guidance": "General advice",
        "Uncertainty and missing data": "Caution",
    }
    for old, new in replacements.items():
        normalized = re.sub(rf"\b{re.escape(old)}\b", new, normalized, flags=re.IGNORECASE)

    for item in retrieved_context:
        for token in filter(None, {item.source_id, item.title}):
            normalized = re.sub(re.escape(token), "the available evidence", normalized, flags=re.IGNORECASE)

    cleanup_patterns = [
        r"the most relevant (document|source|pdf)[^.\n]*[.\n]",
        r"the pdf says[^.\n]*",
        r"the document says[^.\n]*",
        r"the best (document|source|pdf)[^.\n]*",
        r"based on the retrieved context,?",
        r"retrieved context",
        r"this document provides",
        r"the pdf provides",
        r"the available evidence is the [^.\n]*",
        r"###\s*Direct\s*Answer",
        r"###\s*Rationale",
        r"###\s*Caution",
        r"###\s*General\s*Advice",
    ]
    for pattern in cleanup_patterns:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)

    # Contradiction guard: high potassium must not trigger hypokalemia language.
    potassium_value = _to_float(patient_variables.get("potassium"))
    if potassium_value is not None and potassium_value >= 5.0:
        normalized = re.sub(r"[^.\n]*hypokalemia[^.\n]*[.\n]?", "", normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()


def _should_force_deterministic_answer(
    *,
    answer: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    context_assessment: ContextAssessment,
) -> bool:
    issues = _collect_answer_issues(
        answer=answer,
        patient_variables=patient_variables,
        clinical_profile=clinical_profile,
        retrieved_context=[],
    )
    if issues:
        return True
    if not context_assessment.sufficient:
        return True
    abnormal_clusters = _detected_clusters(clinical_profile)
    return any(context_assessment.cluster_coverage.get(cluster, 0) == 0 for cluster in abnormal_clusters)


def _collect_answer_issues(
    *,
    answer: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile | None,
    retrieved_context: list[RetrievedContext],
) -> list[str]:
    issues: list[str] = []
    normalized = answer.strip()
    lowered = normalized.lower()
    required_sections = ["direct answer", "rationale", "caution", "general advice"]
    if not normalized:
        issues.append("Answer is empty.")
    for section in required_sections:
        if section not in lowered:
            issues.append(f"Answer is missing the '{section}' section.")
    if len(normalized.split()) < 35:
        issues.append("Answer is too short to be useful.")
    if any(term in lowered for term in DISALLOWED_SOURCE_REFERENCES):
        issues.append("Answer mentions sources or document-selection language instead of giving direct guidance.")
    if "i don't know" not in lowered and "missing" not in lowered and "uncertain" not in lowered and "limited" not in lowered:
        issues.append("Answer may not clearly communicate uncertainty.")

    potassium_value = _to_float(patient_variables.get("potassium"))
    if potassium_value is not None and potassium_value >= 5.0 and "hypokalemia" in lowered:
        issues.append("Answer contradicts the patient potassium value.")

    if clinical_profile is not None:
        for cluster_name, findings in _detected_clusters(clinical_profile).items():
            if any(finding.status in {"low", "high"} for finding in findings):
                cluster_word = cluster_name.lower().split(" and ")[0]
                mention_targets = {cluster_word, *(finding.label.lower() for finding in findings[:3])}
                if not any(target in lowered for target in mention_targets):
                    issues.append(f"Answer does not acknowledge the '{cluster_name}' abnormality cluster.")

    unsupported_terms = [
        "mra",
        "mineralocorticoid receptor antagonist",
        "aliskiren",
    ]
    if retrieved_context:
        context_text = " ".join(f"{item.title} {item.snippet}" for item in retrieved_context).lower()
        for term in unsupported_terms:
            if term in lowered and term not in context_text:
                issues.append("Answer introduces unsupported treatment-specific wording.")
                break
    return sorted(set(issues))


def _build_deterministic_answer(
    *,
    question: str,
    patient_variables: dict[str, Any],
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext],
    context_assessment: ContextAssessment,
) -> str:
    clusters = _detected_clusters(clinical_profile)
    cluster_lines: list[str] = []
    rationale_lines: list[str] = []
    for cluster_name, findings in clusters.items():
        labels = ", ".join(_finding_phrase(finding) for finding in findings[:3])
        cluster_lines.append(f"- {cluster_name}: {labels}.")
        rationale_lines.append(f"- {cluster_name}: " + "; ".join(finding.summary for finding in findings[:2]))

    evidence_points = _evidence_points(retrieved_context, patient_variables)
    if evidence_points:
        cluster_lines.extend(f"- {point}" for point in evidence_points[:2])

    missing_bits = _missing_details(patient_variables)
    caution_lines = []
    if context_assessment.reasons:
        caution_lines.append(
            "- I don't know the full conclusion because the retrieved evidence is incomplete for some abnormal findings."
        )
    uncovered = [cluster for cluster, count in context_assessment.cluster_coverage.items() if count == 0]
    if uncovered:
        caution_lines.append(f"- Evidence is limited for: {', '.join(uncovered)}.")
    caution_lines.extend(f"- Missing detail: {item}." for item in missing_bits[:3])
    if not caution_lines:
        caution_lines.append("- I don't know the full conclusion because symptoms, medication history, and baseline trends are still missing.")

    general_advice = [
        "- Review these results together with symptoms, medication history, and prior laboratory trends.",
        "- Repeat or trend abnormal values when clinically appropriate, especially renal function, potassium, sodium, and inflammatory markers.",
    ]
    if not context_assessment.sufficient:
        general_advice.append("- Do not over-interpret findings that are not well covered by the retrieved guidance.")

    rationale_block = rationale_lines[:4] if rationale_lines else ["- The answer is based on the interpreted patient findings and the available grounded excerpts."]
    lines = [
        "1. Direct answer",
        *cluster_lines[:4],
        "",
        "2. Rationale",
        *rationale_block,
        "",
        "3. Caution",
        *caution_lines,
        "",
        "4. General advice",
        *general_advice,
    ]
    return "\n".join(lines).strip()


def _evidence_points(retrieved_context: list[RetrievedContext], patient_variables: dict[str, Any]) -> list[str]:
    if not retrieved_context:
        return []
    combined = " ".join(item.snippet for item in retrieved_context).lower()
    points: list[str] = []
    if any(term in combined for term in ["creatinine", "renal function", "electrolytes"]):
        points.append("The grounded guidance supports close monitoring of renal function and electrolytes")
    if "nephrotoxic" in combined:
        points.append("The grounded guidance supports reviewing nephrotoxic medicines or other contributors to renal stress")
    if "potassium" in combined and (_to_float(patient_variables.get("potassium")) or 0) >= 5.0:
        points.append("The grounded guidance supports follow-up of elevated potassium rather than assuming it is benign")
    return points


def _missing_details(patient_variables: dict[str, Any]) -> list[str]:
    expected = ["symptoms", "medication_history", "baseline_creatinine", "prior_results", "diagnosis"]
    missing = [item.replace("_", " ") for item in expected if item not in patient_variables]
    return missing


def _finding_phrase(finding: ClinicalFinding) -> str:
    unit = f" {finding.unit}" if finding.unit else ""
    return f"{finding.label} {finding.status} ({finding.value}{unit})"


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _detected_clusters(clinical_profile: ClinicalProfile) -> dict[str, list[ClinicalFinding]]:
    clusters: dict[str, list[ClinicalFinding]] = {}
    for finding in clinical_profile.abnormal_variables:
        cluster = _cluster_for_finding(finding)
        clusters.setdefault(cluster, []).append(finding)
    return clusters


def _cluster_for_finding(finding: ClinicalFinding) -> str:
    key = finding.key.lower()
    label = finding.label.lower()
    if key in {"creatinine", "urea", "bun", "egfr", "cysc", "cystatin_c", "potassium"} or "creatin" in label:
        return "Renal function and potassium"
    if key in {"ferritin", "hemoglobin", "hb", "haemoglobin", "hematocrit", "haematocrit", "transferrin"}:
        return "Anemia and iron status"
    if key in {"crp", "hscrp", "hs_crp", "il6", "il_6", "c_reactive_protein"} or "reactive protein" in label:
        return "Inflammation"
    if key in {"sodium", "chloride", "magnesium"}:
        return "Electrolytes"
    if key in {"bnp", "nt_pro_bnp", "troponin", "hstnt", "ef", "qrs"}:
        return "Cardiac status"
    if key in {"hba1c", "glucose"}:
        return "Glycemic status"
    if key in {"cholesterol", "ldl", "hdl", "triglycerides"}:
        return "Lipids"
    return "Other findings"


def _context_matches_findings(combined: str, findings: list[ClinicalFinding]) -> bool:
    for finding in findings:
        tokens = {finding.key.lower(), finding.label.lower()}
        if any(token in combined for token in tokens):
            return True
        if finding.key.lower() == "potassium" and "renal function" in combined:
            return True
    return False
