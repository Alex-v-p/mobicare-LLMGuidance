from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from inference.clinical import ClinicalFinding, ClinicalProfile, build_clinical_profile, build_question_from_patient_data
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
class AbnormalityCluster:
    key: str
    label: str
    findings: list[ClinicalFinding]
    terms: list[str]
    query: str


@dataclass(slots=True)
class QueryPlan:
    effective_question: str
    base_query: str
    expanded_queries: list[str]
    clinical_profile: ClinicalProfile
    clusters: list[AbnormalityCluster]


@dataclass(slots=True)
class ContextAssessment:
    sufficient: bool
    confidence: str
    reasons: list[str]
    topical_terms: list[str]
    cluster_coverage: dict[str, int] | None = None


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
        clusters = _build_abnormality_clusters(profile)

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
            for cluster in clusters:
                expanded_queries.append(cluster.query)
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
            clusters=clusters,
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
        clusters: list[AbnormalityCluster] | None = None,
    ) -> ContextAssessment:
        query_terms = _extract_terms(retrieval_query)
        profile_terms = {term.lower() for term in clinical_profile.relevant_terms()}
        combined_texts = [f"{item.title} {item.snippet}".lower() for item in retrieved_context]
        coverage_hits = 0
        for combined in combined_texts:
            if query_terms & _extract_terms(combined):
                coverage_hits += 1
            elif any(term in combined for term in profile_terms):
                coverage_hits += 1

        cluster_coverage: dict[str, int] = {}
        if clusters:
            for cluster in clusters:
                cluster_coverage[cluster.label] = sum(
                    1 for combined in combined_texts if any(term in combined for term in cluster.terms)
                )

        reasons: list[str] = []
        if len(retrieved_context) < minimum_results:
            reasons.append("too_few_context_chunks")
        if coverage_hits == 0:
            reasons.append("no_clear_query_term_overlap")
        if clinical_profile.has_abnormal_variables and not profile_terms:
            reasons.append("no_abnormal_terms_available")
        if cluster_coverage and any(hits == 0 for hits in cluster_coverage.values()):
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
            cluster_coverage=cluster_coverage or None,
        )


class ChunkRelevanceRanker:
    def rank(
        self,
        *,
        contexts: list[RetrievedContext],
        retrieval_query: str,
        clinical_profile: ClinicalProfile,
        limit: int,
        clusters: list[AbnormalityCluster] | None = None,
    ) -> tuple[list[RetrievedContext], list[dict[str, Any]]]:
        query_terms = _extract_terms(retrieval_query)
        profile_terms = {term.lower() for term in clinical_profile.relevant_terms()}
        scored: list[tuple[float, RetrievedContext, dict[str, Any]]] = []
        for item in contexts:
            combined = f"{item.title} {item.snippet}".lower()
            overlap = len(query_terms & _extract_terms(combined))
            profile_overlap = sum(1 for term in profile_terms if term in combined)
            cluster_hits = 0
            cluster_labels: list[str] = []
            for cluster in clusters or []:
                if any(term in combined for term in cluster.terms):
                    cluster_hits += 1
                    cluster_labels.append(cluster.label)
            score = float(overlap) + (1.5 * float(profile_overlap)) + (1.25 * float(cluster_hits))
            details = {
                "chunk_id": item.chunk_id,
                "source_id": item.source_id,
                "query_term_overlap": overlap,
                "clinical_term_overlap": profile_overlap,
                "cluster_hits": cluster_hits,
                "clusters": cluster_labels,
                "score": score,
            }
            scored.append((score, item, details))
        ranked = sorted(scored, key=lambda entry: entry[0], reverse=True)

        selected: list[tuple[float, RetrievedContext, dict[str, Any]]] = []
        seen_ids: set[tuple[str, str | None, str]] = set()
        for cluster in clusters or []:
            for score, item, details in ranked:
                identity = _context_key(item)
                if identity in seen_ids:
                    continue
                if cluster.label in details["clusters"]:
                    selected.append((score, item, details))
                    seen_ids.add(identity)
                    break
        for score, item, details in ranked:
            identity = _context_key(item)
            if identity in seen_ids:
                continue
            selected.append((score, item, details))
            seen_ids.add(identity)
            if len(selected) >= limit:
                break
        selected = selected[:limit]
        return [item for _, item, _ in selected], [details for _, _, details in selected]


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
        clusters: list[AbnormalityCluster] | None = None,
    ) -> ContextAssessment:
        return self._context_judge.assess(
            retrieved_context=retrieved_context,
            retrieval_query=retrieval_query,
            clinical_profile=clinical_profile,
            minimum_results=minimum_results,
            clusters=clusters,
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
            "retrieval_clusters": [cluster.label for cluster in retrieval_plan.clusters],
        }
        if not request.options.use_retrieval:
            assessment = self._context_judge.assess(
                retrieved_context=retrieved_context,
                retrieval_query=retrieval_plan.base_query,
                clinical_profile=retrieval_plan.clinical_profile,
                minimum_results=request.options.retrieval_low_context_min_results,
                clusters=retrieval_plan.clusters,
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

        ranked_contexts, ranking_details = self._chunk_ranker.rank(
            contexts=combined,
            retrieval_query=retrieval_plan.base_query,
            clinical_profile=retrieval_plan.clinical_profile,
            limit=max(request.options.top_k, request.options.retrieval_low_context_min_results),
            clusters=retrieval_plan.clusters,
        )
        final_assessment = self._context_judge.assess(
            retrieved_context=ranked_contexts,
            retrieval_query=retrieval_plan.base_query,
            clinical_profile=retrieval_plan.clinical_profile,
            minimum_results=request.options.retrieval_low_context_min_results,
            clusters=retrieval_plan.clusters,
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
        clusters: list[AbnormalityCluster],
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
            clusters=clusters,
        )
        llm_response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=request.options.temperature,
            max_tokens=request.options.max_tokens,
        )
        normalized_answer = _normalize_generated_answer(
            llm_response.response,
            retrieved_context=retrieved_context,
            clinical_profile=clinical_profile,
            context_assessment=context_assessment,
            clusters=clusters,
        )
        return normalized_answer, len(prompt)


class ResponseVerifier:
    def __init__(self, ollama_client: OllamaClient | None) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        assert self._ollama_client is not None
        return self._ollama_client.with_model(request.options.llm_model)

    def heuristic_verify(self, answer: str, retrieved_context: list[RetrievedContext] | None = None) -> VerificationResult:
        issues: list[str] = []
        normalized = answer.strip()
        lowered = normalized.lower()
        required_sections = ["direct answer", "rationale", "caution", "general advice"]
        if not normalized:
            issues.append("Answer is empty.")
        for section in required_sections:
            if section not in lowered:
                issues.append(f"Answer is missing the '{section}' section.")
        if len(normalized.split()) < 45:
            issues.append("Answer is too short to be useful.")
        if any(term in lowered for term in DISALLOWED_SOURCE_REFERENCES):
            issues.append("Answer mentions sources or document-selection language instead of giving direct guidance.")
        if "main answer" in lowered or "general guidance" in lowered or "uncertainty and missing data" in lowered:
            issues.append("Answer still contains legacy section labels instead of the required final structure.")
        if normalized.lower().count("1. direct answer") > 1:
            issues.append("Answer structure is malformed or duplicated.")
        issues.extend(_unsupported_treatment_specific_issues(normalized, retrieved_context or []))
        if "i don't know" not in lowered and "uncertain" not in lowered and "missing" not in lowered:
            issues.append("Answer may not clearly communicate uncertainty.")
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
        clusters: list[AbnormalityCluster],
    ) -> VerificationResult:
        if not request.options.enable_response_verification or self._ollama_client is None:
            return self.heuristic_verify(answer, retrieved_context)

        prompt = build_verification_prompt(
            question=effective_question,
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context,
            answer=answer,
            clusters=clusters,
        )
        response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=160,
        )
        return self.parse(response.response) or self.heuristic_verify(answer, retrieved_context)


def _build_abnormality_clusters(profile: ClinicalProfile) -> list[AbnormalityCluster]:
    grouped: dict[str, list[ClinicalFinding]] = {}
    labels = {
        "renal": "Renal function and potassium",
        "anemia": "Anemia and iron status",
        "inflammation": "Inflammation",
        "electrolytes": "Electrolytes",
        "cardiac": "Cardiac status",
        "glycemic": "Glycemic control",
        "lipids": "Lipids",
        "other": "Other abnormalities",
    }
    for finding in profile.abnormal_variables:
        group = _cluster_key_for_finding(finding)
        grouped.setdefault(group, []).append(finding)

    clusters: list[AbnormalityCluster] = []
    for key, findings in grouped.items():
        terms = list({_sanitize_term(f.label) for f in findings} | set(_cluster_terms_for_key(key)))
        focus = ", ".join(f.label for f in findings[:3])
        query = f"Clinical management or follow-up guidance for {labels.get(key, key)} with focus on {focus}."
        clusters.append(
            AbnormalityCluster(
                key=key,
                label=labels.get(key, key.title()),
                findings=findings,
                terms=[term.lower() for term in terms if term],
                query=query,
            )
        )
    return clusters


def _cluster_key_for_finding(finding: ClinicalFinding) -> str:
    key = finding.key.lower()
    if key in {"creatinine", "urea", "bun", "egfr", "cysc", "cystatin_c", "potassium"}:
        return "renal"
    if key in {"hemoglobin", "hb", "hematocrit", "ferritin", "transferrin", "tsat", "iron"}:
        return "anemia"
    if key in {"crp", "hscrp", "c_reactive_protein", "il6", "il_6"}:
        return "inflammation"
    if key in {"sodium", "chloride", "magnesium", "calcium"}:
        return "electrolytes"
    if key in {"bnp", "nt_pro_bnp", "hs_tnt", "troponin", "ef", "ejection_fraction", "qrs"}:
        return "cardiac"
    if key in {"hba1c", "glucose"}:
        return "glycemic"
    if key in {"cholesterol", "ldl", "hdl", "triglycerides"}:
        return "lipids"
    return "other"


def _cluster_terms_for_key(key: str) -> list[str]:
    return {
        "renal": ["creatinine", "potassium", "renal", "kidney", "hyperkalaemia", "hyperkalemia", "diuretic", "nephrotoxic"],
        "anemia": ["hemoglobin", "ferritin", "iron", "anemia", "transferrin"],
        "inflammation": ["crp", "c-reactive protein", "inflammation", "il-6"],
        "electrolytes": ["sodium", "electrolyte", "hyponatremia", "potassium"],
        "cardiac": ["heart failure", "bnp", "nt-pro-bnp", "ejection fraction", "troponin"],
        "glycemic": ["glucose", "hba1c", "diabetes", "glycemic"],
        "lipids": ["cholesterol", "ldl", "hdl", "lipid"],
        "other": [],
    }.get(key, [])


def _context_key(item: RetrievedContext) -> tuple[str, str | None, str]:
    return (item.source_id, item.chunk_id, item.snippet)


def _extract_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]{3,}", text.lower()) if term not in _STOPWORDS}


_STOPWORDS = {
    "about", "according", "after", "all", "and", "are", "based", "been", "for", "from",
    "guidance", "has", "have", "into", "most", "not", "that", "the", "their", "this", "treatment",
    "what", "with", "would", "patient", "variables", "management", "follow", "relevant", "document",
    "grounded", "available", "data", "should", "could", "regarding", "focus", "clinical", "main",
    "answer", "direct", "general", "advice", "rationale", "caution",
}

_MEDICATION_TERMS = [
    "aliskiren", "arb", "ace-i", "ace inhibitor", "arni", "spironolactone", "eplerenone", "amiloride",
    "triamterene", "nsaid", "nsaids", "diuretic", "beta blocker", "sglt2", "sacubitril", "valsartan",
]


def _normalize_generated_answer(
    answer: str,
    *,
    retrieved_context: list[RetrievedContext],
    clinical_profile: ClinicalProfile,
    context_assessment: ContextAssessment,
    clusters: list[AbnormalityCluster],
) -> str:
    normalized = answer.strip()
    if not normalized:
        return _build_fallback_answer(clinical_profile, retrieved_context, context_assessment, clusters)

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
    ]
    for pattern in cleanup_patterns:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)

    normalized = _remove_unsupported_treatment_specifics(normalized, retrieved_context)
    normalized = _coerce_section_structure(normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized).strip()
    verification = ResponseVerifier(None).heuristic_verify(normalized, retrieved_context)
    if verification.verdict == "fail":
        return _build_fallback_answer(clinical_profile, retrieved_context, context_assessment, clusters)
    return normalized


def _coerce_section_structure(answer: str) -> str:
    headings = [
        ("1. Direct answer", ["1. direct answer", "direct answer"]),
        ("2. Rationale", ["2. rationale", "rationale"]),
        ("3. Caution", ["3. caution", "caution", "uncertainty and missing data"]),
        ("4. General advice", ["4. general advice", "general advice", "general guidance"]),
    ]
    lowered = answer.lower()
    if all(any(alias in lowered for alias in aliases) for _, aliases in headings):
        return answer
    body = answer.strip()
    body = re.sub(r"(?im)^\s*\d+\.\s*(main answer|general guidance|general advice|uncertainty and missing data|caution|rationale)\s*$", "", body)
    body = re.sub(r"(?im)^\s*(main answer|general guidance|general advice|uncertainty and missing data|caution|rationale)\s*$", "", body)
    body = re.sub(r"\n{2,}", "\n", body).strip()
    return (
        "1. Direct answer\n"
        f"{body or '- Direct evidence-based answer could not be extracted from the model output.'}\n\n"
        "2. Rationale\n"
        "- The answer above should be grounded in the retrieved evidence and interpreted patient findings.\n\n"
        "3. Caution\n"
        "- I don't know the full conclusion because key clinical details may be missing.\n\n"
        "4. General advice\n"
        "- Review the patient values together with symptoms, medication history, and prior results."
    )


def _unsupported_treatment_specific_issues(answer: str, retrieved_context: list[RetrievedContext]) -> list[str]:
    lowered_answer = answer.lower()
    context_text = " ".join(f"{item.title} {item.snippet}" for item in retrieved_context).lower()
    issues: list[str] = []
    for term in _MEDICATION_TERMS:
        if term in lowered_answer and term not in context_text:
            issues.append(f"Answer introduces unsupported treatment-specific term: {term}.")
    return issues


def _remove_unsupported_treatment_specifics(answer: str, retrieved_context: list[RetrievedContext]) -> str:
    context_text = " ".join(f"{item.title} {item.snippet}" for item in retrieved_context).lower()
    parts = re.split(r"(?<=[.!?])\s+", answer)
    kept: list[str] = []
    for part in parts:
        lowered = part.lower()
        if any(term in lowered and term not in context_text for term in _MEDICATION_TERMS):
            continue
        kept.append(part)
    return " ".join(kept).strip()


def _build_fallback_answer(
    clinical_profile: ClinicalProfile,
    retrieved_context: list[RetrievedContext],
    context_assessment: ContextAssessment,
    clusters: list[AbnormalityCluster],
) -> str:
    direct_lines: list[str] = []
    rationale_lines: list[str] = []
    caution_lines: list[str] = []
    general_lines: list[str] = []

    for cluster in clusters:
        snippets = _context_snippets_for_cluster(cluster, retrieved_context)
        if snippets:
            direct_lines.extend(_direct_lines_for_cluster(cluster))
            rationale_lines.append(
                f"- Current evidence is most clearly aligned with {cluster.label.lower()}: {', '.join(f.label for f in cluster.findings[:3])}."
            )
        else:
            caution_lines.append(
                f"- I don't know the specific next step for {cluster.label.lower()} from the current retrieved evidence."
            )

    if not direct_lines:
        if clinical_profile.abnormal_variables:
            direct_lines.append(
                "- Several abnormal findings need follow-up, but the current retrieved evidence is too limited to support a specific management recommendation."
            )
        else:
            direct_lines.append("- No clear abnormal finding was identified from the supplied variables.")

    if clinical_profile.abnormal_variables:
        summarized = ", ".join(f"{f.label} ({f.status})" for f in clinical_profile.abnormal_variables[:6])
        rationale_lines.insert(0, f"- Interpreted abnormal findings include: {summarized}.")
    else:
        rationale_lines.insert(0, "- No interpreted abnormal biomarkers were available to drive focused guidance.")

    if context_assessment.confidence != "high":
        caution_lines.insert(0, "- The retrieved evidence is incomplete, so the answer should be treated cautiously.")
    caution_lines.append("- I don't know the full clinical conclusion without symptoms, medication list, diagnosis, baseline values, and trend over time.")
    caution_lines.append("- Avoid inferring a specific diagnosis or medication change from these values alone.")

    if clusters:
        general_lines.append("- Recheck abnormal results against baseline values and the current medication or supplement list.")
        general_lines.append("- Prioritize the most urgent abnormalities first and correlate them with symptoms and examination findings.")
    else:
        general_lines.append("- Review the values together with symptoms, medical history, medications, and previous measurements.")

    return (
        "1. Direct answer\n" + "\n".join(_dedupe_lines(direct_lines)[:4]) + "\n\n"
        "2. Rationale\n" + "\n".join(_dedupe_lines(rationale_lines)[:4]) + "\n\n"
        "3. Caution\n" + "\n".join(_dedupe_lines(caution_lines)[:4]) + "\n\n"
        "4. General advice\n" + "\n".join(_dedupe_lines(general_lines)[:3])
    )


def _context_snippets_for_cluster(cluster: AbnormalityCluster, retrieved_context: list[RetrievedContext]) -> list[str]:
    snippets: list[str] = []
    for item in retrieved_context:
        combined = f"{item.title} {item.snippet}".lower()
        if any(term in combined for term in cluster.terms):
            snippets.append(item.snippet.strip())
    return snippets[:2]


def _direct_lines_for_cluster(cluster: AbnormalityCluster) -> list[str]:
    if cluster.key == "renal":
        return [
            "- Elevated renal-function markers and potassium warrant close monitoring and medication review.",
            "- Review nephrotoxic drugs, potassium-raising agents, and whether diuretic adjustment is appropriate if congestion is absent.",
        ]
    if cluster.key == "anemia":
        return [
            "- Low hemoglobin and iron-related markers need follow-up because they are compatible with anemia or iron deficiency.",
        ]
    if cluster.key == "inflammation":
        return [
            "- Inflammatory markers are elevated, so an underlying inflammatory or infectious driver should be considered in the clinical context.",
        ]
    if cluster.key == "electrolytes":
        return [
            "- Electrolyte abnormalities should be rechecked and interpreted with symptoms, fluid status, and current treatment.",
        ]
    return [
        f"- The abnormal findings in {cluster.label.lower()} need follow-up, but the exact next step depends on the broader clinical picture.",
    ]


def _sanitize_term(label: str) -> str:
    return re.sub(r"[^a-z0-9\- ]+", "", label.lower())


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(line)
    return result
