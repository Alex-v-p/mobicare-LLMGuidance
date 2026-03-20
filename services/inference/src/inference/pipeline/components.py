from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from inference.clinical import ClinicalProfile, build_clinical_profile, build_question_from_patient_data
from inference.http.clients.ollama_client import OllamaClient
from inference.pipeline.prompts.multistep import (
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


@dataclass(slots=True)
class ContextAssessment:
    sufficient: bool
    confidence: str
    reasons: list[str]
    topical_terms: list[str]


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
        coverage_hits = 0
        for item in retrieved_context:
            combined = f"{item.title} {item.snippet}".lower()
            if query_terms & _extract_terms(combined):
                coverage_hits += 1
            elif any(term in combined for term in profile_terms):
                coverage_hits += 1

        reasons: list[str] = []
        if len(retrieved_context) < minimum_results:
            reasons.append("too_few_context_chunks")
        if coverage_hits == 0:
            reasons.append("no_clear_query_term_overlap")
        if clinical_profile.has_abnormal_variables and not profile_terms:
            reasons.append("no_abnormal_terms_available")

        sufficient = not reasons
        confidence = "high" if sufficient and len(retrieved_context) >= max(3, minimum_results) else "medium"
        if reasons:
            confidence = "low"
        return ContextAssessment(
            sufficient=sufficient,
            confidence=confidence,
            reasons=reasons,
            topical_terms=sorted(profile_terms or query_terms)[:8],
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
        scored: list[tuple[float, RetrievedContext, dict[str, Any]]] = []
        for item in contexts:
            combined = f"{item.title} {item.snippet}".lower()
            overlap = len(query_terms & _extract_terms(combined))
            profile_overlap = sum(1 for term in profile_terms if term in combined)
            score = float(overlap) + (1.5 * float(profile_overlap))
            details = {
                "chunk_id": item.chunk_id,
                "source_id": item.source_id,
                "query_term_overlap": overlap,
                "clinical_term_overlap": profile_overlap,
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
        return llm_response.response.strip(), len(prompt)


class ResponseVerifier:
    def __init__(self, ollama_client: OllamaClient) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        return self._ollama_client.with_model(request.options.llm_model)

    def heuristic_verify(self, answer: str) -> VerificationResult:
        issues: list[str] = []
        normalized = answer.strip()
        lowered = normalized.lower()
        required_sections = ["evidence-based recommendation", "document-grounded general guidance", "uncertainty and missing data"]
        if not normalized:
            issues.append("Answer is empty.")
        for section in required_sections:
            if section not in lowered:
                issues.append(f"Answer is missing the '{section}' section.")
        if len(normalized.split()) < 40:
            issues.append("Answer is too short to be useful.")
        if "insufficient" not in lowered and "uncertain" not in lowered and "missing" not in lowered:
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
    ) -> VerificationResult:
        if not request.options.enable_response_verification:
            return self.heuristic_verify(answer)

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
        return self.parse(response.response) or self.heuristic_verify(answer)


def _context_key(item: RetrievedContext) -> tuple[str, str | None, str]:
    return (item.source_id, item.chunk_id, item.snippet)


def _extract_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]{3,}", text.lower()) if term not in _STOPWORDS}


_STOPWORDS = {
    "about", "according", "after", "all", "and", "are", "based", "been", "for", "from",
    "guidance", "has", "have", "into", "most", "not", "that", "the", "their", "this", "treatment",
    "what", "with", "would", "patient", "variables", "management", "follow", "relevant", "document",
    "grounded", "available", "data", "should", "could", "regarding", "focus",
}
