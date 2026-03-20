from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from inference.clinical import ClinicalProfile, build_clinical_profile, build_question_from_patient_data
from inference.http.clients.ollama_client import OllamaClient
from inference.pipeline.answer_support import (
    collect_answer_issues,
    context_key,
    context_matches_findings,
    detected_clusters,
    extract_terms,
    normalize_generated_answer,
    should_force_deterministic_answer,
    build_deterministic_answer,
    infer_specialty_focus,
)
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
    clusters: list[str]
    specialty_focus: str


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
        abnormal_clusters = detected_clusters(profile)
        specialty_focus = infer_specialty_focus(request.patient_variables, profile)

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
    def __init__(self, ollama_client: OllamaClient) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
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


class ContextJudge:
    def assess(
        self,
        *,
        retrieved_context: list[RetrievedContext],
        retrieval_query: str,
        clinical_profile: ClinicalProfile,
        minimum_results: int,
    ) -> ContextAssessment:
        query_terms = extract_terms(retrieval_query)
        profile_terms = {term.lower() for term in clinical_profile.relevant_terms()}
        abnormal_clusters = detected_clusters(clinical_profile)
        coverage_hits = 0
        cluster_coverage = {cluster: 0 for cluster in abnormal_clusters}
        for item in retrieved_context:
            combined = f"{item.title} {item.snippet}".lower()
            if query_terms & extract_terms(combined):
                coverage_hits += 1
            elif any(term in combined for term in profile_terms):
                coverage_hits += 1
            for cluster_name, findings in abnormal_clusters.items():
                if context_matches_findings(combined, findings):
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
        query_terms = extract_terms(retrieval_query)
        profile_terms = {term.lower() for term in clinical_profile.relevant_terms()}
        abnormal_clusters = detected_clusters(clinical_profile)
        specialty_focus = infer_specialty_focus({}, clinical_profile, contexts)
        scored: list[tuple[float, RetrievedContext, dict[str, Any]]] = []
        for item in contexts:
            combined = f"{item.title} {item.snippet}".lower()
            overlap = len(query_terms & extract_terms(combined))
            profile_overlap = sum(1 for term in profile_terms if term in combined)
            covered_clusters = [
                cluster_name
                for cluster_name, findings in abnormal_clusters.items()
                if context_matches_findings(combined, findings)
            ]
            cluster_hits = len(covered_clusters)
            heart_failure_overlap = int(
                specialty_focus.is_heart_failure
                and any(term in combined for term in {"heart failure", "hf", "hfr ef", "hfref", "congestion", "diuretic", "raas", "arni", "mra", "sglt2"})
            )
            score = float(overlap) + (1.5 * float(profile_overlap)) + float(cluster_hits) + (1.25 * float(heart_failure_overlap))
            details = {
                "chunk_id": item.chunk_id,
                "source_id": item.source_id,
                "query_term_overlap": overlap,
                "clinical_term_overlap": profile_overlap,
                "cluster_hits": cluster_hits,
                "clusters": covered_clusters,
                "heart_failure_overlap": heart_failure_overlap,
                "score": score,
            }
            scored.append((score, item, details))

        ranked = sorted(scored, key=lambda entry: entry[0], reverse=True)
        selected: list[tuple[float, RetrievedContext, dict[str, Any]]] = []
        seen_ids: set[tuple[str, str | None, str]] = set()
        covered_once: set[str] = set()
        for entry in ranked:
            _, item, details = entry
            if any(cluster not in covered_once for cluster in details["clusters"]):
                selected.append(entry)
                seen_ids.add(context_key(item))
                covered_once.update(details["clusters"])
            if len(selected) >= limit:
                break
        for entry in ranked:
            _, item, _ = entry
            if context_key(item) in seen_ids:
                continue
            selected.append(entry)
            if len(selected) >= limit:
                break
        return [item for _, item, _ in selected[:limit]], [details for _, _, details in selected[:limit]]


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
            "specialty_focus": retrieval_plan.specialty_focus,
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
        seen = {context_key(item) for item in combined}

        for query_index, query in enumerate(retrieval_plan.expanded_queries, start=1):
            query_items, query_metadata = await self._retrieve_single_query(
                request=request,
                retrieval_query=query,
            )
            per_query_metadata.append({"query_index": query_index, "query": query, **query_metadata})
            for item in query_items:
                identity = context_key(item)
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
            specialty_focus=infer_specialty_focus(request.patient_variables, clinical_profile, retrieved_context),
        )
        llm_response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=request.options.temperature,
            max_tokens=request.options.max_tokens,
        )
        normalized_answer = normalize_generated_answer(
            llm_response.response,
            retrieved_context=retrieved_context,
            patient_variables=request.patient_variables,
        )
        if should_force_deterministic_answer(
            answer=normalized_answer,
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            context_assessment=context_assessment,
        ):
            normalized_answer = build_deterministic_answer(
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
        issues = collect_answer_issues(
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
