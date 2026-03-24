from __future__ import annotations

from typing import Any

from inference.clinical import ClinicalProfile
from inference.pipeline.support import (
    context_key,
    context_matches_findings,
    detected_clusters,
    extract_terms,
    infer_specialty_focus,
    prioritized_clusters,
)
from inference.pipeline.steps.contracts import ContextAssessment, QueryPlan
from inference.retrieval.dense import DenseRetriever
from inference.retrieval.hybrid import HybridRetriever, HybridRetrievalResult
from shared.contracts.inference import InferenceRequest, RetrievedContext

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
        specialty_focus = infer_specialty_focus({}, clinical_profile, retrieved_context)
        focus_clusters = prioritized_clusters(clinical_profile, specialty_focus, limit=2)
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
        if focus_clusters and any(cluster_coverage.get(cluster, 0) == 0 for cluster in focus_clusters):
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
            score, item, _ = entry
            if context_key(item) in seen_ids:
                continue
            if score <= 0.0 and selected:
                continue
            selected.append(entry)
            if len(selected) >= limit:
                break
        if not selected and ranked:
            selected.append(ranked[0])
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

        output_limit = max(
            request.options.top_k,
            request.options.retrieval_low_context_min_results,
            len(retrieval_plan.clusters),
        )
        ranked_contexts, ranking_details = self._chunk_ranker.rank(
            contexts=combined,
            retrieval_query=retrieval_plan.base_query,
            clinical_profile=retrieval_plan.clinical_profile,
            limit=output_limit,
        )
        final_assessment = self._context_judge.assess(
            retrieved_context=ranked_contexts,
            retrieval_query=retrieval_plan.base_query,
            clinical_profile=retrieval_plan.clinical_profile,
            minimum_results=request.options.retrieval_low_context_min_results,
        )
        return ranked_contexts[:output_limit], {
            **retrieval_metadata,
            "rag_output_count": min(len(ranked_contexts), output_limit),
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
