from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.pipeline.support.drug_dosing import (
    build_drug_retrieval_queries,
    build_grounded_drug_dosing_payload,
    build_snapshot,
    render_drug_dosing_answer,
    select_grounded_rag_context,
    summarize_drug_dosing_warnings,
    verify_grounded_payload,
)
from inference.retrieval.dense import DenseRetriever
from inference.retrieval.hybrid import HybridRetriever
from shared.contracts.inference import InferenceRequest, InferenceResponse, RetrievedContext, VerificationResult


@dataclass(slots=True)
class DrugDosingPipelineDependencies:
    retriever: DenseRetriever
    hybrid_retriever: HybridRetriever
    default_embedding_model: str


class DrugDosingPipelineRunner:
    def __init__(self, dependencies: DrugDosingPipelineDependencies) -> None:
        self._deps = dependencies

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        retrieval_items, retrieval_metadata = await self._retrieve_guideline_context(request)
        payload = build_grounded_drug_dosing_payload(
            patient_variables=request.patient_variables,
            retrieved_context=retrieval_items,
            retrieval_queries=retrieval_metadata["retrieval_queries"],
            family_contexts=retrieval_metadata.get("family_contexts"),
        )
        visible_rag = select_grounded_rag_context(retrieval_items, payload)
        verdict, issues, confidence = verify_grounded_payload(payload)
        metadata = dict(retrieval_metadata)
        metadata.pop("family_contexts", None)
        return InferenceResponse(
            request_id=request.request_id,
            status="ok",
            model="drug-dosing-grounded-v1",
            answer=render_drug_dosing_answer(payload),
            retrieved_context=visible_rag,
            used_variables=request.patient_variables,
            warnings=summarize_drug_dosing_warnings(payload),
            metadata={
                "pipeline_runner": "drug_dosing",
                "drug_dosing_mode": "grounded_hybrid_evidence",
                "guideline_basis": "ESC 2021 supplementary tables 2-7",
                "retrieval_mode": request.options.retrieval_mode,
                "use_graph_augmentation": request.options.use_graph_augmentation,
                **metadata,
                "drug_dosing_payload": payload,
            },
            verification=VerificationResult(
                verdict=verdict,
                issues=issues,
                confidence=confidence,
            ),
        )

    async def _retrieve_guideline_context(self, request: InferenceRequest) -> tuple[list[RetrievedContext], dict[str, Any]]:
        query_specs = build_drug_retrieval_queries(build_snapshot(request.patient_variables))
        per_query: list[dict[str, Any]] = []
        combined: list[RetrievedContext] = list(request.retrieved_context)
        seen = {(item.source_id, item.chunk_id, item.snippet) for item in combined}
        family_contexts: dict[str, list[RetrievedContext]] = {spec["family"]: [] for spec in query_specs}
        if request.options.retrieval_mode == "dense":
            embedding_model = self._resolve_embedding_model(self._deps.retriever, request.options.embedding_model)
        else:
            embedding_model = self._resolve_embedding_model(self._deps.hybrid_retriever, request.options.embedding_model)
        per_query_limit = 3

        for query_index, spec in enumerate(query_specs, start=1):
            query = spec["query"]
            family = spec["family"]
            if request.options.retrieval_mode == "dense":
                items = await self._deps.retriever.retrieve(
                    query=query,
                    limit=per_query_limit,
                    embedding_model=embedding_model,
                )
                metadata = {
                    "query_index": query_index,
                    "family": family,
                    "query": query,
                    "retrieval_mode": "dense",
                    "returned_items": len(items),
                }
            else:
                result = await self._deps.hybrid_retriever.retrieve(
                    query=query,
                    limit=per_query_limit,
                    dense_weight=request.options.hybrid_dense_weight,
                    sparse_weight=request.options.hybrid_sparse_weight,
                    use_graph_augmentation=request.options.use_graph_augmentation,
                    graph_max_extra_nodes=request.options.graph_max_extra_nodes,
                    embedding_model=embedding_model,
                )
                items = list(result.items)
                metadata = {
                    "query_index": query_index,
                    "family": family,
                    "query": query,
                    **result.metadata,
                    "returned_items": len(items),
                }
            per_query.append(metadata)
            for item in items:
                if item not in family_contexts[family]:
                    family_contexts[family].append(item)
                identity = (item.source_id, item.chunk_id, item.snippet)
                if identity in seen:
                    continue
                seen.add(identity)
                combined.append(item)

        overall_limit = max(request.options.top_k, 18)
        combined = combined[:overall_limit]
        return combined, {
            "embedding_model": embedding_model,
            "retrieval_queries": [entry["query"] for entry in per_query],
            "retrieval_attempt_details": per_query,
            "rag_output_count": len(combined),
            "family_contexts": family_contexts,
        }

    def _resolve_embedding_model(self, retriever: Any, requested_embedding_model: str | None) -> str | None:
        resolver = getattr(retriever, "resolve_embedding_model", None)
        if callable(resolver):
            return resolver(requested_embedding_model)
        return requested_embedding_model
