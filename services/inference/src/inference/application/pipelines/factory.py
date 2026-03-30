from __future__ import annotations

from typing import Any

from inference.application.ports import ModelSelectableTextGenerationClient
from inference.infrastructure.http.clients.ollama_client import OllamaClient
from inference.application.pipelines.guidance_pipeline import GuidancePipeline
from inference.application.pipelines.runners import (
    DrugDosingPipelineDependencies,
    DrugDosingPipelineRunner,
    PipelineRunnerRegistry,
    StandardPipelineDependencies,
    StandardPipelineRunner,
)
from inference.application.pipelines.steps import (
    AnswerGenerator,
    ExampleResponseBuilder,
    QueryPlanner,
    QueryRewriter,
    RetrievalOrchestrator,
    ResponseVerifier,
)
from inference.retrieval.dense import DenseRetriever
from inference.retrieval.hybrid import HybridRetriever


def build_guidance_pipeline(
    *,
    retriever: DenseRetriever | None = None,
    hybrid_retriever: HybridRetriever | None = None,
    ollama_client: ModelSelectableTextGenerationClient | None = None,
    query_planner: QueryPlanner | None = None,
    query_rewriter: QueryRewriter | None = None,
    retrieval_orchestrator: RetrievalOrchestrator | None = None,
    answer_generator: AnswerGenerator | None = None,
    response_verifier: ResponseVerifier | None = None,
    example_response_builder: ExampleResponseBuilder | None = None,
) -> GuidancePipeline:
    return GuidancePipeline(
        runner_registry=build_pipeline_runner_registry(
            retriever=retriever,
            hybrid_retriever=hybrid_retriever,
            ollama_client=ollama_client,
            query_planner=query_planner,
            query_rewriter=query_rewriter,
            retrieval_orchestrator=retrieval_orchestrator,
            answer_generator=answer_generator,
            response_verifier=response_verifier,
            example_response_builder=example_response_builder,
        )
    )


def build_pipeline_runner_registry(
    *,
    retriever: DenseRetriever | None = None,
    hybrid_retriever: HybridRetriever | None = None,
    ollama_client: ModelSelectableTextGenerationClient | None = None,
    query_planner: QueryPlanner | None = None,
    query_rewriter: QueryRewriter | None = None,
    retrieval_orchestrator: RetrievalOrchestrator | None = None,
    answer_generator: AnswerGenerator | None = None,
    response_verifier: ResponseVerifier | None = None,
    example_response_builder: ExampleResponseBuilder | None = None,
) -> PipelineRunnerRegistry:
    shared_ollama_client = ollama_client or OllamaClient()
    shared_retriever = retriever or DenseRetriever()
    shared_hybrid_retriever = hybrid_retriever or HybridRetriever()
    default_embedding_model = _resolve_default_embedding_model(shared_retriever, shared_hybrid_retriever)

    standard_dependencies = StandardPipelineDependencies(
        query_planner=query_planner or QueryPlanner(),
        query_rewriter=query_rewriter or QueryRewriter(shared_ollama_client),
        retrieval_orchestrator=retrieval_orchestrator
        or RetrievalOrchestrator(
            retriever=shared_retriever,
            hybrid_retriever=shared_hybrid_retriever,
        ),
        answer_generator=answer_generator or AnswerGenerator(shared_ollama_client),
        response_verifier=response_verifier or ResponseVerifier(shared_ollama_client),
        example_response_builder=example_response_builder or ExampleResponseBuilder(),
        default_llm_model=shared_ollama_client.model,
        default_embedding_model=default_embedding_model,
    )
    drug_dependencies = DrugDosingPipelineDependencies(
        retriever=shared_retriever,
        hybrid_retriever=shared_hybrid_retriever,
        default_embedding_model=default_embedding_model,
    )
    return PipelineRunnerRegistry(
        standard_runner=StandardPipelineRunner(standard_dependencies),
        drug_dosing_runner=DrugDosingPipelineRunner(drug_dependencies),
    )


def _resolve_default_embedding_model(*retrievers: Any) -> str:
    for retriever in retrievers:
        default_model_resolver = getattr(retriever, "get_default_embedding_model", None)
        if callable(default_model_resolver):
            resolved = default_model_resolver()
            if resolved:
                return str(resolved)

        embedding_client = getattr(retriever, "_embedding_client", None)
        resolved = getattr(embedding_client, "model", None)
        if resolved:
            return str(resolved)

        settings = getattr(retriever, "_settings", None)
        resolved = getattr(settings, "ollama_embedding_model", None)
        if resolved:
            return str(resolved)

    raise ValueError("Unable to resolve a default embedding model for guidance pipeline construction")
