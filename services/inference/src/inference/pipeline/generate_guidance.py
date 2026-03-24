from __future__ import annotations

from inference.http.clients.ollama_client import OllamaClient
from inference.pipeline.steps import (
    AnswerGenerator,
    ExampleResponseBuilder,
    QueryPlanner,
    QueryRewriter,
    RetrievalOrchestrator,
    ResponseVerifier,
)
from inference.pipeline.runners import (
    DrugDosingPipelineDependencies,
    DrugDosingPipelineRunner,
    PipelineRunnerRegistry,
    StandardPipelineDependencies,
    StandardPipelineRunner,
)
from inference.retrieval.dense import DenseRetriever
from inference.retrieval.hybrid import HybridRetriever
from shared.contracts.inference import InferenceRequest, InferenceResponse


class GuidancePipeline:
    def __init__(
        self,
        retriever: DenseRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        ollama_client: OllamaClient | None = None,
        query_planner: QueryPlanner | None = None,
        query_rewriter: QueryRewriter | None = None,
        retrieval_orchestrator: RetrievalOrchestrator | None = None,
        answer_generator: AnswerGenerator | None = None,
        response_verifier: ResponseVerifier | None = None,
        example_response_builder: ExampleResponseBuilder | None = None,
    ) -> None:
        shared_ollama_client = ollama_client or OllamaClient()
        shared_retriever = retriever or DenseRetriever()
        shared_hybrid_retriever = hybrid_retriever or HybridRetriever()
        standard_dependencies = StandardPipelineDependencies(
            query_planner=query_planner or QueryPlanner(),
            query_rewriter=query_rewriter or QueryRewriter(shared_ollama_client),
            retrieval_orchestrator=retrieval_orchestrator or RetrievalOrchestrator(
                retriever=shared_retriever,
                hybrid_retriever=shared_hybrid_retriever,
            ),
            answer_generator=answer_generator or AnswerGenerator(shared_ollama_client),
            response_verifier=response_verifier or ResponseVerifier(shared_ollama_client),
            example_response_builder=example_response_builder or ExampleResponseBuilder(),
            default_llm_model=shared_ollama_client.model,
            default_embedding_model=shared_retriever._embedding_client.model,
        )
        drug_dependencies = DrugDosingPipelineDependencies(
            retriever=shared_retriever,
            hybrid_retriever=shared_hybrid_retriever,
            default_embedding_model=shared_retriever._embedding_client.model,
        )
        self._runner_registry = PipelineRunnerRegistry(
            standard_runner=StandardPipelineRunner(standard_dependencies),
            drug_dosing_runner=DrugDosingPipelineRunner(drug_dependencies),
        )

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        runner = self._runner_registry.resolve(request.options.pipeline_variant)
        return await runner.run(request)
