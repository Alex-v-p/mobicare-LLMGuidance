from __future__ import annotations

from inference.http.clients.ollama_client import OllamaClient
from inference.pipeline.components import (
    AnswerGenerator,
    ExampleResponseBuilder,
    QueryRewriter,
    RetrievalOrchestrator,
    ResponseVerifier,
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
        query_rewriter: QueryRewriter | None = None,
        retrieval_orchestrator: RetrievalOrchestrator | None = None,
        answer_generator: AnswerGenerator | None = None,
        response_verifier: ResponseVerifier | None = None,
        example_response_builder: ExampleResponseBuilder | None = None,
    ) -> None:
        shared_ollama_client = ollama_client or OllamaClient()
        shared_retriever = retriever or DenseRetriever()
        shared_hybrid_retriever = hybrid_retriever or HybridRetriever()
        self._query_rewriter = query_rewriter or QueryRewriter(shared_ollama_client)
        self._retrieval_orchestrator = retrieval_orchestrator or RetrievalOrchestrator(
            retriever=shared_retriever,
            hybrid_retriever=shared_hybrid_retriever,
        )
        self._answer_generator = answer_generator or AnswerGenerator(shared_ollama_client)
        self._response_verifier = response_verifier or ResponseVerifier(shared_ollama_client)
        self._example_response_builder = example_response_builder or ExampleResponseBuilder()
        self._default_llm_model = shared_ollama_client.model
        self._default_embedding_model = shared_retriever._embedding_client.model

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        if request.options.use_example_response:
            retrieved_context = list(request.retrieved_context)
            warnings = ["Example response mode enabled; retrieval and model generation were skipped."]
            if not request.patient_variables:
                warnings.append("No patient variables were supplied.")
            if not retrieved_context:
                warnings.append("No retrieval context was supplied.")

            return InferenceResponse(
                request_id=request.request_id,
                status="ok",
                model=request.options.llm_model or "mock-guidance-v1",
                answer=self._example_response_builder.build(request, retrieved_context).strip(),
                retrieved_context=retrieved_context,
                used_variables=request.patient_variables,
                warnings=warnings,
                metadata={
                    "mock_response": True,
                    "use_retrieval": request.options.use_retrieval,
                    "retrieval_top_k": request.options.top_k,
                    "temperature": request.options.temperature,
                    "max_tokens": request.options.max_tokens,
                    "retrieval_mode": request.options.retrieval_mode,
                    "use_graph_augmentation": request.options.use_graph_augmentation,
                    "llm_model": request.options.llm_model or self._default_llm_model,
                    "embedding_model": request.options.embedding_model or self._default_embedding_model,
                },
                verification=None,
            )

        warnings: list[str] = []
        rewrite_result = await self._query_rewriter.rewrite(request)
        if rewrite_result.rewritten:
            warnings.append(f"Query rewritten for retrieval: {rewrite_result.query}")

        retrieved_context, retrieval_metadata = await self._retrieval_orchestrator.retrieve_context(
            request=request,
            retrieval_query=rewrite_result.query,
        )
        if not retrieved_context:
            warnings.append("No relevant retrieval context was found.")

        verification_feedback: list[str] | None = None
        attempt_limit = 1 + (request.options.max_regeneration_attempts if request.options.enable_regeneration else 0)
        final_answer = ""
        final_prompt_length = 0
        final_verification = None

        for attempt in range(1, attempt_limit + 1):
            final_answer, final_prompt_length = await self._answer_generator.generate(
                request=request,
                retrieved_context=retrieved_context,
                rewritten_query=rewrite_result.query if rewrite_result.rewritten else None,
                verification_feedback=verification_feedback,
                attempt_number=attempt,
            )
            final_verification = await self._response_verifier.verify(
                request=request,
                retrieved_context=retrieved_context,
                answer=final_answer,
            )
            if final_verification.verdict == "pass" or attempt == attempt_limit:
                break
            verification_feedback = final_verification.issues
            warnings.append(
                f"Answer regeneration attempt {attempt} triggered due to verification issues: "
                + "; ".join(final_verification.issues)
            )

        return InferenceResponse(
            request_id=request.request_id,
            status="ok",
            model=request.options.llm_model or self._default_llm_model,
            answer=final_answer,
            retrieved_context=retrieved_context,
            used_variables=request.patient_variables,
            warnings=warnings,
            metadata={
                "use_retrieval": request.options.use_retrieval,
                "retrieval_top_k": request.options.top_k,
                "temperature": request.options.temperature,
                "max_tokens": request.options.max_tokens,
                "query_rewritten": rewrite_result.rewritten,
                "retrieval_query": rewrite_result.query,
                "response_regeneration_attempts": attempt,
                "prompt_character_count": final_prompt_length,
                "llm_model": request.options.llm_model or self._default_llm_model,
                "embedding_model": request.options.embedding_model or self._default_embedding_model,
                **retrieval_metadata,
            },
            verification=final_verification,
        )
