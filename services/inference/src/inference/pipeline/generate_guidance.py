from __future__ import annotations

from inference.http.clients.ollama_client import OllamaClient
from inference.pipeline.prompts.basic import build_prompt
from inference.retrieval.dense import DenseRetriever
from inference.retrieval.hybrid import HybridRetriever
from shared.contracts.inference import InferenceRequest, InferenceResponse, RetrievedContext


class GuidancePipeline:
    def __init__(
        self,
        retriever: DenseRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        self._retriever = retriever or DenseRetriever()
        self._hybrid_retriever = hybrid_retriever or HybridRetriever()
        self._ollama_client = ollama_client or OllamaClient()

    def _build_mock_answer(self, request: InferenceRequest, retrieved_context: list[RetrievedContext]) -> str:
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
                model="mock-guidance-v1",
                answer=self._build_mock_answer(request, retrieved_context).strip(),
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
                },
            )

        retrieved_context: list[RetrievedContext] = []
        retrieval_metadata: dict[str, object] = {
            "retrieval_mode": request.options.retrieval_mode,
            "use_graph_augmentation": request.options.use_graph_augmentation,
        }
        if request.options.use_retrieval:
            if request.options.retrieval_mode == "dense":
                retrieved_context = await self._retriever.retrieve(
                    query=request.question,
                    limit=request.options.top_k,
                )
            else:
                hybrid_result = await self._hybrid_retriever.retrieve(
                    query=request.question,
                    limit=request.options.top_k,
                    dense_weight=request.options.hybrid_dense_weight,
                    sparse_weight=request.options.hybrid_sparse_weight,
                    use_graph_augmentation=request.options.use_graph_augmentation,
                    graph_max_extra_nodes=request.options.graph_max_extra_nodes,
                )
                retrieved_context = hybrid_result.items
                retrieval_metadata.update(hybrid_result.metadata)

        prompt = build_prompt(
            question=request.question,
            patient_variables=request.patient_variables,
            retrieved_context=retrieved_context,
        )
        llm_response = await self._ollama_client.generate(
            prompt=prompt,
            temperature=request.options.temperature,
            max_tokens=request.options.max_tokens,
        )

        warnings = []
        if not request.patient_variables:
            warnings.append("No patient variables were supplied.")
        if not retrieved_context:
            warnings.append("No retrieval context was supplied.")

        return InferenceResponse(
            request_id=request.request_id,
            status="ok",
            model=self._ollama_client.model,
            answer=llm_response.response.strip(),
            retrieved_context=retrieved_context,
            used_variables=request.patient_variables,
            warnings=warnings,
            metadata={
                "prompt_chars": len(prompt),
                "use_retrieval": request.options.use_retrieval,
                "retrieval_top_k": request.options.top_k,
                "ollama_done": llm_response.done,
                "done_reason": llm_response.done_reason,
                "eval_count": llm_response.eval_count,
                "mock_response": False,
                "hybrid_dense_weight": request.options.hybrid_dense_weight,
                "hybrid_sparse_weight": request.options.hybrid_sparse_weight,
                "graph_max_extra_nodes": request.options.graph_max_extra_nodes,
                **retrieval_metadata,
            },
        )
