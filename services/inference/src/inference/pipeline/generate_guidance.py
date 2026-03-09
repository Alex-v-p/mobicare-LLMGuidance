from __future__ import annotations

from inference.http.clients.ollama_client import OllamaClient
from inference.pipeline.prompts.basic import build_prompt
from inference.retrieval.dense import DenseRetriever
from shared.contracts.inference import InferenceRequest, InferenceResponse


class GuidancePipeline:
    def __init__(self, retriever: DenseRetriever | None = None, ollama_client: OllamaClient | None = None) -> None:
        self._retriever = retriever or DenseRetriever()
        self._ollama_client = ollama_client or OllamaClient()

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        retrieved_context = []
        if request.options.use_retrieval:
            retrieved_context = await self._retriever.retrieve(
                query=request.question,
                limit=request.options.top_k,
            )
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
            },
        )
