from __future__ import annotations

from inference.http.clients.ollama_client import OllamaClient
from inference.pipeline.prompts.basic import build_prompt
from inference.retrieval.fake_rag import FakeRetriever
from shared.contracts.inference import InferenceRequest, InferenceResponse


class GuidancePipeline:
    def __init__(self, retriever: FakeRetriever | None = None, ollama_client: OllamaClient | None = None) -> None:
        self._retriever = retriever or FakeRetriever()
        self._ollama_client = ollama_client or OllamaClient()

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        retrieved_context = self._retriever.retrieve(
            patient_variables=request.patient_variables,
            use_fake_rag=request.options.use_fake_rag,
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
                "fake_rag": request.options.use_fake_rag,
                "ollama_done": llm_response.done,
                "done_reason": llm_response.done_reason,
                "eval_count": llm_response.eval_count,
            },
        )
