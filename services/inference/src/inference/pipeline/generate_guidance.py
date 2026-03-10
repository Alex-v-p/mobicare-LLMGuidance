from __future__ import annotations

import re
from dataclasses import dataclass

from inference.http.clients.ollama_client import OllamaClient
from inference.pipeline.prompts.multistep import (
    build_generation_prompt,
    build_query_rewrite_prompt,
    build_verification_prompt,
)
from inference.retrieval.dense import DenseRetriever
from inference.retrieval.hybrid import HybridRetriever
from shared.contracts.inference import InferenceRequest, InferenceResponse, RetrievedContext, VerificationResult


@dataclass(slots=True)
class QueryRewriteResult:
    query: str
    rewritten: bool


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

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        return self._ollama_client.with_model(request.options.llm_model)

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

    async def _rewrite_query(self, request: InferenceRequest) -> QueryRewriteResult:
        if not request.options.enable_query_rewriting:
            return QueryRewriteResult(query=request.question, rewritten=False)

        prompt = build_query_rewrite_prompt(request.question, request.patient_variables)
        response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=min(96, request.options.max_tokens),
        )
        rewritten_query = response.response.strip()
        match = re.search(r"REWRITTEN_QUERY\s*:\s*(.+)", rewritten_query, flags=re.IGNORECASE | re.DOTALL)
        normalized = (match.group(1) if match else rewritten_query).strip().splitlines()[0].strip()
        if not normalized:
            return QueryRewriteResult(query=request.question, rewritten=False)
        return QueryRewriteResult(query=normalized, rewritten=normalized != request.question.strip())

    async def _retrieve_context(self, request: InferenceRequest, retrieval_query: str) -> tuple[list[RetrievedContext], dict[str, object]]:
        retrieved_context: list[RetrievedContext] = list(request.retrieved_context)
        retrieval_metadata: dict[str, object] = {
            "retrieval_mode": request.options.retrieval_mode,
            "use_graph_augmentation": request.options.use_graph_augmentation,
        }
        if not request.options.use_retrieval:
            return retrieved_context, retrieval_metadata

        if request.options.retrieval_mode == "dense":
            retrieved_context = await self._retriever.retrieve(
                query=retrieval_query,
                limit=request.options.top_k,
                embedding_model=request.options.embedding_model,
            )
        else:
            hybrid_result = await self._hybrid_retriever.retrieve(
                query=retrieval_query,
                limit=request.options.top_k,
                dense_weight=request.options.hybrid_dense_weight,
                sparse_weight=request.options.hybrid_sparse_weight,
                use_graph_augmentation=request.options.use_graph_augmentation,
                graph_max_extra_nodes=request.options.graph_max_extra_nodes,
                embedding_model=request.options.embedding_model,
            )
            retrieved_context = hybrid_result.items
            retrieval_metadata.update(hybrid_result.metadata)
        return retrieved_context, retrieval_metadata

    async def _generate_answer(
        self,
        *,
        request: InferenceRequest,
        retrieved_context: list[RetrievedContext],
        rewritten_query: str | None,
        verification_feedback: list[str] | None,
        attempt_number: int,
    ) -> tuple[str, int]:
        prompt = build_generation_prompt(
            question=request.question,
            patient_variables=request.patient_variables,
            retrieved_context=retrieved_context,
            rewritten_query=rewritten_query,
            verification_feedback=verification_feedback,
            attempt_number=attempt_number,
        )
        llm_response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=request.options.temperature,
            max_tokens=request.options.max_tokens,
        )
        return llm_response.response.strip(), len(prompt)

    def _heuristic_verify(self, answer: str) -> VerificationResult:
        issues: list[str] = []
        normalized = answer.strip()
        lowered = normalized.lower()
        if not normalized:
            issues.append("Answer is empty.")
        if "1." not in normalized or "2." not in normalized or "3." not in normalized:
            issues.append("Answer does not follow the required 3-part structure.")
        if len(normalized.split()) < 25:
            issues.append("Answer is too short to be useful.")
        if "direct answer" not in lowered and "rationale" not in lowered and "caution" not in lowered:
            issues.append("Answer may not clearly label or separate its parts.")
        return VerificationResult(
            verdict="fail" if issues else "pass",
            issues=issues or ["none"],
            confidence="medium" if issues else "low",
        )

    async def _verify_answer(
        self,
        *,
        request: InferenceRequest,
        retrieved_context: list[RetrievedContext],
        answer: str,
    ) -> VerificationResult:
        if not request.options.enable_response_verification:
            return self._heuristic_verify(answer)

        prompt = build_verification_prompt(
            question=request.question,
            patient_variables=request.patient_variables,
            retrieved_context=retrieved_context,
            answer=answer,
        )
        response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=160,
        )
        return self._parse_verification_response(response.response) or self._heuristic_verify(answer)

    def _parse_verification_response(self, raw: str) -> VerificationResult | None:
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
                    "llm_model": request.options.llm_model or self._ollama_client.model,
                    "embedding_model": request.options.embedding_model or self._retriever._embedding_client.model,
                },
                verification=None,
            )

        warnings: list[str] = []
        rewrite_result = await self._rewrite_query(request)
        retrieved_context, retrieval_metadata = await self._retrieve_context(request, rewrite_result.query)

        answer = ""
        verification = VerificationResult(verdict="pass", issues=["none"], confidence="low")
        prompt_chars = 0
        attempts_used = 0
        max_attempts = 1 + (request.options.max_regeneration_attempts if request.options.enable_regeneration else 0)
        feedback: list[str] | None = None

        for attempt_number in range(1, max_attempts + 1):
            attempts_used = attempt_number
            answer, prompt_chars = await self._generate_answer(
                request=request,
                retrieved_context=retrieved_context,
                rewritten_query=rewrite_result.query if rewrite_result.rewritten else None,
                verification_feedback=feedback,
                attempt_number=attempt_number,
            )
            verification = await self._verify_answer(
                request=request,
                retrieved_context=retrieved_context,
                answer=answer,
            )
            if verification.verdict == "pass":
                break
            if attempt_number >= max_attempts:
                warnings.append("Response verification failed after the final generation attempt.")
                break
            feedback = [issue for issue in verification.issues if issue.lower() != "none"]
            warnings.append(f"Regenerating answer after verification failed on attempt {attempt_number}.")

        if not request.patient_variables:
            warnings.append("No patient variables were supplied.")
        if not retrieved_context:
            warnings.append("No retrieval context was supplied.")

        return InferenceResponse(
            request_id=request.request_id,
            status="ok",
            model=self._get_llm_client(request).model,
            answer=answer,
            retrieved_context=retrieved_context,
            used_variables=request.patient_variables,
            warnings=warnings,
            metadata={
                "prompt_chars": prompt_chars,
                "use_retrieval": request.options.use_retrieval,
                "retrieval_top_k": request.options.top_k,
                "mock_response": False,
                "hybrid_dense_weight": request.options.hybrid_dense_weight,
                "hybrid_sparse_weight": request.options.hybrid_sparse_weight,
                "graph_max_extra_nodes": request.options.graph_max_extra_nodes,
                "query_rewritten": rewrite_result.rewritten,
                "retrieval_query": rewrite_result.query,
                "generation_attempts": attempts_used,
                "regeneration_enabled": request.options.enable_regeneration,
                "response_verification_enabled": request.options.enable_response_verification,
                "llm_model": self._get_llm_client(request).model,
                "embedding_model": request.options.embedding_model or self._retriever._embedding_client.model,
                **retrieval_metadata,
            },
            verification=verification,
        )
