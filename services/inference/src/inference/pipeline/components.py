from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from inference.http.clients.ollama_client import OllamaClient
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


class QueryRewriter:
    def __init__(self, ollama_client: OllamaClient) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        return self._ollama_client.with_model(request.options.llm_model)

    async def rewrite(self, request: InferenceRequest) -> QueryRewriteResult:
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


class RetrievalOrchestrator:
    def __init__(
        self,
        *,
        retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
    ) -> None:
        self._retriever = retriever
        self._hybrid_retriever = hybrid_retriever

    async def retrieve_context(
        self,
        *,
        request: InferenceRequest,
        retrieval_query: str,
    ) -> tuple[list[RetrievedContext], dict[str, object]]:
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
            return retrieved_context, retrieval_metadata

        hybrid_result: HybridRetrievalResult = await self._hybrid_retriever.retrieve(
            query=retrieval_query,
            limit=request.options.top_k,
            dense_weight=request.options.hybrid_dense_weight,
            sparse_weight=request.options.hybrid_sparse_weight,
            use_graph_augmentation=request.options.use_graph_augmentation,
            graph_max_extra_nodes=request.options.graph_max_extra_nodes,
            embedding_model=request.options.embedding_model,
        )
        return hybrid_result.items, {**retrieval_metadata, **hybrid_result.metadata}


class AnswerGenerator:
    def __init__(self, ollama_client: OllamaClient) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        return self._ollama_client.with_model(request.options.llm_model)

    async def generate(
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


class ResponseVerifier:
    def __init__(self, ollama_client: OllamaClient) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        return self._ollama_client.with_model(request.options.llm_model)

    def heuristic_verify(self, answer: str) -> VerificationResult:
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
        retrieved_context: list[RetrievedContext],
        answer: str,
    ) -> VerificationResult:
        if not request.options.enable_response_verification:
            return self.heuristic_verify(answer)

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
        return self.parse(response.response) or self.heuristic_verify(answer)
