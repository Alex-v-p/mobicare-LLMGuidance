from __future__ import annotations

import re
from typing import Any

from inference.clinical import ClinicalProfile
from inference.http.clients.ollama_client import OllamaClient
from inference.pipeline.support import (
    build_deterministic_answer,
    collect_answer_issues,
    infer_specialty_focus,
    is_literal_question_mode,
    is_minimal_unknown_fallback_answer,
    looks_like_generic_clinical_fallback,
    normalize_generated_answer,
    should_force_deterministic_answer,
    synthesize_clinical_state,
)
from inference.pipeline.prompts.multistep import build_generation_prompt, build_verification_prompt
from inference.pipeline.steps.contracts import ContextAssessment
from shared.contracts.inference import InferenceRequest, RetrievedContext, VerificationResult

class AnswerGenerator:
    def __init__(self, ollama_client: OllamaClient) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        return self._ollama_client.with_model(request.options.llm_model)

    async def generate(
        self,
        *,
        request: InferenceRequest,
        effective_question: str,
        clinical_profile: ClinicalProfile,
        retrieved_context: list[RetrievedContext],
        rewritten_query: str | None,
        verification_feedback: list[str] | None,
        attempt_number: int,
        context_assessment: ContextAssessment,
    ) -> tuple[str, int]:
        specialty_focus = infer_specialty_focus(request.patient_variables, clinical_profile, retrieved_context)
        clinical_synthesis = synthesize_clinical_state(
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context,
            context_assessment=context_assessment,
            specialty=specialty_focus,
        )
        prompt = build_generation_prompt(
            question=effective_question,
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context,
            clinical_synthesis=clinical_synthesis,
            actionable_reasoning=list(clinical_synthesis.action_points),
            rewritten_query=rewritten_query,
            verification_feedback=verification_feedback,
            attempt_number=attempt_number,
            allow_general_guidance=request.options.enable_general_guidance_section,
            context_assessment=context_assessment,
            specialty_focus=specialty_focus,
            literal_question_mode=is_literal_question_mode(effective_question, request.patient_variables, clinical_profile),
        )
        llm_response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=request.options.temperature,
            max_tokens=request.options.max_tokens,
        )
        normalized_answer = normalize_generated_answer(
            llm_response.response,
            retrieved_context=retrieved_context,
            patient_variables=request.patient_variables,
        )
        if should_force_deterministic_answer(
            answer=normalized_answer,
            question=effective_question,
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context,
            context_assessment=context_assessment,
        ) and not is_minimal_unknown_fallback_answer(normalized_answer):
            normalized_answer = build_deterministic_answer(
                question=effective_question,
                patient_variables=request.patient_variables,
                clinical_profile=clinical_profile,
                retrieved_context=retrieved_context,
                context_assessment=context_assessment,
                prefer_unknown_fallback=(
                    request.options.enable_unknown_fallback
                    and looks_like_generic_clinical_fallback(normalized_answer)
                ),
            )
        return normalized_answer, len(prompt)


class ResponseVerifier:
    def __init__(self, ollama_client: OllamaClient | None) -> None:
        self._ollama_client = ollama_client

    def _get_llm_client(self, request: InferenceRequest) -> OllamaClient:
        if self._ollama_client is None:
            raise RuntimeError("Response verification requested without an Ollama client.")
        return self._ollama_client.with_model(request.options.llm_model)

    def heuristic_verify(
        self,
        answer: str,
        *,
        question: str = "",
        patient_variables: dict[str, Any] | None = None,
        clinical_profile: ClinicalProfile | None = None,
        retrieved_context: list[RetrievedContext] | None = None,
    ) -> VerificationResult:
        issues = collect_answer_issues(
            answer=answer,
            question=question,
            patient_variables=patient_variables or {},
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context or [],
        )
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
        effective_question: str,
        clinical_profile: ClinicalProfile,
        retrieved_context: list[RetrievedContext],
        answer: str,
    ) -> VerificationResult:
        heuristic = self.heuristic_verify(
            answer,
            question=effective_question,
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context,
        )
        if heuristic.verdict == "fail" or not request.options.enable_response_verification or self._ollama_client is None:
            return heuristic

        prompt = build_verification_prompt(
            question=effective_question,
            patient_variables=request.patient_variables,
            clinical_profile=clinical_profile,
            retrieved_context=retrieved_context,
            answer=answer,
        )
        response = await self._get_llm_client(request).generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=160,
        )
        parsed = self.parse(response.response)
        if parsed is None:
            return heuristic
        if parsed.verdict == "pass" and heuristic.verdict == "fail":
            return heuristic
        if parsed.verdict == "pass":
            return parsed
        combined = sorted({*heuristic.issues, *parsed.issues})
        return VerificationResult(verdict="fail", issues=combined, confidence="medium")
