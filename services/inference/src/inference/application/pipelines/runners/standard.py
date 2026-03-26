from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.application.pipelines.steps import (
    AnswerGenerator,
    ExampleResponseBuilder,
    QueryPlanner,
    QueryRewriter,
    RetrievalOrchestrator,
    ResponseVerifier,
)
from inference.domain.guidance import build_deterministic_answer, looks_like_generic_clinical_fallback
from inference.application.pipelines.steps.contracts import QueryPlan
from shared.contracts.inference import InferenceRequest, InferenceResponse


@dataclass(slots=True)
class StandardPipelineDependencies:
    query_planner: QueryPlanner
    query_rewriter: QueryRewriter
    retrieval_orchestrator: RetrievalOrchestrator
    answer_generator: AnswerGenerator
    response_verifier: ResponseVerifier
    example_response_builder: ExampleResponseBuilder
    default_llm_model: str
    default_embedding_model: str


class StandardPipelineRunner:
    def __init__(self, dependencies: StandardPipelineDependencies) -> None:
        self._deps = dependencies

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        query_plan = self._deps.query_planner.create_plan(request)
        if request.options.use_example_response:
            return self._build_example_response(request, query_plan)

        warnings = self._build_initial_warnings(request, query_plan)
        rewrite_result = await self._deps.query_rewriter.rewrite(request, query_plan.base_query, query_plan.specialty_focus)
        if rewrite_result.rewritten:
            warnings.append(f"Query rewritten for retrieval: {rewrite_result.query}")
            query_plan.expanded_queries[0] = rewrite_result.query

        retrieved_context, retrieval_metadata = await self._deps.retrieval_orchestrator.retrieve_context(
            request=request,
            retrieval_plan=query_plan,
        )
        self._append_retrieval_warnings(warnings, retrieved_context, retrieval_metadata)

        final_answer, final_verification, prompt_length, attempts = await self._generate_verified_answer(
            request=request,
            query_plan=query_plan,
            rewrite_query=(rewrite_result.query if rewrite_result.rewritten else None),
            retrieved_context=retrieved_context,
            warnings=warnings,
        )

        return InferenceResponse(
            request_id=request.request_id,
            status="ok",
            model=request.options.llm_model or self._deps.default_llm_model,
            answer=final_answer,
            retrieved_context=retrieved_context,
            used_variables=request.patient_variables,
            warnings=warnings,
            metadata={
                **self._base_metadata(request),
                "pipeline_runner": "standard",
                "query_rewritten": rewrite_result.rewritten,
                "retrieval_query": rewrite_result.query,
                "effective_question": query_plan.effective_question,
                "clinical_abnormal_variables": [finding.label for finding in query_plan.clinical_profile.abnormal_variables],
                "clinical_unknown_variables": query_plan.clinical_profile.unknown_variables,
                "response_regeneration_attempts": attempts,
                "specialty_focus": query_plan.specialty_focus,
                "prompt_character_count": prompt_length,
                **retrieval_metadata,
            },
            verification=final_verification,
        )

    def _build_example_response(self, request: InferenceRequest, query_plan: QueryPlan) -> InferenceResponse:
        retrieved_context = list(request.retrieved_context)
        warnings = ["Example response mode enabled; retrieval and model generation were skipped."]
        if not request.patient_variables:
            warnings.append("No patient variables were supplied.")
        if not retrieved_context:
            warnings.append("No retrieval context was supplied.")
        if not request.question.strip():
            warnings.append("No explicit question was supplied; a patient-data-driven task was inferred.")

        metadata = {
            **self._base_metadata(request),
            "pipeline_runner": "standard",
            "mock_response": True,
            "effective_question": query_plan.effective_question,
            "clinical_abnormal_variables": [finding.label for finding in query_plan.clinical_profile.abnormal_variables],
        }
        return InferenceResponse(
            request_id=request.request_id,
            status="ok",
            model=request.options.llm_model or "mock-guidance-v1",
            answer=self._deps.example_response_builder.build(request, retrieved_context).strip(),
            retrieved_context=retrieved_context,
            used_variables=request.patient_variables,
            warnings=warnings,
            metadata=metadata,
            verification=None,
        )

    def _base_metadata(self, request: InferenceRequest) -> dict[str, object]:
        effective_embedding_model = request.options.embedding_model or self._deps.default_embedding_model
        return {
            "use_retrieval": request.options.use_retrieval,
            "retrieval_top_k": request.options.top_k,
            "temperature": request.options.temperature,
            "max_tokens": request.options.max_tokens,
            "retrieval_mode": request.options.retrieval_mode,
            "use_graph_augmentation": request.options.use_graph_augmentation,
            "llm_model": request.options.llm_model or self._deps.default_llm_model,
            "embedding_model": effective_embedding_model,
        }

    def _build_initial_warnings(self, request: InferenceRequest, query_plan: QueryPlan) -> list[str]:
        warnings: list[str] = []
        if query_plan.clinical_profile.abnormal_variables:
            abnormal_names = ", ".join(finding.label for finding in query_plan.clinical_profile.abnormal_variables[:5])
            warnings.append(f"Adaptive retrieval focused on clinically relevant findings: {abnormal_names}")
        if not request.question.strip() and request.patient_variables:
            warnings.append("No explicit question was supplied; a patient-data-driven task was inferred.")
        return warnings

    def _append_retrieval_warnings(
        self,
        warnings: list[str],
        retrieved_context,
        retrieval_metadata: dict[str, object],
    ) -> None:
        context_assessment_meta = retrieval_metadata.get("context_assessment") or {}
        context_sufficient = bool(context_assessment_meta.get("sufficient"))
        if not retrieved_context:
            warnings.append("No relevant retrieval context was found.")
        elif not context_sufficient:
            warnings.append("Retrieved context appears weak or incomplete; answer should be treated cautiously.")

    async def _generate_verified_answer(
        self,
        *,
        request: InferenceRequest,
        query_plan: QueryPlan,
        rewrite_query: str | None,
        retrieved_context,
        warnings: list[str],
    ) -> tuple[str, object, int, int]:
        verification_feedback: list[str] | None = None
        attempt_limit = 1 + (request.options.max_regeneration_attempts if request.options.enable_regeneration else 0)
        final_answer = ""
        final_prompt_length = 0
        final_verification = None
        for attempt in range(1, attempt_limit + 1):
            context_assessment = self._deps.retrieval_orchestrator.assess_context(
                retrieved_context=retrieved_context,
                retrieval_query=query_plan.base_query,
                clinical_profile=query_plan.clinical_profile,
                minimum_results=request.options.retrieval_low_context_min_results,
            )
            final_answer, final_prompt_length = await self._deps.answer_generator.generate(
                request=request,
                effective_question=query_plan.effective_question,
                clinical_profile=query_plan.clinical_profile,
                retrieved_context=retrieved_context,
                rewritten_query=rewrite_query,
                verification_feedback=verification_feedback,
                attempt_number=attempt,
                context_assessment=context_assessment,
            )
            final_verification = await self._deps.response_verifier.verify(
                request=request,
                effective_question=query_plan.effective_question,
                clinical_profile=query_plan.clinical_profile,
                retrieved_context=retrieved_context,
                answer=final_answer,
            )
            if final_verification.verdict == "pass":
                return final_answer, final_verification, final_prompt_length, attempt
            if attempt == attempt_limit:
                fallback_answer = build_deterministic_answer(
                    question=query_plan.effective_question,
                    patient_variables=request.patient_variables,
                    clinical_profile=query_plan.clinical_profile,
                    retrieved_context=retrieved_context,
                    context_assessment=context_assessment,
                    prefer_unknown_fallback=(
                        request.options.enable_unknown_fallback
                        and looks_like_generic_clinical_fallback(final_answer)
                    ),
                )
                fallback_verification = self._deps.response_verifier.heuristic_verify(
                    fallback_answer,
                    question=query_plan.effective_question,
                    patient_variables=request.patient_variables,
                    clinical_profile=query_plan.clinical_profile,
                    retrieved_context=retrieved_context,
                )
                return fallback_answer, fallback_verification, final_prompt_length, attempt

            verification_feedback = final_verification.issues
            warnings.append(
                f"Answer regeneration attempt {attempt} triggered due to verification issues: "
                + "; ".join(final_verification.issues)
            )

        return final_answer, final_verification, final_prompt_length, attempt_limit
