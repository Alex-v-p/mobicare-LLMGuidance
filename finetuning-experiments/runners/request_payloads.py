from __future__ import annotations

from typing import Any

from configs.schema import BenchmarkRunConfig
from datasets.schema import BenchmarkCase


def _resolve_case_pipeline_variant(case: BenchmarkCase, config: BenchmarkRunConfig) -> str:
    generation_metadata = case.generation_metadata or {}
    request_options = generation_metadata.get("request_options") or {}
    return str(
        request_options.get("pipeline_variant")
        or generation_metadata.get("pipeline_variant")
        or config.inference.pipeline_variant
        or "standard"
    )


def _resolve_request_question(case: BenchmarkCase) -> str:
    generation_metadata = case.generation_metadata or {}
    return "" if generation_metadata.get("omit_question_from_request") else case.question


def build_guidance_payload(case: BenchmarkCase, config: BenchmarkRunConfig) -> dict[str, Any]:
    return {
        "question": _resolve_request_question(case),
        "patient": {"values": case.patient_variables or {}},
        "options": {
            "use_retrieval": True,
            "top_k": config.inference.top_k,
            "temperature": config.inference.temperature,
            "max_tokens": config.inference.max_tokens,
            "use_example_response": config.inference.use_example_response,
            "retrieval_mode": config.inference.retrieval_mode,
            "hybrid_dense_weight": config.inference.hybrid_dense_weight,
            "hybrid_sparse_weight": config.inference.hybrid_sparse_weight,
            "use_graph_augmentation": config.inference.use_graph_augmentation,
            "graph_max_extra_nodes": config.inference.graph_max_extra_nodes,
            "enable_query_rewriting": config.inference.enable_query_rewriting,
            "enable_response_verification": config.inference.enable_response_verification,
            "enable_regeneration": config.inference.enable_regeneration,
            "max_regeneration_attempts": config.inference.max_regeneration_attempts,
            "llm_model": config.inference.llm_model,
            "embedding_model": config.inference.embedding_model,
            "pipeline_variant": _resolve_case_pipeline_variant(case, config),
        },
    }
