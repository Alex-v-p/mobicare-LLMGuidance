from __future__ import annotations

from typing import Any

from configs.schema import BenchmarkRunConfig
from utils.hashing import fingerprint


def build_ingestion_fingerprint(config: BenchmarkRunConfig) -> str:
    payload: dict[str, Any] = {
        "documents_version": config.documents_version,
        "cleaning_strategy": config.ingestion.cleaning_strategy,
        "cleaning_params": config.ingestion.cleaning_params,
        "chunking_strategy": config.ingestion.chunking_strategy,
        "chunking_params": config.ingestion.chunking_params,
        "embedding_model": config.ingestion.embedding_model,
    }
    return fingerprint(payload, prefix="ingestion")



def build_run_fingerprint(config: BenchmarkRunConfig) -> str:
    payload: dict[str, Any] = {
        "dataset_version": config.dataset_version,
        "dataset_path": config.dataset_path,
        "documents_version": config.documents_version,
        "ingestion_fingerprint": build_ingestion_fingerprint(config),
        "inference": {
            "top_k": config.inference.top_k,
            "temperature": config.inference.temperature,
            "max_tokens": config.inference.max_tokens,
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
            "prompt_engineering_label": config.inference.prompt_engineering_label,
            "use_example_response": config.inference.use_example_response,
            "pipeline_variant": config.inference.pipeline_variant,
        },
        "source_mapping": {
            "max_matches": config.source_mapping.max_matches,
            "page_window": config.source_mapping.page_window,
            "page_offset_candidates": config.source_mapping.page_offset_candidates,
            "semantic_fallback_enabled": config.source_mapping.semantic_fallback_enabled,
            "include_chunk_pairs": config.source_mapping.include_chunk_pairs,
            "llm_second_pass_enabled": config.source_mapping.llm_second_pass_enabled,
            "max_soft_candidates": config.source_mapping.max_soft_candidates,
            "mapping_profile": config.source_mapping.mapping_profile,
            "llm_labeling_profile": config.source_mapping.llm_labeling_profile,
            "max_sequence_length": config.source_mapping.max_sequence_length,
            "semantic_fallback_max_matches": config.source_mapping.semantic_fallback_max_matches,
        },
        "execution": {
            "include_unanswerable": config.execution.include_unanswerable,
            "max_cases": config.execution.max_cases,
        },
    }
    return fingerprint(payload, prefix="run")
