from __future__ import annotations

from pathlib import Path

from .schema import BenchmarkRunConfig

_ALLOWED_RETRIEVAL_MODES = {"dense", "sparse", "hybrid"}



def _fail(errors: list[str]) -> None:
    if errors:
        message = "Benchmark config validation failed:\n- " + "\n- ".join(errors)
        raise ValueError(message)



def validate_run_config(config: BenchmarkRunConfig) -> None:
    errors: list[str] = []

    if not config.label.strip():
        errors.append("label must not be empty.")
    if not config.dataset_path:
        errors.append("dataset_path is required.")
    elif not Path(config.dataset_path).exists():
        errors.append(f"dataset_path does not exist: {config.dataset_path}")

    if config.inference.top_k <= 0:
        errors.append("inference.top_k must be positive.")
    if config.inference.max_tokens <= 0:
        errors.append("inference.max_tokens must be positive.")
    if not 0.0 <= config.inference.temperature <= 2.0:
        errors.append("inference.temperature must be between 0.0 and 2.0.")
    if config.inference.retrieval_mode not in _ALLOWED_RETRIEVAL_MODES:
        errors.append(f"inference.retrieval_mode must be one of {sorted(_ALLOWED_RETRIEVAL_MODES)}.")
    if config.inference.retrieval_mode == "hybrid":
        total_weight = config.inference.hybrid_dense_weight + config.inference.hybrid_sparse_weight
        if total_weight <= 0:
            errors.append("hybrid retrieval weights must sum to more than 0.")
    if config.inference.max_regeneration_attempts < 1:
        errors.append("inference.max_regeneration_attempts must be at least 1.")
    if not config.inference.enable_regeneration and config.inference.max_regeneration_attempts != 1:
        errors.append("max_regeneration_attempts should remain 1 when regeneration is disabled.")

    if config.source_mapping.max_matches <= 0:
        errors.append("source_mapping.max_matches must be positive.")
    if config.source_mapping.page_window < 0:
        errors.append("source_mapping.page_window cannot be negative.")
    if not config.source_mapping.page_offset_candidates:
        errors.append("source_mapping.page_offset_candidates must not be empty.")
    if config.source_mapping.max_soft_candidates < 0:
        errors.append("source_mapping.max_soft_candidates cannot be negative.")

    if config.execution.batch_size <= 0:
        errors.append("execution.batch_size must be positive.")
    if config.execution.poll_interval_seconds <= 0:
        errors.append("execution.poll_interval_seconds must be positive.")
    if config.execution.max_wait_seconds <= 0:
        errors.append("execution.max_wait_seconds must be positive.")
    if config.execution.max_cases is not None and config.execution.max_cases <= 0:
        errors.append("execution.max_cases must be positive when provided.")
    if config.execution.warmup_cases < 0:
        errors.append("execution.warmup_cases cannot be negative.")
    if not config.execution.output_dir:
        errors.append("execution.output_dir is required.")

    if not config.ingestion.cleaning_strategy.strip():
        errors.append("ingestion.cleaning_strategy must not be empty.")
    if not config.ingestion.chunking_strategy.strip():
        errors.append("ingestion.chunking_strategy must not be empty.")
    if not isinstance(config.ingestion.cleaning_params, dict):
        errors.append("ingestion.cleaning_params must be an object.")
    if not isinstance(config.ingestion.chunking_params, dict):
        errors.append("ingestion.chunking_params must be an object.")

    chunk_size = config.ingestion.chunking_params.get("chunk_size")
    chunk_overlap = config.ingestion.chunking_params.get("chunk_overlap")
    if chunk_size is not None and int(chunk_size) <= 0:
        errors.append("ingestion.chunking_params.chunk_size must be positive when provided.")
    if chunk_overlap is not None and int(chunk_overlap) < 0:
        errors.append("ingestion.chunking_params.chunk_overlap cannot be negative.")
    if chunk_size is not None and chunk_overlap is not None and int(chunk_overlap) >= int(chunk_size):
        errors.append("ingestion.chunking_params.chunk_overlap must be smaller than chunk_size.")

    _fail(errors)
