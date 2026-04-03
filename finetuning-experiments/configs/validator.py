from __future__ import annotations

from pathlib import Path

from .schema import BenchmarkRunConfig

_ALLOWED_RETRIEVAL_MODES = {"dense", "sparse", "hybrid"}
_ALLOWED_PERCENTILE_POLICIES = {"all", "success_only"}
_ALLOWED_OUTLIER_POLICIES = {"keep_all", "exclude_failures"}
_ALLOWED_GATEWAY_AUTH_MODES = {"none", "bearer", "gateway_login", "local_jwt"}
_ALLOWED_PIPELINE_VARIANTS = {"standard", "drug_dosing"}
_ALLOWED_SOURCE_MAPPING_PROFILES = {"legacy_v1", "semantic_recovery_v2"}
_ALLOWED_LLM_LABELING_PROFILES = {"heuristic_v1", "semantic_recovery_v2"}



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
    if config.inference.pipeline_variant not in _ALLOWED_PIPELINE_VARIANTS:
        errors.append(f"inference.pipeline_variant must be one of {sorted(_ALLOWED_PIPELINE_VARIANTS)}.")

    if config.source_mapping.max_matches <= 0:
        errors.append("source_mapping.max_matches must be positive.")
    if config.source_mapping.page_window < 0:
        errors.append("source_mapping.page_window cannot be negative.")
    if not config.source_mapping.page_offset_candidates:
        errors.append("source_mapping.page_offset_candidates must not be empty.")
    if config.source_mapping.max_soft_candidates < 0:
        errors.append("source_mapping.max_soft_candidates cannot be negative.")
    if config.source_mapping.mapping_profile not in _ALLOWED_SOURCE_MAPPING_PROFILES:
        errors.append(f"source_mapping.mapping_profile must be one of {sorted(_ALLOWED_SOURCE_MAPPING_PROFILES)}.")
    if config.source_mapping.llm_labeling_profile not in _ALLOWED_LLM_LABELING_PROFILES:
        errors.append(f"source_mapping.llm_labeling_profile must be one of {sorted(_ALLOWED_LLM_LABELING_PROFILES)}.")
    if config.source_mapping.max_sequence_length < 1:
        errors.append("source_mapping.max_sequence_length must be at least 1.")
    if config.source_mapping.semantic_fallback_max_matches < 1:
        errors.append("source_mapping.semantic_fallback_max_matches must be at least 1.")

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
    if config.execution.gateway_auth_mode not in _ALLOWED_GATEWAY_AUTH_MODES:
        errors.append(f"execution.gateway_auth_mode must be one of {sorted(_ALLOWED_GATEWAY_AUTH_MODES)}.")
    if config.execution.gateway_auth_mode == "bearer" and not (config.execution.gateway_auth_token or "").strip():
        errors.append("execution.gateway_auth_token is required when gateway_auth_mode='bearer'.")
    if config.execution.gateway_auth_mode in {"gateway_login", "local_jwt"} and not (config.execution.gateway_auth_email or "").strip():
        errors.append("execution.gateway_auth_email is required when gateway_auth_mode uses a generated or login-based token.")
    if config.execution.gateway_auth_mode == "gateway_login" and not (config.execution.gateway_auth_password or ""):
        errors.append("execution.gateway_auth_password is required when gateway_auth_mode='gateway_login'.")
    if config.execution.gateway_auth_mode == "local_jwt" and not (config.execution.gateway_jwt_secret or "").strip():
        errors.append("execution.gateway_jwt_secret is required when gateway_auth_mode='local_jwt'.")
    if config.execution.gateway_jwt_exp_minutes <= 0:
        errors.append("execution.gateway_jwt_exp_minutes must be positive.")

    rubric = config.evaluation.deterministic_rubric
    rubric_weights = [
        rubric.required_fact_weight,
        rubric.reference_alignment_weight,
        rubric.gold_alignment_weight,
        rubric.context_alignment_weight,
        rubric.groundedness_weight,
        rubric.verification_weight,
    ]
    if any(weight < 0 for weight in rubric_weights):
        errors.append("evaluation.deterministic_rubric weights cannot be negative.")
    if sum(rubric_weights) <= 0:
        errors.append("evaluation.deterministic_rubric weights must sum to more than 0.")

    llm_judge = config.evaluation.llm_judge
    if llm_judge.provider not in {"ollama"}:
        errors.append("evaluation.llm_judge.provider must currently be 'ollama'.")
    if llm_judge.enabled and not str(llm_judge.base_url).strip():
        errors.append("evaluation.llm_judge.base_url is required when llm_judge is enabled.")
    if llm_judge.max_tokens <= 0:
        errors.append("evaluation.llm_judge.max_tokens must be positive.")
    if llm_judge.timeout_seconds <= 0:
        errors.append("evaluation.llm_judge.timeout_seconds must be positive.")
    if not 0.0 <= llm_judge.temperature <= 2.0:
        errors.append("evaluation.llm_judge.temperature must be between 0.0 and 2.0.")

    environment = config.execution.environment
    if not isinstance(environment.capture_enabled, bool):
        errors.append("execution.environment.capture_enabled must be a boolean.")
    if environment.container_names and not all(str(name).strip() for name in environment.container_names):
        errors.append("execution.environment.container_names must not contain empty values.")
    if environment.docker_compose_path and not Path(environment.docker_compose_path).exists():
        errors.append(f"execution.environment.docker_compose_path does not exist: {environment.docker_compose_path}")

    api_test = config.execution.api_test
    if api_test.warmup_requests < 0:
        errors.append("execution.api_test.warmup_requests cannot be negative.")
    if api_test.guidance_repeat_count < 0:
        errors.append("execution.api_test.guidance_repeat_count cannot be negative.")
    if api_test.ingestion_repeat_count < 0:
        errors.append("execution.api_test.ingestion_repeat_count cannot be negative.")
    if api_test.percentile_policy not in _ALLOWED_PERCENTILE_POLICIES:
        errors.append(f"execution.api_test.percentile_policy must be one of {sorted(_ALLOWED_PERCENTILE_POLICIES)}.")
    if api_test.outlier_policy not in _ALLOWED_OUTLIER_POLICIES:
        errors.append(f"execution.api_test.outlier_policy must be one of {sorted(_ALLOWED_OUTLIER_POLICIES)}.")

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
