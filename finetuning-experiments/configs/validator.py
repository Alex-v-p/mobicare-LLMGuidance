from __future__ import annotations

from .schema import BenchmarkRunConfig


def validate_run_config(config: BenchmarkRunConfig) -> None:
    if not config.label.strip():
        raise ValueError("Config label must not be empty.")
    if not config.dataset_path:
        raise ValueError("Config dataset_path is required.")
    if config.inference.top_k <= 0:
        raise ValueError("inference.top_k must be positive.")
    if config.source_mapping.max_matches <= 0:
        raise ValueError("source_mapping.max_matches must be positive.")
