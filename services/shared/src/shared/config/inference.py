from __future__ import annotations

from functools import lru_cache

from shared.config.base import SharedServiceSettings, resolve_runtime_settings


class InferenceSettings(SharedServiceSettings):
    inference_port: int = 8001

    qdrant_collection: str = "guidance_chunks"

    redis_job_queue: str = "guidance_jobs"
    redis_ingestion_job_queue: str = "ingestion_jobs"
    job_ttl_seconds: int = 7 * 24 * 60 * 60
    job_lease_seconds: int = 60

    ollama_model: str = "qwen2.5:0.5b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_timeout_s: float = 120.0
    ollama_embedding_batch_size: int = 8
    ollama_embedding_fallback_concurrency: int = 2

    callback_timeout_s: int = 10
    callback_max_attempts: int = 3

    retrieval_top_k: int = 3
    jobs_dir: str = "/data/jobs"


@lru_cache(maxsize=1)
def get_inference_settings() -> InferenceSettings:
    return resolve_runtime_settings(InferenceSettings)
