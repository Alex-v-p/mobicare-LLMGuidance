from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

import requests

from adapters.ingestion import IngestionClient
from adapters.guidance import GuidanceClient
from adapters.metrics import MetricsClient
from configs.schema import BenchmarkRunConfig
from datasets.schema import BenchmarkCase
from scoring.latency import summarize_latencies, summarize_stage_latencies
from telemetry.stage_recorder import extract_guidance_telemetry, extract_ingestion_telemetry

logger = logging.getLogger(__name__)



def _build_guidance_payload(case: BenchmarkCase, config: BenchmarkRunConfig) -> dict[str, Any]:
    return {
        "question": case.question,
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
        },
    }



def _build_ingestion_payload(config: BenchmarkRunConfig) -> dict[str, Any]:
    return {
        "documents_version": config.documents_version,
        "collection": config.execution.collection,
        "batch_size": config.execution.batch_size,
        "cleaning_strategy": config.ingestion.cleaning_strategy,
        "cleaning_params": config.ingestion.cleaning_params,
        "chunking_strategy": config.ingestion.chunking_strategy,
        "chunking_params": config.ingestion.chunking_params,
        "embedding_model": config.ingestion.embedding_model,
    }



def _timed_call(fn: Any) -> tuple[float, Any, Exception | None]:
    start = time.monotonic()
    try:
        return max(0.0, time.monotonic() - start), fn(), None
    except Exception as exc:  # noqa: BLE001
        return max(0.0, time.monotonic() - start), None, exc



def _categorize_failure(error: Exception | None, record: dict[str, Any] | None, *, endpoint: str) -> str | None:
    if error is not None:
        if isinstance(error, TimeoutError):
            return "timeout"
        if isinstance(error, requests.HTTPError):
            status_code = getattr(error.response, "status_code", None)
            if status_code in {400, 404, 409, 422}:
                return "validation_error"
            return "request_error"
        if isinstance(error, requests.RequestException):
            return "request_error"
        return "unknown_error"

    snapshot = dict(record or {})
    status = str(snapshot.get("status") or "").lower()
    if status == "failed":
        text = str(snapshot.get("error") or snapshot.get("message") or "").lower()
        if "validation" in text or "422" in text:
            return "validation_error"
        if "retriev" in text:
            return "retrieval_failure"
        if "verif" in text:
            return "verification_failure"
        if "callback" in text:
            return "callback_failure"
        if "lease" in text or "worker" in text or "retry" in text:
            return "worker_anomaly"
        return "generation_failure" if endpoint == "guidance_endpoint" else "ingestion_failure"

    rag = snapshot.get("rag") or []
    if endpoint == "guidance_endpoint" and isinstance(rag, list) and not rag:
        return "empty_retrieval"

    verification = snapshot.get("verification") or {}
    verdict = str(verification.get("verdict") or "").lower()
    if verdict in {"fail", "failed", "warning", "warn"}:
        return "verification_failure"
    return None



def _endpoint_summary(
    endpoint_name: str,
    attempts: list[dict[str, Any]],
    *,
    percentile_policy: str,
    outlier_policy: str,
) -> dict[str, Any]:
    all_latencies = [float(item.get("latency_seconds", 0.0)) for item in attempts]
    success_attempts = [item for item in attempts if item.get("status") == "completed"]
    failed_attempts = [item for item in attempts if item.get("status") != "completed"]
    included = success_attempts if percentile_policy == "success_only" else attempts
    latency_summary = summarize_latencies(
        [float(item.get("latency_seconds", 0.0)) for item in included],
        policy=percentile_policy,
        outlier_policy=outlier_policy,
    )
    failure_taxonomy = Counter(
        item.get("failure_category")
        for item in attempts
        if item.get("failure_category")
    )
    queue_delay_values = [
        float(item["telemetry_derived"]["queue_delay_ms"]) / 1000.0
        for item in attempts
        if item.get("telemetry_derived", {}).get("queue_delay_ms") is not None
    ]
    execution_values = [
        float(item["telemetry_derived"]["execution_duration_ms"]) / 1000.0
        for item in attempts
        if item.get("telemetry_derived", {}).get("execution_duration_ms") is not None
    ]
    stage_lists = [list(item.get("telemetry_stages") or []) for item in attempts]
    return {
        **latency_summary,
        "endpoint": endpoint_name,
        "request_count": len(attempts),
        "success_count": len(success_attempts),
        "failure_count": len(failed_attempts),
        "completion_rate": (len(success_attempts) / len(attempts)) if attempts else 0.0,
        "failure_rate": (len(failed_attempts) / len(attempts)) if attempts else 0.0,
        "requests_per_second": (len(success_attempts) / sum(all_latencies)) if sum(all_latencies) > 0 else 0.0,
        "timeout_rate": ((failure_taxonomy.get("timeout", 0)) / len(attempts)) if attempts else 0.0,
        "queue_delay": summarize_latencies(queue_delay_values, policy="success_only", outlier_policy=outlier_policy),
        "execution_duration": summarize_latencies(execution_values, policy="success_only", outlier_policy=outlier_policy),
        "stage_latency_summary": summarize_stage_latencies(stage_lists),
        "failure_taxonomy": dict(sorted(failure_taxonomy.items())),
        "attempts": attempts,
    }



def _choose_cases(cases: list[BenchmarkCase], config: BenchmarkRunConfig) -> list[BenchmarkCase]:
    sample_case_ids = set(config.execution.api_test.sample_case_ids or [])
    if not sample_case_ids:
        return cases or []
    selected = [case for case in cases if case.id in sample_case_ids]
    return selected or cases



def run_api_stage(config: BenchmarkRunConfig, cases: list[BenchmarkCase]) -> dict[str, Any]:
    api_test = config.execution.api_test
    if not api_test.enabled:
        return {}

    selected_cases = _choose_cases(cases, config)
    gateway_client = IngestionClient(base_url=config.execution.gateway_url)
    guidance_client = GuidanceClient(base_url=config.execution.gateway_url)
    metrics_client = MetricsClient(base_url=config.execution.gateway_url)
    guidance_attempts: list[dict[str, Any]] = []
    ingestion_attempts: list[dict[str, Any]] = []

    if api_test.include_guidance_endpoint and selected_cases:
        for warmup_index in range(api_test.warmup_requests):
            case = selected_cases[warmup_index % len(selected_cases)]
            try:
                guidance_client.run_guidance_and_wait(
                    _build_guidance_payload(case, config),
                    poll_interval_seconds=config.execution.poll_interval_seconds,
                    max_wait_seconds=config.execution.max_wait_seconds,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Guidance API warm-up failed for case=%s: %s", case.id, exc)

        for index in range(api_test.guidance_repeat_count):
            case = selected_cases[index % len(selected_cases)]
            latency_seconds, result, error = _timed_call(
                lambda case=case: guidance_client.run_guidance_and_wait(
                    _build_guidance_payload(case, config),
                    poll_interval_seconds=config.execution.poll_interval_seconds,
                    max_wait_seconds=config.execution.max_wait_seconds,
                )
            )
            record = (result.record if result is not None else {"status": "failed", "error": str(error)})
            telemetry = extract_guidance_telemetry(record)
            guidance_attempts.append(
                {
                    "attempt_index": index + 1,
                    "case_id": case.id,
                    "status": str(record.get("status") or ("failed" if error else "unknown")),
                    "latency_seconds": latency_seconds,
                    "failure_category": _categorize_failure(error, record, endpoint="guidance_endpoint"),
                    "telemetry_derived": telemetry.get("derived") or {},
                    "telemetry_stages": telemetry.get("stages") or [],
                    "warning_count": len(record.get("warnings") or []),
                    "rag_count": len(record.get("rag") or []),
                }
            )

    if api_test.include_ingestion_endpoint and api_test.ingestion_repeat_count > 0:
        for warmup_index in range(api_test.warmup_requests):
            if api_test.ingestion_delete_collection_each_run:
                try:
                    gateway_client.delete_collection()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Ingestion API warm-up delete failed: %s", exc)
            try:
                gateway_client.run_and_wait(
                    _build_ingestion_payload(config),
                    poll_interval_seconds=config.execution.poll_interval_seconds,
                    max_wait_seconds=config.execution.max_wait_seconds,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ingestion API warm-up failed: %s", exc)

        for index in range(api_test.ingestion_repeat_count):
            if api_test.ingestion_delete_collection_each_run:
                try:
                    gateway_client.delete_collection()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Ingestion delete before repeat failed: %s", exc)
            latency_seconds, result, error = _timed_call(
                lambda: gateway_client.run_and_wait(
                    _build_ingestion_payload(config),
                    poll_interval_seconds=config.execution.poll_interval_seconds,
                    max_wait_seconds=config.execution.max_wait_seconds,
                )
            )
            record = (result.record if result is not None else {"status": "failed", "error": str(error)})
            telemetry = extract_ingestion_telemetry(record)
            payload = dict(record.get("result") or {})
            ingestion_attempts.append(
                {
                    "attempt_index": index + 1,
                    "status": str(record.get("status") or ("failed" if error else "unknown")),
                    "latency_seconds": latency_seconds,
                    "failure_category": _categorize_failure(error, record, endpoint="ingestion_endpoint"),
                    "telemetry_derived": telemetry.get("derived") or {},
                    "telemetry_stages": telemetry.get("stages") or [],
                    "documents_found": payload.get("documents_found"),
                    "chunks_created": payload.get("chunks_created"),
                    "vectors_upserted": payload.get("vectors_upserted"),
                }
            )

    endpoint_summaries: dict[str, Any] = {}
    if guidance_attempts:
        endpoint_summaries["guidance_endpoint"] = _endpoint_summary(
            "guidance_endpoint",
            guidance_attempts,
            percentile_policy=api_test.percentile_policy,
            outlier_policy=api_test.outlier_policy,
        )
    if ingestion_attempts:
        endpoint_summaries["ingestion_endpoint"] = _endpoint_summary(
            "ingestion_endpoint",
            ingestion_attempts,
            percentile_policy=api_test.percentile_policy,
            outlier_policy=api_test.outlier_policy,
        )

    primary_summary = endpoint_summaries.get("guidance_endpoint") or endpoint_summaries.get("ingestion_endpoint") or {}
    failure_taxonomy = Counter()
    for summary in endpoint_summaries.values():
        failure_taxonomy.update(summary.get("failure_taxonomy") or {})

    metrics_snapshot = {}
    if api_test.metrics_endpoint_path:
        try:
            metrics_snapshot = metrics_client.fetch_json(api_test.metrics_endpoint_path)
        except Exception as exc:  # noqa: BLE001
            metrics_snapshot = {"status": "failed", "error": str(exc)}

    return {
        **primary_summary,
        "mode": "api_test",
        "benchmark_case_api": {},
        "endpoint_summaries": endpoint_summaries,
        "failure_taxonomy": dict(sorted(failure_taxonomy.items())),
        "load_test_metadata": {
            "warmup_requests": api_test.warmup_requests,
            "guidance_repeat_count": api_test.guidance_repeat_count,
            "ingestion_repeat_count": api_test.ingestion_repeat_count,
            "percentile_policy": api_test.percentile_policy,
            "outlier_policy": api_test.outlier_policy,
        },
        "metrics_snapshot": metrics_snapshot,
    }
