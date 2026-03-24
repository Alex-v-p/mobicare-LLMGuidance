from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from adapters.guidance import GuidanceClient
from adapters.guidance_payloads import normalize_guidance_record
from artifacts.models import CURRENT_ARTIFACT_VERSION, RunArtifact
from artifacts.writer import write_run_artifact
from caching.fingerprints import build_run_fingerprint
from caching.run_registry import RunRegistry
from configs.schema import BenchmarkRunConfig
from datasets.loader import load_benchmark_dataset
from datasets.schema import BenchmarkCase
from runners.api_runner import run_api_stage
from runners.request_payloads import build_guidance_payload
from runners.generation_runner import run_generation_stage
from runners.ingestion_runner import run_ingestion_stage
from runners.retrieval_runner import run_retrieval_stage
from scoring.aggregation import summarize_results
from scoring.normalization import normalize_run_metrics
from telemetry.stage_recorder import extract_guidance_telemetry
from utils.datetime import utc_now_iso
from utils.environment import collect_environment_snapshot
from utils.ids import build_run_id

logger = logging.getLogger(__name__)


def _build_cases(raw_dataset: dict[str, Any]) -> list[BenchmarkCase]:
    return [BenchmarkCase(**case) for case in raw_dataset.get("cases", [])]




def _timing_seconds(start: float) -> float:
    return max(0.0, time.monotonic() - start)


def _run_warmups(cases: list[BenchmarkCase], config: BenchmarkRunConfig, guidance_client: GuidanceClient) -> None:
    for case in cases[: config.execution.warmup_cases]:
        logger.info("Warm-up case=%s", case.id)
        try:
            guidance_client.run_guidance_and_wait(
                build_guidance_payload(case, config),
                poll_interval_seconds=config.execution.poll_interval_seconds,
                max_wait_seconds=config.execution.max_wait_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warm-up failed for case=%s: %s", case.id, exc)


def _build_success_case_result(
    *,
    case: BenchmarkCase,
    source_mapping: dict[str, Any] | None,
    guidance_record: dict[str, Any],
    total_latency: float,
    evaluation_config: BenchmarkRunConfig.EvaluationConfig | None = None,
) -> dict[str, Any]:
    normalized_record = normalize_guidance_record(guidance_record)
    retrieval_result = run_retrieval_stage(source_mapping, normalized_record)
    generation_result = run_generation_stage(
        case,
        normalized_record,
        retrieval_result.retrieved_chunks,
        evaluation_config,
    )
    telemetry = extract_guidance_telemetry(normalized_record)
    derived = telemetry.get("derived") or {}
    return {
        "status": "completed",
        "case_id": case.id,
        "question": case.question,
        "answerability": case.answerability,
        "reference_answer": case.reference_answer,
        "gold_passage_text": case.gold_passage_text,
        "source_match_candidates": retrieval_result.source_match_candidates,
        "source_list": retrieval_result.source_list,
        "generated_answer": generation_result.generated_answer,
        "retrieved_chunks": retrieval_result.retrieved_chunks,
        "warnings": generation_result.warnings,
        "verification": generation_result.verification,
        "metadata": generation_result.metadata,
        "retrieval_scores": retrieval_result.retrieval_scores,
        "generation_scores": generation_result.generation_scores,
        "telemetry": telemetry,
        "timings": {
            "total_latency_seconds": total_latency,
            "created_at": normalized_record.get("created_at"),
            "started_at": normalized_record.get("started_at"),
            "completed_at": normalized_record.get("completed_at"),
            "updated_at": normalized_record.get("updated_at"),
            "queue_delay_ms": derived.get("queue_delay_ms"),
            "execution_duration_ms": derived.get("execution_duration_ms"),
            "total_duration_ms": derived.get("total_duration_ms"),
        },
        "raw_endpoint_result": normalized_record,
        "endpoint_envelope": normalized_record.get("endpoint_envelope") or {},
    }


def _build_failed_case_result(
    *,
    case: BenchmarkCase,
    source_mapping: dict[str, Any] | None,
    error: Exception,
    total_latency: float,
    evaluation_config: BenchmarkRunConfig.EvaluationConfig | None = None,
) -> dict[str, Any]:
    retrieval_result = run_retrieval_stage(source_mapping, {})
    generation_result = run_generation_stage(
        case,
        {},
        retrieval_result.retrieved_chunks,
        evaluation_config,
    )
    failed_record = {"status": "failed", "error": str(error)}
    return {
        "status": "failed",
        "case_id": case.id,
        "question": case.question,
        "answerability": case.answerability,
        "reference_answer": case.reference_answer,
        "gold_passage_text": case.gold_passage_text,
        "source_match_candidates": retrieval_result.source_match_candidates,
        "source_list": retrieval_result.source_list,
        "generated_answer": generation_result.generated_answer,
        "retrieved_chunks": retrieval_result.retrieved_chunks,
        "warnings": generation_result.warnings,
        "verification": generation_result.verification,
        "metadata": generation_result.metadata,
        "retrieval_scores": retrieval_result.retrieval_scores,
        "generation_scores": generation_result.generation_scores,
        "telemetry": extract_guidance_telemetry(failed_record),
        "timings": {"total_latency_seconds": total_latency},
        "raw_endpoint_result": failed_record,
        "error": str(error),
    }


def run_benchmark(config: BenchmarkRunConfig) -> Path:
    raw_dataset = load_benchmark_dataset(config.dataset_path)
    if not config.dataset_version:
        config.dataset_version = raw_dataset.get("dataset_id") or raw_dataset.get("dataset_version")

    run_fingerprint = build_run_fingerprint(config)
    registries_root = Path(config.execution.output_dir) / "_registries"
    run_registry = RunRegistry(registries_root)
    existing_run = run_registry.get(run_fingerprint)
    if existing_run and existing_run.status in {"completed", "reused", "skipped"} and existing_run.artifact_path:
        existing_path = Path(existing_run.artifact_path)
        if existing_path.exists():
            logger.info("Reusing completed run for fingerprint=%s from run_id=%s", run_fingerprint, existing_run.run_id)
            run_registry.upsert(
                fingerprint=run_fingerprint,
                status="reused",
                run_id=existing_run.run_id,
                artifact_path=str(existing_path),
                ingestion_fingerprint=existing_run.ingestion_fingerprint,
                metadata={"reused_from_run_id": existing_run.run_id},
            )
            return existing_path

    cases = _build_cases(raw_dataset)
    if not config.execution.include_unanswerable:
        cases = [case for case in cases if case.answerability == "answerable"]
    if config.execution.max_cases:
        cases = cases[: config.execution.max_cases]

    logger.info("Loaded %s benchmark cases", len(cases))

    run_id = build_run_id(config.label)
    run_registry.upsert(fingerprint=run_fingerprint, status="running", run_id=run_id)

    try:
        ingestion_stage = run_ingestion_stage(config, cases, run_id=run_id)
        source_mapping_by_case = {item["case_id"]: item for item in ingestion_stage.assignments}
        guidance_client = GuidanceClient(base_url=config.execution.gateway_url)
        _run_warmups(cases, config, guidance_client)

        per_case_results: list[dict[str, Any]] = []
        for case in cases:
            source_mapping = source_mapping_by_case.get(case.id)
            payload = build_guidance_payload(case, config)
            logger.info("Running case=%s question=%s", case.id, case.question[:100])
            start = time.monotonic()
            try:
                result = guidance_client.run_guidance_and_wait(
                    payload,
                    poll_interval_seconds=config.execution.poll_interval_seconds,
                    max_wait_seconds=config.execution.max_wait_seconds,
                )
                per_case_results.append(
                    _build_success_case_result(
                        case=case,
                        source_mapping=source_mapping,
                        guidance_record=result.record,
                        total_latency=_timing_seconds(start),
                        evaluation_config=config.evaluation,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                per_case_results.append(
                    _build_failed_case_result(
                        case=case,
                        source_mapping=source_mapping,
                        error=exc,
                        total_latency=_timing_seconds(start),
                        evaluation_config=config.evaluation,
                    )
                )

        summaries = summarize_results(per_case_results)
        benchmark_api_summary = summaries.get("api_summary") or {}
        api_summary = benchmark_api_summary
        if config.execution.api_test.enabled:
            api_summary = run_api_stage(config, cases)
            if api_summary:
                api_summary["benchmark_case_api"] = benchmark_api_summary

        environment_snapshot = collect_environment_snapshot(config)

        artifact = RunArtifact(
            artifact_type="run",
            artifact_version=CURRENT_ARTIFACT_VERSION,
            run_id=run_id,
            label=config.label,
            datetime=utc_now_iso(),
            dataset_version=config.dataset_version,
            documents_version=config.documents_version,
            notes=config.notes,
            change_note=config.change_note,
            config=config.to_dict(),
            cache={
                "run_fingerprint": run_fingerprint,
                "ingestion_fingerprint": ingestion_stage.cache.get("fingerprint"),
                "run_registry_status": "completed",
                "ingestion_cache": ingestion_stage.cache,
            },
            environment=environment_snapshot,
            ingestion_summary=ingestion_stage.ingestion_summary,
            source_mapping_summary=ingestion_stage.source_mapping_summary,
            retrieval_summary=summaries.get("retrieval_summary") or {},
            generation_summary=summaries.get("generation_summary") or {},
            api_summary=api_summary,
            normalized_metrics=normalize_run_metrics(
                summaries.get("retrieval_summary"),
                summaries.get("generation_summary"),
                api_summary,
            ),
            per_case_results=per_case_results,
        ).to_dict()
        artifact_path = write_run_artifact(config.execution.output_dir, run_id, artifact)
        run_registry.upsert(
            fingerprint=run_fingerprint,
            status="completed",
            run_id=run_id,
            artifact_path=str(artifact_path),
            ingestion_fingerprint=ingestion_stage.cache.get("fingerprint"),
            metadata={"case_count": len(per_case_results)},
        )
        return artifact_path
    except Exception as exc:  # noqa: BLE001
        run_registry.upsert(fingerprint=run_fingerprint, status="failed", run_id=run_id, error=str(exc))
        raise