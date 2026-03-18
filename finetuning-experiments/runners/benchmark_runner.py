from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from adapters.gateway import GatewayClient
from adapters.guidance import GuidanceClient
from adapters.qdrant import QdrantScrollClient
from artifacts.models import CURRENT_ARTIFACT_VERSION, RunArtifact
from artifacts.writer import write_run_artifact
from configs.schema import BenchmarkRunConfig
from datasets.loader import load_benchmark_dataset
from datasets.schema import BenchmarkCase
from scoring.aggregation import summarize_results
from scoring.generation import score_generation
from scoring.normalization import normalize_run_metrics
from scoring.retrieval import score_retrieval
from source_mapping.matcher import SourceMatcher

from utils.datetime import utc_now_iso
from utils.ids import build_run_id

logger = logging.getLogger(__name__)


def _build_cases(raw_dataset: dict[str, Any]) -> list[BenchmarkCase]:
    return [BenchmarkCase(**case) for case in raw_dataset.get("cases", [])]


def _ingest(config: BenchmarkRunConfig) -> dict[str, Any] | None:
    client = GatewayClient(base_url=config.execution.gateway_url)
    if config.ingestion.delete_collection_first:
        try:
            delete_result = client.delete_ingestion_collection()
            logger.info("Deleted collection collection=%s existed=%s", delete_result.get("collection"), delete_result.get("existed"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed deleting collection before ingestion: %s", exc)
    payload = {
        "options": {
            "cleaning_strategy": config.ingestion.cleaning_strategy,
            "cleaning_params": config.ingestion.cleaning_params,
            "chunking_strategy": config.ingestion.chunking_strategy,
            "chunking_params": config.ingestion.chunking_params,
            "embedding_model": config.ingestion.embedding_model,
        }
    }
    result = client.run_ingestion_and_wait(
        payload,
        poll_interval_seconds=config.execution.poll_interval_seconds,
        max_wait_seconds=config.execution.max_wait_seconds,
    )
    return result.record


def _map_sources(config: BenchmarkRunConfig, cases: list[BenchmarkCase]) -> list[dict[str, Any]]:
    qdrant_client = QdrantScrollClient(url=config.execution.qdrant_url, collection_name=config.execution.collection)
    payloads = qdrant_client.fetch_all_payloads(batch_size=config.execution.batch_size)
    logger.info("Loaded %s payloads from Qdrant", len(payloads))
    matcher = SourceMatcher(
        max_matches=config.source_mapping.max_matches,
        page_window=config.source_mapping.page_window,
        page_offset_candidates=tuple(config.source_mapping.page_offset_candidates),
        semantic_fallback_enabled=config.source_mapping.semantic_fallback_enabled,
        include_chunk_pairs=config.source_mapping.include_chunk_pairs,
    )
    return [matcher.build_chunk_assignment(case=case, mapping_label=config.label, payloads=payloads).to_dict() for case in cases]


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


def _timing_seconds(record: dict[str, Any], start: float) -> float:
    return max(0.0, time.monotonic() - start)


def run_benchmark(config: BenchmarkRunConfig) -> Path:
    raw_dataset = load_benchmark_dataset(config.dataset_path)
    if not config.dataset_version:
        config.dataset_version = raw_dataset.get("dataset_id") or raw_dataset.get("dataset_version")
    cases = _build_cases(raw_dataset)
    if not config.execution.include_unanswerable:
        cases = [case for case in cases if case.answerability == "answerable"]
    if config.execution.max_cases:
        cases = cases[: config.execution.max_cases]

    logger.info("Loaded %s benchmark cases", len(cases))

    ingestion_record = _ingest(config)
    assignments = _map_sources(config, cases)
    assignment_by_case = {item["case_id"]: item for item in assignments}
    guidance_client = GuidanceClient(base_url=config.execution.gateway_url)

    warmup_cases = cases[: config.execution.warmup_cases]
    for case in warmup_cases:
        logger.info("Warm-up case=%s", case.id)
        try:
            guidance_client.run_guidance_and_wait(
                _build_guidance_payload(case, config),
                poll_interval_seconds=config.execution.poll_interval_seconds,
                max_wait_seconds=config.execution.max_wait_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warm-up failed for case=%s: %s", case.id, exc)

    per_case_results: list[dict[str, Any]] = []
    for case in cases:
        expected_matches = assignment_by_case.get(case.id, {}).get("matches", [])
        payload = _build_guidance_payload(case, config)
        logger.info("Running case=%s question=%s", case.id, case.question[:100])
        start = time.monotonic()
        try:
            result = guidance_client.run_guidance_and_wait(
                payload,
                poll_interval_seconds=config.execution.poll_interval_seconds,
                max_wait_seconds=config.execution.max_wait_seconds,
            )
            record = result.record
            total_latency = _timing_seconds(record, start)
            retrieved = record.get("rag") or []
            answer = record.get("answer") or ""
            retrieval_scores = score_retrieval(expected_matches, retrieved)
            generation_scores = score_generation(case.to_dict(), answer, retrieved)
            per_case_results.append(
                {
                    "case_id": case.id,
                    "question": case.question,
                    "answerability": case.answerability,
                    "reference_answer": case.reference_answer,
                    "gold_passage_text": case.gold_passage_text,
                    "source_match_candidates": expected_matches,
                    "generated_answer": answer,
                    "retrieved_chunks": retrieved,
                    "warnings": record.get("warnings") or [],
                    "verification": record.get("verification"),
                    "metadata": record.get("metadata") or {},
                    "retrieval_scores": retrieval_scores,
                    "generation_scores": generation_scores,
                    "timings": {
                        "total_latency_seconds": total_latency,
                        "created_at": record.get("created_at"),
                        "started_at": record.get("started_at"),
                        "completed_at": record.get("completed_at"),
                        "updated_at": record.get("updated_at"),
                    },
                    "raw_endpoint_result": record,
                }
            )
        except Exception as exc:  # noqa: BLE001
            total_latency = _timing_seconds({}, start)
            per_case_results.append(
                {
                    "case_id": case.id,
                    "question": case.question,
                    "answerability": case.answerability,
                    "reference_answer": case.reference_answer,
                    "gold_passage_text": case.gold_passage_text,
                    "source_match_candidates": expected_matches,
                    "generated_answer": "",
                    "retrieved_chunks": [],
                    "warnings": [],
                    "verification": None,
                    "metadata": {},
                    "retrieval_scores": score_retrieval(expected_matches, []),
                    "generation_scores": score_generation(case.to_dict(), "", []),
                    "timings": {"total_latency_seconds": total_latency},
                    "raw_endpoint_result": {"status": "failed", "error": str(exc)},
                    "error": str(exc),
                }
            )

    summaries = summarize_results(per_case_results)
    run_id = build_run_id(config.label)
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
        ingestion_summary=ingestion_record or {},
        source_mapping_summary={
            "mapping_label": config.label,
            "case_chunk_assignments": assignments,
        },
        retrieval_summary=summaries.get("retrieval_summary") or {},
        generation_summary=summaries.get("generation_summary") or {},
        api_summary=summaries.get("api_summary") or {},
        normalized_metrics=normalize_run_metrics(
            summaries.get("retrieval_summary"),
            summaries.get("generation_summary"),
            summaries.get("api_summary"),
        ),
        per_case_results=per_case_results,
    ).to_dict()
    return write_run_artifact(config.execution.output_dir, run_id, artifact)
