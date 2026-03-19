from __future__ import annotations

from pathlib import Path

import logging
from dataclasses import asdict, dataclass
from typing import Any

from adapters.gateway import GatewayClient
from caching.fingerprints import build_ingestion_fingerprint
from caching.ingestion_registry import IngestionRegistry
from adapters.qdrant import QdrantScrollClient
from configs.schema import BenchmarkRunConfig
from datasets.schema import BenchmarkCase
from source_mapping.llm_labeler import LLMLabelerConfig, OptionalLLMLabeler
from source_mapping.matcher import SourceMatcher
from telemetry.stage_recorder import extract_ingestion_telemetry
from scoring.ingestion import summarize_ingestion_payloads

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionStageResult:
    ingestion_summary: dict[str, Any]
    source_mapping_summary: dict[str, Any]
    assignments: list[dict[str, Any]]
    cache: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _build_ingestion_payload(config: BenchmarkRunConfig) -> dict[str, Any]:
    return {
        "options": {
            "cleaning_strategy": config.ingestion.cleaning_strategy,
            "cleaning_params": config.ingestion.cleaning_params,
            "chunking_strategy": config.ingestion.chunking_strategy,
            "chunking_params": config.ingestion.chunking_params,
            "embedding_model": config.ingestion.embedding_model,
        }
    }



def _extract_ingestion_summary(result_record: dict[str, Any], *, job_id: str, status: str) -> dict[str, Any]:
    payload = (result_record.get("result") or {}) if isinstance(result_record, dict) else {}
    telemetry = extract_ingestion_telemetry(result_record)
    derived = telemetry.get("derived") or {}
    return {
        "job_id": job_id,
        "status": status,
        "collection": payload.get("collection") or result_record.get("collection"),
        "documents_found": payload.get("documents_found"),
        "chunks_created": payload.get("chunks_created"),
        "vectors_upserted": payload.get("vectors_upserted"),
        "cleaning_strategy": payload.get("cleaning_strategy"),
        "chunking_strategy": payload.get("chunking_strategy"),
        "embedding_model": payload.get("embedding_model"),
        "created_at": result_record.get("created_at"),
        "started_at": result_record.get("started_at"),
        "completed_at": result_record.get("completed_at"),
        "updated_at": result_record.get("updated_at"),
        "queue_delay_ms": derived.get("queue_delay_ms"),
        "execution_duration_ms": derived.get("execution_duration_ms"),
        "total_duration_ms": derived.get("total_duration_ms"),
        "telemetry": telemetry,
        "raw_endpoint_result": result_record,
    }



def _map_sources(config: BenchmarkRunConfig, cases: list[BenchmarkCase]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
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
    llm_labeler = OptionalLLMLabeler(
        LLMLabelerConfig(
            enabled=config.source_mapping.llm_second_pass_enabled,
            max_candidates=config.source_mapping.max_soft_candidates,
        )
    )
    assignments = [
        matcher.build_case_source_mapping(
            case=case,
            mapping_label=config.label,
            strategy=config.ingestion.chunking_strategy,
            payloads=payloads,
            llm_labeler=llm_labeler,
            max_soft_candidates=config.source_mapping.max_soft_candidates,
        ).to_dict()
        for case in cases
    ]
    label_totals = {
        label: sum(len((item.get("source_list") or {}).get(label) or []) for item in assignments)
        for label in ("direct_evidence", "partial_direct_evidence", "supporting", "tangential", "irrelevant")
    }
    ingestion_metrics = summarize_ingestion_payloads(payloads)
    summary = {
        "mapping_label": config.label,
        "strategy": config.ingestion.chunking_strategy,
        "case_count": len(assignments),
        "payload_count": len(payloads),
        "label_totals": label_totals,
        "ingestion_metrics": ingestion_metrics,
        "matcher": {
            "max_matches": config.source_mapping.max_matches,
            "page_window": config.source_mapping.page_window,
            "page_offset_candidates": list(config.source_mapping.page_offset_candidates),
            "semantic_fallback_enabled": config.source_mapping.semantic_fallback_enabled,
            "include_chunk_pairs": config.source_mapping.include_chunk_pairs,
            "llm_second_pass_enabled": config.source_mapping.llm_second_pass_enabled,
            "max_soft_candidates": config.source_mapping.max_soft_candidates,
        },
        "case_chunk_assignments": assignments,
    }
    return assignments, summary, payloads



def run_ingestion_stage(config: BenchmarkRunConfig, cases: list[BenchmarkCase], *, run_id: str) -> IngestionStageResult:
    fingerprint = build_ingestion_fingerprint(config)
    registry = IngestionRegistry(Path(config.execution.output_dir) / "_registries")
    cached = registry.get(fingerprint)
    if cached:
        logger.info("Reusing cached ingestion for fingerprint=%s from run_id=%s", fingerprint, cached.run_id)
        ingestion_summary = dict(cached.ingestion_summary)
        ingestion_summary.setdefault("ingestion_metrics", (cached.source_mapping_summary or {}).get("ingestion_metrics") or {})
        ingestion_summary.setdefault("cache", {})
        ingestion_summary["cache"].update({
            "fingerprint": fingerprint,
            "status": "reused",
            "source_run_id": cached.run_id,
            "registry_updated_at": cached.updated_at,
        })
        return IngestionStageResult(
            ingestion_summary=ingestion_summary,
            source_mapping_summary=cached.source_mapping_summary,
            assignments=cached.assignments,
            cache={"fingerprint": fingerprint, "status": "reused", "source_run_id": cached.run_id},
        )

    client = GatewayClient(base_url=config.execution.gateway_url)
    if config.ingestion.delete_collection_first:
        try:
            delete_result = client.delete_ingestion_collection()
            logger.info(
                "Deleted collection collection=%s existed=%s",
                delete_result.get("collection"),
                delete_result.get("existed"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed deleting collection before ingestion: %s", exc)

    ingestion_result = client.run_ingestion_and_wait(
        _build_ingestion_payload(config),
        poll_interval_seconds=config.execution.poll_interval_seconds,
        max_wait_seconds=config.execution.max_wait_seconds,
    )
    assignments, source_mapping_summary, payloads = _map_sources(config, cases)
    ingestion_summary = _extract_ingestion_summary(
        ingestion_result.record,
        job_id=ingestion_result.job_id,
        status=ingestion_result.status,
    )
    ingestion_summary["ingestion_metrics"] = summarize_ingestion_payloads(
        payloads,
        documents_found=ingestion_summary.get("documents_found"),
        failed_documents_count=(ingestion_summary.get("raw_endpoint_result") or {}).get("failed_documents_count"),
        failed_chunks_count=(ingestion_summary.get("raw_endpoint_result") or {}).get("failed_chunks_count"),
    )
    ingestion_summary.setdefault("cache", {})
    ingestion_summary["cache"].update({"fingerprint": fingerprint, "status": "created", "source_run_id": run_id})
    registry.put(
        fingerprint=fingerprint,
        run_id=run_id,
        documents_version=config.documents_version,
        ingestion_summary=ingestion_summary,
        source_mapping_summary=source_mapping_summary,
        assignments=assignments,
    )
    return IngestionStageResult(
        ingestion_summary=ingestion_summary,
        source_mapping_summary=source_mapping_summary,
        assignments=assignments,
        cache={"fingerprint": fingerprint, "status": "created", "source_run_id": run_id},
    )
