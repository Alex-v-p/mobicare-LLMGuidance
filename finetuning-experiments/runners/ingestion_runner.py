from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from adapters.gateway import GatewayClient
from adapters.qdrant import QdrantScrollClient
from configs.schema import BenchmarkRunConfig
from datasets.schema import BenchmarkCase
from source_mapping.matcher import SourceMatcher

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionStageResult:
    ingestion_summary: dict[str, Any]
    source_mapping_summary: dict[str, Any]
    assignments: list[dict[str, Any]]

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
        "raw_endpoint_result": result_record,
    }


def _map_sources(config: BenchmarkRunConfig, cases: list[BenchmarkCase]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    assignments = [matcher.build_chunk_assignment(case=case, mapping_label=config.label, payloads=payloads).to_dict() for case in cases]
    summary = {
        "mapping_label": config.label,
        "case_count": len(assignments),
        "payload_count": len(payloads),
        "matcher": {
            "max_matches": config.source_mapping.max_matches,
            "page_window": config.source_mapping.page_window,
            "page_offset_candidates": list(config.source_mapping.page_offset_candidates),
            "semantic_fallback_enabled": config.source_mapping.semantic_fallback_enabled,
            "include_chunk_pairs": config.source_mapping.include_chunk_pairs,
        },
        "case_chunk_assignments": assignments,
    }
    return assignments, summary


def run_ingestion_stage(config: BenchmarkRunConfig, cases: list[BenchmarkCase]) -> IngestionStageResult:
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
    assignments, source_mapping_summary = _map_sources(config, cases)
    return IngestionStageResult(
        ingestion_summary=_extract_ingestion_summary(
            ingestion_result.record,
            job_id=ingestion_result.job_id,
            status=ingestion_result.status,
        ),
        source_mapping_summary=source_mapping_summary,
        assignments=assignments,
    )
