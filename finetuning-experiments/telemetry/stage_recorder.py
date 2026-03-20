from __future__ import annotations

from datetime import datetime
from typing import Any

from telemetry.stage_models import TelemetryEnvelope, TelemetryStage


def _parse_iso(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None



def _duration_ms(start_value: Any, end_value: Any) -> float | None:
    start = _parse_iso(start_value)
    end = _parse_iso(end_value)
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds() * 1000.0)



def _coerce_stage(raw_stage: dict[str, Any]) -> TelemetryStage:
    return TelemetryStage(
        name=str(raw_stage.get("name") or raw_stage.get("stage") or "unknown"),
        status=str(raw_stage.get("status") or "unknown"),
        duration_ms=(float(raw_stage["duration_ms"]) if raw_stage.get("duration_ms") is not None else None),
        started_at=raw_stage.get("started_at"),
        completed_at=raw_stage.get("completed_at"),
        metadata=dict(raw_stage.get("metadata") or {}),
    )



def _derive_common_fields(record: dict[str, Any]) -> dict[str, float | None]:
    queue_delay_ms = _duration_ms(record.get("created_at"), record.get("started_at"))
    execution_duration_ms = _duration_ms(record.get("started_at"), record.get("completed_at"))
    total_duration_ms = _duration_ms(record.get("created_at"), record.get("completed_at"))
    return {
        "queue_delay_ms": queue_delay_ms,
        "execution_duration_ms": execution_duration_ms,
        "total_duration_ms": total_duration_ms,
    }



def extract_guidance_telemetry(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    explicit_stages = metadata.get("stages") or record.get("stages") or []
    if explicit_stages:
        envelope = TelemetryEnvelope(
            stages=[_coerce_stage(stage) for stage in explicit_stages if isinstance(stage, dict)],
            derived=_derive_common_fields(record),
            source="endpoint",
        )
        return envelope.to_dict()

    stages: list[TelemetryStage] = []
    derived = _derive_common_fields(record)
    queue_delay_ms = derived.get("queue_delay_ms")
    execution_duration_ms = derived.get("execution_duration_ms")

    if queue_delay_ms is not None:
        stages.append(
            TelemetryStage(
                name="job_queue",
                status="completed",
                duration_ms=queue_delay_ms,
                started_at=record.get("created_at"),
                completed_at=record.get("started_at"),
            )
        )

    retrieval_mode = metadata.get("retrieval_mode")
    if metadata.get("query_rewritten"):
        stages.append(
            TelemetryStage(
                name="query_rewrite",
                status="completed",
                metadata={"query_rewritten": True},
            )
        )
    elif metadata.get("use_retrieval"):
        stages.append(
            TelemetryStage(
                name="query_rewrite",
                status="skipped",
                metadata={"query_rewritten": False},
            )
        )

    if metadata.get("use_retrieval"):
        if retrieval_mode == "hybrid":
            if metadata.get("dense_candidates") is not None:
                stages.append(
                    TelemetryStage(
                        name="retrieval_dense",
                        status="completed",
                        metadata={"candidate_count": metadata.get("dense_candidates")},
                    )
                )
            if metadata.get("sparse_candidates") is not None:
                stages.append(
                    TelemetryStage(
                        name="retrieval_sparse",
                        status="completed",
                        metadata={"candidate_count": metadata.get("sparse_candidates")},
                    )
                )
        else:
            stages.append(
                TelemetryStage(
                    name=f"retrieval_{retrieval_mode or 'default'}",
                    status="completed",
                )
            )

        if metadata.get("graph_augmented") or metadata.get("use_graph_augmentation"):
            stages.append(
                TelemetryStage(
                    name="graph_augmentation",
                    status="completed" if metadata.get("graph_augmented") else "requested",
                    metadata={
                        "graph_nodes_added": metadata.get("graph_nodes_added"),
                        "graph_edges_used": list(metadata.get("graph_edges_used") or []),
                    },
                )
            )

    generation_stage_metadata = {
        "llm_model": metadata.get("llm_model"),
        "prompt_character_count": metadata.get("prompt_character_count"),
        "response_regeneration_attempts": metadata.get("response_regeneration_attempts"),
    }
    stages.append(
        TelemetryStage(
            name="generation",
            status=str(record.get("status") or "unknown"),
            duration_ms=execution_duration_ms,
            started_at=record.get("started_at"),
            completed_at=record.get("completed_at"),
            metadata={key: value for key, value in generation_stage_metadata.items() if value is not None},
        )
    )

    verification = record.get("verification")
    if verification is not None:
        stages.append(
            TelemetryStage(
                name="verification",
                status="completed",
                metadata={
                    "verdict": verification.get("verdict"),
                    "confidence": verification.get("confidence"),
                    "issue_count": len(verification.get("issues") or []),
                },
            )
        )
    elif metadata.get("enable_response_verification"):
        stages.append(TelemetryStage(name="verification", status="skipped"))

    return TelemetryEnvelope(stages=stages, derived=derived, source="derived").to_dict()



def extract_ingestion_telemetry(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict((record.get("result") or {}) if isinstance(record, dict) else {})
    explicit_stages = payload.get("stages") or record.get("stages") or []
    if explicit_stages:
        envelope = TelemetryEnvelope(
            stages=[_coerce_stage(stage) for stage in explicit_stages if isinstance(stage, dict)],
            derived=_derive_common_fields(record),
            source="endpoint",
        )
        return envelope.to_dict()

    derived = _derive_common_fields(record)
    stages: list[TelemetryStage] = []
    queue_delay_ms = derived.get("queue_delay_ms")
    execution_duration_ms = derived.get("execution_duration_ms")

    if queue_delay_ms is not None:
        stages.append(
            TelemetryStage(
                name="job_queue",
                status="completed",
                duration_ms=queue_delay_ms,
                started_at=record.get("created_at"),
                completed_at=record.get("started_at"),
            )
        )

    stages.append(
        TelemetryStage(
            name="ingestion_execution",
            status=str(record.get("status") or "unknown"),
            duration_ms=execution_duration_ms,
            started_at=record.get("started_at"),
            completed_at=record.get("completed_at"),
            metadata={
                "documents_found": payload.get("documents_found"),
                "chunks_created": payload.get("chunks_created"),
                "vectors_upserted": payload.get("vectors_upserted"),
                "cleaning_strategy": payload.get("cleaning_strategy"),
                "chunking_strategy": payload.get("chunking_strategy"),
                "embedding_model": payload.get("embedding_model"),
            },
        )
    )

    return TelemetryEnvelope(stages=stages, derived=derived, source="derived").to_dict()
