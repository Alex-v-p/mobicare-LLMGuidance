from __future__ import annotations

from datetime import datetime
from typing import Any

from adapters.guidance_payloads import normalize_guidance_record
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
    normalized = normalize_guidance_record(record)
    metadata = dict(normalized.get("metadata") or {})
    explicit_stages = metadata.get("stages") or normalized.get("stages") or []
    if explicit_stages:
        envelope = TelemetryEnvelope(
            stages=[_coerce_stage(stage) for stage in explicit_stages if isinstance(stage, dict)],
            derived=_derive_common_fields(normalized),
            source="endpoint",
        )
        return envelope.to_dict()

    stages: list[TelemetryStage] = []
    derived = _derive_common_fields(normalized)
    queue_delay_ms = derived.get("queue_delay_ms")
    execution_duration_ms = derived.get("execution_duration_ms")

    if queue_delay_ms is not None:
        stages.append(
            TelemetryStage(
                name="job_queue",
                status="completed",
                duration_ms=queue_delay_ms,
                started_at=normalized.get("created_at"),
                completed_at=normalized.get("started_at"),
            )
        )

    retrieval_mode = metadata.get("retrieval_mode")
    retrieval_attempts = metadata.get("retrieval_attempts")
    retrieval_attempt_details = [item for item in (metadata.get("retrieval_attempt_details") or []) if isinstance(item, dict)]
    retrieval_ranking = [item for item in (metadata.get("retrieval_ranking") or []) if isinstance(item, dict)]
    context_assessment = dict(metadata.get("context_assessment") or {})

    if metadata.get("query_rewritten"):
        stages.append(
            TelemetryStage(
                name="query_rewrite",
                status="completed",
                metadata={
                    "query_rewritten": True,
                    "effective_question": metadata.get("effective_question"),
                    "retrieval_query": metadata.get("retrieval_query"),
                },
            )
        )
    elif metadata.get("use_retrieval"):
        stages.append(
            TelemetryStage(
                name="query_rewrite",
                status="skipped",
                metadata={
                    "query_rewritten": False,
                    "effective_question": metadata.get("effective_question"),
                    "retrieval_query": metadata.get("retrieval_query"),
                },
            )
        )

    if metadata.get("use_retrieval"):
        if retrieval_attempt_details:
            dense_candidates = [float(item.get("dense_candidates", 0)) for item in retrieval_attempt_details if item.get("dense_candidates") is not None]
            sparse_candidates = [float(item.get("sparse_candidates", 0)) for item in retrieval_attempt_details if item.get("sparse_candidates") is not None]
            stages.append(
                TelemetryStage(
                    name="retrieval_planning",
                    status="completed",
                    metadata={
                        "attempt_count": retrieval_attempts or len(retrieval_attempt_details),
                        "query_count": len(metadata.get("retrieval_queries") or []),
                        "cluster_count": len(metadata.get("retrieval_clusters") or []),
                        "specialty_focus": metadata.get("specialty_focus"),
                    },
                )
            )
            stages.append(
                TelemetryStage(
                    name=f"retrieval_{retrieval_mode or 'default'}",
                    status="completed",
                    metadata={
                        "attempt_count": retrieval_attempts or len(retrieval_attempt_details),
                        "returned_items": sum(int(item.get("returned_items") or 0) for item in retrieval_attempt_details),
                        "average_dense_candidates": (sum(dense_candidates) / len(dense_candidates)) if dense_candidates else None,
                        "average_sparse_candidates": (sum(sparse_candidates) / len(sparse_candidates)) if sparse_candidates else None,
                    },
                )
            )
        elif retrieval_mode == "hybrid":
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
                    metadata={"attempt_count": retrieval_attempts},
                )
            )

        graph_edges_used = list(metadata.get("graph_edges_used") or [])
        if not graph_edges_used and retrieval_attempt_details:
            for attempt in retrieval_attempt_details:
                graph_edges_used.extend(list(attempt.get("graph_edges_used") or []))
        graph_nodes_added = metadata.get("graph_nodes_added")
        if graph_nodes_added is None and retrieval_attempt_details:
            graph_nodes_added = sum(int(item.get("graph_nodes_added") or 0) for item in retrieval_attempt_details)

        if metadata.get("graph_augmented") or metadata.get("use_graph_augmentation") or graph_edges_used or graph_nodes_added:
            stages.append(
                TelemetryStage(
                    name="graph_augmentation",
                    status="completed" if (metadata.get("graph_augmented") or graph_edges_used or graph_nodes_added) else "requested",
                    metadata={
                        "graph_nodes_added": graph_nodes_added,
                        "graph_edges_used": graph_edges_used,
                    },
                )
            )

        if retrieval_ranking or context_assessment:
            stages.append(
                TelemetryStage(
                    name="context_assessment",
                    status="completed" if context_assessment else "unknown",
                    metadata={
                        "sufficient": context_assessment.get("sufficient"),
                        "confidence": context_assessment.get("confidence"),
                        "reason_count": len(context_assessment.get("reasons") or []),
                        "ranking_count": len(retrieval_ranking),
                        "top_score": retrieval_ranking[0].get("score") if retrieval_ranking else None,
                    },
                )
            )

    generation_stage_metadata = {
        "llm_model": metadata.get("llm_model") or normalized.get("model"),
        "prompt_character_count": metadata.get("prompt_character_count"),
        "response_regeneration_attempts": metadata.get("response_regeneration_attempts"),
        "response_shape": (normalized.get("endpoint_envelope") or {}).get("response_shape"),
        "warning_count": len(normalized.get("warnings") or []),
    }
    stages.append(
        TelemetryStage(
            name="generation",
            status=str(normalized.get("status") or "unknown"),
            duration_ms=execution_duration_ms,
            started_at=normalized.get("started_at"),
            completed_at=normalized.get("completed_at"),
            metadata={key: value for key, value in generation_stage_metadata.items() if value is not None},
        )
    )

    verification = normalized.get("verification")
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
