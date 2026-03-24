from __future__ import annotations

from typing import Any


_GUIDANCE_ENVELOPE_KEYS = {
    "job_id",
    "request_id",
    "status",
    "model",
    "error",
    "result_object_key",
    "callback_attempts",
    "callback_last_status",
    "callback_last_error",
    "worker_id",
    "lease_expires_at",
    "created_at",
    "started_at",
    "completed_at",
    "updated_at",
}


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def extract_retrieved_context(record: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = _coerce_dict(record)
    rag = snapshot.get("rag")
    if isinstance(rag, list):
        return [dict(item) for item in rag if isinstance(item, dict)]
    retrieved_context = snapshot.get("retrieved_context")
    if isinstance(retrieved_context, list):
        return [dict(item) for item in retrieved_context if isinstance(item, dict)]
    return []


def infer_response_shape(record: dict[str, Any]) -> str:
    snapshot = _coerce_dict(record)
    metadata = _coerce_dict(snapshot.get("metadata"))
    if any(key in snapshot for key in ("job_id", "result_object_key", "worker_id", "callback_attempts")):
        return "async_job_expanded"
    if metadata.get("retrieval_attempts") is not None or metadata.get("retrieval_ranking") is not None:
        return "multistep_diagnostics"
    if "rag" in snapshot:
        return "api_job_status"
    if "retrieved_context" in snapshot:
        return "inference_response"
    return "unknown"


def extract_endpoint_envelope(record: dict[str, Any]) -> dict[str, Any]:
    snapshot = _coerce_dict(record)
    envelope = {key: snapshot.get(key) for key in _GUIDANCE_ENVELOPE_KEYS if key in snapshot}
    envelope["response_shape"] = infer_response_shape(snapshot)
    return envelope


def normalize_guidance_record(record: dict[str, Any]) -> dict[str, Any]:
    snapshot = _coerce_dict(record)
    metadata = _coerce_dict(snapshot.get("metadata"))
    normalized = dict(snapshot)
    normalized["rag"] = extract_retrieved_context(snapshot)
    normalized["warnings"] = list(snapshot.get("warnings") or [])
    normalized["metadata"] = metadata
    if snapshot.get("verification") is not None:
        normalized["verification"] = _coerce_dict(snapshot.get("verification"))
    normalized["endpoint_envelope"] = extract_endpoint_envelope(snapshot)
    diagnostics = {
        "retrieval_attempts": metadata.get("retrieval_attempts"),
        "retrieval_attempt_details": _coerce_list_of_dicts(metadata.get("retrieval_attempt_details")),
        "retrieval_ranking": _coerce_list_of_dicts(metadata.get("retrieval_ranking")),
        "context_assessment": _coerce_dict(metadata.get("context_assessment")),
        "retrieval_clusters": list(metadata.get("retrieval_clusters") or []),
        "retrieval_queries": list(metadata.get("retrieval_queries") or []),
        "clinical_abnormal_variables": list(metadata.get("clinical_abnormal_variables") or []),
        "clinical_unknown_variables": list(metadata.get("clinical_unknown_variables") or []),
    }
    normalized["diagnostics"] = diagnostics
    return normalized
