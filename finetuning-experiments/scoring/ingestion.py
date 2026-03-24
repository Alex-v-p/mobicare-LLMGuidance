from __future__ import annotations

from statistics import median
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default



def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default



def _extract_chunk_text(payload: dict[str, Any]) -> str:
    for key in ("text", "chunk_text", "content", "page_content", "snippet"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    metadata = payload.get("metadata") or {}
    for key in ("text", "chunk_text", "content", "page_content", "snippet"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""



def _extract_doc_id(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") or {}
    for key in ("document_id", "doc_id", "source_document_id", "source_id", "file_id"):
        value = payload.get(key)
        if value is not None:
            return str(value)
        value = metadata.get(key)
        if value is not None:
            return str(value)
    for key in ("document_name", "source", "filename", "file_name"):
        value = payload.get(key)
        if value is not None:
            return str(value)
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return "unknown"



def _extract_page(payload: dict[str, Any]) -> int | None:
    metadata = payload.get("metadata") or {}
    for key in ("page", "page_number", "page_index"):
        raw = payload.get(key, metadata.get(key))
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None



def summarize_ingestion_payloads(
    payloads: list[dict[str, Any]],
    *,
    documents_found: int | None = None,
    failed_documents_count: int | None = None,
    failed_chunks_count: int | None = None,
) -> dict[str, Any]:
    chunk_lengths = [len(_extract_chunk_text(payload).split()) for payload in payloads]
    chunk_count = len(payloads)
    doc_chunk_counts: dict[str, int] = {}
    pages_by_doc: dict[str, set[int]] = {}
    for payload in payloads:
        doc_id = _extract_doc_id(payload)
        doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
        page = _extract_page(payload)
        if page is not None:
            pages_by_doc.setdefault(doc_id, set()).add(page)

    chunks_per_document_values = list(doc_chunk_counts.values())
    document_count = _safe_int(documents_found, len(doc_chunk_counts) or 0)
    avg_chunks_per_document = (
        sum(chunks_per_document_values) / len(chunks_per_document_values) if chunks_per_document_values else 0.0
    )
    page_coverage_ratio = (
        len(pages_by_doc) / document_count if document_count > 0 else 0.0
    )

    return {
        "chunk_count_observed": chunk_count,
        "document_count_observed": len(doc_chunk_counts),
        "average_chunk_length": (sum(chunk_lengths) / len(chunk_lengths)) if chunk_lengths else 0.0,
        "median_chunk_length": median(chunk_lengths) if chunk_lengths else 0.0,
        "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0.0,
        "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0.0,
        "chunks_per_document": {
            "average": avg_chunks_per_document,
            "min": min(chunks_per_document_values) if chunks_per_document_values else 0.0,
            "max": max(chunks_per_document_values) if chunks_per_document_values else 0.0,
        },
        "page_coverage_ratio": page_coverage_ratio,
        "failed_documents_count": _safe_int(failed_documents_count, 0),
        "failed_chunks_count": _safe_int(failed_chunks_count, 0),
    }
