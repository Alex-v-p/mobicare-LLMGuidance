from __future__ import annotations

from typing import Any

from shared.contracts.inference import RetrievedContext

from inference.storage.qdrant_store import (
    MissingCollectionEmbeddingModelError,
    MissingCollectionError,
    QdrantVectorStore,
)


class RetrievalCollectionNotReadyError(RuntimeError):
    pass


class RetrievalEmbeddingModelError(RuntimeError):
    pass


def ensure_collection_ready(vector_store: QdrantVectorStore) -> None:
    if not vector_store.collection_exists():
        raise RetrievalCollectionNotReadyError(
            f"Qdrant collection '{vector_store.collection_name}' does not exist yet. Run document ingestion first."
        )
    if not vector_store.collection_has_points():
        raise RetrievalCollectionNotReadyError(
            f"Qdrant collection '{vector_store.collection_name}' is empty. Run document ingestion first."
        )


def search_qdrant(
    *,
    vector_store: QdrantVectorStore,
    query_vector: list[float],
    limit: int,
):
    ensure_collection_ready(vector_store)
    try:
        return vector_store.search(query_vector=query_vector, limit=limit)
    except MissingCollectionError as exc:
        raise RetrievalCollectionNotReadyError(str(exc)) from exc


def resolve_collection_embedding_model(
    *,
    vector_store: QdrantVectorStore,
    requested_embedding_model: str | None,
) -> str:
    ensure_collection_ready(vector_store)
    try:
        stored_embedding_model = vector_store.get_collection_embedding_model()
    except MissingCollectionError as exc:
        raise RetrievalCollectionNotReadyError(str(exc)) from exc
    except MissingCollectionEmbeddingModelError as exc:
        raise RetrievalEmbeddingModelError(str(exc)) from exc

    if requested_embedding_model and requested_embedding_model != stored_embedding_model:
        raise RetrievalEmbeddingModelError(
            f"Guidance request asked for embedding model '{requested_embedding_model}', but "
            f"collection '{vector_store.collection_name}' was ingested with '{stored_embedding_model}'. "
            "Remove the request override or re-ingest the collection with the requested embedding model."
        )
    return stored_embedding_model


def payload_identity(payload: dict[str, Any]) -> str:
    return str(payload.get("chunk_id") or payload.get("source_id") or "unknown")



def payload_to_context(payload: dict[str, Any]) -> RetrievedContext:
    return RetrievedContext(
        source_id=str(payload.get("source_id") or payload.get("chunk_id") or "unknown"),
        title=str(payload.get("title") or payload.get("object_name") or "Untitled"),
        snippet=str(payload.get("text") or ""),
        chunk_id=str(payload.get("chunk_id")) if payload.get("chunk_id") is not None else None,
        page_number=int(payload.get("page_number")) if payload.get("page_number") is not None else None,
    )
