from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference.retrieval.common import RetrievalCollectionNotReadyError, payload_identity, payload_to_context, search_qdrant
from inference.retrieval.dense import DenseRetriever
from inference.retrieval.graph import ChunkGraphAugmenter
from inference.retrieval.hybrid import HybridRetriever
from inference.retrieval.sparse import SparseKeywordRetriever
from inference.storage.qdrant_store import MissingCollectionError


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.selected_model = None

    def with_model(self, model):
        self.selected_model = model
        return self

    async def embed(self, query):
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self, *, exists=True, has_points=True, search_hits=None, payloads=None, missing=False):
        self.collection_name = "guidance"
        self.exists = exists
        self.has_points = has_points
        self.search_hits = search_hits or []
        self.payloads = payloads or []
        self.missing = missing

    def collection_exists(self):
        return self.exists

    def collection_has_points(self):
        return self.has_points

    def search(self, query_vector, limit):
        if self.missing:
            raise MissingCollectionError("missing")
        return self.search_hits[:limit]

    def count_points(self):
        return len(self.payloads)

    def get_all_payloads(self):
        return list(self.payloads)


def test_common_payload_helpers_and_collection_ready_error_mapping():
    payload = {"chunk_id": "c1", "source_id": "s1", "title": "T", "text": "Body", "page_number": 3}
    assert payload_identity(payload) == "c1"
    context = payload_to_context(payload)
    assert context.chunk_id == "c1"
    assert context.page_number == 3

    with pytest.raises(RetrievalCollectionNotReadyError):
        search_qdrant(vector_store=FakeVectorStore(exists=False), query_vector=[0.1], limit=1)
    with pytest.raises(RetrievalCollectionNotReadyError):
        search_qdrant(vector_store=FakeVectorStore(missing=True), query_vector=[0.1], limit=1)


async def test_dense_retriever_embeds_query_and_ignores_empty_payloads():
    hits = [SimpleNamespace(payload={"source_id": "s1", "title": "Guide", "text": "HF advice", "chunk_id": "c1"}, score=0.9), SimpleNamespace(payload=None, score=0.2)]
    retriever = DenseRetriever(embedding_client=FakeEmbeddingClient(), vector_store=FakeVectorStore(search_hits=hits))

    results = await retriever.retrieve("heart failure", limit=2, embedding_model="embed-z")

    assert len(results) == 1
    assert results[0].title == "Guide"


def test_sparse_keyword_retriever_ranks_matching_documents_and_reuses_cache():
    retriever = SparseKeywordRetriever()
    documents = [
        {"chunk_id": "a", "source_id": "s1", "title": "HF", "text": "heart failure ace inhibitor treatment"},
        {"chunk_id": "b", "source_id": "s2", "title": "AF", "text": "atrial fibrillation anticoagulation"},
    ]

    hits1 = retriever.search("heart failure treatment", documents, limit=2)
    cached_index = retriever._cached_index
    hits2 = retriever.search("heart failure", documents, limit=2)

    assert hits1[0].source_id == "s1"
    assert retriever._cached_index is cached_index
    assert hits2[0].title == "HF"


def test_graph_augmenter_adds_adjacent_relevant_chunks():
    ranked = [{"chunk_id": "c2", "source_id": "s1", "title": "Guide", "text": "ace inhibitor for heart failure", "chunk_index": 1}]
    corpus = [
        {"chunk_id": "c1", "source_id": "s1", "title": "Guide", "text": "introduction", "chunk_index": 0},
        ranked[0],
        {"chunk_id": "c3", "source_id": "s1", "title": "Guide", "text": "more heart failure treatment", "chunk_index": 2},
    ]

    items, metadata = ChunkGraphAugmenter().expand(query="heart failure treatment", ranked_payloads=ranked, corpus_payloads=corpus, max_extra_nodes=1)

    assert len(items) == 2
    assert metadata["graph_augmented"] is True
    assert metadata["graph_nodes_added"] == 1
    assert metadata["graph_edges_used"]


async def test_hybrid_retriever_fuses_dense_and_sparse_results_and_can_augment_graph():
    payloads = [
        {"chunk_id": "c1", "source_id": "s1", "title": "Guide", "text": "heart failure ace inhibitor", "chunk_index": 0},
        {"chunk_id": "c2", "source_id": "s1", "title": "Guide", "text": "beta blocker recommendation", "chunk_index": 1},
    ]
    dense_hits = [SimpleNamespace(payload=payloads[0], score=0.9), SimpleNamespace(payload=payloads[1], score=0.2)]
    retriever = HybridRetriever(
        embedding_client=FakeEmbeddingClient(),
        vector_store=FakeVectorStore(search_hits=dense_hits, payloads=payloads),
    )

    result = await retriever.retrieve(query="heart failure", limit=1, use_graph_augmentation=True, graph_max_extra_nodes=1)

    assert result.items[0].chunk_id == "c1"
    assert result.metadata["retrieval_mode"] == "hybrid"
    assert result.metadata["dense_candidates"] == 2
    assert result.metadata["sparse_candidates"] >= 1
