from __future__ import annotations

from inference.indexing.ingestion_service import IngestionService
from inference.indexing.models import TextChunk
from inference.indexing.vector_indexer import VectorIndexingService
from shared.contracts.ingestion import IngestDocumentsRequest


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.model = "embed-default"
        self.selected_model = None

    def with_model(self, model):
        self.selected_model = model
        return self

    async def embed_many(self, texts):
        return [[float(len(text))] for text in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.collection_name = "guidance"
        self.ensure_calls = []
        self.upsert_calls = []

    def ensure_collection(self, vector_size: int) -> None:
        self.ensure_calls.append(vector_size)

    def upsert_chunks(self, chunks, embeddings):
        self.upsert_calls.append((chunks, embeddings))
        return len(chunks)


class FakeDocStore:
    documents_bucket = "docs"
    documents_prefix = "guidelines"

    def __init__(self):
        self.ensure_called = 0

    def ensure_bucket_exists(self):
        self.ensure_called += 1


class FakeLoader:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def load_all(self, *, split_pdf_pages: bool = False):
        self.calls.append(split_pdf_pages)
        return self.docs


class FakePreparer:
    def __init__(self, source_documents, chunks):
        self.source_documents = source_documents
        self.chunks = chunks
        self.calls = []

    def prepare_documents(self, *, documents, options):
        self.calls.append((documents, options))
        return self.source_documents, self.chunks


class FakeIndexer:
    def __init__(self, result: int):
        self.result = result
        self.calls = []

    async def index_chunks(self, *, chunks, embedding_model=None):
        self.calls.append((chunks, embedding_model))
        return self.result


async def test_vector_indexer_indexes_only_non_empty_chunks():
    embedding_client = FakeEmbeddingClient()
    vector_store = FakeVectorStore()
    service = VectorIndexingService(embedding_client=embedding_client, vector_store=vector_store)
    chunks = [
        TextChunk(chunk_id="1", source_id="a", title="A", text="hello", metadata={}),
        TextChunk(chunk_id="2", source_id="a", title="A", text="   ", metadata={}),
    ]

    indexed = await service.index_chunks(chunks, embedding_model="embed-x")

    assert indexed == 1
    assert embedding_client.selected_model == "embed-x"
    assert vector_store.ensure_calls == [1]
    assert len(vector_store.upsert_calls[0][0]) == 1


async def test_ingestion_service_orchestrates_loader_preparer_and_indexer():
    doc_store = FakeDocStore()
    loaded_documents = [object()]
    source_documents = [object(), object()]
    chunks = [TextChunk(chunk_id="1", source_id="a", title="A", text="hello", metadata={})]
    loader = FakeLoader(loaded_documents)
    preparer = FakePreparer(source_documents, chunks)
    indexer = FakeIndexer(1)
    embedding_client = FakeEmbeddingClient()
    vector_store = FakeVectorStore()
    service = IngestionService(
        document_store=doc_store,
        document_loader=loader,
        embedding_client=embedding_client,
        vector_store=vector_store,
        document_preparer=preparer,
        vector_indexer=indexer,
    )
    request = IngestDocumentsRequest()
    request.options.chunking_strategy = "page_indexed"
    request.options.embedding_model = None

    response = await service.ingest(request)

    assert doc_store.ensure_called == 1
    assert loader.calls == [True]
    assert preparer.calls[0][0] == loaded_documents
    assert indexer.calls[0][0] == chunks
    assert response.documents_found == 2
    assert response.chunks_created == 1
    assert response.vectors_upserted == 1
    assert response.embedding_model == "embed-default"
