from __future__ import annotations

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.indexing.document_loader import DocumentLoader
from inference.indexing.document_preparer import DocumentPreparationService
from inference.indexing.vector_indexer import VectorIndexingService
from inference.storage.minio_documents import MinioDocumentStore
from inference.storage.qdrant_store import QdrantVectorStore
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionResponse


class IngestionService:
    def __init__(
        self,
        document_store: MinioDocumentStore | None = None,
        document_loader: DocumentLoader | None = None,
        embedding_client: OllamaEmbeddingsClient | None = None,
        vector_store: QdrantVectorStore | None = None,
        document_preparer: DocumentPreparationService | None = None,
        vector_indexer: VectorIndexingService | None = None,
    ) -> None:
        shared_document_store = document_store or MinioDocumentStore()
        shared_document_loader = document_loader or DocumentLoader(shared_document_store)
        shared_embedding_client = embedding_client or OllamaEmbeddingsClient()
        shared_vector_store = vector_store or QdrantVectorStore()
        self._document_store = shared_document_store
        self._document_loader = shared_document_loader
        self._document_preparer = document_preparer or DocumentPreparationService()
        self._vector_indexer = vector_indexer or VectorIndexingService(
            embedding_client=shared_embedding_client,
            vector_store=shared_vector_store,
        )
        self._vector_store = shared_vector_store
        self._embedding_client = shared_embedding_client

    async def ingest(self, request: IngestDocumentsRequest | None = None) -> IngestionResponse:
        request = request or IngestDocumentsRequest()
        options = request.options
        self._document_store.ensure_bucket_exists()

        split_pdf_pages = options.chunking_strategy == "page_indexed"
        loaded_documents = self._document_loader.load_all(split_pdf_pages=split_pdf_pages)
        source_documents, chunks = self._document_preparer.prepare_documents(
            documents=loaded_documents,
            options=options,
        )

        vectors_upserted = await self._vector_indexer.index_chunks(
            chunks=chunks,
            embedding_model=options.embedding_model,
        )

        return IngestionResponse(
            documents_bucket=self._document_store.documents_bucket,
            documents_prefix=self._document_store.documents_prefix,
            documents_found=len(source_documents),
            chunks_created=len(chunks),
            vectors_upserted=vectors_upserted,
            collection=self._vector_store.collection_name,
            cleaning_strategy=options.cleaning_strategy,
            chunking_strategy=options.chunking_strategy,
            cleaning_params=options.cleaning_params,
            chunking_params=options.chunking_params,
            embedding_model=options.embedding_model or self._embedding_client.model,
        )
