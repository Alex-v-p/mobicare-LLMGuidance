from __future__ import annotations

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.indexing.chunking.basic_chunker import BasicChunker
from inference.indexing.document_loader import DocumentLoader
from inference.indexing.models import SourceDocument, TextChunk
from inference.storage.minio_documents import MinioDocumentStore
from inference.storage.qdrant_store import QdrantVectorStore
from shared.contracts.ingestion import IngestionResponse


class IngestionService:
    def __init__(
        self,
        document_store: MinioDocumentStore | None = None,
        document_loader: DocumentLoader | None = None,
        chunker: BasicChunker | None = None,
        embedding_client: OllamaEmbeddingsClient | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self._document_store = document_store or MinioDocumentStore()
        self._document_loader = document_loader or DocumentLoader(self._document_store)
        self._chunker = chunker or BasicChunker()
        self._embedding_client = embedding_client or OllamaEmbeddingsClient()
        self._vector_store = vector_store or QdrantVectorStore()

    async def ingest(self) -> IngestionResponse:
        self._document_store.ensure_bucket_exists()
        loaded_documents = self._document_loader.load_all()

        source_documents = [
            SourceDocument(
                source_id=document.path,
                title=document.title,
                text=document.text,
                metadata={"object_name": document.path},
            )
            for document in loaded_documents
        ]

        chunks: list[TextChunk] = []
        for document in source_documents:
            chunks.extend(self._chunker.chunk(document))

        if not chunks:
            return IngestionResponse(
                documents_bucket=self._document_store.documents_bucket,
                documents_prefix=self._document_store.documents_prefix,
                documents_found=len(source_documents),
                chunks_created=0,
                vectors_upserted=0,
                collection=self._vector_store.collection_name,
            )

        embeddings = await self._embedding_client.embed_many([chunk.text for chunk in chunks])
        self._vector_store.ensure_collection(vector_size=len(embeddings[0]))
        vectors_upserted = self._vector_store.upsert_chunks(chunks, embeddings)

        return IngestionResponse(
            documents_bucket=self._document_store.documents_bucket,
            documents_prefix=self._document_store.documents_prefix,
            documents_found=len(source_documents),
            chunks_created=len(chunks),
            vectors_upserted=vectors_upserted,
            collection=self._vector_store.collection_name,
        )
