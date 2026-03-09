from __future__ import annotations

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.indexing.chunking.basic_chunker import BasicChunker
from inference.indexing.models import SourceDocument
from inference.storage.minio_documents import MinioDocumentStore
from inference.storage.qdrant_store import QdrantVectorStore
from shared.contracts.ingestion import IngestionResponse


class IngestionService:
    def __init__(
        self,
        document_store: MinioDocumentStore | None = None,
        chunker: BasicChunker | None = None,
        embedding_client: OllamaEmbeddingsClient | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self._document_store = document_store or MinioDocumentStore()
        self._chunker = chunker or BasicChunker()
        self._embedding_client = embedding_client or OllamaEmbeddingsClient()
        self._vector_store = vector_store or QdrantVectorStore()

    @property
    def collection_name(self) -> str:
        return self._vector_store.collection_name

    async def ingest(self, recreate_collection: bool = False, bucket: str | None = None, prefix: str | None = None) -> IngestionResponse:
        minio_docs = self._document_store.list_documents(bucket=bucket, prefix=prefix)
        documents = [
            SourceDocument(
                source_id=doc.object_name.replace("/", "_").replace(".", "_"),
                title=doc.title,
                text=doc.text,
                metadata=doc.metadata,
            )
            for doc in minio_docs
        ]

        chunks = []
        for document in documents:
            chunks.extend(self._chunker.chunk(document))

        if not chunks:
            return IngestionResponse(
                bucket=bucket or self._document_store.default_bucket,
                prefix=prefix if prefix is not None else self._document_store.default_prefix,
                documents_found=len(documents),
                chunks_created=0,
                vectors_upserted=0,
                collection=self.collection_name,
            )

        embeddings = await self._embedding_client.embed_many([chunk.text for chunk in chunks])
        self._vector_store.ensure_collection(vector_size=len(embeddings[0]), recreate=recreate_collection)
        upserted = self._vector_store.upsert_chunks(chunks, embeddings)
        return IngestionResponse(
            bucket=bucket or self._document_store.default_bucket,
            prefix=prefix if prefix is not None else self._document_store.default_prefix,
            documents_found=len(documents),
            chunks_created=len(chunks),
            vectors_upserted=upserted,
            collection=self.collection_name,
        )
