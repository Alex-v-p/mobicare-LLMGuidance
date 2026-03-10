from __future__ import annotations

from inference.embeddings.ollama_embeddings import OllamaEmbeddingsClient
from inference.indexing.chunking.factory import ChunkerFactory
from inference.indexing.cleaning.factory import CleanerFactory
from inference.indexing.document_loader import DocumentLoader
from inference.indexing.chunking.utils import normalize_for_offset_matching
from inference.indexing.models import SourceDocument, TextChunk
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
    ) -> None:
        self._document_store = document_store or MinioDocumentStore()
        self._document_loader = document_loader or DocumentLoader(self._document_store)
        self._embedding_client = embedding_client or OllamaEmbeddingsClient()
        self._vector_store = vector_store or QdrantVectorStore()

    async def ingest(self, request: IngestDocumentsRequest | None = None) -> IngestionResponse:
        request = request or IngestDocumentsRequest()
        options = request.options
        self._document_store.ensure_bucket_exists()

        split_pdf_pages = options.chunking_strategy == "page_indexed"
        loaded_documents = self._document_loader.load_all(split_pdf_pages=split_pdf_pages)
        cleaner = CleanerFactory.create(options.cleaning_strategy)
        chunker = ChunkerFactory.create(options.chunking_strategy, options.chunking_params)

        source_documents = []
        for document in loaded_documents:
            cleaned_document = cleaner.clean(
                SourceDocument(
                    source_id=document.path,
                    title=document.title,
                    text=document.text,
                    metadata={**document.metadata, "cleaning_strategy": options.cleaning_strategy},
                )
            )
            source_documents.append(self._enrich_with_page_ranges(cleaned_document))

        chunks: list[TextChunk] = []
        for document in source_documents:
            chunks.extend(chunker.chunk(document))

        if not chunks:
            return IngestionResponse(
                documents_bucket=self._document_store.documents_bucket,
                documents_prefix=self._document_store.documents_prefix,
                documents_found=len(source_documents),
                chunks_created=0,
                vectors_upserted=0,
                collection=self._vector_store.collection_name,
                cleaning_strategy=options.cleaning_strategy,
                chunking_strategy=options.chunking_strategy,
                cleaning_params=options.cleaning_params,
                chunking_params=options.chunking_params,
            )

        safe_chunks = [chunk for chunk in chunks if chunk.text and chunk.text.strip()]

        if not safe_chunks:
            return IngestionResponse(
                documents_bucket=self._document_store.documents_bucket,
                documents_prefix=self._document_store.documents_prefix,
                documents_found=len(source_documents),
                chunks_created=0,
                vectors_upserted=0,
                collection=self._vector_store.collection_name,
                cleaning_strategy=options.cleaning_strategy,
                chunking_strategy=options.chunking_strategy,
                cleaning_params=options.cleaning_params,
                chunking_params=options.chunking_params,
            )

        embeddings = await self._embedding_client.embed_many([chunk.text for chunk in safe_chunks])
        self._vector_store.ensure_collection(vector_size=len(embeddings[0]))
        vectors_upserted = self._vector_store.upsert_chunks(safe_chunks, embeddings)

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
        )

    def _enrich_with_page_ranges(self, document: SourceDocument) -> SourceDocument:
        metadata = dict(document.metadata)

        if metadata.get("page_number") is not None:
            metadata["normalized_source_text"] = normalize_for_offset_matching(document.text)
            return SourceDocument(
                source_id=document.source_id,
                title=document.title,
                text=document.text,
                metadata=metadata,
            )

        raw_text = document.text or ""
        if not raw_text.strip():
            return document

        pages = [page for page in raw_text.split("\n\n") if page.strip()]
        if not pages:
            metadata["normalized_source_text"] = normalize_for_offset_matching(raw_text)
            return SourceDocument(
                source_id=document.source_id,
                title=document.title,
                text=document.text,
                metadata=metadata,
            )

        page_ranges = []
        cursor = 0
        normalized_pages: list[str] = []

        for index, page_text in enumerate(pages, start=1):
            normalized_page = normalize_for_offset_matching(page_text)
            if not normalized_page:
                continue

            start = cursor
            end = start + len(normalized_page)
            page_ranges.append(
                {
                    "page_number": index,
                    "start": start,
                    "end": end,
                }
            )
            normalized_pages.append(normalized_page)
            cursor = end + 1

        normalized_source_text = " ".join(normalized_pages)
        metadata["page_ranges"] = page_ranges
        metadata["normalized_source_text"] = normalized_source_text

        return SourceDocument(
            source_id=document.source_id,
            title=document.title,
            text=document.text,
            metadata=metadata,
        )