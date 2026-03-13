from __future__ import annotations

from inference.indexing.chunking.factory import ChunkerFactory
from inference.indexing.chunking.utils import build_page_ranges, normalize_for_offset_matching
from inference.indexing.cleaning.factory import CleanerFactory
from inference.indexing.models import SourceDocument, TextChunk
from inference.indexing.document_loader import LoadedDocument
from shared.contracts.ingestion import IngestionOptions


class DocumentPreparationService:
    def prepare_documents(
        self,
        *,
        documents: list[LoadedDocument],
        options: IngestionOptions,
    ) -> tuple[list[SourceDocument], list[TextChunk]]:
        cleaner = CleanerFactory.create(options.cleaning_strategy)
        chunker = ChunkerFactory.create(options.chunking_strategy, options.chunking_params)

        source_documents: list[SourceDocument] = []
        for document in documents:
            cleaned_document = cleaner.clean(
                SourceDocument(
                    source_id=document.path,
                    title=document.title,
                    text=document.text,
                    metadata={**document.metadata, "cleaning_strategy": options.cleaning_strategy},
                )
            )
            source_documents.append(self._enrich_with_page_ranges(cleaned_document, original_metadata=document.metadata))

        chunks: list[TextChunk] = []
        for document in source_documents:
            chunks.extend(chunker.chunk(document))
        return source_documents, chunks

    def _enrich_with_page_ranges(self, document: SourceDocument, *, original_metadata: dict[str, object]) -> SourceDocument:
        metadata = dict(document.metadata)
        metadata["normalized_source_text"] = normalize_for_offset_matching(document.text)

        if metadata.get("page_number") is not None:
            return SourceDocument(
                source_id=document.source_id,
                title=document.title,
                text=document.text,
                metadata=metadata,
            )

        raw_page_texts = original_metadata.get("raw_page_texts")
        if isinstance(raw_page_texts, list) and raw_page_texts:
            page_ranges, normalized_source_text = build_page_ranges(str(page) for page in raw_page_texts)
            if page_ranges:
                metadata["page_ranges"] = page_ranges
                metadata["normalized_source_text"] = normalized_source_text
            return SourceDocument(
                source_id=document.source_id,
                title=document.title,
                text=document.text,
                metadata=metadata,
            )

        raw_text = document.text or ""
        if not raw_text.strip():
            return document

        page_ranges, normalized_source_text = build_page_ranges(page for page in raw_text.split("\n\n") if page.strip())
        if page_ranges:
            metadata["page_ranges"] = page_ranges
            metadata["normalized_source_text"] = normalized_source_text

        return SourceDocument(
            source_id=document.source_id,
            title=document.title,
            text=document.text,
            metadata=metadata,
        )
