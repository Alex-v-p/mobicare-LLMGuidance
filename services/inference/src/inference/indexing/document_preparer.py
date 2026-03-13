from __future__ import annotations

from inference.indexing.chunking.factory import ChunkerFactory
from inference.indexing.chunking.utils import normalize_for_offset_matching
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
            source_documents.append(self._enrich_with_page_ranges(cleaned_document))

        chunks: list[TextChunk] = []
        for document in source_documents:
            chunks.extend(chunker.chunk(document))
        return source_documents, chunks

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
            page_ranges.append({"page_number": index, "start": start, "end": end})
            normalized_pages.append(normalized_page)
            cursor = end + 1

        metadata["page_ranges"] = page_ranges
        metadata["normalized_source_text"] = " ".join(normalized_pages)

        return SourceDocument(
            source_id=document.source_id,
            title=document.title,
            text=document.text,
            metadata=metadata,
        )
