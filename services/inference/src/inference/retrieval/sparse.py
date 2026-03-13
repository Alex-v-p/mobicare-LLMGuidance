from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from inference.retrieval.common import payload_identity

_TOKEN_RE = re.compile(r"\b[a-z0-9]{2,}\b", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "this", "these", "those",
    "can", "could", "should", "would", "may", "might", "about", "into", "than", "then", "them", "they", "their",
    "there", "here", "what", "when", "where", "which", "who", "why", "how", "your", "you", "our", "we",
}


@dataclass(slots=True)
class SparseHit:
    source_id: str
    title: str
    snippet: str
    score: float
    payload: dict[str, Any]


@dataclass(slots=True)
class _IndexedDocument:
    payload: dict[str, Any]
    tokens: list[str]
    frequencies: Counter[str]
    length: int


@dataclass(slots=True)
class _SparseCorpusIndex:
    signature: tuple[str, ...]
    documents: list[_IndexedDocument]
    doc_freq: Counter[str]
    average_doc_length: float


class SparseKeywordRetriever:
    def __init__(self) -> None:
        self._cached_index: _SparseCorpusIndex | None = None

    def tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS]

    def search(self, query: str, documents: list[dict[str, Any]], limit: int = 3) -> list[SparseHit]:
        query_terms = self.tokenize(query)
        if not query_terms or not documents:
            return []

        index = self._build_or_reuse_index(documents)
        if not index.documents:
            return []

        hits: list[SparseHit] = []
        doc_count = len(index.documents)
        average_doc_length = max(index.average_doc_length, 1.0)
        k1 = 1.5
        b = 0.75

        for indexed_document in index.documents:
            score = 0.0
            for term in query_terms:
                tf = indexed_document.frequencies.get(term, 0)
                if tf == 0:
                    continue
                df = index.doc_freq.get(term, 0)
                idf = math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
                denom = tf + k1 * (1 - b + b * (indexed_document.length / average_doc_length))
                score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))

            if score <= 0:
                continue

            payload = indexed_document.payload
            hits.append(
                SparseHit(
                    source_id=str(payload.get("source_id") or payload.get("chunk_id") or "unknown"),
                    title=str(payload.get("title") or payload.get("object_name") or "Untitled"),
                    snippet=str(payload.get("text") or ""),
                    score=score,
                    payload=payload,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def _build_or_reuse_index(self, documents: list[dict[str, Any]]) -> _SparseCorpusIndex:
        signature = tuple(payload_identity(document) for document in documents)
        if self._cached_index is not None and self._cached_index.signature == signature:
            return self._cached_index

        indexed_documents: list[_IndexedDocument] = []
        doc_freq: Counter[str] = Counter()
        total_tokens = 0

        for payload in documents:
            tokens = self.tokenize(str(payload.get("text") or ""))
            if not tokens:
                continue
            frequencies = Counter(tokens)
            indexed_documents.append(
                _IndexedDocument(
                    payload=payload,
                    tokens=tokens,
                    frequencies=frequencies,
                    length=len(tokens),
                )
            )
            total_tokens += len(tokens)
            doc_freq.update(set(tokens))

        average_doc_length = total_tokens / max(len(indexed_documents), 1)
        self._cached_index = _SparseCorpusIndex(
            signature=signature,
            documents=indexed_documents,
            doc_freq=doc_freq,
            average_doc_length=average_doc_length,
        )
        return self._cached_index
