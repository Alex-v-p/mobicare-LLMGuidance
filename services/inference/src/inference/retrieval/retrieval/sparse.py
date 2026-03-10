from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

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


class SparseKeywordRetriever:
    def tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS]

    def search(self, query: str, documents: list[dict[str, Any]], limit: int = 3) -> list[SparseHit]:
        query_terms = self.tokenize(query)
        if not query_terms or not documents:
            return []

        doc_tokens: list[list[str]] = [self.tokenize(str(doc.get("text") or "")) for doc in documents]
        doc_count = len(documents)
        avg_doc_len = sum(len(tokens) for tokens in doc_tokens) / max(doc_count, 1)
        doc_freq: Counter[str] = Counter()
        for tokens in doc_tokens:
            doc_freq.update(set(tokens))

        hits: list[SparseHit] = []
        k1 = 1.5
        b = 0.75

        for doc, tokens in zip(documents, doc_tokens):
            if not tokens:
                continue
            freqs = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
                denom = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1)))
                score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))

            if score <= 0:
                continue

            hits.append(
                SparseHit(
                    source_id=str(doc.get("source_id") or doc.get("chunk_id") or "unknown"),
                    title=str(doc.get("title") or doc.get("object_name") or "Untitled"),
                    snippet=str(doc.get("text") or ""),
                    score=score,
                    payload=doc,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]
