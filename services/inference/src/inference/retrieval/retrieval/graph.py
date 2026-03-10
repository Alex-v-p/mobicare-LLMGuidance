from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from shared.contracts.inference import RetrievedContext

from inference.retrieval.sparse import SparseKeywordRetriever


@dataclass(slots=True)
class GraphNode:
    node_id: str
    payload: dict[str, Any]


class ChunkGraphAugmenter:
    def __init__(self) -> None:
        self._tokenizer = SparseKeywordRetriever()

    def expand(
        self,
        *,
        query: str,
        ranked_payloads: list[dict[str, Any]],
        corpus_payloads: list[dict[str, Any]],
        max_extra_nodes: int = 2,
    ) -> tuple[list[RetrievedContext], dict[str, Any]]:
        if not ranked_payloads:
            return [], {"graph_augmented": False, "graph_nodes_added": 0, "graph_edges_used": []}

        source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for payload in corpus_payloads:
            source_groups[str(payload.get("source_id") or "")].append(payload)

        for items in source_groups.values():
            items.sort(key=lambda item: int(item.get("chunk_index") or 0))

        seen = {str(item.get("chunk_id") or item.get("source_id") or id(item)) for item in ranked_payloads}
        results = [self._to_context(item) for item in ranked_payloads]
        edge_descriptions: list[str] = []
        query_terms = set(self._tokenizer.tokenize(query))

        for payload in ranked_payloads:
            if len(results) >= len(ranked_payloads) + max_extra_nodes:
                break

            group = source_groups.get(str(payload.get("source_id") or ""), [])
            try:
                index = group.index(payload)
            except ValueError:
                continue

            candidates = []
            if index - 1 >= 0:
                candidates.append(("prev", group[index - 1]))
            if index + 1 < len(group):
                candidates.append(("next", group[index + 1]))

            for relation, candidate in candidates:
                candidate_id = str(candidate.get("chunk_id") or candidate.get("source_id") or id(candidate))
                if candidate_id in seen:
                    continue
                candidate_terms = set(self._tokenizer.tokenize(str(candidate.get("text") or "")))
                overlap = len(query_terms & candidate_terms)
                if overlap == 0 and relation == "prev":
                    # allow one structural back-link even when lexical overlap is weak
                    overlap = 1
                if overlap <= 0:
                    continue
                seen.add(candidate_id)
                results.append(self._to_context(candidate))
                edge_descriptions.append(f"{payload.get('chunk_id')} -> {candidate.get('chunk_id')} ({relation})")
                if len(results) >= len(ranked_payloads) + max_extra_nodes:
                    break

        return results, {
            "graph_augmented": len(results) > len(ranked_payloads),
            "graph_nodes_added": max(0, len(results) - len(ranked_payloads)),
            "graph_edges_used": edge_descriptions,
        }

    def _to_context(self, payload: dict[str, Any]) -> RetrievedContext:
        return RetrievedContext(
            source_id=str(payload.get("source_id") or payload.get("chunk_id") or "unknown"),
            title=str(payload.get("title") or payload.get("object_name") or "Untitled"),
            snippet=str(payload.get("text") or ""),
        )
