from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from shared.contracts.inference import RetrievedContext

from inference.retrieval.common import payload_identity, payload_to_context
from inference.retrieval.sparse import SparseKeywordRetriever


@dataclass(slots=True)
class _CandidateEdge:
    relation: str
    candidate: dict[str, Any]
    score: int
    edge_description: str


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

        query_terms = set(self._tokenizer.tokenize(query))
        seen = {payload_identity(item) for item in ranked_payloads}
        results = [payload_to_context(item) for item in ranked_payloads]
        edge_descriptions: list[str] = []

        candidate_edges: list[_CandidateEdge] = []
        for payload in ranked_payloads:
            candidate_edges.extend(self._collect_adjacent_candidates(payload, source_groups, query_terms))

        candidate_edges.sort(key=lambda item: item.score, reverse=True)
        for edge in candidate_edges:
            candidate_id = payload_identity(edge.candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            results.append(payload_to_context(edge.candidate))
            edge_descriptions.append(edge.edge_description)
            if len(results) >= len(ranked_payloads) + max_extra_nodes:
                break

        return results, {
            "graph_augmented": len(results) > len(ranked_payloads),
            "graph_nodes_added": max(0, len(results) - len(ranked_payloads)),
            "graph_edges_used": edge_descriptions,
        }

    def _collect_adjacent_candidates(
        self,
        payload: dict[str, Any],
        source_groups: dict[str, list[dict[str, Any]]],
        query_terms: set[str],
    ) -> list[_CandidateEdge]:
        group = source_groups.get(str(payload.get("source_id") or ""), [])
        if not group:
            return []
        try:
            index = group.index(payload)
        except ValueError:
            return []

        anchor_terms = set(self._tokenizer.tokenize(str(payload.get("text") or "")))
        adjacent: list[tuple[str, dict[str, Any]]] = []
        if index - 1 >= 0:
            adjacent.append(("prev", group[index - 1]))
        if index + 1 < len(group):
            adjacent.append(("next", group[index + 1]))

        edges: list[_CandidateEdge] = []
        for relation, candidate in adjacent:
            candidate_terms = set(self._tokenizer.tokenize(str(candidate.get("text") or "")))
            query_overlap = len(query_terms & candidate_terms)
            anchor_overlap = len(anchor_terms & candidate_terms)
            contiguous_bonus = 1 if self._is_adjacent(payload, candidate) else 0
            score = (3 * query_overlap) + anchor_overlap + contiguous_bonus
            if score <= 0:
                continue
            edges.append(
                _CandidateEdge(
                    relation=relation,
                    candidate=candidate,
                    score=score,
                    edge_description=f"{payload.get('chunk_id')} -> {candidate.get('chunk_id')} ({relation})",
                )
            )
        return edges

    def _is_adjacent(self, anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
        try:
            anchor_index = int(anchor.get("chunk_index") or 0)
            candidate_index = int(candidate.get("chunk_index") or 0)
        except (TypeError, ValueError):
            return False
        return abs(anchor_index - candidate_index) == 1
