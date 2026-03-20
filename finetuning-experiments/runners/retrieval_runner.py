from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from adapters.guidance_payloads import extract_retrieved_context
from scoring.retrieval import score_retrieval


@dataclass(slots=True)
class RetrievalStageResult:
    retrieved_chunks: list[dict[str, Any]]
    retrieval_scores: dict[str, Any]
    source_match_candidates: list[dict[str, Any]]
    source_list: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_retrieval_stage(source_mapping: dict[str, Any] | None, guidance_record: dict[str, Any]) -> RetrievalStageResult:
    mapping = source_mapping or {}
    source_list = dict(mapping.get("source_list") or {})
    expected_matches = list(mapping.get("matches") or [])
    retrieved_chunks = extract_retrieved_context(guidance_record)
    return RetrievalStageResult(
        retrieved_chunks=retrieved_chunks,
        retrieval_scores=score_retrieval(mapping, retrieved_chunks),
        source_match_candidates=expected_matches,
        source_list=source_list,
    )
