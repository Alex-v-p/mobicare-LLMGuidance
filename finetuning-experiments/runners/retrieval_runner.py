from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from scoring.retrieval import score_retrieval


@dataclass(slots=True)
class RetrievalStageResult:
    retrieved_chunks: list[dict[str, Any]]
    retrieval_scores: dict[str, Any]
    source_match_candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_retrieval_stage(source_mapping: dict[str, Any] | None, guidance_record: dict[str, Any]) -> RetrievalStageResult:
    expected_matches = list((source_mapping or {}).get("matches") or [])
    retrieved_chunks = list(guidance_record.get("rag") or [])
    return RetrievalStageResult(
        retrieved_chunks=retrieved_chunks,
        retrieval_scores=score_retrieval(expected_matches, retrieved_chunks),
        source_match_candidates=expected_matches,
    )
