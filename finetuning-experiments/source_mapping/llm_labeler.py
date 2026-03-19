from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import SoftEvidenceLabel


@dataclass(slots=True)
class LLMLabelerConfig:
    enabled: bool = False
    max_candidates: int = 12


class OptionalLLMLabeler:
    """Optional second-pass classifier.

    This intentionally uses deterministic heuristics as a safe fallback when no
    external classifier is configured. It keeps the source-mapping pipeline shape
    ready for a future true LLM-backed classifier without coupling increment 4 to
    an inference dependency.
    """

    def __init__(self, config: LLMLabelerConfig | None = None) -> None:
        self._config = config or LLMLabelerConfig()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def classify(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        return [self._classify_candidate(item) for item in candidates[: self._config.max_candidates]]

    def _classify_candidate(self, item: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(item.get("metadata") or {})
        semantic = float(item.get("semantic_score", 0.0) or 0.0)
        lexical = float(item.get("lexical_score", 0.0) or 0.0)
        coverage = float(metadata.get("passage_coverage", 0.0) or 0.0)
        key_terms = float(metadata.get("key_term_coverage", 0.0) or 0.0)
        reference = float(metadata.get("reference_answer_coverage", 0.0) or 0.0)
        boundary_hits = int(metadata.get("boundary_ngram_hits", 0) or 0)

        label: SoftEvidenceLabel
        if coverage >= 0.35 and (semantic >= 0.75 or lexical >= 0.45 or reference >= 0.45):
            label = "supporting"
        elif coverage >= 0.16 or semantic >= 0.55 or lexical >= 0.28 or key_terms >= 0.28 or boundary_hits >= 1:
            label = "tangential"
        else:
            label = "irrelevant"

        enriched = dict(item)
        enriched["label"] = label
        enriched["llm_label"] = label
        metadata["llm_second_pass"] = self.enabled
        metadata["llm_label_reason"] = "heuristic_fallback_classifier"
        enriched["metadata"] = metadata
        return enriched
