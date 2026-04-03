from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import SoftEvidenceLabel


@dataclass(slots=True)
class LLMLabelerConfig:
    enabled: bool = False
    max_candidates: int = 12
    profile: str = "heuristic_v1"


class OptionalLLMLabeler:
    """Optional second-pass classifier.

    This intentionally uses deterministic heuristics as a safe fallback when no
    external classifier is configured. It keeps the source-mapping pipeline shape
    ready for a future true LLM-backed classifier without coupling the scoring
    experiments to an inference dependency.
    """

    def __init__(self, config: LLMLabelerConfig | None = None) -> None:
        self._config = config or LLMLabelerConfig()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def profile(self) -> str:
        return self._config.profile

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
        page_distance = metadata.get("page_distance")
        near_page = isinstance(page_distance, int) and page_distance <= 2

        label: SoftEvidenceLabel
        profile_name = (self._config.profile or "heuristic_v1").strip().lower()
        if profile_name == "semantic_recovery_v2":
            if (coverage >= 0.28 and (semantic >= 0.64 or lexical >= 0.40)) or (reference >= 0.42 and key_terms >= 0.35) or (semantic >= 0.74 and (near_page or boundary_hits >= 1)):
                label = "supporting"
            elif coverage >= 0.14 or semantic >= 0.48 or lexical >= 0.24 or key_terms >= 0.24 or boundary_hits >= 1 or near_page:
                label = "tangential"
            else:
                label = "irrelevant"
        else:
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
        metadata["llm_label_reason"] = f"heuristic_{profile_name}"
        metadata["llm_labeling_profile"] = self._config.profile
        enriched["metadata"] = metadata
        return enriched
