from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MappingThresholds:
    min_combined_score: float = 0.48
    min_lexical_score: float = 0.50
    min_passage_coverage: float = 0.72
    min_partial_passage_coverage: float = 0.60
    min_ordered_token_ratio: float = 0.40
    min_exact_substring_ratio: float = 0.66
    min_anchor_pair_coverage: float = 0.45
    min_anchor_coverage_any: float = 0.65
    min_key_term_coverage: float = 0.45
    semantic_fallback_min: float = 0.93
    supporting_combined_min: float = 0.28
    supporting_semantic_min: float = 0.70
    tangential_combined_min: float = 0.14
    tangential_semantic_min: float = 0.52

    @classmethod
    def for_profile(cls, profile: str | None) -> "MappingThresholds":
        profile_name = (profile or "legacy_v1").strip().lower()
        if profile_name == "semantic_recovery_v2":
            return cls(
                min_combined_score=0.44,
                min_lexical_score=0.46,
                min_passage_coverage=0.68,
                min_partial_passage_coverage=0.54,
                min_ordered_token_ratio=0.36,
                min_exact_substring_ratio=0.58,
                min_anchor_pair_coverage=0.36,
                min_anchor_coverage_any=0.56,
                min_key_term_coverage=0.38,
                semantic_fallback_min=0.82,
                supporting_combined_min=0.24,
                supporting_semantic_min=0.62,
                tangential_combined_min=0.12,
                tangential_semantic_min=0.42,
            )
        return cls()
