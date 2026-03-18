from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MappingThresholds:
    min_combined_score: float = 0.50
    min_lexical_score: float = 0.50
    min_passage_coverage: float = 0.72
    min_partial_passage_coverage: float = 0.60
    min_ordered_token_ratio: float = 0.40
    min_exact_substring_ratio: float = 0.70
    min_anchor_pair_coverage: float = 0.45
    min_anchor_coverage_any: float = 0.65
    min_key_term_coverage: float = 0.50
    semantic_fallback_min: float = 0.93
