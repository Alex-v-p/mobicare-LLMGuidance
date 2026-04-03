from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?:[\r\n]+|(?<=[.!?;:])\s+)")
_NEGATION_TOKENS = {"no", "not", "without", "never", "none", "absent", "absence", "denies", "denied"}
_STOPWORDS = {
    "the", "a", "an", "to", "and", "or", "of", "in", "for", "on", "with", "is", "are", "was", "were", "by", "that",
    "this", "these", "those", "from", "as", "at", "be", "it", "its", "into", "than", "then", "their", "there",
}


def _norm(text: str | None) -> str:
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def _informative(text: str | None) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 2]


def _fact_hit(answer_norm: str, fact: str) -> bool:
    fact_norm = _norm(fact)
    return bool(fact_norm and fact_norm in answer_norm)




def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None

def _coverage(tokens: list[str], haystack: set[str]) -> float:
    unique = set(tokens)
    return len(unique & haystack) / len(unique) if unique else 0.0


def _set_f1(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return (2 * precision * recall) / (precision + recall) if precision + recall else 0.0


def _sentence_candidates(text: str | None) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    segments = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(raw) if segment and segment.strip()]
    return segments or [raw]


def _ordered_overlap_ratio(fact_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not fact_tokens or not candidate_tokens:
        return 0.0
    matcher = SequenceMatcher(a=fact_tokens, b=candidate_tokens, autojunk=False)
    match = matcher.find_longest_match(0, len(fact_tokens), 0, len(candidate_tokens))
    return match.size / len(fact_tokens) if fact_tokens else 0.0


def _soft_token_overlap(tokens: list[str], candidate_tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    if not candidate_tokens:
        return 0.0
    scores: list[float] = []
    for token in tokens:
        if token.isdigit():
            scores.append(1.0 if token in candidate_tokens else 0.0)
            continue
        best = 0.0
        for candidate in candidate_tokens:
            if token == candidate:
                best = 1.0
                break
            best = max(best, SequenceMatcher(None, token, candidate, autojunk=False).ratio())
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def _fact_acronym(tokens: list[str]) -> str:
    return "".join(token[0] for token in tokens if token and token[0].isalpha())


def _candidate_windows(answer: str | None, fact_token_count: int) -> list[str]:
    segments = _sentence_candidates(answer)
    candidates: list[str] = []
    seen: set[str] = set()
    target = max(1, fact_token_count)
    window_sizes = sorted({max(2, target - 1), target, target + 1, target + 2})
    for segment in segments:
        segment_norm = _norm(segment)
        if not segment_norm or segment_norm in seen:
            continue
        seen.add(segment_norm)
        candidates.append(segment_norm)
        tokens = segment_norm.split()
        if not tokens:
            continue
        for size in window_sizes:
            if size >= len(tokens):
                continue
            step = 1 if len(tokens) <= 32 else 2
            for start in range(0, len(tokens) - size + 1, step):
                window = " ".join(tokens[start:start + size])
                if window and window not in seen:
                    seen.add(window)
                    candidates.append(window)
    answer_norm = _norm(answer)
    if answer_norm and answer_norm not in seen:
        candidates.append(answer_norm)
    return candidates


def _fact_threshold_v2(fact_tokens: list[str], *, negative: bool) -> float:
    token_count = len(fact_tokens)
    if token_count <= 1:
        threshold = 0.9
    elif token_count == 2:
        threshold = 0.72
    elif token_count == 3:
        threshold = 0.7
    else:
        threshold = 0.68
    if negative:
        threshold += 0.08
    return min(0.95, threshold)


def _fact_match_analysis_v2(answer: str, fact: str, *, negative: bool = False) -> dict[str, Any]:
    fact_norm = _norm(fact)
    fact_tokens = _informative(fact) or fact_norm.split()
    fact_token_count = len(fact_tokens) if fact_tokens else max(1, len(fact_norm.split()))
    numeric_tokens = [token for token in fact_norm.split() if token.isdigit()]
    acronym = _fact_acronym(fact_tokens) if len(fact_tokens) >= 2 else ""
    threshold = _fact_threshold_v2(fact_tokens, negative=negative)
    if not fact_norm:
        return {
            "fact": fact,
            "match": False,
            "score": 0.0,
            "threshold": threshold,
            "preview": "",
            "method": "empty_fact",
        }

    best: dict[str, Any] = {
        "score": 0.0,
        "preview": "",
        "method": "window_similarity",
        "soft_overlap": 0.0,
        "token_f1": 0.0,
        "ordered_overlap": 0.0,
        "numeric_coverage": 1.0 if not numeric_tokens else 0.0,
        "abbreviation_bonus": 0.0,
        "ngram_bonus": 0.0,
        "negation_penalty": 0.0,
        "exact_match": False,
    }

    for candidate in _candidate_windows(answer, fact_token_count):
        candidate_tokens = candidate.split()
        if not candidate_tokens:
            continue
        candidate_set = set(candidate_tokens)
        soft_overlap = _soft_token_overlap(fact_tokens, candidate_tokens)
        token_f1 = _set_f1(set(fact_tokens), candidate_set)
        ordered_overlap = _ordered_overlap_ratio(fact_tokens, candidate_tokens)
        numeric_coverage = _coverage(numeric_tokens, candidate_set) if numeric_tokens else 1.0
        ngram_bonus = 0.0
        if len(fact_tokens) >= 2:
            bigram = " ".join(fact_tokens[:2])
            if bigram and bigram in candidate:
                ngram_bonus = 1.0
            elif len(fact_tokens) >= 3:
                trigram = " ".join(fact_tokens[:3])
                if trigram and trigram in candidate:
                    ngram_bonus = 1.0
        abbreviation_bonus = 1.0 if acronym and acronym in candidate_set else 0.0
        exact_match = fact_norm in candidate
        negation_penalty = 0.12 if negative and any(token in _NEGATION_TOKENS for token in candidate_tokens) else 0.0
        score = max(
            1.0 if exact_match else 0.0,
            min(
                1.0,
                (0.45 * soft_overlap)
                + (0.10 * token_f1)
                + (0.15 * ordered_overlap)
                + (0.12 * numeric_coverage)
                + (0.03 * ngram_bonus)
                + (0.15 * abbreviation_bonus),
            ),
        )
        if numeric_tokens and numeric_coverage < 1.0 and not exact_match:
            score = min(score, 0.55 + (0.2 * numeric_coverage))
        score = max(0.0, score - negation_penalty)
        if score > best["score"]:
            best = {
                "score": score,
                "preview": candidate[:180],
                "method": "exact_substring" if exact_match else "window_similarity",
                "soft_overlap": soft_overlap,
                "token_f1": token_f1,
                "ordered_overlap": ordered_overlap,
                "numeric_coverage": numeric_coverage,
                "abbreviation_bonus": abbreviation_bonus,
                "ngram_bonus": ngram_bonus,
                "negation_penalty": negation_penalty,
                "exact_match": exact_match,
            }

    result = {
        "fact": fact,
        "match": bool(best["score"] >= threshold),
        "score": round(float(best["score"]), 4),
        "threshold": round(float(threshold), 4),
        "preview": best["preview"],
        "method": best["method"],
        "soft_overlap": round(float(best["soft_overlap"]), 4),
        "token_f1": round(float(best["token_f1"]), 4),
        "ordered_overlap": round(float(best["ordered_overlap"]), 4),
        "numeric_coverage": round(float(best["numeric_coverage"]), 4),
        "abbreviation_bonus": round(float(best["abbreviation_bonus"]), 4),
        "ngram_bonus": round(float(best["ngram_bonus"]), 4),
        "negation_penalty": round(float(best["negation_penalty"]), 4),
        "exact_match": bool(best["exact_match"]),
    }
    return result


def _verification_verdict(verification: dict[str, Any] | None) -> str:
    return str((verification or {}).get("verdict") or "").strip().lower()


def _verification_confidence(verification: dict[str, Any] | None) -> str:
    return str((verification or {}).get("confidence") or "").strip().lower()


def _verification_score(verification: dict[str, Any] | None) -> float:
    verdict = _verification_verdict(verification)
    confidence = _verification_confidence(verification)
    if verdict in {"pass", "passed", "ok", "success"}:
        return {"high": 1.0, "medium": 0.9, "low": 0.8}.get(confidence, 0.85)
    if verdict in {"partial", "warning", "mixed"}:
        return {"high": 0.65, "medium": 0.55, "low": 0.45}.get(confidence, 0.55)
    if verdict in {"fail", "failed", "error"}:
        return {"high": 0.1, "medium": 0.2, "low": 0.3}.get(confidence, 0.2)
    return 0.5


def _verification_transport_pass(verification: dict[str, Any] | None) -> bool | None:
    verdict = _verification_verdict(verification)
    if not verdict:
        return None
    return verdict in {"pass", "passed", "ok", "success"}


def _verification_band_from_payload(verification: dict[str, Any] | None) -> str | None:
    verdict = _verification_verdict(verification)
    if verdict in {"pass", "passed", "ok", "success"}:
        return "pass"
    if verdict in {"partial", "warning", "mixed"}:
        return "partial"
    if verdict in {"fail", "failed", "error"}:
        return "fail"
    return None


def _intrinsic_generation_quality_score(generation_scores: dict[str, Any]) -> float | None:
    components: list[float] = []
    for key in (
        "required_fact_recall",
        "groundedness_score",
        "faithfulness_to_retrieved_context",
        "answer_similarity",
    ):
        value = generation_scores.get(key)
        if value is None:
            continue
        try:
            components.append(float(value))
        except (TypeError, ValueError):
            continue
    hallucination_rate = generation_scores.get("hallucination_rate")
    if hallucination_rate is not None:
        try:
            components.append(max(0.0, 1.0 - float(hallucination_rate)))
        except (TypeError, ValueError):
            pass
    if not components:
        return None
    return sum(components) / len(components)


def _expected_verification_band(intrinsic_quality: float | None) -> str | None:
    if intrinsic_quality is None:
        return None
    if intrinsic_quality >= 0.7:
        return "pass"
    if intrinsic_quality >= 0.4:
        return "partial"
    return "fail"


def _band_distance(left: str | None, right: str | None) -> int | None:
    if left is None or right is None:
        return None
    order = {"fail": 0, "partial": 1, "pass": 2}
    if left not in order or right not in order:
        return None
    return abs(order[left] - order[right])


def _verification_alignment_score(expected_band: str | None, observed_band: str | None, verification: dict[str, Any] | None) -> tuple[float | None, str | None]:
    distance = _band_distance(expected_band, observed_band)
    if distance is None:
        return None, None
    confidence = _verification_confidence(verification)
    if distance == 0:
        return 1.0, "aligned"
    if distance == 1:
        penalty = {"high": 0.15, "medium": 0.08, "low": 0.03}.get(confidence, 0.05)
        return max(0.0, 0.65 - penalty), "partially_aligned"
    penalty = {"high": 0.15, "medium": 0.08, "low": 0.03}.get(confidence, 0.05)
    return max(0.0, 0.2 - penalty), "misaligned"


def _grounded_fact_pass_fields(generation_scores: dict[str, Any]) -> tuple[bool, bool | None, str | None]:
    deterministic_applicable = bool(
        generation_scores.get("deterministic_rubric_applicable")
        or (generation_scores.get("deterministic_rubric") or {}).get("applicable")
        or (generation_scores.get("deterministic_rubric") or {}).get("enabled")
    )
    if not deterministic_applicable:
        return False, None, "not_applicable"

    required_total = int(generation_scores.get("required_fact_total") or 0)
    forbidden_violations = float(generation_scores.get("forbidden_fact_violations") or 0.0)
    required_fact_recall = generation_scores.get("required_fact_recall")
    groundedness_score = generation_scores.get("groundedness_score")
    context_faithfulness = generation_scores.get("faithfulness_to_retrieved_context")
    effective_generation_score = generation_scores.get("effective_generation_score")
    if effective_generation_score is None:
        effective_generation_score = generation_scores.get("primary_generation_score")
    if effective_generation_score is None:
        effective_generation_score = generation_scores.get("deterministic_rubric_score")

    try:
        required_fact_recall = None if required_fact_recall is None else float(required_fact_recall)
    except (TypeError, ValueError):
        required_fact_recall = None
    try:
        groundedness_score = None if groundedness_score is None else float(groundedness_score)
    except (TypeError, ValueError):
        groundedness_score = None
    try:
        context_faithfulness = None if context_faithfulness is None else float(context_faithfulness)
    except (TypeError, ValueError):
        context_faithfulness = None
    try:
        effective_generation_score = None if effective_generation_score is None else float(effective_generation_score)
    except (TypeError, ValueError):
        effective_generation_score = None

    meets_required_facts = True if required_total <= 0 else (required_fact_recall or 0.0) >= 0.6
    meets_groundedness = True if groundedness_score is None else groundedness_score >= 0.5
    meets_context_faithfulness = True if context_faithfulness is None else context_faithfulness >= 0.45
    meets_effective_generation = True if effective_generation_score is None else effective_generation_score >= 0.5

    passed = bool(
        forbidden_violations <= 0
        and meets_required_facts
        and meets_groundedness
        and meets_context_faithfulness
        and meets_effective_generation
    )
    if passed:
        return True, True, "passed"
    if forbidden_violations > 0:
        return True, False, "forbidden_fact_violation"
    if not meets_required_facts:
        return True, False, "insufficient_fact_recall"
    if not meets_groundedness:
        return True, False, "insufficient_groundedness"
    if not meets_context_faithfulness:
        return True, False, "insufficient_context_faithfulness"
    if not meets_effective_generation:
        return True, False, "insufficient_generation_score"
    return True, False, "failed_requirements"


def _grade_from_score(score: float) -> str:
    if score >= 0.85:
        return "excellent"
    if score >= 0.7:
        return "good"
    if score >= 0.5:
        return "partial"
    return "poor"


def _nested_score(payload: dict[str, Any], container_key: str, score_key: str = "score") -> Any:
    container = payload.get(container_key) or {}
    if not isinstance(container, dict):
        return None
    return container.get(score_key)


def _resolve_effective_generation_fields(generation_scores: dict[str, Any]) -> tuple[Any, Any, Any]:
    deterministic_score = generation_scores.get("deterministic_rubric_score")
    if deterministic_score is None:
        deterministic_score = _nested_score(generation_scores, "deterministic_rubric")
    if deterministic_score is None:
        deterministic_score = generation_scores.get("answer_quality_score")
    if deterministic_score is None:
        deterministic_score = generation_scores.get("judge_score")

    deterministic_grade = generation_scores.get("deterministic_rubric_grade")
    if deterministic_grade is None:
        deterministic_grade = _nested_score(generation_scores, "deterministic_rubric", "grade")
    if deterministic_grade is None:
        deterministic_grade = generation_scores.get("answer_quality_grade")
    if deterministic_grade is None:
        deterministic_grade = generation_scores.get("judge_grade")

    llm_score = generation_scores.get("llm_judge_score")
    if llm_score is None:
        llm_score = _nested_score(generation_scores, "llm_judge")

    llm_grade = generation_scores.get("llm_judge_grade")
    if llm_grade is None:
        llm_grade = _nested_score(generation_scores, "llm_judge", "overall_grade")
    if llm_grade is None:
        llm_grade = _nested_score(generation_scores, "llm_judge", "grade")

    evaluation_profile = str(generation_scores.get("evaluation_profile") or "").strip().lower()
    effective_score = generation_scores.get("effective_generation_score")
    effective_grade = generation_scores.get("effective_generation_grade")
    effective_source = generation_scores.get("effective_generation_score_source")

    llm_primary_opt_in = bool(generation_scores.get("llm_judge_use_as_primary"))
    if effective_score is None:
        if llm_primary_opt_in and llm_score is not None:
            effective_score = llm_score
            effective_grade = llm_grade
            effective_source = "llm_judge"
        elif evaluation_profile == "observation_only" and llm_score is not None:
            effective_score = llm_score
            effective_grade = llm_grade
            effective_source = "llm_judge"
        elif deterministic_score is not None:
            effective_score = deterministic_score
            effective_grade = deterministic_grade
            effective_source = "deterministic_rubric"
        elif llm_score is not None:
            effective_score = llm_score
            effective_grade = llm_grade
            effective_source = "llm_judge"

    return deterministic_score, deterministic_grade, llm_score, llm_grade, effective_score, effective_grade, effective_source


def finalize_generation_score_fields(generation_scores: dict[str, Any], verification: dict[str, Any] | None = None) -> dict[str, Any]:
    deterministic_score, deterministic_grade, llm_score, llm_grade, effective_score, effective_grade, effective_source = _resolve_effective_generation_fields(generation_scores)

    generation_scores["deterministic_rubric_score"] = deterministic_score
    generation_scores["deterministic_rubric_grade"] = deterministic_grade
    generation_scores.setdefault("deterministic_rubric_score_v2", generation_scores.get("deterministic_rubric_score_v2"))
    generation_scores.setdefault("deterministic_rubric_grade_v2", generation_scores.get("deterministic_rubric_grade_v2"))
    generation_scores.setdefault("effective_generation_score_v2", generation_scores.get("effective_generation_score_v2", generation_scores.get("deterministic_rubric_score_v2")))
    generation_scores.setdefault("effective_generation_grade_v2", generation_scores.get("effective_generation_grade_v2", generation_scores.get("deterministic_rubric_grade_v2")))
    generation_scores.setdefault("effective_generation_score_v2_source", generation_scores.get("effective_generation_score_v2_source", "deterministic_rubric_v2" if generation_scores.get("effective_generation_score_v2") is not None else None))
    generation_scores.setdefault("primary_generation_score_v2", generation_scores.get("primary_generation_score_v2", generation_scores.get("effective_generation_score_v2")))
    generation_scores.setdefault("primary_generation_grade_v2", generation_scores.get("primary_generation_grade_v2", generation_scores.get("effective_generation_grade_v2")))
    generation_scores.setdefault("primary_generation_score_v2_source", generation_scores.get("primary_generation_score_v2_source", generation_scores.get("effective_generation_score_v2_source")))
    generation_scores.setdefault("required_fact_recall_v2", generation_scores.get("required_fact_recall_v2"))
    generation_scores.setdefault("required_fact_support_score_v2", generation_scores.get("required_fact_support_score_v2"))
    generation_scores.setdefault("required_fact_hits_v2", generation_scores.get("required_fact_hits_v2"))
    generation_scores.setdefault("required_fact_misses_v2", generation_scores.get("required_fact_misses_v2"))
    generation_scores.setdefault("required_fact_details_v2", generation_scores.get("required_fact_details_v2"))
    generation_scores.setdefault("forbidden_fact_violations_v2", generation_scores.get("forbidden_fact_violations_v2"))
    generation_scores.setdefault("forbidden_fact_risk_score_v2", generation_scores.get("forbidden_fact_risk_score_v2"))
    generation_scores.setdefault("forbidden_fact_details_v2", generation_scores.get("forbidden_fact_details_v2"))
    generation_scores.setdefault("fact_scoring_version", generation_scores.get("fact_scoring_version"))
    generation_scores["llm_judge_score"] = llm_score
    generation_scores["llm_judge_grade"] = llm_grade
    llm_judge_payload = generation_scores.get("llm_judge") or {}
    llm_available = generation_scores.get("llm_judge_available")
    if llm_available is None:
        llm_available = bool(
            generation_scores.get("llm_judge_requested") and (
                llm_score is not None or bool(llm_judge_payload) or not str(llm_judge_payload.get("error") or "").strip()
            )
        )
        if str(llm_judge_payload.get("error") or "").strip():
            llm_available = False
        if llm_judge_payload.get("available") is False:
            llm_available = False
    generation_scores["llm_judge_available"] = llm_available
    generation_scores["llm_judge_error"] = _first_defined(generation_scores.get("llm_judge_error"), llm_judge_payload.get("error"))
    generation_scores["llm_judge_profile"] = _first_defined(generation_scores.get("llm_judge_profile"), llm_judge_payload.get("judge_profile"))
    generation_scores["llm_judge_enabled"] = bool(generation_scores.get("llm_judge_requested") or llm_judge_payload.get("enabled"))
    generation_scores["llm_judge_use_as_primary"] = bool(generation_scores.get("llm_judge_use_as_primary"))
    generation_scores["effective_generation_score"] = effective_score
    generation_scores["effective_generation_grade"] = effective_grade
    generation_scores["effective_generation_score_source"] = effective_source
    generation_scores["primary_generation_score"] = generation_scores.get("primary_generation_score", effective_score)
    generation_scores["primary_generation_grade"] = generation_scores.get("primary_generation_grade", effective_grade)
    generation_scores["primary_generation_score_source"] = generation_scores.get("primary_generation_score_source", effective_source)

    if generation_scores.get("answer_quality_score") is None:
        generation_scores["answer_quality_score"] = deterministic_score
    if generation_scores.get("answer_quality_grade") is None:
        generation_scores["answer_quality_grade"] = deterministic_grade
    if generation_scores.get("judge_score") is None:
        generation_scores["judge_score"] = deterministic_score
    if generation_scores.get("judge_grade") is None:
        generation_scores["judge_grade"] = deterministic_grade
    if generation_scores.get("judge_score_source") is None:
        generation_scores["judge_score_source"] = "legacy_alias_of_deterministic_rubric" if deterministic_score is not None else effective_source

    intrinsic_quality = generation_scores.get("verification_intrinsic_quality_score")
    if intrinsic_quality is None:
        intrinsic_quality = _intrinsic_generation_quality_score(generation_scores)
    observed_band = generation_scores.get("verification_observed_verdict_band")
    if observed_band is None:
        observed_band = _verification_band_from_payload(verification)
    expected_band = generation_scores.get("verification_expected_verdict_band")
    if expected_band is None:
        expected_band = _expected_verification_band(intrinsic_quality)
    alignment_score = generation_scores.get("verification_alignment_score")
    alignment_label = generation_scores.get("verification_alignment_label")
    if alignment_score is None and verification is not None:
        alignment_score, alignment_label = _verification_alignment_score(expected_band, observed_band, verification)
    if alignment_label is None and alignment_score is not None:
        alignment_label = "aligned" if alignment_score >= 0.75 else ("partially_aligned" if alignment_score >= 0.45 else "misaligned")

    generation_scores["verification_transport_pass"] = generation_scores.get("verification_transport_pass")
    if generation_scores["verification_transport_pass"] is None:
        generation_scores["verification_transport_pass"] = _verification_transport_pass(verification)
    generation_scores["verification_observed_verdict_band"] = observed_band
    generation_scores["verification_expected_verdict_band"] = expected_band
    generation_scores["verification_intrinsic_quality_score"] = intrinsic_quality
    generation_scores["verification_alignment_score"] = alignment_score
    generation_scores["verification_alignment_label"] = alignment_label

    grounded_fact_applicable, grounded_fact_pass, grounded_fact_reason = _grounded_fact_pass_fields(generation_scores)
    generation_scores["grounded_fact_pass_applicable"] = grounded_fact_applicable
    generation_scores["grounded_fact_pass"] = grounded_fact_pass
    generation_scores["grounded_fact_pass_reason"] = grounded_fact_reason
    return generation_scores


def _is_observation_only(case: dict[str, Any]) -> bool:
    generation_metadata = case.get("generation_metadata") or {}
    tags = {str(tag).strip().lower() for tag in (case.get("tags") or [])}
    return bool(
        generation_metadata.get("evaluation_profile") == "observation_only"
        or generation_metadata.get("request_mode") == "biomarker_only"
        or generation_metadata.get("omit_question_from_request")
        or "observation-case" in tags
        or "biomarker-only" in tags
    )


def _metric_or_none(value: float, *, applicable: bool) -> float | None:
    return value if applicable else None


def _default_weights(rubric_config: Any | None) -> dict[str, float]:
    weights = {
        "required_fact": 0.35,
        "reference_alignment": 0.15,
        "gold_alignment": 0.10,
        "context_alignment": 0.15,
        "groundedness": 0.15,
        "verification": 0.10,
    }
    if rubric_config is None:
        return weights
    return {
        "required_fact": float(getattr(rubric_config, "required_fact_weight", weights["required_fact"])),
        "reference_alignment": float(getattr(rubric_config, "reference_alignment_weight", weights["reference_alignment"])),
        "gold_alignment": float(getattr(rubric_config, "gold_alignment_weight", weights["gold_alignment"])),
        "context_alignment": float(getattr(rubric_config, "context_alignment_weight", weights["context_alignment"])),
        "groundedness": float(getattr(rubric_config, "groundedness_weight", weights["groundedness"])),
        "verification": float(getattr(rubric_config, "verification_weight", weights["verification"])),
    }


def _applied_weights(base_weights: dict[str, float], applicability: dict[str, bool]) -> dict[str, float]:
    return {
        key: weight
        for key, weight in base_weights.items()
        if applicability.get(key, False) and weight > 0
    }


def score_generation(
    case: dict[str, Any],
    answer: str,
    retrieved_context: list[dict[str, Any]],
    verification: dict[str, Any] | None = None,
    rubric_config: Any | None = None,
) -> dict[str, Any]:
    answer_norm = _norm(answer)
    context_norm = _norm(" ".join((item.get("snippet") or "") for item in retrieved_context))
    observation_only = _is_observation_only(case)

    required = case.get("required_facts", []) or []
    forbidden = case.get("forbidden_facts", []) or []
    required_hits = sum(1 for fact in required if _fact_hit(answer_norm, fact))
    forbidden_hits = sum(1 for fact in forbidden if _fact_hit(answer_norm, fact))
    required_v2_details = [_fact_match_analysis_v2(answer, fact, negative=False) for fact in required]
    forbidden_v2_details = [_fact_match_analysis_v2(answer, fact, negative=True) for fact in forbidden]
    required_hits_v2 = sum(1 for item in required_v2_details if item.get("match"))
    forbidden_hits_v2 = sum(1 for item in forbidden_v2_details if item.get("match"))

    answer_tokens = _informative(answer)
    answer_token_set = set(answer_tokens)
    reference_token_set = set(_informative(case.get("reference_answer", "")))
    gold_token_set = set(_informative(case.get("gold_passage_text", "")))
    context_tokens = set(_informative(context_norm))

    reference_token_f1 = _set_f1(answer_token_set, reference_token_set)
    gold_token_f1 = _set_f1(answer_token_set, gold_token_set)
    context_token_f1 = _set_f1(answer_token_set, context_tokens)

    faith_gold = _coverage(answer_tokens, gold_token_set)
    faith_context = _coverage(answer_tokens, context_tokens)
    unsupported_tokens = sorted(answer_token_set - context_tokens)
    unsupported = max(0, len(unsupported_tokens))
    unsupported_rate = (unsupported / len(answer_token_set)) if answer_token_set else 0.0
    groundedness = max(0.0, 1.0 - unsupported_rate)

    verification_payload = verification or {}
    verification_present = bool(verification_payload)
    verification_score = _verification_score(verification_payload) if verification_present else 0.5

    required_fact_applicable = bool(required)
    forbidden_fact_applicable = bool(forbidden)
    required_fact_recall = (required_hits / len(required)) if required_fact_applicable else None
    required_fact_support_score_v2 = (
        sum(float(item.get("score") or 0.0) for item in required_v2_details) / len(required_v2_details)
        if required_fact_applicable else None
    )
    required_fact_recall_v2 = (required_hits_v2 / len(required)) if required_fact_applicable else None
    forbidden_fact_risk_score_v2 = (
        max(float(item.get("score") or 0.0) for item in forbidden_v2_details)
        if forbidden_fact_applicable and forbidden_v2_details else None
    )

    base_weights = _default_weights(rubric_config)
    dimension_applicability = {
        "required_fact": required_fact_applicable,
        "reference_alignment": bool(reference_token_set),
        "gold_alignment": bool(gold_token_set),
        "context_alignment": bool(context_tokens),
        "groundedness": bool(answer_token_set),
        "verification": verification_present,
    }
    applied_weights = _applied_weights(base_weights, dimension_applicability)
    weight_total = sum(applied_weights.values())

    deterministic_applicable = bool(not observation_only and getattr(rubric_config, "enabled", True))
    deterministic_rubric_score: float | None = None
    deterministic_grade: str | None = None

    required_fact_subscore = _metric_or_none(required_fact_recall or 0.0, applicable=required_fact_applicable)
    required_fact_subscore_v2 = _metric_or_none(required_fact_support_score_v2 or 0.0, applicable=required_fact_applicable)
    subscores = {
        "required_fact": required_fact_subscore,
        "required_fact_recall": required_fact_subscore,
        "reference_alignment": _metric_or_none(reference_token_f1, applicable=dimension_applicability["reference_alignment"]),
        "gold_alignment": _metric_or_none(gold_token_f1, applicable=dimension_applicability["gold_alignment"]),
        "context_alignment": _metric_or_none(max(faith_context, context_token_f1), applicable=dimension_applicability["context_alignment"]),
        "groundedness": _metric_or_none(groundedness, applicable=dimension_applicability["groundedness"]),
        "verification": _metric_or_none(verification_score, applicable=dimension_applicability["verification"]),
    }
    subscores_v2 = {
        **subscores,
        "required_fact": required_fact_subscore_v2,
        "required_fact_support_score_v2": required_fact_subscore_v2,
    }

    deterministic_rubric_score_v2: float | None = None
    deterministic_grade_v2: str | None = None
    deterministic_v2: dict[str, Any] | None = None

    if deterministic_applicable and weight_total > 0:
        weighted_sum = sum(
            float(subscores[key] or 0.0) * applied_weights[key]
            for key in applied_weights
        )
        deterministic_rubric_score = weighted_sum / weight_total
        if forbidden_hits:
            deterministic_rubric_score = max(0.0, deterministic_rubric_score - min(0.4, 0.2 * forbidden_hits))
        deterministic_rubric_score = max(0.0, min(1.0, deterministic_rubric_score))
        deterministic_grade = _grade_from_score(deterministic_rubric_score)

        weighted_sum_v2 = sum(
            float(subscores_v2[key] or 0.0) * applied_weights[key]
            for key in applied_weights
        )
        deterministic_rubric_score_v2 = weighted_sum_v2 / weight_total
        if forbidden_hits_v2:
            deterministic_rubric_score_v2 = max(0.0, deterministic_rubric_score_v2 - min(0.4, 0.2 * forbidden_hits_v2))
        deterministic_rubric_score_v2 = max(0.0, min(1.0, deterministic_rubric_score_v2))
        deterministic_grade_v2 = _grade_from_score(deterministic_rubric_score_v2)

    exact_pass_applicable = bool(deterministic_applicable and (required_fact_applicable or forbidden_fact_applicable))
    exact_pass = None
    if exact_pass_applicable:
        meets_required_facts = True if not required_fact_applicable else (required_fact_recall or 0.0) >= 0.999
        exact_pass = bool(
            meets_required_facts
            and forbidden_hits == 0
            and unsupported == 0
        )

    skipped_reason = None
    if not deterministic_applicable:
        skipped_reason = "observation_only_case" if observation_only else "deterministic_rubric_disabled"
    elif weight_total <= 0:
        skipped_reason = "no_applicable_dimensions"

    deterministic = {
        "enabled": deterministic_applicable,
        "applicable": deterministic_applicable,
        "skipped_reason": skipped_reason,
        "score": deterministic_rubric_score,
        "grade": deterministic_grade,
        "weights": base_weights,
        "applied_weights": applied_weights,
        "applicable_dimensions": dimension_applicability,
        "subscores": subscores,
        "required_fact_hits": required_hits,
        "required_fact_total": len(required),
        "required_fact_misses": max(0, len(required) - required_hits),
        "forbidden_fact_violations": forbidden_hits,
        "forbidden_fact_total": len(forbidden),
    }
    deterministic_v2 = {
        "enabled": deterministic_applicable,
        "applicable": deterministic_applicable,
        "skipped_reason": skipped_reason,
        "score": deterministic_rubric_score_v2,
        "grade": deterministic_grade_v2,
        "weights": base_weights,
        "applied_weights": applied_weights,
        "applicable_dimensions": dimension_applicability,
        "subscores": subscores_v2,
        "required_fact_hits": required_hits_v2,
        "required_fact_total": len(required),
        "required_fact_misses": max(0, len(required) - required_hits_v2),
        "forbidden_fact_violations": forbidden_hits_v2,
        "forbidden_fact_total": len(forbidden),
        "version": "heuristic_fact_v2",
    }

    generation_scores = {
        "evaluation_profile": "observation_only" if observation_only else "standard",
        "deterministic_rubric_applicable": deterministic_applicable,
        "answer_similarity": reference_token_f1,
        "answer_similarity_legacy_note": "Token-F1 against the reference answer. Prefer deterministic_rubric.score and effective_generation_score.",
        "reference_token_f1": reference_token_f1,
        "gold_token_f1": gold_token_f1,
        "context_token_f1": context_token_f1,
        "deterministic_rubric": deterministic,
        "deterministic_rubric_score": deterministic_rubric_score,
        "deterministic_rubric_grade": deterministic_grade,
        "deterministic_rubric_v2": deterministic_v2,
        "deterministic_rubric_score_v2": deterministic_rubric_score_v2,
        "deterministic_rubric_grade_v2": deterministic_grade_v2,
        "effective_generation_score_v2": deterministic_rubric_score_v2,
        "effective_generation_grade_v2": deterministic_grade_v2,
        "effective_generation_score_v2_source": "deterministic_rubric_v2" if deterministic_rubric_score_v2 is not None else None,
        "primary_generation_score_v2": deterministic_rubric_score_v2,
        "primary_generation_grade_v2": deterministic_grade_v2,
        "primary_generation_score_v2_source": "deterministic_rubric_v2" if deterministic_rubric_score_v2 is not None else None,
        "fact_scoring_version": "legacy_v1_plus_heuristic_v2",
        "answer_quality_score": deterministic_rubric_score,
        "answer_quality_grade": deterministic_grade,
        "judge_score": deterministic_rubric_score,
        "judge_grade": deterministic_grade,
        "judge_score_source": "legacy_alias_of_deterministic_rubric",
        "llm_judge_score": None,
        "llm_judge_grade": None,
        "effective_generation_score": deterministic_rubric_score,
        "effective_generation_grade": deterministic_grade,
        "effective_generation_score_source": "deterministic_rubric" if deterministic_rubric_score is not None else None,
        "primary_generation_score": deterministic_rubric_score,
        "primary_generation_grade": deterministic_grade,
        "primary_generation_score_source": "deterministic_rubric" if deterministic_rubric_score is not None else None,
        "verification_score": _metric_or_none(verification_score, applicable=verification_present),
        "fact_recall_applicable": required_fact_applicable,
        "required_fact_recall": required_fact_recall,
        "required_fact_hits": required_hits,
        "required_fact_total": len(required),
        "required_fact_misses": max(0, len(required) - required_hits),
        "required_fact_recall_v2": required_fact_recall_v2,
        "required_fact_support_score_v2": required_fact_support_score_v2,
        "required_fact_hits_v2": required_hits_v2,
        "required_fact_misses_v2": max(0, len(required) - required_hits_v2),
        "required_fact_details_v2": required_v2_details,
        "forbidden_fact_applicable": forbidden_fact_applicable,
        "forbidden_fact_violations": forbidden_hits,
        "forbidden_fact_total": len(forbidden),
        "forbidden_fact_violations_v2": forbidden_hits_v2,
        "forbidden_fact_risk_score_v2": forbidden_fact_risk_score_v2,
        "forbidden_fact_details_v2": forbidden_v2_details,
        "faithfulness_to_gold_passage": _metric_or_none(faith_gold, applicable=bool(gold_token_set)),
        "faithfulness_to_retrieved_context": _metric_or_none(faith_context, applicable=bool(context_tokens)),
        "groundedness_score": _metric_or_none(groundedness, applicable=bool(answer_token_set)),
        "hallucination_unsupported_token_count": unsupported,
        "hallucination_rate": unsupported_rate,
        "unsupported_tokens": unsupported_tokens[:25],
        "retrieved_context_chunk_count": len(retrieved_context),
        "exact_pass_applicable": exact_pass_applicable,
        "exact_pass": exact_pass,
        "grounded_fact_pass_applicable": deterministic_applicable,
        "grounded_fact_pass": None,
        "grounded_fact_pass_reason": None,
    }
    return finalize_generation_score_fields(generation_scores, verification=verification)
