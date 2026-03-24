from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

from datasets.schema import BenchmarkCase
from .llm_labeler import LLMLabelerConfig, OptionalLLMLabeler
from .models import ALL_LABELS, CaseSourceMapping, SourceEvidenceItem
from .thresholds import MappingThresholds


logger = logging.getLogger(__name__)
_RAW_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_OCR_BREAK_RE = re.compile(r"(?<=[a-z])\s*[-/–—]\s*(?=[a-z])")
_OCR_CODE_RE = re.compile(r"\bc\s*0\b|/c0|\\c0|\bc0\b")
_OCR_SINGLE_LETTER_SPLIT_RE = re.compile(r"\b([a-z])\s+([a-z])\b")
_OCR_DIGIT_UNIT_RE = re.compile(r"(?<=\d)\s+(?=[a-z]{1,3}\b)")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "vs", "with",
    "was", "were", "which", "within", "than", "then", "no", "not", "other", "results", "trial",
    "study", "section", "data", "supplementary", "table", "major", "clinical", "patients", "patient",
}


@dataclass(slots=True)
class _ChunkRecord:
    chunk_id: str
    text: str
    tokens: list[str]
    normalized_text: str
    compact_text: str
    page_number: int | None
    chunk_index: int | None
    document_key: str
    payload: dict[str, Any]


@dataclass(slots=True)
class _WindowMetrics:
    chunk_ids: list[str]
    text: str
    normalized_text: str
    page_numbers: list[int]
    page_distance: int | None
    exact_substring_ratio: float
    longest_common_ratio: float
    ordered_token_ratio: float
    passage_coverage: float
    anchor_start_coverage: float
    anchor_end_coverage: float
    reference_answer_coverage: float
    key_term_coverage: float
    boundary_ngram_hits: int
    rare_phrase_hit_count: int
    semantic_score: float
    lexical_score: float
    combined_score: float
    reasons: list[str]
    acceptance_rule: str | None
    window_start_token: int
    window_end_token: int
    from_chunk_pair: bool
    from_chunk_triplet: bool


@dataclass(slots=True)
class _NormalizedText:
    text: str
    compact: str
    tokens: list[str]


def _normalize_for_matching(text: str | None) -> _NormalizedText:
    if not text:
        return _NormalizedText(text="", compact="", tokens=[])

    value = unicodedata.normalize("NFKD", text)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = value.replace("ﬁ", "fi").replace("ﬂ", "fl")
    value = value.replace("þ", "k")
    value = value.replace("&", " and ")
    value = _OCR_CODE_RE.sub(" ", value)
    value = _OCR_BREAK_RE.sub("", value)
    value = re.sub(r"(?<=\d)\s+(?=\d)", "", value)
    value = _OCR_DIGIT_UNIT_RE.sub("", value)
    value = re.sub(r"(?<=\b[a-z])\s+(?=[a-z]\b)", "", value)
    value = re.sub(r"\b([a-z]{1,3})\s+(?=[a-z]{1,3}\b)", lambda m: m.group(0).replace(" ", ""), value)
    previous = None
    while previous != value:
        previous = value
        value = _OCR_SINGLE_LETTER_SPLIT_RE.sub(r"\1\2", value)
    value = _NON_ALNUM_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    tokens = value.split() if value else []
    compact = "".join(tokens)
    return _NormalizedText(text=value, compact=compact, tokens=tokens)


def normalize_text(text: str | None) -> str:
    return _normalize_for_matching(text).text


def tokenize(text: str | None) -> list[str]:
    return _normalize_for_matching(text).tokens


def informative_tokens(text: str | None) -> list[str]:
    return [token for token in tokenize(text) if len(token) > 2 and token not in _STOPWORDS]


def _multiset_coverage_ratio(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    query_counter = Counter(query_tokens)
    candidate_counter = Counter(candidate_tokens)
    matched = sum(min(count, candidate_counter.get(token, 0)) for token, count in query_counter.items())
    return matched / max(sum(query_counter.values()), 1)


def _unique_coverage_ratio(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    unique = {token for token in query_tokens if token}
    if not unique:
        return 0.0
    candidate_set = set(candidate_tokens)
    return len(unique & candidate_set) / len(unique)


def _longest_common_ratio(gold_text: str, candidate_text: str) -> float:
    if not gold_text or not candidate_text:
        return 0.0
    match = SequenceMatcher(a=gold_text, b=candidate_text).find_longest_match(0, len(gold_text), 0, len(candidate_text))
    return match.size / max(len(gold_text), 1)


def _ordered_token_ratio(gold_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not gold_tokens or not candidate_tokens:
        return 0.0
    matcher = SequenceMatcher(a=gold_tokens, b=candidate_tokens)
    lcs = sum(block.size for block in matcher.get_matching_blocks())
    return lcs / max(len(gold_tokens), 1)


def _exact_substring_ratio(gold_compact: str, candidate_compact: str) -> float:
    if not gold_compact or not candidate_compact:
        return 0.0
    if gold_compact in candidate_compact:
        return 1.0
    if candidate_compact in gold_compact:
        return len(candidate_compact) / max(len(gold_compact), 1)
    match = SequenceMatcher(a=gold_compact, b=candidate_compact).find_longest_match(0, len(gold_compact), 0, len(candidate_compact))
    return match.size / max(len(gold_compact), 1)


def _semantic_score(gold_text: str, candidate_text: str) -> float:
    if not gold_text or not candidate_text:
        return 0.0
    return SequenceMatcher(a=gold_text, b=candidate_text).ratio()


def _boundary_ngram_hits(text: str | None, candidate_text: str) -> int:
    tokens = informative_tokens(text)
    if len(tokens) < 3:
        return 0
    hits = 0
    for size in (8, 7, 6, 5, 4, 3):
        if len(tokens) >= size and " ".join(tokens[:size]) in candidate_text:
            hits += 1
            break
    for size in (8, 7, 6, 5, 4, 3):
        if len(tokens) >= size and " ".join(tokens[-size:]) in candidate_text:
            hits += 1
            break
    return hits




def _find_subsequence_positions(haystack: list[str], needle: list[str], *, min_prefix: int = 3) -> list[int]:
    if not haystack or not needle:
        return []
    sizes: list[int] = []
    max_len = min(len(needle), 8)
    for size in range(max_len, min_prefix - 1, -1):
        sizes.append(size)
    positions: set[int] = set()
    for size in sizes:
        prefix = needle[:size]
        for index in range(0, max(len(haystack) - size + 1, 0)):
            if haystack[index:index + size] == prefix:
                positions.add(index)
        if positions:
            break
    return sorted(positions)


def _find_token_positions(haystack: list[str], tokens: list[str]) -> list[int]:
    if not haystack or not tokens:
        return []
    positions: set[int] = set()
    candidate_tokens = [token for token in tokens if token]
    for token in candidate_tokens[:10]:
        try:
            positions.add(haystack.index(token))
        except ValueError:
            continue
    return sorted(positions)
def _rare_phrase_hits(gold_text: str | None, candidate_text: str) -> int:
    tokens = informative_tokens(gold_text)
    if len(tokens) < 3:
        return 0
    frequencies = Counter(tokens)
    ranked = sorted({token for token in tokens}, key=lambda token: (frequencies[token], -len(token), token))
    hits = 0
    for pivot in ranked[:8]:
        indexes = [idx for idx, token in enumerate(tokens) if token == pivot]
        for index in indexes[:2]:
            start = max(0, index - 2)
            end = min(len(tokens), index + 3)
            phrase_tokens = tokens[start:end]
            if len(phrase_tokens) >= 3 and " ".join(phrase_tokens) in candidate_text:
                hits += 1
                break
    return hits


class SourceMatcher:
    def __init__(
        self,
        thresholds: MappingThresholds | None = None,
        *,
        max_matches: int = 5,
        page_window: int = 2,
        page_offset_candidates: tuple[int, ...] = (0, -1, -2, 1, -3, 2, 3),
        semantic_fallback_enabled: bool = False,
        include_chunk_pairs: bool = True,
        max_sequence_length: int = 3,
        window_step_ratio: float = 0.15,
    ) -> None:
        self._thresholds = thresholds or MappingThresholds()
        self._max_matches = max(1, max_matches)
        self._page_window = max(0, page_window)
        self._page_offset_candidates = page_offset_candidates
        self._semantic_fallback_enabled = semantic_fallback_enabled
        self._include_chunk_pairs = include_chunk_pairs
        self._max_sequence_length = max(1, max_sequence_length)
        self._window_step_ratio = min(max(window_step_ratio, 0.05), 0.5)

    def build_case_source_mapping(
        self,
        *,
        case: BenchmarkCase,
        mapping_label: str,
        payloads: Iterable[dict[str, Any]],
        strategy: str = "unknown",
        llm_labeler: OptionalLLMLabeler | None = None,
        max_soft_candidates: int = 12,
    ) -> CaseSourceMapping:
        chunks = self._filter_and_prepare_chunks(case, payloads)
        logger.info("Case %s: evaluating %s document-local chunks for mapping_label=%s", case.id, len(chunks), mapping_label)

        if not chunks or not (case.gold_passage_text or "").strip():
            return CaseSourceMapping(
                case_id=case.id,
                mapping_label=mapping_label,
                strategy=strategy,
                source_list={label: [] for label in ALL_LABELS},
            )

        candidate_units = self._build_candidate_units(chunks)
        metrics = [self._score_candidate_unit(case, unit) for unit in candidate_units]
        metrics = [metric for metric in metrics if metric is not None]

        deduped: dict[tuple[str, ...], _WindowMetrics] = {}
        for metric in metrics:
            key = tuple(metric.chunk_ids)
            current = deduped.get(key)
            if current is None or metric.combined_score > current.combined_score:
                deduped[key] = metric

        ranked = sorted(deduped.values(), key=self._metrics_sort_key, reverse=True)
        strict_matches = self._select_matches(ranked)
        source_list = self._build_source_list(
            ranked,
            strict_matches,
            llm_labeler=llm_labeler,
            max_soft_candidates=max_soft_candidates,
        )
        return CaseSourceMapping(
            case_id=case.id,
            mapping_label=mapping_label,
            strategy=strategy,
            source_list=source_list,
            metadata={
                "strict_match_count": len(source_list["direct_evidence"]) + len(source_list["partial_direct_evidence"]),
                "soft_match_count": len(source_list["supporting"]) + len(source_list["tangential"]),
                "llm_second_pass_enabled": bool(llm_labeler and llm_labeler.enabled),
            },
        )

    def _filter_and_prepare_chunks(self, case: BenchmarkCase, payloads: Iterable[dict[str, Any]]) -> list[_ChunkRecord]:
        rows: list[_ChunkRecord] = []
        for payload in payloads:
            if not self._matches_document(case, payload):
                continue
            text = str(payload.get("text") or "")
            chunk_id = str(payload.get("chunk_id") or "")
            if not text.strip() or not chunk_id:
                continue
            normalized = _normalize_for_matching(text)
            rows.append(
                _ChunkRecord(
                    chunk_id=chunk_id,
                    text=text,
                    tokens=normalized.tokens,
                    normalized_text=normalized.text,
                    compact_text=normalized.compact,
                    page_number=self._safe_int(payload.get("page_number")),
                    chunk_index=self._safe_int(payload.get("chunk_index")),
                    document_key=str(payload.get("source_id") or payload.get("title") or ""),
                    payload=dict(payload),
                )
            )
        rows.sort(key=lambda row: (row.page_number if row.page_number is not None else 10**9, row.chunk_index if row.chunk_index is not None else 10**9, row.chunk_id))
        return rows

    def _build_candidate_units(self, chunks: list[_ChunkRecord]) -> list[list[_ChunkRecord]]:
        units: list[list[_ChunkRecord]] = []
        max_len = 1 if not self._include_chunk_pairs else max(self._max_sequence_length, 2)
        for start in range(len(chunks)):
            current: list[_ChunkRecord] = []
            for end in range(start, min(len(chunks), start + max_len)):
                chunk = chunks[end]
                if current and current[-1].document_key != chunk.document_key:
                    break
                if current:
                    prev = current[-1]
                    if prev.chunk_index is not None and chunk.chunk_index is not None and chunk.chunk_index != prev.chunk_index + 1:
                        break
                current.append(chunk)
                units.append(list(current))
        return units

    def _score_candidate_unit(self, case: BenchmarkCase, unit: list[_ChunkRecord]) -> _WindowMetrics | None:
        unit_tokens: list[str] = []
        page_numbers: list[int] = []
        chunk_ids: list[str] = []
        for chunk in unit:
            unit_tokens.extend(chunk.tokens)
            chunk_ids.append(chunk.chunk_id)
            if chunk.page_number is not None:
                page_numbers.append(chunk.page_number)
        if not unit_tokens:
            return None

        gold_norm = _normalize_for_matching(case.gold_passage_text or case.gold_passage_normalized)
        if not gold_norm.tokens:
            return None
        reference_norm = _normalize_for_matching(case.reference_answer)
        start_norm = _normalize_for_matching(case.anchor_start_text)
        end_norm = _normalize_for_matching(case.anchor_end_text)
        hint_tokens = [t for t in case.retrieval_hints.get("key_terms", []) if isinstance(t, str)]
        hints_norm = _normalize_for_matching(" ".join(hint_tokens))

        gold_len = len(gold_norm.tokens)
        window_sizes = sorted({
            max(14, int(gold_len * 0.70)),
            max(18, int(gold_len * 0.85)),
            max(24, int(gold_len * 1.0)),
            max(30, int(gold_len * 1.15)),
            max(36, int(gold_len * 1.3)),
            max(42, int(gold_len * 1.45)),
        })
        step = max(2, int(gold_len * self._window_step_ratio))

        starts: set[int] = {0}
        if len(unit_tokens) > max(window_sizes):
            starts.update(range(0, max(len(unit_tokens) - min(window_sizes) + 1, 1), step))
            starts.add(max(0, len(unit_tokens) - max(window_sizes)))

        for positions in (
            _find_subsequence_positions(unit_tokens, start_norm.tokens),
            _find_subsequence_positions(unit_tokens, end_norm.tokens),
            _find_subsequence_positions(unit_tokens, gold_norm.tokens),
            _find_subsequence_positions(unit_tokens, reference_norm.tokens, min_prefix=2),
            _find_token_positions(unit_tokens, hints_norm.tokens),
        ):
            for position in positions:
                starts.add(max(0, position - max(2, gold_len // 10)))
                starts.add(max(0, position - max(4, gold_len // 5)))
                starts.add(position)

        starts = sorted(starts)

        best: _WindowMetrics | None = None
        for start in starts:
            for window_size in window_sizes:
                end = min(len(unit_tokens), start + window_size)
                if end - start < 12:
                    continue
                window_tokens = unit_tokens[start:end]
                window_norm = " ".join(window_tokens)
                window_compact = "".join(window_tokens)
                exact_ratio = _exact_substring_ratio(gold_norm.compact, window_compact)
                common_ratio = _longest_common_ratio(gold_norm.compact, window_compact)
                ordered_ratio = _ordered_token_ratio(gold_norm.tokens, window_tokens)
                passage_coverage = _multiset_coverage_ratio(gold_norm.tokens, window_tokens)
                start_coverage = _multiset_coverage_ratio(start_norm.tokens, window_tokens)
                end_coverage = _multiset_coverage_ratio(end_norm.tokens, window_tokens)
                reference_coverage = _multiset_coverage_ratio(reference_norm.tokens, window_tokens)
                key_term_coverage = _unique_coverage_ratio(hints_norm.tokens, window_tokens)
                boundary_hits = _boundary_ngram_hits(case.anchor_start_text, window_norm) + _boundary_ngram_hits(case.anchor_end_text, window_norm)
                phrase_hits = _rare_phrase_hits(case.gold_passage_text, window_norm)
                semantic = _semantic_score(gold_norm.text, window_norm)
                page_distance = self._page_distance(case.source_page, min(page_numbers) if page_numbers else None)
                page_bonus = 1.0 / (1.0 + abs(page_distance)) if page_distance is not None else 0.0

                lexical_score = (
                    0.30 * passage_coverage
                    + 0.24 * ordered_ratio
                    + 0.20 * exact_ratio
                    + 0.07 * max(start_coverage, end_coverage)
                    + 0.08 * key_term_coverage
                    + 0.06 * reference_coverage
                    + 0.025 * min((boundary_hits / 2.0), 1.0)
                    + 0.02 * min((phrase_hits / 3.0), 1.0)
                    + 0.015 * page_bonus
                )
                combined_score = lexical_score
                if self._semantic_fallback_enabled and lexical_score < self._thresholds.min_lexical_score:
                    combined_score = 0.93 * lexical_score + 0.07 * semantic

                reasons: list[str] = []
                if exact_ratio >= 0.95:
                    reasons.append("full_passage_substring")
                elif exact_ratio >= 0.45:
                    reasons.append("strong_compact_substring")
                if start_coverage >= 0.6:
                    reasons.append("strong_start_anchor_coverage")
                if end_coverage >= 0.6:
                    reasons.append("strong_end_anchor_coverage")
                if passage_coverage >= 0.7:
                    reasons.append("high_passage_token_coverage")
                if ordered_ratio >= 0.45:
                    reasons.append("strong_ordered_reconstruction")
                if common_ratio >= 0.4:
                    reasons.append("strong_common_span")
                if key_term_coverage >= 0.6:
                    reasons.append("high_key_term_coverage")
                if reference_coverage >= 0.5:
                    reasons.append("reference_answer_overlap")
                if reference_coverage >= 0.9 and key_term_coverage >= 0.5:
                    reasons.append("definition_answer_lock")
                if boundary_hits >= 1:
                    reasons.append("boundary_ngram_hit")
                if phrase_hits >= 1:
                    reasons.append("rare_phrase_hit")
                if page_distance is not None and page_distance <= self._page_window:
                    reasons.append("near_expected_page")
                if len(unit) == 2:
                    reasons.append("adjacent_chunk_pair")
                elif len(unit) >= 3:
                    reasons.append("adjacent_chunk_triplet")

                acceptance_rule = self._determine_acceptance_rule(
                    exact_ratio=exact_ratio,
                    ordered_ratio=ordered_ratio,
                    passage_coverage=passage_coverage,
                    start_coverage=start_coverage,
                    end_coverage=end_coverage,
                    reference_coverage=reference_coverage,
                    key_term_coverage=key_term_coverage,
                    page_distance=page_distance,
                )

                metric = _WindowMetrics(
                    chunk_ids=chunk_ids,
                    text=" ".join(window_tokens),
                    normalized_text=window_norm,
                    page_numbers=page_numbers,
                    page_distance=page_distance,
                    exact_substring_ratio=exact_ratio,
                    longest_common_ratio=common_ratio,
                    ordered_token_ratio=ordered_ratio,
                    passage_coverage=passage_coverage,
                    anchor_start_coverage=start_coverage,
                    anchor_end_coverage=end_coverage,
                    reference_answer_coverage=reference_coverage,
                    key_term_coverage=key_term_coverage,
                    boundary_ngram_hits=boundary_hits,
                    rare_phrase_hit_count=phrase_hits,
                    semantic_score=semantic,
                    lexical_score=lexical_score,
                    combined_score=combined_score,
                    reasons=reasons,
                    acceptance_rule=acceptance_rule,
                    window_start_token=start,
                    window_end_token=end,
                    from_chunk_pair=len(unit) == 2,
                    from_chunk_triplet=len(unit) >= 3,
                )
                if best is None or self._metrics_sort_key(metric) > self._metrics_sort_key(best):
                    best = metric
        return best

    def _determine_acceptance_rule(
        self,
        *,
        exact_ratio: float,
        ordered_ratio: float,
        passage_coverage: float,
        start_coverage: float,
        end_coverage: float,
        key_term_coverage: float,
        reference_coverage: float,
        page_distance: int | None,
    ) -> str | None:
        anchor_max = max(start_coverage, end_coverage)
        both_anchors = min(start_coverage, end_coverage)
        near_page = page_distance is not None and page_distance <= self._page_window + 1

        if exact_ratio >= self._thresholds.min_exact_substring_ratio:
            return "exact_or_compact_substring"
        if passage_coverage >= self._thresholds.min_passage_coverage and ordered_ratio >= self._thresholds.min_ordered_token_ratio:
            return "strong_reconstruction"
        if reference_coverage >= 0.92 and key_term_coverage >= 0.50 and ordered_ratio >= 0.35:
            return "definition_answer_lock"
        if passage_coverage >= 0.58 and ordered_ratio >= 0.50 and key_term_coverage >= 0.60 and anchor_max >= 0.45:
            return "keyed_reconstruction"
        if passage_coverage >= self._thresholds.min_partial_passage_coverage and both_anchors >= self._thresholds.min_anchor_pair_coverage and near_page:
            return "anchor_reconstruction_near_page"
        if passage_coverage >= self._thresholds.min_partial_passage_coverage and anchor_max >= self._thresholds.min_anchor_coverage_any and key_term_coverage >= self._thresholds.min_key_term_coverage and near_page:
            return "hinted_partial_reconstruction"
        return None

    def _select_matches(self, ranked: list[_WindowMetrics]) -> list[_WindowMetrics]:
        selected: list[_WindowMetrics] = []
        seen_chunks: set[tuple[str, ...]] = set()
        for metric in ranked:
            if metric.acceptance_rule is None:
                continue
            if metric.combined_score < self._thresholds.min_combined_score:
                continue
            key = tuple(metric.chunk_ids)
            if key in seen_chunks:
                continue
            selected.append(metric)
            seen_chunks.add(key)
            if len(selected) >= self._max_matches:
                break

        if selected or not self._semantic_fallback_enabled:
            return selected

        for metric in ranked:
            if metric.semantic_score >= self._thresholds.semantic_fallback_min and metric.passage_coverage >= 0.35:
                return [metric]
        return []

    def _build_source_list(
        self,
        ranked: list[_WindowMetrics],
        strict_matches: list[_WindowMetrics],
        *,
        llm_labeler: OptionalLLMLabeler | None,
        max_soft_candidates: int,
    ) -> dict[str, list[SourceEvidenceItem]]:
        buckets: dict[str, list[SourceEvidenceItem]] = {label: [] for label in ALL_LABELS}
        seen: set[tuple[str, ...]] = set()
        for metric in strict_matches:
            label = self._determine_strict_label(metric)
            item = self._to_source_evidence(metric, label=label)
            buckets[label].append(item)
            seen.add(tuple(metric.chunk_ids))

        unresolved: list[dict[str, Any]] = []
        for metric in ranked:
            key = tuple(metric.chunk_ids)
            if key in seen:
                continue
            if len(unresolved) >= max_soft_candidates:
                break
            unresolved.append(self._to_source_evidence(metric, label="irrelevant").to_dict())
            seen.add(key)

        classifier = llm_labeler or OptionalLLMLabeler(LLMLabelerConfig(enabled=False, max_candidates=max_soft_candidates))
        for item in classifier.classify(unresolved):
            label = str(item.get("label") or "irrelevant")
            if label not in buckets:
                label = "irrelevant"
            buckets[label].append(
                SourceEvidenceItem(
                    chunk_ids=list(item.get("chunk_ids") or []),
                    label=label,
                    combined_score=float(item.get("combined_score", 0.0) or 0.0),
                    lexical_score=float(item.get("lexical_score", 0.0) or 0.0),
                    semantic_score=float(item.get("semantic_score", 0.0) or 0.0),
                    metadata=dict(item.get("metadata") or {}),
                    llm_label=item.get("llm_label"),
                )
            )
        return buckets

    def _determine_strict_label(self, metric: _WindowMetrics) -> str:
        if (
            metric.exact_substring_ratio >= 0.9
            or metric.passage_coverage >= self._thresholds.min_passage_coverage
            or metric.acceptance_rule in {"exact_or_compact_substring", "strong_reconstruction", "definition_answer_lock"}
        ):
            return "direct_evidence"
        return "partial_direct_evidence"

    def _to_source_evidence(self, metric: _WindowMetrics, *, label: str) -> SourceEvidenceItem:
        page_span = [min(metric.page_numbers), max(metric.page_numbers)] if metric.page_numbers else []
        preview_words = metric.text.split()
        preview = " ".join(preview_words[:60])
        if len(preview_words) > 60:
            preview += " ..."
        return SourceEvidenceItem(
            chunk_ids=metric.chunk_ids,
            label=label,
            combined_score=metric.combined_score,
            lexical_score=metric.lexical_score,
            semantic_score=metric.semantic_score,
            metadata={
                "page_numbers": metric.page_numbers,
                "page_span": page_span,
                "page_distance": metric.page_distance,
                "window_start_token": metric.window_start_token,
                "window_end_token": metric.window_end_token,
                "exact_substring_ratio": round(metric.exact_substring_ratio, 4),
                "longest_common_ratio": round(metric.longest_common_ratio, 4),
                "ordered_token_ratio": round(metric.ordered_token_ratio, 4),
                "passage_coverage": round(metric.passage_coverage, 4),
                "anchor_start_coverage": round(metric.anchor_start_coverage, 4),
                "anchor_end_coverage": round(metric.anchor_end_coverage, 4),
                "reference_answer_coverage": round(metric.reference_answer_coverage, 4),
                "key_term_coverage": round(metric.key_term_coverage, 4),
                "boundary_ngram_hits": metric.boundary_ngram_hits,
                "rare_phrase_hit_count": metric.rare_phrase_hit_count,
                "acceptance_rule": metric.acceptance_rule,
                "window_preview": preview,
                "reasons": metric.reasons,
                "from_chunk_pair": metric.from_chunk_pair,
                "from_chunk_triplet": metric.from_chunk_triplet,
            },
        )

    @staticmethod
    def _metrics_sort_key(metric: _WindowMetrics) -> tuple[float, float, float, float, float, float, float]:
        page_bonus = 1.0 / (1.0 + abs(metric.page_distance)) if metric.page_distance is not None else 0.0
        return (
            metric.combined_score,
            metric.passage_coverage,
            metric.ordered_token_ratio,
            metric.exact_substring_ratio,
            metric.longest_common_ratio,
            max(metric.anchor_start_coverage, metric.anchor_end_coverage),
            page_bonus,
        )

    def _matches_document(self, case: BenchmarkCase, payload: dict[str, Any]) -> bool:
        source_doc_id = (case.source_document_id or "").lower()
        source_doc_name = (case.source_document_name or "").lower().replace(".pdf", "")
        payload_source_id = str(payload.get("source_id") or "").lower()
        payload_title = str(payload.get("title") or "").lower().replace(".pdf", "")
        return bool(
            (source_doc_id and source_doc_id in payload_source_id)
            or (source_doc_name and source_doc_name in payload_source_id)
            or (source_doc_name and source_doc_name in payload_title)
        )

    def _page_distance(self, source_page: int | None, payload_page: int | None) -> int | None:
        if source_page is None or payload_page is None:
            return None
        distances = [abs((source_page + offset) - payload_page) for offset in self._page_offset_candidates]
        return min(distances) if distances else abs(source_page - payload_page)



    def build_chunk_assignment(
        self,
        *,
        case: BenchmarkCase,
        mapping_label: str,
        payloads: Iterable[dict[str, Any]],
    ) -> CaseSourceMapping:
        return self.build_case_source_mapping(
            case=case,
            mapping_label=mapping_label,
            payloads=payloads,
            strategy="unknown",
        )

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
