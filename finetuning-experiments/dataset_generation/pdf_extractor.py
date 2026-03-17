from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from .config import PdfExtractionConfig
from .models import ExtractedPassage

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_REFERENCE_HEADING_RE = re.compile(r"^(references|bibliography|citations?)$", re.IGNORECASE)
_TABLE_HEADING_RE = re.compile(r"^(supplementary\s+(table|figure|text)\s+\d+.*)$", re.IGNORECASE)
_UPPER_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9\s\-–—/(),.+]+\??$")
_DOTTED_LINE_RE = re.compile(r"^[.\s\d]+$")
_SECTION_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\s+.+$")

_HEADING_KEYWORDS = {
    "WHY?",
    "IN WHOM AND WHEN?",
    "WHICH DRUG AND WHAT DOSE?",
    "WHICH ACE-I AND WHAT DOSE?",
    "WHAT DOSE?",
    "WHERE?",
    "HOW TO USE?",
    "PROBLEM SOLVING",
    "ADVICE TO PATIENT",
    "INDICATIONS:",
    "CONTRAINDICATIONS:",
    "CAUTIONS/SEEK SPECIALIST ADVICE:",
}


class PdfPassageExtractor:
    def __init__(self, config: PdfExtractionConfig | None = None) -> None:
        self.config = config or PdfExtractionConfig()

    def extract(self, pdf_path: str | Path) -> list[ExtractedPassage]:
        path = Path(pdf_path)
        reader = PdfReader(str(path))
        passages: list[ExtractedPassage] = []
        reference_section_seen = False

        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            if not raw_text.strip():
                continue

            if self._page_starts_reference_section(raw_text):
                reference_section_seen = True

            if reference_section_seen or self._skip_page(raw_text, page_number):
                continue

            blocks = self._extract_blocks(raw_text)
            current_heading: str | None = None
            local_index = 0

            for block in blocks:
                if self._is_heading(block):
                    current_heading = self._normalize_inline(block)
                    if self.config.skip_reference_sections and _REFERENCE_HEADING_RE.match(current_heading):
                        reference_section_seen = True
                    continue

                if reference_section_seen:
                    continue

                normalized = self._normalize_inline(block)
                if not normalized or not self._valid_passage(normalized):
                    continue

                for chunk in self._split_long_block(normalized):
                    if not self._valid_passage(chunk):
                        continue
                    local_index += 1
                    passages.append(self._make_passage(path, page_number, local_index, chunk, current_heading))

        return passages


    def _page_starts_reference_section(self, raw_text: str) -> bool:
        normalized = self._normalize_inline(raw_text).lower()
        if "references" not in normalized and "bibliography" not in normalized:
            return False
        citation_markers = len(re.findall(r"\b\d+\.\s+[A-Z]", raw_text))
        return citation_markers >= 3

    def _skip_page(self, raw_text: str, page_number: int) -> bool:
        if page_number <= self.config.skip_initial_pages:
            return True

        normalized = self._normalize_inline(raw_text).lower()
        if not normalized:
            return True

        if self.config.skip_reference_sections and any(marker in normalized for marker in self.config.reference_page_markers):
            return True

        return False

    def _extract_blocks(self, raw_text: str) -> list[str]:
        text = raw_text.replace("\r", "\n")
        blocks = [self._normalize_block(part) for part in re.split(r"\n\s*\n+", text) if self._normalize_block(part)]

        if len(blocks) <= 2:
            blocks = self._line_grouped_blocks(text)

        final_blocks: list[str] = []
        pending: list[str] = []
        pending_words = 0

        for block in blocks:
            if self._is_heading(block):
                if pending_words >= self.config.min_words:
                    final_blocks.append(self._normalize_inline(" ".join(pending)))
                pending = []
                pending_words = 0
                final_blocks.append(block)
                continue

            words = len(block.split())
            if words < self.config.min_words and self.config.merge_small_paragraphs:
                pending.append(block)
                pending_words += words
                if pending_words >= self.config.min_words:
                    final_blocks.append(self._normalize_inline(" ".join(pending)))
                    pending = []
                    pending_words = 0
                continue

            if pending:
                block = self._normalize_inline(" ".join([*pending, block]))
                pending = []
                pending_words = 0

            final_blocks.append(block)

        if pending and pending_words >= self.config.min_words:
            final_blocks.append(self._normalize_inline(" ".join(pending)))

        return [block for block in final_blocks if block]

    def _line_grouped_blocks(self, text: str) -> list[str]:
        lines = [self._clean_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]

        blocks: list[str] = []
        current: list[str] = []
        current_words = 0
        target_words = self.config.target_block_words

        for line in lines:
            if self._is_heading(line):
                if current:
                    blocks.append(self._normalize_inline(" ".join(current)))
                    current = []
                    current_words = 0
                blocks.append(line)
                continue

            line_words = len(line.split())
            if current and (
                current_words >= target_words
                or self._starts_new_logical_block(line, current_words)
            ):
                blocks.append(self._normalize_inline(" ".join(current)))
                current = []
                current_words = 0

            current.append(line)
            current_words += line_words

            if current_words >= self.config.max_words:
                blocks.append(self._normalize_inline(" ".join(current)))
                current = []
                current_words = 0

        if current:
            blocks.append(self._normalize_inline(" ".join(current)))

        return blocks

    def _clean_line(self, line: str) -> str:
        normalized = self._normalize_inline(line)
        if not normalized:
            return ""

        lowered = normalized.lower()
        if lowered == "esc 2021":
            return ""
        if lowered.startswith("supplementary data"):
            return ""
        if lowered.startswith("no supplementary data for this section"):
            return ""
        if normalized.isdigit():
            return ""
        if _DOTTED_LINE_RE.match(normalized):
            return ""
        if len(normalized) <= 2:
            return ""
        return normalized

    def _normalize_block(self, text: str) -> str:
        lines = [self._clean_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        return self._normalize_inline(" ".join(lines))

    def _normalize_inline(self, text: str) -> str:
        cleaned = (
            text.replace("\u00a0", " ")
            .replace("•", " • ")
            .replace("◦", " ")
            .replace("/C0", "-")
            .replace("þ", "+")
        )
        cleaned = _WHITESPACE_RE.sub(" ", cleaned)
        return cleaned.strip()

    def _split_long_block(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.config.max_words:
            return [text]

        sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()]
        if len(sentences) <= 1:
            return self._split_by_word_window(words)

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0
        for sentence in sentences:
            count = len(sentence.split())
            if current and current_words + count > self.config.max_words:
                chunks.append(self._normalize_inline(" ".join(current)))
                current = []
                current_words = 0
            current.append(sentence)
            current_words += count
        if current:
            chunks.append(self._normalize_inline(" ".join(current)))
        return chunks

    def _split_by_word_window(self, words: list[str]) -> list[str]:
        chunks: list[str] = []
        step = max(self.config.min_words, self.config.max_words - self.config.word_window_overlap)
        for start in range(0, len(words), step):
            chunk_words = words[start : start + self.config.max_words]
            if len(chunk_words) < self.config.min_words:
                break
            chunks.append(" ".join(chunk_words))
        return chunks

    def _starts_new_logical_block(self, line: str, current_words: int) -> bool:
        if current_words < self.config.min_words:
            return False
        if line.startswith("•"):
            return False
        if _TABLE_HEADING_RE.match(line):
            return True
        if self._is_heading(line):
            return True
        if line.endswith(":"):
            return True
        if re.match(r"^\d+[.)]\s+", line):
            return True
        return False

    def _is_heading(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if len(stripped.split()) > 18:
            return False
        if _REFERENCE_HEADING_RE.match(stripped):
            return True
        if _TABLE_HEADING_RE.match(stripped):
            return True
        if _SECTION_NUMBER_RE.match(stripped) and len(stripped.split()) <= 12:
            return True
        if stripped in _HEADING_KEYWORDS:
            return True
        return bool(_UPPER_HEADING_RE.match(stripped))

    def _valid_passage(self, text: str) -> bool:
        words = text.split()
        if len(words) < self.config.min_words:
            return False
        if len(set(word.lower() for word in words)) < self.config.min_unique_words:
            return False
        lowered = text.lower()
        if any(marker in lowered for marker in self.config.reject_passage_markers):
            return False
        if lowered.startswith("authors/") or lowered.startswith("document reviewers"):
            return False
        return True

    def _make_passage(
        self,
        path: Path,
        page_number: int,
        local_index: int,
        text: str,
        section_title: str | None,
    ) -> ExtractedPassage:
        return ExtractedPassage(
            passage_id=f"{path.stem}_p{page_number:03d}_{local_index:03d}",
            document_id=path.stem,
            document_name=path.name,
            text=text,
            page=page_number,
            block_index=local_index,
            section_title=section_title,
            metadata={},
        )
