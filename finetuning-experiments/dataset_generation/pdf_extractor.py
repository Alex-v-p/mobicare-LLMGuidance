from __future__ import annotations

import re
from pathlib import Path
from pypdf import PdfReader

from .config import ExtractionConfig
from .models import ExtractedPassage


class PdfPassageExtractor:
    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config

    def extract(self, pdf_path: str | Path) -> list[ExtractedPassage]:
        reader = PdfReader(str(pdf_path))
        document_name = Path(pdf_path).name
        document_id = Path(pdf_path).stem
        passages: list[ExtractedPassage] = []
        for page_index, page in enumerate(reader.pages, start=1):
            if page_index <= self.config.skip_first_pages:
                continue
            raw_text = page.extract_text() or ""
            if not raw_text.strip():
                continue
            if self.config.stop_at_references and self._looks_like_references_page(raw_text):
                break
            blocks = self._extract_blocks(raw_text)
            block_index = 0
            for block in blocks:
                words = len(block.split())
                if words < self.config.min_words:
                    continue
                for chunk in self._split_long_block(block):
                    chunk = self._normalize(chunk)
                    if len(chunk.split()) < self.config.min_words:
                        continue
                    block_index += 1
                    passages.append(
                        ExtractedPassage(
                            passage_id=f"{document_id}_p{page_index:03d}_{block_index:03d}",
                            text=chunk,
                            normalized_text=self._normalize_for_matching(chunk),
                            document_id=document_id,
                            document_name=document_name,
                            page=page_index,
                            block_index=block_index,
                            section_title=self._guess_section_title(chunk),
                            metadata={"word_count": len(chunk.split()), "char_count": len(chunk)},
                        )
                    )
        return passages

    def _extract_blocks(self, raw_text: str) -> list[str]:
        text = raw_text.replace("\r", "")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        blocks: list[str] = []
        current: list[str] = []
        for line in lines:
            if self._is_noise_line(line):
                continue
            if current and (self._looks_like_heading(line) or len(" ".join(current).split()) >= self.config.max_words):
                blocks.append(" ".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append(" ".join(current))
        return blocks

    def _split_long_block(self, block: str) -> list[str]:
        words = block.split()
        if len(words) <= self.config.max_words:
            return [block]
        chunks: list[str] = []
        step = self.config.max_words
        i = 0
        while i < len(words):
            chunk_words = words[i : i + step]
            chunks.append(" ".join(chunk_words))
            i += step
        return chunks

    def _looks_like_heading(self, line: str) -> bool:
        stripped = line.strip()
        return bool(stripped.isupper() or re.match(r"^(IN WHOM AND WHEN\?|HOW TO USE\?|PROBLEM SOLVING|ADVICE TO PATIENT)", stripped))

    def _is_noise_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if re.fullmatch(r"\d+", stripped):
            return True
        if stripped.lower().startswith("esc guidelines"):
            return True
        return False

    def _looks_like_references_page(self, text: str) -> bool:
        lower = text.lower()
        return "references" in lower and lower.count("doi") + lower.count("et al") > 3

    def _guess_section_title(self, text: str) -> str | None:
        for marker in ["IN WHOM AND WHEN?", "HOW TO USE?", "PROBLEM SOLVING", "ADVICE TO PATIENT"]:
            if marker.lower() in text.lower():
                return marker
        first = text.split(".")[0].strip()
        return first[:80] if first else None

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_for_matching(self, text: str) -> str:
        text = text.lower()
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()
