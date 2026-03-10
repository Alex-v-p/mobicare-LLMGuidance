from __future__ import annotations

import re
from collections import Counter

from inference.indexing.cleaning.base import DocumentCleaner
from inference.indexing.models import SourceDocument


class NoOpCleaner(DocumentCleaner):
    def clean(self, document: SourceDocument) -> SourceDocument:
        return document


class BasicCleaner(DocumentCleaner):
    def clean(self, document: SourceDocument) -> SourceDocument:
        text = document.text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return SourceDocument(
            source_id=document.source_id,
            title=document.title,
            text=text.strip(),
            metadata=dict(document.metadata),
        )


class DeepCleaner(DocumentCleaner):
    _dot_lines = re.compile(r"(?m)^\s*(?:\.|·|•|_|\-){4,}\s*$")
    _page_markers = re.compile(r"(?mi)^\s*(?:page\s+)?\d+\s*/\s*\d+\s*$")
    _esc_header = re.compile(
        r"(?mi)^\s*(?:ESC Guidelines(?: \d+)?|ESC 20\d{2}|European Heart Journal.*|ESC GUIDELINES|Supplementary (?:Table|Figure|text).*)\s*$"
    )
    _doi_line = re.compile(r"(?mi)\bdoi\s*:\s*\S+")
    _url = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
    _email = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
    _hyphen_break = re.compile(r"(\w)-\n(\w)")
    _mid_sentence_linebreak = re.compile(r"(?<![\n.!?])\n(?!\n)")
    _spaces = re.compile(r"[ \t]+")
    _many_newlines = re.compile(r"\n{3,}")
    _toc_line = re.compile(
        r"(?mi)^\s*(?:\d+(?:\.\d+)*\s+)?[A-Z]?[A-Za-z][^\n]{5,}\.{3,}\s*\d+\s*$"
    )
    _reference_line = re.compile(
        r"(?mi)^\s*\d+\.\s+[A-Z][A-Za-z'`’\-]+.*(?:\d{4}|N Engl J Med|Lancet|Eur Heart J|JAMA|Circulation).*$"
    )
    _author_affiliation_line = re.compile(
        r"(?mi)^\s*(?:Authors?/Task Force Members?|Document Reviewers?|Author/Task Force Member affiliations?|ESC Clinical Practice Guidelines Committee|Working Groups:|Councils:|Associations:|Patient Forum:).*$"
    )
    _copyright_line = re.compile(
        r"(?mi)^\s*(?:Disclaimer:|The ESC Guidelines represent|No commercial use is authorized|Permission can be obtained|All rights reserved|This article has been co-published|The articles are identical except).*$"
    )
    _abbrev_definition = re.compile(
        r"(?m)^[A-Z][A-Z0-9/\-\+]{1,20}\s{1,}[A-Z][^\n]{2,}$"
    )
    _mostly_symbol_line = re.compile(r"^[\W_]+$")

    def clean(self, document: SourceDocument) -> SourceDocument:
        text = document.text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("￾", "")
        text = text.replace("þ", "+")
        text = self._hyphen_break.sub(r"\1\2", text)

        lines = [self._normalize_line(line) for line in text.split("\n")]
        lines = self._drop_obvious_noise(lines)
        lines = self._drop_repeated_short_lines(lines)
        lines = self._drop_low_value_sections(lines)

        text = "\n".join(lines)
        text = self._doi_line.sub("", text)
        text = self._url.sub("", text)
        text = self._email.sub("", text)

        text = self._mid_sentence_linebreak.sub(" ", text)
        text = self._spaces.sub(" ", text)
        text = self._many_newlines.sub("\n\n", text)
        text = re.sub(r" ?([,:;])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"[ ]+\n", "\n", text)

        text = self._post_filter_whole_document(text)

        return SourceDocument(
            source_id=document.source_id,
            title=document.title,
            text=text.strip(),
            metadata=dict(document.metadata),
        )

    def _normalize_line(self, line: str) -> str:
        line = line.strip()
        line = line.replace("–", "-").replace("—", "-")
        line = line.replace("•", "-")
        line = self._spaces.sub(" ", line)
        return line

    def _drop_obvious_noise(self, lines: list[str]) -> list[str]:
        kept: list[str] = []

        for line in lines:
            if not line:
                kept.append("")
                continue

            if self._dot_lines.match(line):
                continue
            if self._page_markers.match(line):
                continue
            if self._esc_header.match(line):
                continue
            if self._toc_line.match(line):
                continue
            if self._author_affiliation_line.match(line):
                continue
            if self._copyright_line.match(line):
                continue
            if self._mostly_symbol_line.match(line) and len(line) > 2:
                continue

            # Kill standalone contact / web / DOI lines
            if self._url.search(line) and len(line) < 200:
                continue
            if self._email.search(line):
                continue
            if "doi:" in line.lower():
                continue

            # Very short junk like repeated dots or broken OCR fragments
            if len(line) <= 2 and not re.search(r"[A-Za-z0-9]", line):
                continue

            kept.append(line)

        return kept

    def _drop_repeated_short_lines(self, lines: list[str]) -> list[str]:
        normalized_nonempty = [line.lower() for line in lines if line and len(line) <= 80]
        counts = Counter(normalized_nonempty)

        kept: list[str] = []
        for line in lines:
            if line and len(line) <= 80 and counts[line.lower()] >= 3:
                # removes repeated short headers/footers across pages
                continue
            kept.append(line)
        return kept

    def _drop_low_value_sections(self, lines: list[str]) -> list[str]:
        """
        Removes long runs of lines that are mostly:
        - abbreviations pages
        - references pages
        - author/contact boilerplate
        """
        if not lines:
            return lines

        blocks = self._split_into_blocks(lines)
        cleaned_blocks: list[str] = []

        for block in blocks:
            if self._is_reference_block(block):
                continue
            if self._is_abbreviation_block(block):
                continue
            if self._is_boilerplate_block(block):
                continue
            cleaned_blocks.append(block)

        output: list[str] = []
        for i, block in enumerate(cleaned_blocks):
            output.extend(block.split("\n"))
            if i < len(cleaned_blocks) - 1:
                output.append("")

        return output

    def _split_into_blocks(self, lines: list[str]) -> list[str]:
        blocks: list[str] = []
        current: list[str] = []

        for line in lines:
            if not line:
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
            else:
                current.append(line)

        if current:
            blocks.append("\n".join(current).strip())

        return [b for b in blocks if b.strip()]

    def _is_reference_block(self, block: str) -> bool:
        lines = [l for l in block.split("\n") if l.strip()]
        if len(lines) < 4:
            return False

        ref_hits = sum(1 for line in lines if self._reference_line.match(line))
        journal_hits = sum(
            1
            for line in lines
            if re.search(r"\b(?:N Engl J Med|Lancet|Eur Heart J|JAMA|Circulation)\b", line)
        )

        return ref_hits >= max(4, len(lines) // 3) or journal_hits >= max(4, len(lines) // 3)

    def _is_abbreviation_block(self, block: str) -> bool:
        lines = [l for l in block.split("\n") if l.strip()]
        if len(lines) < 8:
            return False

        abbrev_hits = sum(1 for line in lines if self._abbrev_definition.match(line))
        upper_heavy = sum(
            1 for line in lines
            if len(line) < 80 and re.fullmatch(r"[A-Z0-9/\-\+\(\)\., ]+", line) is not None
        )

        return abbrev_hits >= max(6, len(lines) // 2) or upper_heavy >= max(6, len(lines) // 2)

    def _is_boilerplate_block(self, block: str) -> bool:
        lowered = block.lower()

        boilerplate_markers = [
            "all rights reserved",
            "no commercial use is authorized",
            "permission can be obtained",
            "disclaimer:",
            "document reviewers:",
            "authors/task force members",
            "patient forum:",
            "author/task force member affiliations",
        ]

        hits = sum(1 for marker in boilerplate_markers if marker in lowered)
        return hits >= 2

    def _post_filter_whole_document(self, text: str) -> str:
        # Remove "15 References" section and everything after it if present
        text = re.sub(r"(?is)\n\s*15\s+References\b.*$", "", text)

        # Remove "Abbreviations and acronyms" section if it survived
        text = re.sub(
            r"(?is)\n\s*1\s+Abbreviations and acronyms\b.*?(?=\n\s*\d+\s+[A-Z]|\Z)",
            "\n",
            text,
        )

        # Remove obvious table of contents section if it survived
        text = re.sub(
            r"(?is)\n\s*Table of contents\b.*?(?=\n\s*\d+\s+[A-Z]|\Z)",
            "\n",
            text,
        )

        return text


class MedicalGuidelineDeepCleaner(DocumentCleaner):
    _spaces = re.compile(r"[ \t]+")
    _many_newlines = re.compile(r"\n{3,}")
    _hyphen_break = re.compile(r"(\w)-\n(\w)")
    _mid_sentence_linebreak = re.compile(r"(?<![\n.!?])\n(?!\n)")

    _dot_lines = re.compile(r"(?m)^\s*(?:\.|·|•|_|\-){4,}\s*$")
    _page_markers = re.compile(r"(?mi)^\s*(?:page\s+)?\d+\s*(?:/\s*\d+)?\s*$")
    _url = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
    _email = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
    _doi_inline = re.compile(r"(?i)\bdoi\s*:\s*\S+")
    _doi_url = re.compile(r"(?i)\b(?:https?://)?doi\.org/\S+")

    _journal_header = re.compile(
        r"(?mi)^\s*(?:European Heart Journal.*|ESC Guidelines(?: \d+)?|ESC 20\d{2}|Supplementary (?:data|figure|table|text).*)\s*$"
    )
    _toc_line = re.compile(
        r"(?mi)^\s*(?:\d+(?:\.\d+)*\s+)?[A-Z]?[A-Za-z][^\n]{5,}\.{3,}\s*\d+\s*$"
    )
    _reference_line = re.compile(
        r"(?mi)^\s*\d+\.\s+.+(?:\d{4}|N Engl J Med|Lancet|Eur Heart J|JAMA|Circulation|BMJ|Chest).*$"
    )
    _author_affiliation_line = re.compile(
        r"(?mi)^\s*(?:Authors?/Task Force Members?|Document Reviewers?|Author/Task Force Member affiliations?|ESC Clinical Practice Guidelines Committee|Working Groups:|Councils:|Associations:|Patient Forum:|Affiliations?:).*$"
    )
    _copyright_line = re.compile(
        r"(?mi)^\s*(?:Disclaimer:|The ESC Guidelines represent|No commercial use is authorized|Permission can be obtained|All rights reserved|This article has been co-published|The articles are identical except|Published on behalf of|For permissions, please email).*$"
    )
    _section_header = re.compile(
        r"(?i)^\s*(?:references|bibliography|acknowledg?e?ments|funding|conflict[s]? of interest|abbreviations(?: and acronyms)?|table of contents)\s*$"
    )
    _abbrev_definition = re.compile(
        r"(?m)^[A-Z][A-Z0-9/\-\+]{1,20}\s{1,}[A-Z][^\n]{2,}$"
    )

    def clean(self, document: SourceDocument) -> SourceDocument:
        text = document.text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("￾", "")
        text = text.replace("–", "-").replace("—", "-")
        text = self._hyphen_break.sub(r"\1\2", text)

        lines = [self._normalize_line(line) for line in text.split("\n")]
        lines = self._drop_line_noise(lines)
        lines = self._drop_repeated_short_lines(lines)
        lines = self._drop_low_value_blocks(lines)

        text = "\n".join(lines)
        text = self._url.sub("", text)
        text = self._email.sub("", text)
        text = self._doi_inline.sub("", text)
        text = self._doi_url.sub("", text)

        text = self._mid_sentence_linebreak.sub(" ", text)
        text = self._spaces.sub(" ", text)
        text = self._many_newlines.sub("\n\n", text)
        text = re.sub(r"[ ]+\n", "\n", text)
        text = re.sub(r" ?([,:;])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)

        text = self._trim_trailing_low_value_sections(text)

        return SourceDocument(
            source_id=document.source_id,
            title=document.title,
            text=text.strip(),
            metadata=dict(document.metadata),
        )

    def _normalize_line(self, line: str) -> str:
        line = line.strip()
        line = line.replace("•", "-")
        line = self._spaces.sub(" ", line)
        return line

    def _drop_line_noise(self, lines: list[str]) -> list[str]:
        kept: list[str] = []

        for line in lines:
            if not line:
                kept.append("")
                continue

            lowered = line.lower()

            if self._dot_lines.match(line):
                continue
            if self._page_markers.match(line):
                continue
            if self._journal_header.match(line):
                continue
            if self._toc_line.match(line):
                continue
            if self._author_affiliation_line.match(line):
                continue
            if self._copyright_line.match(line):
                continue

            if self._url.search(line) and len(line) < 250:
                continue
            if self._email.search(line):
                continue
            if "doi:" in lowered or "doi.org/" in lowered:
                continue

            if len(line) <= 2 and not re.search(r"[A-Za-z0-9]", line):
                continue

            kept.append(line)

        return kept

    def _drop_repeated_short_lines(self, lines: list[str]) -> list[str]:
        repeated_candidates = [line.lower() for line in lines if line and len(line) <= 100]
        counts = Counter(repeated_candidates)

        kept: list[str] = []
        for line in lines:
            if line and len(line) <= 100 and counts[line.lower()] >= 3:
                continue
            kept.append(line)
        return kept

    def _drop_low_value_blocks(self, lines: list[str]) -> list[str]:
        blocks = self._split_blocks(lines)
        kept_blocks: list[str] = []

        for block in blocks:
            if self._is_reference_block(block):
                continue
            if self._is_abbreviation_block(block):
                continue
            if self._is_metadata_block(block):
                continue
            kept_blocks.append(block)

        output: list[str] = []
        for i, block in enumerate(kept_blocks):
            output.extend(block.split("\n"))
            if i < len(kept_blocks) - 1:
                output.append("")

        return output

    def _split_blocks(self, lines: list[str]) -> list[str]:
        blocks: list[str] = []
        current: list[str] = []

        for line in lines:
            if not line:
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
            else:
                current.append(line)

        if current:
            blocks.append("\n".join(current).strip())

        return [block for block in blocks if block.strip()]

    def _is_reference_block(self, block: str) -> bool:
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) < 4:
            return False

        ref_hits = sum(1 for line in lines if self._reference_line.match(line))
        journal_hits = sum(
            1
            for line in lines
            if re.search(r"\b(?:N Engl J Med|Lancet|Eur Heart J|JAMA|Circulation|BMJ|Chest)\b", line)
        )

        lowered = block.lower()
        has_reference_header = bool(
            re.search(r"(?im)^\s*(?:references|bibliography)\s*$", block)
        )

        return has_reference_header or ref_hits >= max(4, len(lines) // 3) or journal_hits >= max(4, len(lines) // 3)

    def _is_abbreviation_block(self, block: str) -> bool:
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) < 8:
            return False

        abbrev_hits = sum(1 for line in lines if self._abbrev_definition.match(line))
        upper_heavy = sum(
            1
            for line in lines
            if len(line) < 100 and re.fullmatch(r"[A-Z0-9/\-\+\(\)\., ]+", line) is not None
        )

        lowered = block.lower()
        has_abbrev_header = bool(
            re.search(r"(?im)^\s*abbreviations(?: and acronyms)?\s*$", block)
        )

        return has_abbrev_header or abbrev_hits >= max(6, len(lines) // 2) or upper_heavy >= max(6, len(lines) // 2)

    def _is_metadata_block(self, block: str) -> bool:
        lowered = block.lower()

        markers = [
            "all rights reserved",
            "no commercial use is authorized",
            "permission can be obtained",
            "document reviewers",
            "authors/task force members",
            "author/task force member affiliations",
            "patient forum",
            "published on behalf of",
            "for permissions, please email",
            "conflict of interest",
            "supplementary data",
        ]

        hits = sum(1 for marker in markers if marker in lowered)
        return hits >= 2

    def _trim_trailing_low_value_sections(self, text: str) -> str:
        section_patterns = [
            r"(?is)\n\s*references\s*\n.*$",
            r"(?is)\n\s*bibliography\s*\n.*$",
            r"(?is)\n\s*acknowledg?e?ments\s*\n.*$",
        ]

        for pattern in section_patterns:
            text = re.sub(pattern, "", text)

        return text