from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets.schema import BenchmarkCase

from .config import DatasetGenerationSettings
from .llm_client import LlmClientFactory
from .models import ExtractedPassage
from .pdf_extractor import PdfPassageExtractor
from .prompts import build_generation_prompt


class DatasetGenerator:
    def __init__(self, settings: DatasetGenerationSettings) -> None:
        self.settings = settings
        self.random = random.Random(settings.generation.seed)
        generation = settings.generation
        provider = generation.provider or os.getenv("DATASET_GENERATION_PROVIDER") or "ollama"
        if provider == "ollama":
            base_url = generation.api_base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
            api_key = None
        else:
            base_url = generation.api_base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:11434/v1"
            api_key = generation.api_key or os.getenv("OPENAI_API_KEY")
        model = generation.model or os.getenv("OPENAI_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5:7b"
        self.provider = provider
        self.model = model
        self.prompt_version = "dataset_generation_v1"
        self.client = LlmClientFactory.create(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=generation.temperature,
        )
        self.extractor = PdfPassageExtractor(settings.extraction)

    def build_dataset(self, pdf_path: str | Path) -> list[BenchmarkCase]:
        passages = self.extractor.extract(pdf_path)
        if not passages:
            raise ValueError(f"No passages extracted from {pdf_path}")
        selected = self._select_passages(passages, self.settings.generation.dataset_size)
        question_types = self._expand_mix(self.settings.generation.dataset_size)
        cases: list[BenchmarkCase] = []
        for index, (passage, question_type) in enumerate(zip(selected, question_types, strict=True), start=1):
            cases.append(self._generate_case(passage, question_type, index))
        return cases

    def save_dataset(self, cases: list[BenchmarkCase], output_path: str | Path) -> None:
        source_documents = sorted(
            {
                (case.source_document_id, case.source_document_name)
                for case in cases
            }
        )
        output = {
            "schema_version": "1.1",
            "dataset_type": "benchmark_dataset",
            "dataset_size": len(cases),
            "question_mix": self.settings.generation.mix,
            "created_at": datetime.now(UTC).isoformat(),
            "generation_metadata": {
                "provider": self.provider,
                "generator_model": self.model,
                "prompt_version": self.prompt_version,
                "extractor_version": "pdf_extractor_v2",
            },
            "source_documents": [
                {"document_id": doc_id, "document_name": doc_name}
                for doc_id, doc_name in source_documents
            ],
            "cases": [case.to_dict() for case in cases],
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    def _select_passages(self, passages: list[ExtractedPassage], dataset_size: int) -> list[ExtractedPassage]:
        if len(passages) < dataset_size:
            raise ValueError(f"Only {len(passages)} passages available, but dataset_size={dataset_size}")

        by_page: dict[int, list[ExtractedPassage]] = {}
        for passage in passages:
            page = passage.page or 0
            by_page.setdefault(page, []).append(passage)

        for bucket in by_page.values():
            self.random.shuffle(bucket)

        selected: list[ExtractedPassage] = []
        ordered_pages = sorted(by_page)
        while len(selected) < dataset_size and ordered_pages:
            next_pages: list[int] = []
            for page in ordered_pages:
                bucket = by_page[page]
                if bucket:
                    selected.append(bucket.pop())
                    if len(selected) == dataset_size:
                        break
                if bucket:
                    next_pages.append(page)
            ordered_pages = next_pages

        return selected

    def _expand_mix(self, total: int) -> list[str]:
        mix = self._normalized_mix(self.settings.generation.mix)
        counts = {key: int(value * total) for key, value in mix.items()}
        while sum(counts.values()) < total:
            for key in sorted(mix, key=lambda item: mix[item], reverse=True):
                counts[key] += 1
                if sum(counts.values()) == total:
                    break
        question_types = [kind for kind, count in counts.items() for _ in range(count)]
        self.random.shuffle(question_types)
        return question_types

    def _normalized_mix(self, mix: dict[str, float]) -> dict[str, float]:
        total = sum(mix.values())
        if total <= 0:
            raise ValueError("Question mix proportions must sum to more than zero.")
        return {key: value / total for key, value in mix.items()}

    def _generate_case(self, passage: ExtractedPassage, question_type: str, index: int) -> BenchmarkCase:
        prompt = build_generation_prompt(passage, question_type)
        last_error: Exception | None = None
        for attempt in range(1, self.settings.generation.retries + 1):
            try:
                result = self.client.chat_json(self.settings.generation.system_prompt, prompt)
                return self._coerce_case(result, passage, question_type, index)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(2 * attempt, 5))
        raise RuntimeError(f"Failed to generate case for {passage.passage_id}") from last_error

    def _coerce_case(
        self,
        data: dict[str, Any],
        passage: ExtractedPassage,
        question_type: str,
        index: int,
    ) -> BenchmarkCase:
        question = self._clean_text(str(data["question"]))
        answer = self._clean_text(str(data["reference_answer"]))
        required_facts = [self._clean_text(str(item)) for item in data.get("required_facts", []) if str(item).strip()]
        forbidden_facts = [self._clean_text(str(item)) for item in data.get("forbidden_facts", []) if str(item).strip()]
        tags = [self._slug_like(str(item)) for item in data.get("tags", []) if str(item).strip()]
        tags = sorted(set(tags))
        patient_variables = data.get("patient_variables", {})
        if not isinstance(patient_variables, dict):
            patient_variables = {}
        normalized_passage = self._normalize_for_matching(passage.text)
        return BenchmarkCase(
            id=f"case-{index:03d}",
            question=question,
            question_type=question_type.replace("_", "-"),
            patient_variables=patient_variables,
            gold_passage_id=passage.passage_id,
            gold_passage_text=passage.text,
            gold_passage_normalized=normalized_passage,
            gold_passage_hash=f"sha256:{hashlib.sha256(normalized_passage.encode('utf-8')).hexdigest()}",
            anchor_start_text=self._anchor_start(passage.text),
            anchor_end_text=self._anchor_end(passage.text),
            source_document_id=passage.document_id,
            source_document_name=passage.document_name,
            source_page=passage.page,
            source_block_index=passage.block_index,
            reference_answer=answer,
            required_facts=required_facts,
            forbidden_facts=forbidden_facts,
            tags=tags,
            generation_metadata={
                "generator_model": self.model,
                "prompt_version": self.prompt_version,
            },
            passage_metadata={
                "word_count": len(passage.text.split()),
                "char_count": len(passage.text),
            },
        )

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _slug_like(self, text: str) -> str:
        return re.sub(r"[^a-z0-9-]+", "-", text.strip().lower()).strip("-")

    def _normalize_for_matching(self, text: str) -> str:
        text = text.lower().replace("-", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _anchor_start(self, text: str, words: int = 8) -> str:
        return " ".join(text.split()[:words]).strip()

    def _anchor_end(self, text: str, words: int = 8) -> str:
        return " ".join(text.split()[-words:]).strip()

    def print_summary(self, cases: list[BenchmarkCase]) -> None:
        counts = Counter(case.question_type for case in cases)
        print(f"Generated {len(cases)} cases")
        print("Question mix:")
        for name, count in sorted(counts.items()):
            print(f"  {name}: {count}")
