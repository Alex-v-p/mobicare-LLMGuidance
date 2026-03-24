from __future__ import annotations

import concurrent.futures as cf
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import random
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any

from datasets.schema import BenchmarkCase

from .config import DatasetGenerationSettings
from .llm_client import OllamaClient
from .models import ExtractedPassage
from .pdf_extractor import PdfPassageExtractor
from .prompts import build_answerable_prompt, build_unanswerable_prompt

_UNANSWERABLE_TOPICS = [
    "management of ulcerative colitis flare during pregnancy",
    "first-line antibiotics for pediatric otitis media",
    "treatment of diabetic ketoacidosis in children",
    "staging and treatment of melanoma",
    "management of acute appendicitis",
    "vaccination schedule for infants",
    "migraine prophylaxis options",
    "thyroid storm treatment protocol",
    "anticoagulation reversal in intracranial hemorrhage",
    "COPD inhaler step-up therapy",
]


class DatasetGenerator:
    def __init__(self, settings: DatasetGenerationSettings) -> None:
        self.settings = settings
        self.random = random.Random(settings.generation.seed)
        self.client = OllamaClient(
            base_url=settings.generation.api_base_url,
            model=settings.generation.model,
            timeout_seconds=settings.generation.timeout_seconds,
            options=settings.generation.ollama_options,
        )
        self.extractor = PdfPassageExtractor(settings.extraction)
        self._write_lock = Lock()
        self._jsonl_path = Path(settings.output_jsonl_path) if settings.output_jsonl_path else Path(settings.output_path).with_suffix(".jsonl")
        self._failed_jsonl_path = self._jsonl_path.with_name(f"{self._jsonl_path.stem}.failed.jsonl")

    def build_dataset(self, pdf_path: str | Path) -> list[BenchmarkCase]:
        passages = self.extractor.extract(pdf_path)
        answerable_count, unanswerable_count = self._case_kind_counts(self.settings.generation.dataset_size)
        if len(passages) < answerable_count:
            raise ValueError(f"Only {len(passages)} passages available, but need {answerable_count} answerable cases")
        selected = self._select_passages(passages, answerable_count)
        question_types = self._expand_mix(self.settings.generation.dataset_size)
        case_kinds = ["answerable"] * answerable_count + ["unanswerable"] * unanswerable_count
        paired_specs = list(zip(question_types, case_kinds, strict=True))
        self.random.shuffle(paired_specs)

        jobs: list[tuple[int, ExtractedPassage | None, str, str]] = []
        passage_iter = iter(selected)
        unanswerable_topics = self._sample_unanswerable_topics(unanswerable_count)
        topic_iter = iter(unanswerable_topics)
        for index, (question_type, case_kind) in enumerate(paired_specs, start=1):
            if case_kind == "answerable":
                jobs.append((index, next(passage_iter), question_type, case_kind))
            else:
                jobs.append((index, None, question_type, next(topic_iter)))

        existing = self._load_existing_cases() if self.settings.generation.resume_from_jsonl else {}
        results: dict[int, BenchmarkCase] = {self._case_index(case.id): case for case in existing.values()}
        pending_jobs = [job for job in jobs if job[0] not in results]
        if not pending_jobs:
            return [results[index] for index in sorted(results)]

        with cf.ThreadPoolExecutor(max_workers=max(1, self.settings.generation.concurrency)) as executor:
            future_map = {
                executor.submit(self._generate_job, index, passage, question_type, kind_or_topic): (index, passage, question_type, kind_or_topic)
                for index, passage, question_type, kind_or_topic in pending_jobs
            }
            for future in cf.as_completed(future_map):
                index, passage, question_type, kind_or_topic = future_map[future]
                try:
                    case = future.result()
                    results[index] = case
                    self._append_case_jsonl(case)
                except Exception as exc:  # noqa: BLE001
                    self._append_failed_case(index, question_type, kind_or_topic, passage, exc)
                    if not self.settings.generation.continue_on_error:
                        raise
                    print(f"[WARN] Skipping case-{index:03d}: {exc}")
        cases = [results[index] for index in sorted(results)]
        if not cases:
            raise RuntimeError("No benchmark cases were generated successfully")
        return cases

    def _load_existing_cases(self) -> dict[str, BenchmarkCase]:
        if not self._jsonl_path.exists():
            return {}
        loaded: dict[str, BenchmarkCase] = {}
        for line in self._jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                case = BenchmarkCase(**payload)
            except Exception:  # noqa: BLE001
                continue
            loaded[case.id] = case
        if loaded:
            print(f"[INFO] Resuming with {len(loaded)} existing cases from {self._jsonl_path}")
        return loaded

    def _append_case_jsonl(self, case: BenchmarkCase) -> None:
        with self._write_lock:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")

    def _append_failed_case(
        self,
        index: int,
        question_type: str,
        kind_or_topic: str,
        passage: ExtractedPassage | None,
        error: Exception,
    ) -> None:
        payload = {
            "id": f"case-{index:03d}",
            "question_type": question_type,
            "case_kind_or_topic": kind_or_topic,
            "gold_passage_id": passage.passage_id if passage else None,
            "source_page": passage.page if passage else None,
            "error": str(error),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._write_lock:
            self._failed_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._failed_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _case_index(self, case_id: str) -> int:
        return int(case_id.split("-")[-1])

    def _generate_job(self, index: int, passage: ExtractedPassage | None, question_type: str, kind_or_topic: str) -> BenchmarkCase:
        if passage is None:
            return self._generate_unanswerable_case(question_type, kind_or_topic, index)
        return self._generate_answerable_case(passage, question_type, index)

    def _case_kind_counts(self, total: int) -> tuple[int, int]:
        mix = self._normalized_mix(self.settings.generation.case_kind_mix)
        answerable = round(total * mix.get("answerable", 0.8))
        answerable = min(max(answerable, 0), total)
        return answerable, total - answerable

    def _sample_unanswerable_topics(self, total: int) -> list[str]:
        topics = []
        for i in range(total):
            topics.append(_UNANSWERABLE_TOPICS[i % len(_UNANSWERABLE_TOPICS)])
        self.random.shuffle(topics)
        return topics

    def _select_passages(self, passages: list[ExtractedPassage], desired: int) -> list[ExtractedPassage]:
        by_page: dict[int, list[ExtractedPassage]] = {}
        for passage in passages:
            by_page.setdefault(passage.page, []).append(passage)
        for bucket in by_page.values():
            bucket.sort(key=lambda item: item.block_index)
            self.random.shuffle(bucket)
        selected: list[ExtractedPassage] = []
        pages = sorted(by_page)
        while len(selected) < desired and pages:
            remaining: list[int] = []
            for page in pages:
                bucket = by_page[page]
                if bucket:
                    selected.append(bucket.pop())
                    if len(selected) == desired:
                        break
                if bucket:
                    remaining.append(page)
            pages = remaining
        return selected

    def _expand_mix(self, total: int) -> list[str]:
        mix = self._normalized_mix(self.settings.generation.question_mix)
        counts = {key: int(value * total) for key, value in mix.items()}
        while sum(counts.values()) < total:
            for key in sorted(mix, key=lambda k: mix[k], reverse=True):
                counts[key] += 1
                if sum(counts.values()) == total:
                    break
        items = [kind for kind, count in counts.items() for _ in range(count)]
        self.random.shuffle(items)
        return items

    def _normalized_mix(self, mix: dict[str, float]) -> dict[str, float]:
        total = sum(mix.values())
        if total <= 0:
            raise ValueError("Mix values must sum to more than zero")
        return {k: v / total for k, v in mix.items()}

    def _generate_answerable_case(self, passage: ExtractedPassage, question_type: str, index: int) -> BenchmarkCase:
        prompt = build_answerable_prompt(passage, question_type)
        data = self._retry_json(prompt)
        normalized = passage.normalized_text
        return BenchmarkCase(
            id=f"case-{index:03d}",
            dataset_version=self.settings.generation.dataset_version,
            question=self._clean_text(data["question"]),
            question_type=question_type,
            reasoning_type=self._slug_like(data.get("reasoning_type", "single-hop")),
            difficulty=self._slug_like(data.get("difficulty", "easy")),
            answerability="answerable",
            expected_behavior="answer_from_context",
            expected_abstention_style=None,
            case_weight=1.0,
            review_status="auto_generated_unreviewed",
            patient_variables=data.get("patient_variables", {}) if isinstance(data.get("patient_variables", {}), dict) else {},
            gold_passage_id=passage.passage_id,
            gold_passage_text=passage.text,
            gold_passage_normalized=normalized,
            gold_passage_hash=f"sha256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}",
            anchor_start_text=self._anchor_start(passage.text),
            anchor_end_text=self._anchor_end(passage.text),
            source_document_id=passage.document_id,
            source_document_name=passage.document_name,
            source_page=passage.page,
            source_block_index=passage.block_index,
            reference_answer=self._clean_text(data["reference_answer"]),
            required_facts=self._clean_list(data.get("required_facts", [])),
            forbidden_facts=self._clean_list(data.get("forbidden_facts", [])),
            query_variants=self._clean_list(data.get("query_variants", [])),
            tags=self._slug_list(data.get("tags", [])),
            retrieval_hints=self._normalize_retrieval_hints(data.get("retrieval_hints", {}), passage),
            unanswerable_reason=None,
            generation_metadata={
                "generator_model": self.settings.generation.model,
                "prompt_version": self.settings.generation.prompt_version,
                "provider": self.settings.generation.provider,
            },
            passage_metadata={
                "word_count": len(passage.text.split()),
                "char_count": len(passage.text),
                "section_title": passage.section_title,
            },
            hallucination_metadata=self._normalize_hallucination_metadata(data.get("hallucination_metadata", {}), False),
        )

    def _generate_unanswerable_case(self, question_type: str, topic: str, index: int) -> BenchmarkCase:
        prompt = build_unanswerable_prompt(question_type, topic)
        data = self._retry_json(prompt)
        return BenchmarkCase(
            id=f"case-{index:03d}",
            dataset_version=self.settings.generation.dataset_version,
            question=self._clean_text(data["question"]),
            question_type=question_type,
            reasoning_type=self._slug_like(data.get("reasoning_type", "abstention")),
            difficulty=self._slug_like(data.get("difficulty", "medium")),
            answerability="unanswerable",
            expected_behavior="abstain_or_insufficient_context",
            expected_abstention_style="state-not-in-passage",
            case_weight=1.0,
            review_status="auto_generated_unreviewed",
            patient_variables=data.get("patient_variables", {}) if isinstance(data.get("patient_variables", {}), dict) else {},
            gold_passage_id=None,
            gold_passage_text=None,
            gold_passage_normalized=None,
            gold_passage_hash=None,
            anchor_start_text=None,
            anchor_end_text=None,
            source_document_id=None,
            source_document_name=None,
            source_page=None,
            source_block_index=None,
            reference_answer="The provided context does not specify this.",
            required_facts=[],
            forbidden_facts=self._clean_list(data.get("forbidden_facts", [])),
            query_variants=self._clean_list(data.get("query_variants", [])),
            tags=self._slug_list(data.get("tags", [])),
            retrieval_hints={
                "key_terms": self._clean_list(data.get("retrieval_hints", {}).get("key_terms", [])),
                "expected_section": None,
                "document_scope": "out_of_scope",
            },
            unanswerable_reason=self._clean_text(data.get("unanswerable_reason", "The benchmark defines this as an out-of-scope question.")),
            generation_metadata={
                "generator_model": self.settings.generation.model,
                "prompt_version": self.settings.generation.prompt_version,
                "provider": self.settings.generation.provider,
            },
            passage_metadata={"word_count": 0, "char_count": 0, "section_title": None},
            hallucination_metadata=self._normalize_hallucination_metadata(data.get("hallucination_metadata", {}), True),
        )

    def _retry_json(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.generation.retries + 1):
            try:
                return self.client.chat_json(
                    "You are a careful generator of structured benchmark cases. Return only valid JSON.",
                    prompt,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(attempt, 3))
        raise RuntimeError("Failed to generate benchmark case") from last_error

    def save_dataset(self, cases: list[BenchmarkCase], output_path: str | Path, output_jsonl_path: str | None = None) -> None:
        cases = sorted(cases, key=lambda case: case.id)
        payload = {
            "schema_version": "1.4",
            "dataset_type": "benchmark_dataset",
            "dataset_id": self.settings.generation.dataset_id,
            "dataset_size": len(cases),
            "question_mix": self.settings.generation.question_mix,
            "case_kind_mix": self.settings.generation.case_kind_mix,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generation_metadata": {
                "provider": self.settings.generation.provider,
                "generator_model": self.settings.generation.model,
                "prompt_version": self.settings.generation.prompt_version,
                "extractor_version": "pdf_extractor_v3",
                "concurrency": self.settings.generation.concurrency,
                "ollama_options": self.settings.generation.ollama_options,
                "failed_cases_jsonl": str(self._failed_jsonl_path),
            },
            "source_documents": sorted(
                [
                    {"document_id": case.source_document_id, "document_name": case.source_document_name}
                    for case in cases
                    if case.source_document_id and case.source_document_name
                ],
                key=lambda item: (item["document_id"], item["document_name"]),
            ),
            "summary": self._summary(cases),
            "cases": [case.to_dict() for case in cases],
        }
        payload["source_documents"] = list({(d["document_id"], d["document_name"]): d for d in payload["source_documents"]}.values())
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        jsonl_path = Path(output_jsonl_path) if output_jsonl_path else out.with_suffix(".jsonl")
        jsonl_path.write_text(
            "\n".join(json.dumps(case.to_dict(), ensure_ascii=False) for case in cases) + "\n",
            encoding="utf-8",
        )

    def _summary(self, cases: list[BenchmarkCase]) -> dict[str, Any]:
        return {
            "question_type_counts": dict(sorted(Counter(case.question_type for case in cases).items())),
            "answerability_counts": dict(sorted(Counter(case.answerability for case in cases).items())),
            "difficulty_counts": dict(sorted(Counter(case.difficulty for case in cases).items())),
            "reasoning_type_counts": dict(sorted(Counter(case.reasoning_type for case in cases).items())),
            "average_passage_word_count": round(
                sum(case.passage_metadata.get("word_count", 0) for case in cases) / max(1, len(cases)), 1
            ),
        }

    def _normalize_retrieval_hints(self, data: Any, passage: ExtractedPassage) -> dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        return {
            "key_terms": self._clean_list(data.get("key_terms", [])),
            "expected_section": data.get("expected_section") or passage.section_title,
            "document_scope": data.get("document_scope") or "same_page_preferred",
        }

    def _normalize_hallucination_metadata(self, data: Any, is_test: bool) -> dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        return {
            "risk_level": self._slug_like(data.get("risk_level", "medium")),
            "likely_failure_modes": self._clean_list(data.get("likely_failure_modes", [])),
            "unsupported_targets": self._clean_list(data.get("unsupported_targets", [])),
            "is_hallucination_test": bool(data.get("is_hallucination_test", is_test)),
            "case_kind": "unanswerable" if is_test else "answerable",
            "expected_behavior": "abstain_or_insufficient_context" if is_test else "answer_from_context",
        }

    def _clean_text(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _clean_list(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        return [self._clean_text(v) for v in values if self._clean_text(v)]

    def _slug_list(self, values: Any) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in self._clean_list(values):
            slug = self._slug_like(value)
            if slug and slug not in seen:
                seen.add(slug)
                result.append(slug)
        return result

    def _slug_like(self, value: Any) -> str:
        text = self._clean_text(value).lower().replace("_", "-")
        text = re.sub(r"[^a-z0-9-]+", "-", text)
        text = re.sub(r"-{2,}", "-", text)
        return text.strip("-")

    def _anchor_start(self, text: str) -> str:
        return " ".join(text.split()[:8])

    def _anchor_end(self, text: str) -> str:
        return " ".join(text.split()[-8:])
