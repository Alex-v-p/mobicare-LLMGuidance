from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets.schema import BenchmarkCase
from dataset_generation.llm_client import OllamaClient


DEFAULT_QUESTION_MIX: dict[str, float] = {
    "clinical-scenario": 0.55,
    "slightly-indirect": 0.25,
    "factual": 0.20,
}


BIOMARKER_CASE_PROMPT = """You are creating a benchmark case for a medical guidance system.
The real request sent to the system will contain ONLY patient biomarker/context variables and NO explicit question.
However, for benchmark evaluation you must still generate a hidden evaluation question and a reference answer.

Return ONLY valid JSON with keys:
question, question_type, reasoning_type, difficulty, reference_answer, required_facts, forbidden_facts, query_variants, tags, retrieval_hints, hallucination_metadata.

Rules:
- The generated question must be answerable from the biomarker/context profile plus generally relevant heart-failure-oriented clinical interpretation behavior.
- Focus on next-step priorities, safety checks, interpretation, monitoring, and escalation thinking.
- These cases are primarily observational/behavioral, not strict paragraph-grounded quote matching.
- Do not require an exact drug recommendation unless the profile strongly supports that.
- Use short, concrete required_facts.
- query_variants should have 0-2 alternatives.
- retrieval_hints must include key_terms, expected_section, document_scope.
- hallucination_metadata must include risk_level, likely_failure_modes, unsupported_targets.
- question_type must equal the requested value.
- reasoning_type should be one of interpretation, safety-check, prioritization, monitoring, scenario-application.
- difficulty should be easy, medium, or hard.

Requested question_type: {question_type}
Patient/context variables:
{patient_lines}
"""


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_clean_text(v) for v in values if _clean_text(v)]


def _slug_like(value: Any) -> str:
    text = _clean_text(value).lower().replace("_", "-")
    text = re.sub(r"[^a-z0-9-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


def _slug_list(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in _clean_list(values):
        slug = _slug_like(value)
        if slug and slug not in seen:
            seen.add(slug)
            result.append(slug)
    return result


def _normalize_hallucination_metadata(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    return {
        "risk_level": _slug_like(data.get("risk_level", "medium")),
        "likely_failure_modes": _clean_list(data.get("likely_failure_modes", [])),
        "unsupported_targets": _clean_list(data.get("unsupported_targets", [])),
        "is_hallucination_test": bool(data.get("is_hallucination_test", False)),
        "case_kind": "biomarker_only",
        "expected_behavior": "answer_from_biomarkers_without_explicit_question",
    }


def _normalize_retrieval_hints(data: Any, patient_variables: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    fallback_terms = [str(key).replace("_", " ") for key in patient_variables.keys()][:6]
    return {
        "key_terms": _clean_list(data.get("key_terms", [])) or fallback_terms,
        "expected_section": data.get("expected_section") or "biomarker-driven guidance",
        "document_scope": data.get("document_scope") or "behavioral-observation",
    }


def _mix_counts(total: int, mix: dict[str, float]) -> list[str]:
    if total <= 0:
        return []
    mix_total = sum(mix.values())
    if mix_total <= 0:
        raise ValueError("question_mix must sum to more than zero")
    normalized = {key: value / mix_total for key, value in mix.items()}
    counts = {key: int(value * total) for key, value in normalized.items()}
    while sum(counts.values()) < total:
        for key in sorted(normalized, key=lambda item: normalized[item], reverse=True):
            counts[key] += 1
            if sum(counts.values()) == total:
                break
    return [key for key, count in counts.items() for _ in range(count)]


def _load_marker_payload(marker_ranges_path: Path) -> dict[str, Any]:
    return json.loads(marker_ranges_path.read_text(encoding="utf-8"))


def _find_numeric_band(payload: dict[str, Any]) -> dict[str, Any] | None:
    for band in payload.get("bands", []):
        if band.get("low") is not None or band.get("high") is not None:
            return band
    if payload.get("low") is not None or payload.get("high") is not None:
        return payload
    return None


def _sample_value(rng: random.Random, low: float | None, high: float | None, abnormal: str | None) -> float:
    if low is None and high is None:
        return round(rng.uniform(1.0, 100.0), 1)
    if low is None:
        if abnormal == "high":
            return round(high * rng.uniform(1.15, 1.8), 2)
        return round(high * rng.uniform(0.4, 0.95), 2)
    if high is None:
        if abnormal == "low":
            return round(low * rng.uniform(0.4, 0.9), 2)
        return round(low * rng.uniform(1.05, 1.6), 2)
    width = max(high - low, max(abs(high), abs(low), 1.0) * 0.25)
    if abnormal == "low":
        return round(low - rng.uniform(0.1 * width, 0.6 * width), 2)
    if abnormal == "high":
        return round(high + rng.uniform(0.1 * width, 0.6 * width), 2)
    return round(rng.uniform(low, high), 2)


def _pick_marker_definitions(raw_markers: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    usable: list[tuple[str, dict[str, Any]]] = []
    for key, payload in raw_markers.items():
        kind = str(payload.get("kind") or "numeric")
        if kind != "numeric":
            continue
        if _find_numeric_band(payload) is None:
            continue
        usable.append((key, payload))
    return usable


def _sample_patient_profile(raw_markers: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    usable = _pick_marker_definitions(raw_markers)
    if len(usable) < 6:
        raise ValueError("Not enough usable biomarker definitions found")

    patient: dict[str, Any] = {
        "age": int(rng.randint(45, 88)),
        "gender": rng.choice(["male", "female"]),
    }

    chosen = rng.sample(usable, k=rng.randint(5, 8))
    abnormal_budget = rng.randint(2, 4)
    abnormal_positions = set(rng.sample(range(len(chosen)), k=min(abnormal_budget, len(chosen))))

    for idx, (key, payload) in enumerate(chosen):
        band = _find_numeric_band(payload)
        if band is None:
            continue
        direction = str(band.get("direction") or payload.get("direction") or "outside_range")
        if idx in abnormal_positions:
            if direction == "high_is_bad":
                abnormal = "high"
            elif direction == "low_is_bad":
                abnormal = "low"
            else:
                abnormal = rng.choice(["low", "high"])
        else:
            abnormal = None
        patient[key] = _sample_value(
            rng,
            float(band["low"]) if band.get("low") is not None else None,
            float(band["high"]) if band.get("high") is not None else None,
            abnormal,
        )

    return patient


def _patient_lines(patient_variables: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in sorted(patient_variables.items()))


def _build_prompt(patient_variables: dict[str, Any], question_type: str) -> str:
    return BIOMARKER_CASE_PROMPT.format(
        question_type=question_type,
        patient_lines=_patient_lines(patient_variables),
    )


def _summary(cases: list[BenchmarkCase]) -> dict[str, Any]:
    return {
        "question_type_counts": dict(sorted(Counter(case.question_type for case in cases).items())),
        "answerability_counts": dict(sorted(Counter(case.answerability for case in cases).items())),
        "difficulty_counts": dict(sorted(Counter(case.difficulty for case in cases).items())),
        "reasoning_type_counts": dict(sorted(Counter(case.reasoning_type for case in cases).items())),
        "average_biomarker_count": round(
            sum(max(0, len(case.patient_variables) - 2) for case in cases) / max(1, len(cases)), 1
        ),
    }


def _load_existing_cases(base_dataset_path: str | Path | None) -> tuple[dict[str, Any] | None, list[BenchmarkCase]]:
    if not base_dataset_path:
        return None, []
    base_payload = json.loads(Path(base_dataset_path).read_text(encoding="utf-8"))
    existing_cases = [BenchmarkCase(**case) for case in base_payload.get("cases") or []]
    return base_payload, existing_cases


def _assign_unique_ids(new_cases: list[BenchmarkCase], existing_cases: list[BenchmarkCase]) -> None:
    seen_ids = {case.id for case in existing_cases}
    next_index = 1
    for case in new_cases:
        original_id = case.id
        while case.id in seen_ids:
            case.id = f"bio-case-{next_index:03d}"
            next_index += 1
        seen_ids.add(case.id)
        if case.id != original_id:
            case.generation_metadata["original_case_id"] = original_id


def build_biomarker_dataset(
    *,
    marker_ranges_path: str | Path,
    output_path: str | Path,
    output_jsonl_path: str | Path | None,
    dataset_id: str,
    dataset_version: str,
    dataset_size: int,
    base_dataset_path: str | Path | None = None,
    model: str,
    api_base_url: str,
    timeout_seconds: int,
    seed: int,
    prompt_version: str,
    question_mix: dict[str, float] | None = None,
    ollama_options: dict[str, Any] | None = None,
) -> list[BenchmarkCase]:
    rng = random.Random(seed)
    raw_markers = _load_marker_payload(Path(marker_ranges_path))
    client = OllamaClient(
        base_url=api_base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        options=ollama_options or {"num_predict": 500, "num_ctx": 4096, "temperature": 0.2},
    )
    question_types = _mix_counts(dataset_size, question_mix or DEFAULT_QUESTION_MIX)
    rng.shuffle(question_types)

    new_cases: list[BenchmarkCase] = []
    for index, question_type in enumerate(question_types, start=1):
        patient_variables = _sample_patient_profile(raw_markers, rng)
        prompt = _build_prompt(patient_variables, question_type)
        data = client.chat_json(
            "You are a careful generator of structured biomarker-only benchmark cases. Return only valid JSON.",
            prompt,
        )
        new_cases.append(
            BenchmarkCase(
                id=f"bio-case-{index:03d}",
                dataset_version=dataset_version,
                question=_clean_text(data.get("question")),
                question_type=question_type,
                reasoning_type=_slug_like(data.get("reasoning_type", "interpretation")),
                difficulty=_slug_like(data.get("difficulty", "medium")),
                answerability="answerable",
                expected_behavior="answer_from_biomarkers_without_explicit_question",
                expected_abstention_style=None,
                case_weight=0.75,
                review_status="auto_generated_unreviewed",
                patient_variables=patient_variables,
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
                reference_answer=_clean_text(data.get("reference_answer")),
                required_facts=_clean_list(data.get("required_facts", [])),
                forbidden_facts=_clean_list(data.get("forbidden_facts", [])),
                query_variants=_clean_list(data.get("query_variants", [])),
                tags=sorted(set(_slug_list(data.get("tags", [])) + ["biomarker-only", "observation-case"])),
                retrieval_hints=_normalize_retrieval_hints(data.get("retrieval_hints", {}), patient_variables),
                unanswerable_reason=None,
                generation_metadata={
                    "provider": "ollama",
                    "generator_model": model,
                    "prompt_version": prompt_version,
                    "case_source": "biomarker_generator_v1",
                    "request_mode": "biomarker_only",
                    "omit_question_from_request": True,
                    "evaluation_intent": "behavior_observation",
                    "evaluation_profile": "observation_only",
                    "marker_ranges_path": str(marker_ranges_path),
                },
                passage_metadata={
                    "word_count": 0,
                    "char_count": 0,
                    "section_title": None,
                    "biomarker_count": max(0, len(patient_variables) - 2),
                },
                hallucination_metadata=_normalize_hallucination_metadata(data.get("hallucination_metadata", {})),
            )
        )

    base_payload, existing_cases = _load_existing_cases(base_dataset_path)
    _assign_unique_ids(new_cases, existing_cases)
    combined_cases = existing_cases + new_cases

    payload = {
        "schema_version": str((base_payload or {}).get("schema_version") or "1.4"),
        "dataset_type": "benchmark_dataset",
        "dataset_id": str((base_payload or {}).get("dataset_id") or dataset_id),
        "dataset_version": str((base_payload or {}).get("dataset_version") or dataset_version),
        "dataset_size": len(combined_cases),
        "question_mix": (base_payload or {}).get("question_mix") or question_mix or DEFAULT_QUESTION_MIX,
        "case_kind_mix": (base_payload or {}).get("case_kind_mix") or {"answerable": 1.0},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_metadata": {
            **dict((base_payload or {}).get("generation_metadata") or {}),
            "biomarker_augmentation": {
                "provider": "ollama",
                "generator_model": model,
                "prompt_version": prompt_version,
                "generator_mode": "biomarker_only_append",
                "marker_ranges_path": str(marker_ranges_path),
                "ollama_options": ollama_options or {"num_predict": 500, "num_ctx": 4096, "temperature": 0.2},
                "request_question_policy": "omit_question_from_request",
                "added_case_count": len(new_cases),
                "base_dataset_path": str(base_dataset_path) if base_dataset_path else None,
            },
        },
        "source_documents": list((base_payload or {}).get("source_documents") or []),
        "summary": {
            **_summary(combined_cases),
            "normal_case_count": len(existing_cases),
            "biomarker_observation_case_count": len(new_cases),
        },
        "cases": [case.to_dict() for case in combined_cases],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    jsonl = Path(output_jsonl_path) if output_jsonl_path else output.with_suffix(".jsonl")
    jsonl.write_text("\n".join(json.dumps(case.to_dict(), ensure_ascii=False) for case in combined_cases) + "\n", encoding="utf-8")
    return combined_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-ranges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--base-dataset", default=None)
    parser.add_argument("--dataset-id", default="benchmark_biomarker_v1")
    parser.add_argument("--dataset-version", default="benchmark_biomarker_v1")
    parser.add_argument("--dataset-size", type=int, default=25)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--api-base-url", default="http://localhost:11434")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-version", default="biomarker_dataset_generation_v1")
    args = parser.parse_args()

    cases = build_biomarker_dataset(
        marker_ranges_path=args.marker_ranges,
        output_path=args.output,
        output_jsonl_path=args.output_jsonl,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        dataset_size=args.dataset_size,
        base_dataset_path=args.base_dataset,
        model=args.model,
        api_base_url=args.api_base_url,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
        prompt_version=args.prompt_version,
    )
    print(f"Wrote {len(cases)} total cases")
    print(f"JSON: {args.output}")
    print(f"JSONL: {args.output_jsonl or str(Path(args.output).with_suffix('.jsonl'))}")


if __name__ == "__main__":
    main()
