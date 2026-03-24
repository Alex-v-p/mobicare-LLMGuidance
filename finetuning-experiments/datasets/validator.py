from __future__ import annotations

from typing import Any

from .schema import BenchmarkCase


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


_REQUIRED_CASE_FIELDS = {
    "id",
    "question",
    "reference_answer",
}


def _fail(errors: list[str]) -> None:
    if errors:
        message = "Benchmark dataset validation failed:\n- " + "\n- ".join(errors)
        raise ValueError(message)


def validate_benchmark_dataset(raw_dataset: dict[str, Any]) -> None:
    errors: list[str] = []

    if not isinstance(raw_dataset, dict):
        errors.append("Dataset root must be a JSON object.")
        _fail(errors)

    dataset_type = raw_dataset.get("dataset_type")
    if dataset_type and dataset_type != "benchmark_dataset":
        errors.append(f"dataset_type must be 'benchmark_dataset', got {dataset_type!r}.")

    cases = raw_dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("Dataset must contain a non-empty 'cases' list.")
        _fail(errors)

    dataset_size = raw_dataset.get("dataset_size")
    if dataset_size is not None and dataset_size != len(cases):
        errors.append(f"dataset_size={dataset_size} does not match actual case count {len(cases)}.")

    seen_ids: set[str] = set()
    dataset_version = raw_dataset.get("dataset_id") or raw_dataset.get("dataset_version")

    for index, case in enumerate(cases, start=1):
        prefix = f"cases[{index - 1}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        missing = sorted(field for field in _REQUIRED_CASE_FIELDS if not case.get(field))
        if missing:
            errors.append(f"{prefix} is missing required field(s): {', '.join(missing)}.")
            continue
        case_id = str(case.get("id", "")).strip()
        if case_id in seen_ids:
            errors.append(f"Duplicate case id found: {case_id!r}.")
        seen_ids.add(case_id)
        if case.get("expected_chunk_ids"):
            errors.append(f"{prefix} must not contain expected_chunk_ids; chunk ids are strategy-dependent.")
        if dataset_version and not case.get("dataset_version"):
            case["dataset_version"] = dataset_version
        try:
            benchmark_case = BenchmarkCase(**case)
        except TypeError as exc:
            errors.append(f"{prefix} has invalid fields: {exc}.")
            continue
        if benchmark_case.case_weight <= 0:
            errors.append(f"{prefix}.case_weight must be positive.")
        if benchmark_case.answerability == "unanswerable" and benchmark_case.gold_passage_id:
            errors.append(f"{prefix} is unanswerable but still has a gold_passage_id.")
        if benchmark_case.answerability == "answerable" and not benchmark_case.gold_passage_text and not _is_observation_only(case):
            errors.append(f"{prefix} is answerable but has no gold_passage_text.")

    _fail(errors)
