from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "if", "in", "is",
    "it", "of", "on", "or", "should", "that", "the", "their", "this", "to", "what", "when", "which", "with",
    "according", "passage", "patient", "patients"
}


def normalize_text(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slug_like(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", text.strip().lower()).strip("-")


def derive_key_terms(question: str, passage: str) -> list[str]:
    q_terms = [t for t in re.findall(r"[a-z0-9]+", question.lower()) if t not in STOPWORDS and len(t) > 2]
    p_norm = normalize_text(passage)
    result: list[str] = []
    for term in q_terms:
        if term in p_norm and term not in result:
            result.append(term)
    return result[:6]


def infer_reasoning_type(case: dict[str, Any]) -> str:
    if case.get("answerability") == "unanswerable":
        return "abstention"
    q = case.get("question", "").lower()
    if "dose" in q or "dosing" in q:
        return "dosage-lookup"
    if q.startswith("can ") or "contraind" in q:
        return "contraindication-check"
    if any(key in q for key in ["compared", "difference", "versus"]):
        return "comparison"
    if any(key in q for key in ["what is", "what are", "define"]):
        return "definition" if "what is" in q else "single-hop"
    if any(key in q for key in ["if a patient", "if patient", "what should be done"]):
        return "scenario-application"
    return "single-hop"


def infer_difficulty(case: dict[str, Any]) -> str:
    wc = int(case.get("passage_metadata", {}).get("word_count", 0) or 0)
    q = case.get("question", "")
    if case.get("answerability") == "unanswerable":
        return "medium" if wc < 120 else "hard"
    if wc < 70 and len(q.split()) < 14:
        return "easy"
    if wc < 120:
        return "medium"
    return "hard"


def suspicious_unanswerable(case: dict[str, Any]) -> bool:
    if case.get("answerability") != "unanswerable":
        return False
    q = normalize_text(case.get("question", ""))
    p = case.get("gold_passage_normalized") or normalize_text(case.get("gold_passage_text", ""))
    if "expected benefits" in q and "expected bene" in normalize_text(case.get("gold_passage_text", "")):
        return True
    q_terms = [t for t in re.findall(r"[a-z0-9]+", q) if t not in STOPWORDS and len(t) > 3]
    overlap = sum(1 for t in q_terms if t in p)
    if overlap >= max(3, len(q_terms) // 2) and any(word in q for word in ["what should", "what is", "what are", "can "]):
        return True
    return False


def refine_case(case: dict[str, Any], dataset_version: str) -> dict[str, Any]:
    case.setdefault("dataset_version", dataset_version)
    case.setdefault("case_weight", 1.0)
    case.setdefault("query_variants", [])
    case.setdefault("retrieval_hints", {})
    case.setdefault("hallucination_metadata", {})
    case.setdefault("generation_metadata", {})
    case.setdefault("passage_metadata", {})
    case.setdefault("required_facts", [])
    case.setdefault("forbidden_facts", [])
    case.setdefault("tags", [])
    case["reasoning_type"] = infer_reasoning_type(case)
    case["difficulty"] = infer_difficulty(case)
    if case.get("answerability") == "unanswerable":
        case.setdefault("expected_abstention_style", "brief_insufficient_context")
    else:
        case["expected_abstention_style"] = None

    expected_section = case.get("passage_metadata", {}).get("section_title") or ""
    key_terms = derive_key_terms(case.get("question", ""), case.get("gold_passage_text", ""))
    case["retrieval_hints"] = {
        "key_terms": key_terms,
        "expected_section": expected_section,
        "document_scope": "same_page_preferred",
    }

    if suspicious_unanswerable(case):
        if case["id"] == "case-009":
            case["answerability"] = "answerable"
            case["expected_behavior"] = "answer_from_context"
            case["expected_abstention_style"] = None
            case["unanswerable_reason"] = None
            case["reference_answer"] = (
                "Improved symptoms, prevention of worsening heart failure leading to hospital admission, and increased survival."
            )
            case["required_facts"] = [
                "improved symptoms",
                "prevention of worsening heart failure leading to hospital admission",
                "increased survival",
            ]
            case["review_status"] = "corrected"
            case["hallucination_metadata"]["is_hallucination_test"] = False
            case["hallucination_metadata"]["case_kind"] = "answerable"
            case["hallucination_metadata"]["expected_behavior"] = "answer_from_context"
            case["reasoning_type"] = "single-hop"
            case["difficulty"] = "easy"
        else:
            case["review_status"] = "flagged_suspicious_answerability"
    else:
        case.setdefault("review_status", "auto_generated_unreviewed")

    return case


def refine_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    dataset_version = dataset.get("dataset_id") or "benchmark_v1"
    dataset["schema_version"] = "1.3"
    dataset["dataset_id"] = dataset_version
    for case in dataset.get("cases", []):
        refine_case(case, dataset_version)

    # summary refresh
    q_counts: dict[str, int] = {}
    a_counts: dict[str, int] = {}
    d_counts: dict[str, int] = {}
    r_counts: dict[str, int] = {}
    review_counts: dict[str, int] = {}
    for case in dataset.get("cases", []):
        q_counts[case["question_type"]] = q_counts.get(case["question_type"], 0) + 1
        a_counts[case["answerability"]] = a_counts.get(case["answerability"], 0) + 1
        d_counts[case["difficulty"]] = d_counts.get(case["difficulty"], 0) + 1
        r_counts[case["reasoning_type"]] = r_counts.get(case["reasoning_type"], 0) + 1
        review_counts[case["review_status"]] = review_counts.get(case["review_status"], 0) + 1
    dataset["summary"] = {
        "question_type_counts": dict(sorted(q_counts.items())),
        "answerability_counts": dict(sorted(a_counts.items())),
        "difficulty_counts": dict(sorted(d_counts.items())),
        "reasoning_type_counts": dict(sorted(r_counts.items())),
        "review_status_counts": dict(sorted(review_counts.items())),
        "average_passage_word_count": round(
            sum(int(case.get("passage_metadata", {}).get("word_count", 0) or 0) for case in dataset.get("cases", []))
            / max(len(dataset.get("cases", [])), 1),
            2,
        ),
    }
    gm = dataset.setdefault("generation_metadata", {})
    gm.setdefault("prompt_version", "dataset_generation_v3")
    return dataset


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refine and validate an existing benchmark dataset JSON.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    refined = refine_dataset(data)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(refined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved refined dataset to {out_path}")


if __name__ == "__main__":
    main()
