from __future__ import annotations

import math
import re
from typing import Any, Iterable

from shared.contracts.inference import RetrievedContext

from .models import DrugEvidence


def combined_text(contexts: Iterable[RetrievedContext]) -> str:
    return "\n".join(f"{item.title} {item.snippet}".lower() for item in contexts)


def find_agent_line(contexts: list[RetrievedContext], agent: str) -> str:
    normalized_agent = agent.lower().replace("_", " ")
    for item in contexts:
        for line in item.snippet.splitlines():
            lowered = line.lower()
            if normalized_agent in lowered:
                return line.strip()
    return "\n".join(item.snippet for item in contexts)


def parse_labeled_single_dose(line: str, label: str) -> tuple[str | None, float | None, str | None]:
    if not line:
        return None, None, None
    pattern = re.compile(rf"{re.escape(label)}\s+(\d+(?:\.\d+)?)\s*mg\s*([a-z.]+)", re.IGNORECASE)
    match = pattern.search(line)
    if not match:
        return None, None, None
    value = float(match.group(1))
    frequency = match.group(2).strip().lower()
    return format_numeric_dose(value, frequency), value, frequency


def parse_labeled_target_dose(line: str) -> tuple[str | None, float | None, str | None]:
    if not line:
        return None, None, None
    pattern = re.compile(r"target dose\s+(\d+(?:\.\d+)?)(?:\s*[–-]\s*(\d+(?:\.\d+)?))?\s*mg\s*([a-z.]+)", re.IGNORECASE)
    match = pattern.search(line)
    if not match:
        return None, None, None
    first = float(match.group(1))
    second = float(match.group(2)) if match.group(2) else first
    frequency = match.group(3).strip().lower()
    target = max(first, second)
    label = f"{trim_numeric(first)}-{trim_numeric(second)} mg {frequency}" if second != first else f"{trim_numeric(target)} mg {frequency}"
    return label, target, frequency


def parse_arni_main_start(text: str) -> tuple[str | None, float | None, str | None]:
    match = re.search(r"starting dose\s+(\d+)/(\d+)\s*mg\s*([a-z.]+)", text, re.IGNORECASE)
    if not match:
        return None, None, None
    return f"{match.group(1)}/{match.group(2)} mg {match.group(3).lower()}", float(match.group(1)), match.group(3).lower()


def parse_target_combo(text: str) -> tuple[str | None, float | None, str | None]:
    match = re.search(r"target dose\s+(\d+)/(\d+)\s*mg\s*([a-z.]+)", text, re.IGNORECASE)
    if not match:
        return None, None, None
    return f"{match.group(1)}/{match.group(2)} mg {match.group(3).lower()}", float(match.group(1)), match.group(3).lower()


def parse_reduced_combo(text: str) -> tuple[str | None, float | None, str | None]:
    match = re.search(r"(24/26\s*mg\s*[a-z.]+)\s+in selected patients", text, re.IGNORECASE)
    if match:
        dose_text = match.group(1).lower().replace("  ", " ")
        return dose_text, 24.0, dose_text.rsplit(" ", 1)[-1]
    match = re.search(r"reduced starting dose\s*\(?24/26\s*mg\s*([a-z.]+)\)?", text, re.IGNORECASE)
    if match:
        return f"24/26 mg {match.group(1).lower()}", 24.0, match.group(1).lower()
    return None, None, None


def parse_loop_start_range(line: str) -> tuple[str | None, float | None, float | None]:
    if not line:
        return None, None, None
    match = re.search(r"starting dose\s*(\d+(?:\.\d+)?)\s*[–-]?\s*(\d+(?:\.\d+)?)?\s*mg", line, re.IGNORECASE)
    if not match:
        return None, None, None
    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) else low
    label = f"{trim_numeric(low)}-{trim_numeric(high)} mg daily" if high != low else f"{trim_numeric(low)} mg daily"
    return label, low, high


def parse_loop_usual_range(line: str) -> str | None:
    if not line:
        return None
    match = re.search(r"usual dose\s*(\d+(?:\.\d+)?)\s*[–-]?\s*(\d+(?:\.\d+)?)?\s*mg", line, re.IGNORECASE)
    if not match:
        return None
    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) else low
    return f"{trim_numeric(low)}-{trim_numeric(high)} mg daily" if high != low else f"{trim_numeric(low)} mg daily"


def parse_combo_lead_value(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("mg", "")
    match = re.search(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return parse_numeric_value(value)


def combo_from_lead_value(value: float | None, evidence: DrugEvidence) -> str | None:
    if value is None:
        return None
    mapping = {
        evidence.reduced_start_value: evidence.reduced_start_dose,
        evidence.start_low_value: evidence.starting_dose,
        evidence.target_value: evidence.target_dose,
    }
    for key, dose in mapping.items():
        if key is not None and dose and abs(value - key) < 0.01:
            return dose
    return None


def first_raw(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
        for existing_key, value in payload.items():
            if existing_key.lower() == key.lower():
                return value
    return None


def first_float(payload: dict[str, Any], *keys: str) -> float | None:
    return parse_numeric_value(first_raw(payload, *keys))


def first_str(payload: dict[str, Any], *keys: str) -> str | None:
    value = first_raw(payload, *keys)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def first_bool(payload: dict[str, Any], *keys: str) -> bool | None:
    value = first_raw(payload, *keys)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "present", "positive"}:
        return True
    if text in {"0", "false", "no", "n", "absent", "negative"}:
        return False
    return None


def normalize_agent(value: str | None, default: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "metoprolol": "metoprolol_succinate",
        "metoprolol_cr_xl": "metoprolol_succinate",
        "metoprolol_succinate_(cr/xl)": "metoprolol_succinate",
        "sacubitril_valsartan": "sacubitril/valsartan",
        "sac_val": "sacubitril/valsartan",
        "spironolacton": "spironolactone",
        "mra": default,
        "bb": default,
        "loop": default,
    }
    return aliases.get(normalized, normalized)


def agent_or_default(value: str | None, default: str) -> tuple[str, bool]:
    if value:
        if value in {"sacubitril/valsartan", "sacubitril_valsartan"}:
            return "sacubitril/valsartan", False
        return value, False
    return default, True


def parse_numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return numeric
    text = str(value).strip().lower().replace("mg", "")
    if not text:
        return None
    if "/" in text:
        left = text.split("/", 1)[0].strip()
        try:
            return float(left)
        except ValueError:
            return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def format_numeric_dose(value: float | None, frequency: str | None) -> str | None:
    if value is None:
        return None
    freq = (frequency or "").strip()
    if not freq:
        return f"{trim_numeric(value)} mg"
    return f"{trim_numeric(value)} mg {freq}"


def trim_numeric(value: float) -> str:
    if abs(value - round(value)) < 0.001:
        return str(int(round(value)))
    return (f"{value:.3f}").rstrip("0").rstrip(".")


def gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def lt(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def between(value: float | None, lower: float, upper: float) -> bool:
    return value is not None and lower <= value <= upper
