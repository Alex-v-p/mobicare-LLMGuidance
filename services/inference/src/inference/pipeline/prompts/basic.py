from __future__ import annotations

from typing import Any, Dict, List

from shared.contracts.inference import RetrievedContext


def build_prompt(question: str, patient_variables: Dict[str, Any], retrieved_context: List[RetrievedContext]) -> str:
    patient_lines = [f"- {key}: {value}" for key, value in sorted(patient_variables.items()) if value is not None]
    if not patient_lines:
        patient_lines = ["- No patient variables were provided."]

    context_lines = [f"[{item.source_id}] {item.title}: {item.snippet}" for item in retrieved_context]
    if not context_lines:
        context_lines = ["No retrieval context was provided."]

    return f"""
You are a clinical guidance prototype for internal testing only.
You are NOT making a final medical decision.
Use the provided retrieved context and patient variables.
Be cautious, concise, and explicit about uncertainty.
Do not invent values that are missing.

User question:
{question}

Patient variables:
{chr(10).join(patient_lines)}

Retrieved context:
{chr(10).join(context_lines)}

Respond in plain text with 3 short parts:
1. A direct answer.
2. A short rationale tied to the given variables.
3. A short caution mentioning missing or risky data if relevant.
""".strip()
