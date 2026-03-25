from __future__ import annotations

from shared.contracts.inference import InferenceRequest, RetrievedContext

class ExampleResponseBuilder:
    def build(self, request: InferenceRequest, retrieved_context: list[RetrievedContext]) -> str:
        question = request.question.strip() or "No question was supplied."
        variable_lines = [f"- {key}: {value}" for key, value in sorted(request.patient_variables.items())]
        context_lines = [f"- {item.title}: {item.snippet}" for item in retrieved_context[:3]]

        parts = [
            "[MOCK RESPONSE]",
            f"Question: {question}",
            "",
            "This is an example guidance response generated without running retrieval or the LLM.",
        ]

        if variable_lines:
            parts.extend(["", "Patient variables:", *variable_lines])

        if context_lines:
            parts.extend(["", "Provided context:", *context_lines])

        parts.extend(
            [
                "",
                "Example guidance:",
                "Use this output for UI testing, API integration checks, or local development on low-spec hardware.",
            ]
        )
        return "\n".join(parts)
