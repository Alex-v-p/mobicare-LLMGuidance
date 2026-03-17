from __future__ import annotations

import json
import re
from typing import Any

import requests


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: int, options: dict[str, Any] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.options = options or {}

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": "json",
                "options": self.options,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        content = body.get("message", {}).get("content", "")
        return self._parse_json_response(content)

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        cleaned = self._strip_code_fences(content)
        candidates = [cleaned]
        extracted = self._extract_first_json_object(cleaned)
        if extracted and extracted != cleaned:
            candidates.append(extracted)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                repaired = self._trim_to_last_balanced_brace(candidate)
                if repaired and repaired != candidate:
                    try:
                        parsed = json.loads(repaired)
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
            if isinstance(parsed, dict):
                return parsed
        preview = cleaned[:500].replace("\n", "\\n")
        raise ValueError(f"Model did not return valid JSON. Preview: {preview}")

    def _strip_code_fences(self, value: str) -> str:
        value = value.strip()
        value = re.sub(r"^```json\s*", "", value)
        value = re.sub(r"^```\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
        return value.strip()

    def _extract_first_json_object(self, value: str) -> str | None:
        start = value.find("{")
        if start < 0:
            return None
        in_string = False
        escape = False
        depth = 0
        for index in range(start, len(value)):
            char = value[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return value[start : index + 1]
        return value[start:]

    def _trim_to_last_balanced_brace(self, value: str) -> str | None:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            return None
        return value[start : end + 1]
