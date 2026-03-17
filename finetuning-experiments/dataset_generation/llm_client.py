from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import requests


class BaseLlmClient(ABC):
    @abstractmethod
    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        return json.loads(cleaned)


class OpenAICompatibleClient(BaseLlmClient):
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.2) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=240)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return self._parse_json_content(content)


class OllamaClient(BaseLlmClient):
    def __init__(self, base_url: str, model: str, temperature: float = 0.2) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
            },
        }
        response = requests.post(url, headers=headers, json=payload, timeout=240)
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"]
        return self._parse_json_content(content)


class LlmClientFactory:
    @staticmethod
    def create(
        provider: str,
        base_url: str,
        model: str,
        temperature: float,
        api_key: str | None = None,
    ) -> BaseLlmClient:
        normalized = provider.strip().lower()
        if normalized == "ollama":
            return OllamaClient(base_url=base_url, model=model, temperature=temperature)
        if normalized in {"openai", "openai_compatible", "openai-compatible"}:
            return OpenAICompatibleClient(
                base_url=base_url,
                api_key=api_key or "",
                model=model,
                temperature=temperature,
            )
        raise ValueError(
            "Unsupported generation provider. Expected one of: ollama, openai_compatible. "
            f"Got: {provider!r}"
        )
