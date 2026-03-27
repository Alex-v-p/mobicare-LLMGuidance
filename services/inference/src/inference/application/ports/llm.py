from __future__ import annotations

from typing import Protocol

from shared.contracts.inference import OllamaGenerateResponse


class TextGenerationClient(Protocol):
    async def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> OllamaGenerateResponse: ...


class ModelSelectableTextGenerationClient(TextGenerationClient, Protocol):
    @property
    def model(self) -> str: ...

    def with_model(self, model: str | None) -> TextGenerationClient: ...


__all__ = ["ModelSelectableTextGenerationClient", "TextGenerationClient"]
