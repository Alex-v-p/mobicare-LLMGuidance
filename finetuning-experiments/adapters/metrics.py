from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class MetricsClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout_seconds: int = 15) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def fetch_json(self, path: str) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self._base_url}/{path.lstrip('/')}"
        logger.info("Fetching metrics payload from %s", url)
        response = requests.get(url, timeout=self._timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {"value": payload}
        return payload
