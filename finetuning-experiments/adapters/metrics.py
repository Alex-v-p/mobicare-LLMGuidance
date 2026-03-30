from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class MetricsClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout_seconds: int = 15,
        *,
        auth_token: str | None = None,
        verify_ssl: bool | str = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._auth_token = (auth_token or "").strip() or None
        self._verify_ssl = verify_ssl

    def fetch_json(self, path: str) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self._base_url}/{path.lstrip('/')}"
        logger.info("Fetching metrics payload from %s", url)
        response = requests.get(
            url,
            headers=self._auth_headers() or None,
            timeout=self._timeout_seconds,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {"value": payload}
        return payload

    def _auth_headers(self) -> dict[str, str]:
        if not self._auth_token:
            return {}
        return {"Authorization": f"Bearer {self._auth_token}"}
