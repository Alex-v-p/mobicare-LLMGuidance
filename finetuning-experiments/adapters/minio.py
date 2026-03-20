
from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class MinioClient:
    """Best-effort adapter for environment capture; avoids hard dependency on a working MinIO setup."""

    def __init__(self, base_url: str, timeout_seconds: int = 10) -> None:
        self._base_url = base_url.rstrip('/')
        self._timeout_seconds = timeout_seconds

    def probe(self, bucket: str | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            'url': self._base_url,
            'bucket': bucket,
            'reachable': False,
        }
        try:
            response = requests.get(f"{self._base_url}/minio/health/live", timeout=self._timeout_seconds)
            info['reachable'] = response.ok
            info['health_status_code'] = response.status_code
        except Exception as exc:  # noqa: BLE001
            info['error'] = str(exc)
            return info
        return info
