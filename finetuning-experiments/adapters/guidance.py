from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from adapters.guidance_payloads import extract_retrieved_context

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GatewayGuidanceResult:
    job_id: str
    status: str
    record: dict[str, Any]


class GuidanceClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout_seconds: int = 60, bearer_token: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._bearer_token = bearer_token.strip() if isinstance(bearer_token, str) and bearer_token.strip() else None

    def submit_guidance_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/guidance/jobs"
        logger.info("Submitting guidance job to %s", url)
        response = requests.post(url, json=payload, headers=self._build_headers(), timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.json()

    def get_guidance_job(self, job_id: str) -> dict[str, Any]:
        url = f"{self._base_url}/guidance/jobs/{job_id}"
        response = requests.get(url, headers=self._build_headers(), timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.json()

    def run_guidance_and_wait(
        self,
        payload: dict[str, Any],
        *,
        poll_interval_seconds: float = 1.0,
        max_wait_seconds: float = 1800,
    ) -> GatewayGuidanceResult:
        accepted = self.submit_guidance_job(payload)
        job_id = accepted["job_id"]
        logger.info("Accepted guidance job %s", job_id)
        start = time.monotonic()

        while True:
            record = self.get_guidance_job(job_id)
            status = str(record.get("status", "unknown"))
            rag = extract_retrieved_context(record)
            logger.info(
                "Guidance job %s status=%s rag=%s warnings=%s",
                job_id,
                status,
                len(rag),
                len(record.get("warnings") or []),
            )
            if status == "completed":
                return GatewayGuidanceResult(job_id=job_id, status=status, record=record)
            if status == "failed":
                return GatewayGuidanceResult(job_id=job_id, status=status, record=record)
            if time.monotonic() - start > max_wait_seconds:
                raise TimeoutError(f"Timed out waiting for guidance job {job_id}")
            time.sleep(poll_interval_seconds)

    def _build_headers(self) -> dict[str, str]:
        if not self._bearer_token:
            return {}
        return {"Authorization": f"Bearer {self._bearer_token}"}
