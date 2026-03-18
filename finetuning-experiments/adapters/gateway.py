from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GatewayIngestionResult:
    job_id: str
    status: str
    record: dict[str, Any]


class GatewayClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout_seconds: int = 30) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def submit_ingestion_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/ingestion/jobs"
        logger.info("Submitting ingestion job to %s", url)
        response = requests.post(url, json=payload, timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.json()

    def get_ingestion_job(self, job_id: str) -> dict[str, Any]:
        url = f"{self._base_url}/ingestion/jobs/{job_id}"
        response = requests.get(url, timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.json()

    def delete_ingestion_collection(self) -> dict[str, Any]:
        url = f"{self._base_url}/ingestion/collection"
        logger.info("Deleting ingestion collection via %s", url)
        response = requests.delete(url, timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.json()

    def run_ingestion_and_wait(
        self,
        payload: dict[str, Any],
        *,
        poll_interval_seconds: float = 2.0,
        max_wait_seconds: float = 1800,
    ) -> GatewayIngestionResult:
        accepted = self.submit_ingestion_job(payload)
        job_id = accepted["job_id"]
        logger.info("Accepted ingestion job %s", job_id)
        start = time.monotonic()

        while True:
            record = self.get_ingestion_job(job_id)
            status = str(record.get("status", "unknown"))
            logger.info(
                "Ingestion job %s status=%s chunks=%s vectors=%s",
                job_id,
                status,
                ((record.get("result") or {}).get("chunks_created")),
                ((record.get("result") or {}).get("vectors_upserted")),
            )
            if status == "completed":
                return GatewayIngestionResult(job_id=job_id, status=status, record=record)
            if status == "failed":
                raise RuntimeError(f"Ingestion job {job_id} failed: {record.get('error')}")
            if time.monotonic() - start > max_wait_seconds:
                raise TimeoutError(f"Timed out waiting for ingestion job {job_id}")
            time.sleep(poll_interval_seconds)
