
from __future__ import annotations

from typing import Any

from adapters.gateway import GatewayClient, GatewayIngestionResult


class IngestionClient:
    """Thin ingestion adapter that isolates experiment runners from gateway endpoint details."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout_seconds: int = 30) -> None:
        self._gateway = GatewayClient(base_url=base_url, timeout_seconds=timeout_seconds)

    def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._gateway.submit_ingestion_job(payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._gateway.get_ingestion_job(job_id)

    def delete_collection(self) -> dict[str, Any]:
        return self._gateway.delete_ingestion_collection()

    def run_and_wait(
        self,
        payload: dict[str, Any],
        *,
        poll_interval_seconds: float = 2.0,
        max_wait_seconds: float = 1800,
    ) -> GatewayIngestionResult:
        return self._gateway.run_ingestion_and_wait(
            payload,
            poll_interval_seconds=poll_interval_seconds,
            max_wait_seconds=max_wait_seconds,
        )
