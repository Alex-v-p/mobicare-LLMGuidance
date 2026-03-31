from __future__ import annotations

import io
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


@dataclass(slots=True)
class GatewayAPIResponse:
    status_code: int
    body: dict[str, Any]
    headers: dict[str, str]


class GatewayClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout_seconds: int = 30,
        *,
        auth_token: str | None = None,
        verify_ssl: bool | str = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._auth_token = (auth_token or "").strip() or None
        self._verify_ssl = verify_ssl

    def submit_ingestion_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/ingestion/jobs"
        logger.info("Submitting ingestion job to %s", url)
        response = requests.post(
            url,
            json=payload,
            headers=self._auth_headers() or None,
            timeout=self._timeout_seconds,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        return response.json()

    def get_ingestion_job(self, job_id: str) -> dict[str, Any]:
        url = f"{self._base_url}/ingestion/jobs/{job_id}"
        response = requests.get(
            url,
            headers=self._auth_headers() or None,
            timeout=self._timeout_seconds,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        return response.json()

    def delete_ingestion_collection(self) -> dict[str, Any]:
        url = f"{self._base_url}/ingestion/collection"
        logger.info("Deleting ingestion collection via %s", url)
        response = requests.delete(
            url,
            headers=self._auth_headers() or None,
            timeout=self._timeout_seconds,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        return response.json()

    def list_documents(self, *, offset: int = 0, limit: int = 100) -> GatewayAPIResponse:
        return self._request_json("GET", "/documents", params={"offset": offset, "limit": limit})

    def upload_document(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None = None,
        overwrite: bool = True,
    ) -> GatewayAPIResponse:
        url = f"{self._base_url}/documents"
        logger.info("Uploading document %s via %s", filename, url)
        response = requests.post(
            url,
            params={"overwrite": str(overwrite).lower()},
            files={"file": (filename, io.BytesIO(content), content_type or "application/octet-stream")},
            headers=self._auth_headers() or None,
            timeout=self._timeout_seconds,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        body = response.json() if response.content else {}
        return GatewayAPIResponse(
            status_code=response.status_code,
            body=body,
            headers={key: value for key, value in response.headers.items()},
        )

    def delete_document(self, object_name: str) -> GatewayAPIResponse:
        return self._request_json("DELETE", f"/documents/{object_name}")

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


    def test_authentication(self) -> GatewayAPIResponse:
        return self._request_json("GET", "/auth/example-protected")

    def list_clinical_configs(self) -> GatewayAPIResponse:
        return self._request_json("GET", "/clinical-configs")

    def get_clinical_config(self, config_name: str) -> GatewayAPIResponse:
        return self._request_json("GET", f"/clinical-configs/{config_name}")

    def list_clinical_config_versions(self, config_name: str) -> GatewayAPIResponse:
        return self._request_json("GET", f"/clinical-configs/{config_name}/versions")

    def create_clinical_config(
        self,
        config_name: str,
        payload: dict[str, Any],
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> GatewayAPIResponse:
        return self._request_json(
            "POST",
            f"/clinical-configs/{config_name}",
            json_body={"payload": payload},
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )

    def update_clinical_config(
        self,
        config_name: str,
        payload: dict[str, Any],
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> GatewayAPIResponse:
        return self._request_json(
            "PUT",
            f"/clinical-configs/{config_name}",
            json_body={"payload": payload},
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )

    def delete_clinical_config(
        self,
        config_name: str,
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> GatewayAPIResponse:
        return self._request_json(
            "DELETE",
            f"/clinical-configs/{config_name}",
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )

    def rollback_clinical_config(
        self,
        config_name: str,
        version_id: str,
        *,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> GatewayAPIResponse:
        return self._request_json(
            "POST",
            f"/clinical-configs/{config_name}/rollback",
            json_body={"version_id": version_id},
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        expected_etag: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> GatewayAPIResponse:
        headers = self._build_concurrency_headers(
            expected_etag=expected_etag,
            expected_checksum_sha256=expected_checksum_sha256,
        )
        headers.update(self._auth_headers())
        url = f"{self._base_url}{path}"
        logger.info("%s %s", method, url)
        response = requests.request(
            method,
            url,
            json=json_body,
            params=params,
            headers=headers or None,
            timeout=self._timeout_seconds,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        body = response.json() if response.content else {}
        return GatewayAPIResponse(
            status_code=response.status_code,
            body=body,
            headers={key: value for key, value in response.headers.items()},
        )

    def _auth_headers(self) -> dict[str, str]:
        if not self._auth_token:
            return {}
        return {"Authorization": f"Bearer {self._auth_token}"}

    @staticmethod
    def _build_concurrency_headers(
        *,
        expected_etag: str | None,
        expected_checksum_sha256: str | None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        if expected_etag:
            headers["If-Match"] = expected_etag
        if expected_checksum_sha256:
            headers["X-Content-SHA256"] = expected_checksum_sha256
        return headers
