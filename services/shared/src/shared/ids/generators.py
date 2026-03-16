from __future__ import annotations

from uuid import uuid4


def new_request_id() -> str:
    return f"req_{uuid4()}"


def new_job_id() -> str:
    return f"job_{uuid4()}"


def new_ingestion_job_id() -> str:
    return f"ingest_job_{uuid4()}"
