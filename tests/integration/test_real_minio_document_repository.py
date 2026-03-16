from __future__ import annotations

import io

import pytest
from minio import Minio

from api.repositories.document_repository import DocumentRepository
from tests.support.docker import managed_container, reserve_tcp_port, wait_for_http_json, require_docker


@pytest.mark.integration
def test_real_minio_document_repository_round_trip():
    require_docker()
    port = reserve_tcp_port()
    with managed_container(
        image="quay.io/minio/minio:latest",
        ports={port: 9000},
        env={"MINIO_ROOT_USER": "minioadmin", "MINIO_ROOT_PASSWORD": "minioadmin"},
        command=["server", "/data"],
    ):
        wait_for_http_json(f"http://127.0.0.1:{port}/minio/health/ready")
        client = Minio(f"127.0.0.1:{port}", access_key="minioadmin", secret_key="minioadmin", secure=False)
        bucket = "guidance-documents"
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

        repository = DocumentRepository(client=client, documents_bucket=bucket, documents_prefix="guidelines")
        uploaded = repository.upload_document(filename="esc.pdf", content_stream=io.BytesIO(b"pdf-bytes"), size_bytes=9)
        documents, total = repository.list_documents()
        blob = repository.get_document("esc.pdf")
        deleted = repository.delete_document("esc.pdf")

    assert uploaded.object_name == "guidelines/esc.pdf"
    assert total == 1
    assert documents[0].object_name == "guidelines/esc.pdf"
    assert blob.content == b"pdf-bytes"
    assert deleted.object_name == "guidelines/esc.pdf"
