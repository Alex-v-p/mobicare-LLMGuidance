from __future__ import annotations

from minio import Minio

from shared.config import Settings


def create_minio_client(settings: Settings) -> Minio:
    return Minio(
        settings.minio_client_endpoint,
        access_key=settings.minio_root_user,
        secret_key=settings.minio_root_password,
        secure=settings.minio_secure,
    )
