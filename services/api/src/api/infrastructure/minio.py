from __future__ import annotations

from minio import Minio

from shared.bootstrap import create_minio_client_from_settings
from shared.config import ApiSettings


def create_minio_client(settings: ApiSettings) -> Minio:
    return create_minio_client_from_settings(settings)
