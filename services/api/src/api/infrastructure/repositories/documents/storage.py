from __future__ import annotations

from typing import BinaryIO

from minio import Minio
from minio.datatypes import Object
from minio.helpers import ObjectWriteResult
from urllib3.response import HTTPResponse

from api.infrastructure.repositories.documents.errors import DocumentNotFoundError, map_storage_error
from shared.bootstrap import ensure_minio_bucket
from api.infrastructure.repositories.documents.models import DocumentLocation


class DocumentStorage:
    def __init__(self, *, client: Minio, documents_bucket: str, list_prefix: str | None = None) -> None:
        self._client = client
        self._documents_bucket = documents_bucket
        self._list_prefix = list_prefix

    def ensure_bucket_exists(self) -> None:
        try:
            ensure_minio_bucket(self._client, self._documents_bucket)
        except Exception as exc:
            raise map_storage_error(exc, self._documents_bucket) from exc

    def list_objects(self) -> list[Object]:
        self.ensure_bucket_exists()
        objects: list[Object] = []
        try:
            for obj in self._client.list_objects(self._documents_bucket, prefix=self._list_prefix, recursive=True):
                if not obj.is_dir:
                    objects.append(obj)
        except Exception as exc:
            raise map_storage_error(exc, self._documents_bucket) from exc
        return objects

    def object_exists(self, location: DocumentLocation) -> bool:
        try:
            self.stat_object(location)
            return True
        except DocumentNotFoundError:
            return False

    def stat_object(self, location: DocumentLocation) -> object:
        self.ensure_bucket_exists()
        try:
            return self._client.stat_object(location.bucket, location.object_name)
        except Exception as exc:
            raise map_storage_error(exc, location.object_name) from exc

    def get_object_bytes(self, location: DocumentLocation) -> bytes:
        self.ensure_bucket_exists()
        response: HTTPResponse | None = None
        try:
            response = self._client.get_object(location.bucket, location.object_name)
            return response.read()
        except Exception as exc:
            raise map_storage_error(exc, location.object_name) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def put_object(self, *, location: DocumentLocation, content_stream: BinaryIO, size_bytes: int, content_type: str) -> ObjectWriteResult:
        self.ensure_bucket_exists()
        try:
            return self._client.put_object(location.bucket, location.object_name, data=content_stream, length=size_bytes, content_type=content_type)
        except Exception as exc:
            raise map_storage_error(exc, location.object_name) from exc

    def remove_object(self, location: DocumentLocation) -> None:
        self.ensure_bucket_exists()
        try:
            self._client.remove_object(location.bucket, location.object_name)
        except Exception as exc:
            raise map_storage_error(exc, location.object_name) from exc
