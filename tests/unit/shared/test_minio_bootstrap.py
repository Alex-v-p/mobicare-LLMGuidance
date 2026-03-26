from __future__ import annotations

from types import SimpleNamespace

from minio.error import S3Error

from shared.bootstrap.minio import bootstrap_minio_resources, ensure_minio_bucket
from shared.config import Settings


class FakeMinio:
    def __init__(self, *, existing_buckets: set[str] | None = None, clinical_bucket_objects: list[str] | None = None) -> None:
        self.existing_buckets = set(existing_buckets or set())
        self.clinical_bucket_objects = list(clinical_bucket_objects or [])
        self.created_buckets: list[str] = []
        self.put_calls: list[tuple[str, str, str]] = []

    def bucket_exists(self, bucket: str) -> bool:
        return bucket in self.existing_buckets

    def make_bucket(self, bucket: str) -> None:
        self.created_buckets.append(bucket)
        self.existing_buckets.add(bucket)

    def list_objects(self, bucket: str, prefix=None, recursive=True):
        if bucket != "guidance-config":
            return iter([])
        return iter(SimpleNamespace(object_name=name, is_dir=False) for name in self.clinical_bucket_objects)

    def put_object(self, bucket: str, object_name: str, data, length: int, content_type: str):
        self.put_calls.append((bucket, object_name, content_type))
        if bucket == "guidance-config":
            self.clinical_bucket_objects.append(object_name)
        return None


class RacingFakeMinio(FakeMinio):
    def make_bucket(self, bucket: str) -> None:
        raise S3Error(
            code="BucketAlreadyOwnedByYou",
            message="already exists",
            resource=f"/{bucket}",
            request_id="req-1",
            host_id="host-1",
            response=None,
        )



def test_ensure_minio_bucket_tolerates_parallel_creation() -> None:
    client = RacingFakeMinio()

    changed = ensure_minio_bucket(client, "guidance-documents")

    assert changed is False



def test_bootstrap_minio_resources_creates_buckets_and_seeds_defaults_for_empty_clinical_bucket() -> None:
    settings = Settings()
    client = FakeMinio(existing_buckets={settings.minio_results_bucket})

    report = bootstrap_minio_resources(settings=settings, client=client)

    assert report.buckets_created == [settings.minio_documents_bucket, settings.clinical_config_bucket]
    assert sorted(report.clinical_configs_seeded) == ["drug_dosing_catalog", "marker_ranges"]
    assert {object_name for bucket, object_name, _ in client.put_calls if bucket == settings.clinical_config_bucket} == {
        "clinical/drug_dosing_catalog.json",
        "clinical/marker_ranges.json",
    }



def test_bootstrap_minio_resources_does_not_reseed_non_empty_clinical_bucket() -> None:
    settings = Settings()
    client = FakeMinio(
        existing_buckets={settings.minio_documents_bucket, settings.minio_results_bucket, settings.clinical_config_bucket},
        clinical_bucket_objects=["clinical/_versions/marker_ranges/20260101.json"],
    )

    report = bootstrap_minio_resources(settings=settings, client=client)

    assert report.buckets_created == []
    assert report.clinical_configs_seeded == []
    assert client.put_calls == []
