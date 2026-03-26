from __future__ import annotations

from types import SimpleNamespace

from shared.config.settings import Settings

from inference.clinical.config_repository import (
    ClinicalConfigRepository,
    clear_clinical_config_cache,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        return None

    def release_conn(self) -> None:
        return None


class _FakeMinio:
    def __init__(self, payloads: dict[tuple[str, str], bytes] | None = None, *, existing_buckets: set[str] | None = None) -> None:
        self._payloads = dict(payloads or {})
        self._existing_buckets = set(existing_buckets or set())
        self.calls: list[tuple[str, str]] = []
        self.created_buckets: list[str] = []
        self.put_calls: list[tuple[str, str]] = []

    def bucket_exists(self, bucket: str) -> bool:
        return bucket in self._existing_buckets

    def make_bucket(self, bucket: str) -> None:
        self.created_buckets.append(bucket)
        self._existing_buckets.add(bucket)

    def list_objects(self, bucket: str, prefix=None, recursive=True):
        object_names = [name for current_bucket, name in self._payloads if current_bucket == bucket]
        return iter(SimpleNamespace(object_name=name, is_dir=False) for name in object_names)

    def put_object(self, bucket: str, object_name: str, data, length: int, content_type: str):
        payload = data.read()
        self._payloads[(bucket, object_name)] = payload
        self.put_calls.append((bucket, object_name))
        return None

    def get_object(self, bucket: str, object_name: str):
        self.calls.append((bucket, object_name))
        return _FakeResponse(self._payloads[(bucket, object_name)])



def test_clinical_config_repository_loads_packaged_defaults():
    clear_clinical_config_cache()
    repo = ClinicalConfigRepository(settings=Settings(clinical_config_source="packaged"))

    payload = repo.load_marker_ranges_payload()

    assert "potassium" in payload
    assert payload["potassium"]["label"]



def test_clinical_config_repository_loads_minio_override_and_uses_cache():
    clear_clinical_config_cache()
    settings = Settings(
        clinical_config_source="minio",
        clinical_config_bucket="guidance-config",
        clinical_config_prefix="clinical",
        clinical_drug_dosing_catalog_object_name="drug_dosing_catalog.json",
        clinical_config_cache_seconds=60,
    )
    fake_minio = _FakeMinio(
        {
            (
                "guidance-config",
                "clinical/drug_dosing_catalog.json",
            ): b'{"default_agents": {"beta_blocker": "carvedilol"}, "family_query_order": ["beta_blocker"], "families": {"beta_blocker": {"keywords": ["beta-blocker"], "query_template": "{agent} beta-blocker dose"}}, "family_priority": {"beta_blocker": 1}}'
        },
        existing_buckets={"guidance-config", "guidance-documents", "guidance-job-results"},
    )
    repo = ClinicalConfigRepository(settings=settings, client=fake_minio)

    first = repo.load_drug_dosing_catalog_payload()
    second = repo.load_drug_dosing_catalog_payload()

    assert first["default_agents"]["beta_blocker"] == "carvedilol"
    assert second["default_agents"]["beta_blocker"] == "carvedilol"
    assert fake_minio.calls == [("guidance-config", "clinical/drug_dosing_catalog.json")]



def test_clinical_config_repository_bootstraps_minio_defaults_when_bucket_is_empty():
    clear_clinical_config_cache()
    settings = Settings(
        clinical_config_source="minio",
        clinical_config_bucket="guidance-config",
        clinical_config_prefix="clinical",
        clinical_config_cache_seconds=0,
    )
    fake_minio = _FakeMinio(payloads={}, existing_buckets=set())
    repo = ClinicalConfigRepository(settings=settings, client=fake_minio)

    payload = repo.load_marker_ranges_payload()

    assert fake_minio.created_buckets == ["guidance-documents", "guidance-job-results", "guidance-config"]
    assert ("guidance-config", "clinical/marker_ranges.json") in fake_minio.put_calls
    assert payload["potassium"]["label"]
