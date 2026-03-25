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
    def __init__(self, payloads: dict[tuple[str, str], bytes]) -> None:
        self._payloads = payloads
        self.calls: list[tuple[str, str]] = []

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
        }
    )
    repo = ClinicalConfigRepository(settings=settings, client=fake_minio)

    first = repo.load_drug_dosing_catalog_payload()
    second = repo.load_drug_dosing_catalog_payload()

    assert first["default_agents"]["beta_blocker"] == "carvedilol"
    assert second["default_agents"]["beta_blocker"] == "carvedilol"
    assert fake_minio.calls == [("guidance-config", "clinical/drug_dosing_catalog.json")]
