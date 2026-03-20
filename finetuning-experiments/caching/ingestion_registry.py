from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.datetime import utc_now_iso
from utils.json import read_json, write_json


@dataclass(slots=True)
class IngestionCacheEntry:
    fingerprint: str
    created_at: str
    updated_at: str
    run_id: str
    documents_version: str | None
    ingestion_summary: dict[str, Any]
    source_mapping_summary: dict[str, Any]
    assignments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "run_id": self.run_id,
            "documents_version": self.documents_version,
            "ingestion_summary": self.ingestion_summary,
            "source_mapping_summary": self.source_mapping_summary,
            "assignments": self.assignments,
        }


class IngestionRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "ingestion_registry.json"

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"entries": {}}
        raw = read_json(self.path)
        if not isinstance(raw, dict):
            return {"entries": {}}
        raw.setdefault("entries", {})
        return raw

    def _save(self, payload: dict[str, Any]) -> None:
        write_json(self.path, payload)

    def get(self, fingerprint: str) -> IngestionCacheEntry | None:
        payload = self._load()
        entry = (payload.get("entries") or {}).get(fingerprint)
        if not isinstance(entry, dict):
            return None
        return IngestionCacheEntry(
            fingerprint=fingerprint,
            created_at=entry.get("created_at") or utc_now_iso(),
            updated_at=entry.get("updated_at") or entry.get("created_at") or utc_now_iso(),
            run_id=entry.get("run_id") or "",
            documents_version=entry.get("documents_version"),
            ingestion_summary=entry.get("ingestion_summary") or {},
            source_mapping_summary=entry.get("source_mapping_summary") or {},
            assignments=entry.get("assignments") or [],
        )

    def put(
        self,
        *,
        fingerprint: str,
        run_id: str,
        documents_version: str | None,
        ingestion_summary: dict[str, Any],
        source_mapping_summary: dict[str, Any],
        assignments: list[dict[str, Any]],
    ) -> IngestionCacheEntry:
        payload = self._load()
        existing = (payload.get("entries") or {}).get(fingerprint) or {}
        now = utc_now_iso()
        entry = IngestionCacheEntry(
            fingerprint=fingerprint,
            created_at=existing.get("created_at") or now,
            updated_at=now,
            run_id=run_id,
            documents_version=documents_version,
            ingestion_summary=ingestion_summary,
            source_mapping_summary=source_mapping_summary,
            assignments=assignments,
        )
        payload.setdefault("entries", {})[fingerprint] = entry.to_dict()
        self._save(payload)
        return entry
