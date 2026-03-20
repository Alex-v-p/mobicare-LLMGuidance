from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from utils.datetime import utc_now_iso
from utils.json import read_json, write_json

RunStatus = Literal["pending", "running", "completed", "failed", "skipped", "reused"]


@dataclass(slots=True)
class RunRegistryEntry:
    fingerprint: str
    status: RunStatus
    run_id: str
    created_at: str
    updated_at: str
    artifact_path: str | None = None
    ingestion_fingerprint: str | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "status": self.status,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifact_path": self.artifact_path,
            "ingestion_fingerprint": self.ingestion_fingerprint,
            "error": self.error,
            "metadata": self.metadata or {},
        }


class RunRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "run_registry.json"

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

    def get(self, fingerprint: str) -> RunRegistryEntry | None:
        payload = self._load()
        entry = (payload.get("entries") or {}).get(fingerprint)
        if not isinstance(entry, dict):
            return None
        return RunRegistryEntry(
            fingerprint=fingerprint,
            status=entry.get("status") or "pending",
            run_id=entry.get("run_id") or "",
            created_at=entry.get("created_at") or utc_now_iso(),
            updated_at=entry.get("updated_at") or entry.get("created_at") or utc_now_iso(),
            artifact_path=entry.get("artifact_path"),
            ingestion_fingerprint=entry.get("ingestion_fingerprint"),
            error=entry.get("error"),
            metadata=entry.get("metadata") or {},
        )

    def upsert(
        self,
        *,
        fingerprint: str,
        status: RunStatus,
        run_id: str,
        artifact_path: str | None = None,
        ingestion_fingerprint: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunRegistryEntry:
        payload = self._load()
        existing = (payload.get("entries") or {}).get(fingerprint) or {}
        now = utc_now_iso()
        entry = RunRegistryEntry(
            fingerprint=fingerprint,
            status=status,
            run_id=run_id,
            created_at=existing.get("created_at") or now,
            updated_at=now,
            artifact_path=artifact_path if artifact_path is not None else existing.get("artifact_path"),
            ingestion_fingerprint=ingestion_fingerprint if ingestion_fingerprint is not None else existing.get("ingestion_fingerprint"),
            error=error,
            metadata=metadata if metadata is not None else existing.get("metadata") or {},
        )
        payload.setdefault("entries", {})[fingerprint] = entry.to_dict()
        self._save(payload)
        return entry
