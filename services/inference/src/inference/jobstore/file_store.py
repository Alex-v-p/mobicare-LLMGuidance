from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from shared.contracts.inference import JobRecord, utc_now_iso


class FileJobStore:
    def __init__(self, root_dir: str | None = None) -> None:
        self._root = Path(root_dir or os.getenv("JOBS_DIR", "/data/jobs"))
        self._queued = self._root / "queued"
        self._running = self._root / "running"
        self._done = self._root / "done"
        self._failed = self._root / "failed"
        for directory in (self._queued, self._running, self._done, self._failed):
            directory.mkdir(parents=True, exist_ok=True)

    def _path(self, directory: Path, request_id: str) -> Path:
        return directory / f"{request_id}.json"

    def _write(self, path: Path, record: JobRecord) -> None:
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(record.model_dump(mode="json"), indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def enqueue(self, record: JobRecord) -> None:
        path = self._path(self._queued, record.request_id)
        if path.exists() or self.find(record.request_id) is not None:
            raise FileExistsError(f"Job {record.request_id} already exists")
        self._write(path, record)

    def find_path(self, request_id: str) -> Optional[Path]:
        for directory in (self._queued, self._running, self._done, self._failed):
            path = self._path(directory, request_id)
            if path.exists():
                return path
        return None

    def find(self, request_id: str) -> Optional[JobRecord]:
        path = self.find_path(request_id)
        if path is None:
            return None
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def claim_next(self) -> Optional[JobRecord]:
        for path in sorted(self._queued.glob("*.json")):
            claimed_path = self._path(self._running, path.stem)
            try:
                path.replace(claimed_path)
            except FileNotFoundError:
                continue
            record = JobRecord.model_validate_json(claimed_path.read_text(encoding="utf-8"))
            record.status = "running"
            record.started_at = record.started_at or utc_now_iso()
            record.updated_at = utc_now_iso()
            self._write(claimed_path, record)
            return record
        return None

    def update(self, record: JobRecord) -> None:
        current_path = self.find_path(record.request_id)
        if current_path is None:
            raise FileNotFoundError(f"Job {record.request_id} not found")

        target_dir = {
            "queued": self._queued,
            "running": self._running,
            "completed": self._done,
            "failed": self._failed,
        }[record.status]
        target_path = self._path(target_dir, record.request_id)
        record.updated_at = utc_now_iso()
        self._write(target_path, record)
        if current_path != target_path and current_path.exists():
            current_path.unlink()
