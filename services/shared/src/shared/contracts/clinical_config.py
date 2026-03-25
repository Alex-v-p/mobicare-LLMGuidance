from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ClinicalConfigName = Literal["marker_ranges", "drug_dosing_catalog"]


class ClinicalConfigMetadata(BaseModel):
    config_name: ClinicalConfigName
    bucket: str
    object_name: str
    exists_in_minio: bool
    content_type: str = "application/json"
    size_bytes: int | None = None
    etag: str | None = None
    checksum_sha256: str | None = None
    last_modified: datetime | None = None


class ClinicalConfigVersionMetadata(BaseModel):
    config_name: ClinicalConfigName
    version_id: str
    bucket: str
    object_name: str
    reason: Literal["create", "update", "delete", "rollback"]
    source_etag: str | None = None
    source_checksum_sha256: str | None = None
    created_at: datetime
    size_bytes: int | None = None


class ClinicalConfigListResponse(BaseModel):
    configs: list[ClinicalConfigMetadata] = Field(default_factory=list)


class ClinicalConfigReadResponse(BaseModel):
    config: ClinicalConfigMetadata
    payload: dict[str, Any] = Field(default_factory=dict)
    source: Literal["minio"] = "minio"


class ClinicalConfigWriteRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


class ClinicalConfigWriteResponse(BaseModel):
    config: ClinicalConfigMetadata
    status: Literal["created", "updated"]
    archived_version: ClinicalConfigVersionMetadata | None = None


class ClinicalConfigDeleteResponse(BaseModel):
    config_name: ClinicalConfigName
    bucket: str
    object_name: str
    status: Literal["deleted"] = "deleted"
    archived_version: ClinicalConfigVersionMetadata | None = None


class ClinicalConfigVersionListResponse(BaseModel):
    config_name: ClinicalConfigName
    versions: list[ClinicalConfigVersionMetadata] = Field(default_factory=list)


class ClinicalConfigRollbackRequest(BaseModel):
    version_id: str = Field(min_length=1)


class ClinicalConfigRollbackResponse(BaseModel):
    config: ClinicalConfigMetadata
    status: Literal["rolled_back"] = "rolled_back"
    restored_from_version: ClinicalConfigVersionMetadata
    archived_version: ClinicalConfigVersionMetadata | None = None
