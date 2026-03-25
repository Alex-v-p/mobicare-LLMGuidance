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
    last_modified: datetime | None = None


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


class ClinicalConfigDeleteResponse(BaseModel):
    config_name: ClinicalConfigName
    bucket: str
    object_name: str
    status: Literal["deleted"] = "deleted"
