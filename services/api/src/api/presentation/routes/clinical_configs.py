from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Response, status

from api.application.services.clinical_config_service import ClinicalConfigService
from api.dependencies import get_clinical_config_service
from api.presentation.routes.error_mapping import (
    clinical_config_create_errors,
    clinical_config_delete_errors,
    clinical_config_list_errors,
    clinical_config_read_errors,
    clinical_config_rollback_errors,
    clinical_config_update_errors,
    clinical_config_version_errors,
)
from shared.contracts.clinical_config import (
    ClinicalConfigDeleteResponse,
    ClinicalConfigListResponse,
    ClinicalConfigReadResponse,
    ClinicalConfigRollbackRequest,
    ClinicalConfigRollbackResponse,
    ClinicalConfigVersionListResponse,
    ClinicalConfigWriteRequest,
    ClinicalConfigWriteResponse,
)

router = APIRouter(prefix="/clinical-configs", tags=["clinical-configs"])


@router.get("", response_model=ClinicalConfigListResponse)
async def list_clinical_configs(
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigListResponse:
    with clinical_config_list_errors():
        return service.list_configs()


@router.get("/{config_name}", response_model=ClinicalConfigReadResponse)
async def get_clinical_config(
    config_name: str,
    response: Response,
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigReadResponse:
    with clinical_config_read_errors(config_name=config_name):
        result = service.get_config(config_name)
        _apply_concurrency_headers(response, etag=result.config.etag, checksum_sha256=result.config.checksum_sha256)
        return result


@router.get("/{config_name}/versions", response_model=ClinicalConfigVersionListResponse)
async def list_clinical_config_versions(
    config_name: str,
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigVersionListResponse:
    with clinical_config_version_errors(config_name=config_name):
        return service.list_versions(config_name)


@router.post("/{config_name}", response_model=ClinicalConfigWriteResponse, status_code=status.HTTP_201_CREATED)
async def create_clinical_config(
    config_name: str,
    request: ClinicalConfigWriteRequest,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
    x_content_sha256: str | None = Header(default=None, alias="X-Content-SHA256"),
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigWriteResponse:
    with clinical_config_create_errors(config_name=config_name):
        result = service.create_config(
            config_name,
            request.payload,
            expected_etag=if_match,
            expected_checksum_sha256=x_content_sha256,
        )
        _apply_concurrency_headers(response, etag=result.config.etag, checksum_sha256=result.config.checksum_sha256)
        return result


@router.put("/{config_name}", response_model=ClinicalConfigWriteResponse)
async def upsert_clinical_config(
    config_name: str,
    request: ClinicalConfigWriteRequest,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
    x_content_sha256: str | None = Header(default=None, alias="X-Content-SHA256"),
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigWriteResponse:
    with clinical_config_update_errors(config_name=config_name):
        result = service.upsert_config(
            config_name,
            request.payload,
            expected_etag=if_match,
            expected_checksum_sha256=x_content_sha256,
        )
        _apply_concurrency_headers(response, etag=result.config.etag, checksum_sha256=result.config.checksum_sha256)
        return result


@router.post("/{config_name}/rollback", response_model=ClinicalConfigRollbackResponse)
async def rollback_clinical_config(
    config_name: str,
    request: ClinicalConfigRollbackRequest,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
    x_content_sha256: str | None = Header(default=None, alias="X-Content-SHA256"),
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigRollbackResponse:
    with clinical_config_rollback_errors(config_name=config_name):
        result = service.rollback_config(
            config_name,
            request.version_id,
            expected_etag=if_match,
            expected_checksum_sha256=x_content_sha256,
        )
        _apply_concurrency_headers(response, etag=result.config.etag, checksum_sha256=result.config.checksum_sha256)
        return result


@router.delete("/{config_name}", response_model=ClinicalConfigDeleteResponse)
async def delete_clinical_config(
    config_name: str,
    if_match: str | None = Header(default=None, alias="If-Match"),
    x_content_sha256: str | None = Header(default=None, alias="X-Content-SHA256"),
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigDeleteResponse:
    with clinical_config_delete_errors(config_name=config_name):
        return service.delete_config(
            config_name,
            expected_etag=if_match,
            expected_checksum_sha256=x_content_sha256,
        )



def _apply_concurrency_headers(response: Response, *, etag: str | None, checksum_sha256: str | None) -> None:
    if etag:
        response.headers["ETag"] = etag
    if checksum_sha256:
        response.headers["X-Content-SHA256"] = checksum_sha256
