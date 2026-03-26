from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Response, status

from api.application.services.clinical_config_service import ClinicalConfigService
from api.dependencies import get_clinical_config_service
from api.errors import BadRequestError, ConflictError, NotFoundError, ServiceUnavailableError
from api.repositories.clinical_config_repository import (
    ClinicalConfigAlreadyExistsError,
    ClinicalConfigNotFoundError,
    ClinicalConfigOptimisticLockError,
    ClinicalConfigRepositoryError,
    ClinicalConfigVersionNotFoundError,
    InvalidClinicalConfigError,
    UnknownClinicalConfigError,
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
    try:
        return service.list_configs()
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message) from exc


@router.get("/{config_name}", response_model=ClinicalConfigReadResponse)
async def get_clinical_config(
    config_name: str,
    response: Response,
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigReadResponse:
    try:
        result = service.get_config(config_name)
        _apply_concurrency_headers(response, etag=result.config.etag, checksum_sha256=result.config.checksum_sha256)
        return result
    except (ClinicalConfigNotFoundError, UnknownClinicalConfigError) as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@router.get("/{config_name}/versions", response_model=ClinicalConfigVersionListResponse)
async def list_clinical_config_versions(
    config_name: str,
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigVersionListResponse:
    try:
        return service.list_versions(config_name)
    except (ClinicalConfigNotFoundError, ClinicalConfigVersionNotFoundError, UnknownClinicalConfigError) as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@router.post("/{config_name}", response_model=ClinicalConfigWriteResponse, status_code=status.HTTP_201_CREATED)
async def create_clinical_config(
    config_name: str,
    request: ClinicalConfigWriteRequest,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
    x_content_sha256: str | None = Header(default=None, alias="X-Content-SHA256"),
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigWriteResponse:
    try:
        result = service.create_config(
            config_name,
            request.payload,
            expected_etag=if_match,
            expected_checksum_sha256=x_content_sha256,
        )
        _apply_concurrency_headers(response, etag=result.config.etag, checksum_sha256=result.config.checksum_sha256)
        return result
    except InvalidClinicalConfigError as exc:
        raise BadRequestError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except (ClinicalConfigAlreadyExistsError, ClinicalConfigOptimisticLockError) as exc:
        raise ConflictError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except UnknownClinicalConfigError as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@router.put("/{config_name}", response_model=ClinicalConfigWriteResponse)
async def upsert_clinical_config(
    config_name: str,
    request: ClinicalConfigWriteRequest,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
    x_content_sha256: str | None = Header(default=None, alias="X-Content-SHA256"),
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigWriteResponse:
    try:
        result = service.upsert_config(
            config_name,
            request.payload,
            expected_etag=if_match,
            expected_checksum_sha256=x_content_sha256,
        )
        _apply_concurrency_headers(response, etag=result.config.etag, checksum_sha256=result.config.checksum_sha256)
        return result
    except InvalidClinicalConfigError as exc:
        raise BadRequestError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigOptimisticLockError as exc:
        raise ConflictError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except UnknownClinicalConfigError as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@router.post("/{config_name}/rollback", response_model=ClinicalConfigRollbackResponse)
async def rollback_clinical_config(
    config_name: str,
    request: ClinicalConfigRollbackRequest,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
    x_content_sha256: str | None = Header(default=None, alias="X-Content-SHA256"),
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigRollbackResponse:
    try:
        result = service.rollback_config(
            config_name,
            request.version_id,
            expected_etag=if_match,
            expected_checksum_sha256=x_content_sha256,
        )
        _apply_concurrency_headers(response, etag=result.config.etag, checksum_sha256=result.config.checksum_sha256)
        return result
    except (ClinicalConfigNotFoundError, ClinicalConfigVersionNotFoundError, UnknownClinicalConfigError) as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigOptimisticLockError as exc:
        raise ConflictError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@router.delete("/{config_name}", response_model=ClinicalConfigDeleteResponse)
async def delete_clinical_config(
    config_name: str,
    if_match: str | None = Header(default=None, alias="If-Match"),
    x_content_sha256: str | None = Header(default=None, alias="X-Content-SHA256"),
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigDeleteResponse:
    try:
        return service.delete_config(
            config_name,
            expected_etag=if_match,
            expected_checksum_sha256=x_content_sha256,
        )
    except (ClinicalConfigNotFoundError, UnknownClinicalConfigError) as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigOptimisticLockError as exc:
        raise ConflictError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc



def _apply_concurrency_headers(response: Response, *, etag: str | None, checksum_sha256: str | None) -> None:
    if etag:
        response.headers["ETag"] = etag
    if checksum_sha256:
        response.headers["X-Content-SHA256"] = checksum_sha256
