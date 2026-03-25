from __future__ import annotations

from fastapi import APIRouter, Depends, status

from api.application.services.clinical_config_service import ClinicalConfigService
from api.dependencies import get_clinical_config_service
from api.errors import BadRequestError, ConflictError, NotFoundError, ServiceUnavailableError
from api.repositories.clinical_config_repository import (
    ClinicalConfigAlreadyExistsError,
    ClinicalConfigNotFoundError,
    ClinicalConfigRepositoryError,
    InvalidClinicalConfigError,
    UnknownClinicalConfigError,
)
from shared.contracts.clinical_config import (
    ClinicalConfigDeleteResponse,
    ClinicalConfigListResponse,
    ClinicalConfigReadResponse,
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
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigReadResponse:
    try:
        return service.get_config(config_name)  # type: ignore[arg-type]
    except (ClinicalConfigNotFoundError, UnknownClinicalConfigError) as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@router.post("/{config_name}", response_model=ClinicalConfigWriteResponse, status_code=status.HTTP_201_CREATED)
async def create_clinical_config(
    config_name: str,
    request: ClinicalConfigWriteRequest,
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigWriteResponse:
    try:
        return service.create_config(config_name, request.payload)  # type: ignore[arg-type]
    except InvalidClinicalConfigError as exc:
        raise BadRequestError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigAlreadyExistsError as exc:
        raise ConflictError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except UnknownClinicalConfigError as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@router.put("/{config_name}", response_model=ClinicalConfigWriteResponse)
async def upsert_clinical_config(
    config_name: str,
    request: ClinicalConfigWriteRequest,
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigWriteResponse:
    try:
        return service.upsert_config(config_name, request.payload)  # type: ignore[arg-type]
    except InvalidClinicalConfigError as exc:
        raise BadRequestError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except UnknownClinicalConfigError as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc


@router.delete("/{config_name}", response_model=ClinicalConfigDeleteResponse)
async def delete_clinical_config(
    config_name: str,
    service: ClinicalConfigService = Depends(get_clinical_config_service),
) -> ClinicalConfigDeleteResponse:
    try:
        return service.delete_config(config_name)  # type: ignore[arg-type]
    except (ClinicalConfigNotFoundError, UnknownClinicalConfigError) as exc:
        raise NotFoundError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc
    except ClinicalConfigRepositoryError as exc:
        raise ServiceUnavailableError(code=exc.code, message=exc.message, details={"config_name": config_name}) from exc