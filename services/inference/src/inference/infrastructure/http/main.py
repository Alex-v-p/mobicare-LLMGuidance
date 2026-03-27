from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from inference.infrastructure.http.dependencies import get_document_store, get_guidance_job_result_store, get_ingestion_job_result_store, get_retrieval_state_controller
from inference.infrastructure.http.exceptions import register_exception_handlers
from inference.infrastructure.http.routes.guidance import router as guidance_router
from inference.infrastructure.http.routes.health import router as health_router
from inference.infrastructure.http.routes.ingestion import router as ingestion_router
from inference.infrastructure.http.security import require_internal_service_request
from shared.bootstrap.minio import bootstrap_minio_resources_on_startup
from shared.config import InferenceSettings, get_inference_settings
from shared.observability import configure_logging
from shared.observability.middleware import MetricsMiddleware, RequestContextMiddleware


SERVICE_NAME = "inference"


def create_app(settings: InferenceSettings | None = None) -> FastAPI:
    resolved = settings or get_inference_settings()
    configure_logging(SERVICE_NAME)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        bootstrap_minio_resources_on_startup(
            settings=resolved,
            client=get_document_store().client,
            service=SERVICE_NAME,
        )
        get_guidance_job_result_store().ensure_bucket()
        get_ingestion_job_result_store().ensure_bucket()
        await get_retrieval_state_controller().refresh_from_vector_store()
        try:
            yield
        finally:
            await get_retrieval_state_controller().close()

    app = FastAPI(
        title="mobicare-llm inference",
        version="0.1.0",
        docs_url=None if not resolved.expose_api_docs else "/docs",
        redoc_url=None if not resolved.expose_api_docs else "/redoc",
        openapi_url=None if not resolved.expose_api_docs else "/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(MetricsMiddleware, service_name=SERVICE_NAME)
    register_exception_handlers(app)

    protected_dependencies = [Depends(require_internal_service_request)] if resolved.enable_internal_service_auth else []

    app.include_router(health_router)
    app.include_router(guidance_router, dependencies=protected_dependencies)
    app.include_router(ingestion_router, dependencies=protected_dependencies)
    return app


app = create_app()
