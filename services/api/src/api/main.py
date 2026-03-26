from fastapi import Depends, FastAPI

from api.auth.dependencies import get_current_user
from api.exception_handlers import register_exception_handlers
from api.routes.auth import router as auth_router
from api.routes.clinical_configs import router as clinical_configs_router
from api.routes.documents import router as documents_router
from api.routes.guidance import router as guidance_router
from api.routes.health import router as health_router
from api.routes.ingestion import router as ingestion_router
from shared.config import Settings, get_settings
from shared.observability import configure_logging
from shared.observability.middleware import MetricsMiddleware, RequestContextMiddleware


SERVICE_NAME = "Gateway API"


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(SERVICE_NAME)
    app = FastAPI(
        title="mobicare-llm API",
        version="0.1.0",
        docs_url=None if not resolved.expose_api_docs else "/docs",
        redoc_url=None if not resolved.expose_api_docs else "/redoc",
        openapi_url=None if not resolved.expose_api_docs else "/openapi.json",
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(MetricsMiddleware, service_name=SERVICE_NAME)
    register_exception_handlers(app)

    protected_dependencies = [Depends(get_current_user)] if resolved.require_public_auth else []

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(clinical_configs_router, dependencies=protected_dependencies)
    app.include_router(guidance_router, dependencies=protected_dependencies)
    app.include_router(ingestion_router, dependencies=protected_dependencies)
    app.include_router(documents_router, dependencies=protected_dependencies)
    return app


app = create_app()
