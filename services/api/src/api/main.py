from fastapi import FastAPI

from api.exception_handlers import register_exception_handlers
from api.routes.documents import router as documents_router
from api.routes.guidance import router as guidance_router
from api.routes.health import router as health_router
from api.routes.ingestion import router as ingestion_router
from shared.observability import configure_logging
from shared.observability.middleware import MetricsMiddleware, RequestContextMiddleware


SERVICE_NAME = "api"


def create_app() -> FastAPI:
    configure_logging(SERVICE_NAME)
    app = FastAPI(title="mobicare-llm API", version="0.1.0")
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(MetricsMiddleware, service_name=SERVICE_NAME)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(guidance_router)
    app.include_router(ingestion_router)
    app.include_router(documents_router)
    return app


app = create_app()
