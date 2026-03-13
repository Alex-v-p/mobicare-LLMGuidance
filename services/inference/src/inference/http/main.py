from fastapi import FastAPI

from inference.http.exceptions import register_exception_handlers
from inference.http.routes.guidance import router as guidance_router
from inference.http.routes.health import router as health_router
from inference.http.routes.ingestion import router as ingestion_router
from shared.observability.middleware import RequestContextMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="mobicare-llm inference", version="0.1.0")
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(guidance_router)
    app.include_router(ingestion_router)
    return app


app = create_app()
