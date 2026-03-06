from fastapi import FastAPI

from api.routes.health import router as health_router
from api.routes.guidance import router as guidance_router


def create_app() -> FastAPI:
    app = FastAPI(title="mobicare-llm API", version="0.1.0")
    app.include_router(health_router)
    app.include_router(guidance_router)
    return app


app = create_app()
