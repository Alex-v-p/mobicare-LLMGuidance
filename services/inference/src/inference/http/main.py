from fastapi import FastAPI

from inference.http.routes.health import router as health_router
from inference.http.routes.generate import router as generate_router


def create_app() -> FastAPI:
    app = FastAPI(title="mobicare-llm inference", version="0.1.0")
    app.include_router(health_router)
    app.include_router(generate_router)
    return app


app = create_app()
