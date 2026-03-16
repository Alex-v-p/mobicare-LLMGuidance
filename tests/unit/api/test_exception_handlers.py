from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.errors import AppError
from api.exception_handlers import register_exception_handlers


def create_test_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/app-error")
    async def app_error():
        raise AppError(code="BAD_THING", message="boom", status_code=409, details={"a": 1})

    @app.get("/http-error")
    async def http_error():
        raise HTTPException(status_code=418, detail="teapot")

    @app.get("/unexpected")
    async def unexpected():
        raise RuntimeError("kaboom")

    @app.get("/validation/{value}")
    async def validation(value: int):
        return {"value": value}

    return app


def test_app_error_response_contains_request_id_and_error_payload():
    with TestClient(create_test_app(), raise_server_exceptions=False) as client:
        response = client.get("/app-error")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "BAD_THING"
    assert response.headers["X-Request-ID"]


def test_http_exception_response_uses_normalized_payload():
    with TestClient(create_test_app(), raise_server_exceptions=False) as client:
        response = client.get("/http-error")

    assert response.status_code == 418
    assert response.json()["error"]["message"] == "teapot"


def test_validation_error_response_contains_field_details():
    with TestClient(create_test_app(), raise_server_exceptions=False) as client:
        response = client.get("/validation/not-an-int")

    assert response.status_code == 422
    field = response.json()["error"]["details"]["fields"][0]
    assert field["field"] == "path.value"


def test_unexpected_exception_returns_internal_server_error_payload():
    with TestClient(create_test_app(), raise_server_exceptions=False) as client:
        response = client.get("/unexpected")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_SERVER_ERROR"
