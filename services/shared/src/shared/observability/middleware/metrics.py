from __future__ import annotations

from time import perf_counter

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from shared.observability.metrics import get_metrics_registry


class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, service_name: str) -> None:
        super().__init__(app)
        self._service_name = service_name
        self._metrics = get_metrics_registry()

    async def dispatch(self, request: Request, call_next):
        start = perf_counter()
        response = await call_next(request)
        duration = perf_counter() - start
        labels = {
            "service": self._service_name,
            "method": request.method,
            "path": request.url.path,
            "status_code": str(response.status_code),
        }
        self._metrics.inc("app_requests_total", labels=labels)
        self._metrics.observe("app_request_duration_seconds", duration, labels=labels)
        return response
