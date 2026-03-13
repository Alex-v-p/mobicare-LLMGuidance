from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from shared.observability.request_context import (
    REQUEST_ID_HEADER,
    REQUEST_ID_STATE_KEY,
    new_request_id,
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        setattr(request.state, REQUEST_ID_STATE_KEY, request_id)
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
