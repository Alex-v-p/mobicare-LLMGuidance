from __future__ import annotations

from fastapi import Request

from shared.ids import new_request_id

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_STATE_KEY = "request_id"


def get_or_create_request_id(request: Request) -> str:
    request_id = request.headers.get(REQUEST_ID_HEADER) or getattr(
        request.state,
        REQUEST_ID_STATE_KEY,
        None,
    )
    if not request_id:
        request_id = new_request_id()
        setattr(request.state, REQUEST_ID_STATE_KEY, request_id)
    return request_id
