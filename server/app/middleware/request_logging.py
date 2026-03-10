from __future__ import annotations

import time

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "{method} {path} {status} {duration:.1f}ms client={client}",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration=duration_ms,
            client=request.client.host if request.client else "-",
        )
        return response
