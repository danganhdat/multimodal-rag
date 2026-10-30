"""Request middleware: trace_id, timing, JSON error responses."""

import logging
import time

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .observe import new_trace_id, span, write_trace

logger = logging.getLogger(__name__)


class TraceMiddleware(BaseHTTPMiddleware):
    """Attaches trace_id to every request, times it, catches unhandled errors."""

    async def dispatch(self, request: Request, call_next):
        trace_id = new_trace_id()
        request.state.trace_id = trace_id
        t0 = time.perf_counter()

        try:
            with span(trace_id, "request", method=request.method, path=request.url.path):
                response: Response = await call_next(request)

            latency_ms = (time.perf_counter() - t0) * 1000
            response.headers["X-Trace-Id"] = trace_id
            response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"

            if request.url.path.startswith("/api"):
                logger.info(
                    "%s %s → %s %.0fms [%s]",
                    request.method, request.url.path,
                    response.status_code, latency_ms, trace_id,
                )
                write_trace(
                    trace_id,
                    endpoint=request.url.path,
                    status=str(response.status_code),
                    latency_ms=round(latency_ms, 2),
                )

            return response

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                "%s %s → 500 %.0fms [%s]",
                request.method, request.url.path, latency_ms, trace_id,
            )

            write_trace(
                trace_id,
                endpoint=request.url.path,
                status="500",
                latency_ms=round(latency_ms, 2),
                error_label=type(exc).__name__,
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": str(exc),
                        "trace_id": trace_id,
                    }
                },
                headers={"X-Trace-Id": trace_id},
            )
