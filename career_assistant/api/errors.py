"""Typed-error → HTTP-status mapping + request logging for the API (Phase 14).

Domain code raises *typed* errors (the ingest family, ``ValueError`` for bad/unknown input,
``RuntimeError`` for an unmet runtime dependency like the PDF browser). These handlers turn each
into a clean JSON error with the right status, so a bad upload or a missing id never surfaces as a
500. Routes still raise ``HTTPException(404)`` directly for not-found, which passes through
unchanged.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from career_assistant.ingest.errors import (
    EmptyResumeError,
    ScannedPDFError,
    UnsupportedResumeError,
)

logger = logging.getLogger("career_assistant.api")


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def register_error_handlers(app: FastAPI) -> None:
    """Install the typed-error handlers (most specific first via exception MRO lookup)."""

    @app.exception_handler(EmptyResumeError)
    async def _empty(_request: Request, exc: EmptyResumeError) -> JSONResponse:
        return _error(422, f"Resume is empty or too short: {exc}")

    @app.exception_handler(ScannedPDFError)
    async def _scanned(_request: Request, exc: ScannedPDFError) -> JSONResponse:
        return _error(422, f"Resume appears to be a scanned image (no extractable text): {exc}")

    @app.exception_handler(UnsupportedResumeError)
    async def _unsupported(_request: Request, exc: UnsupportedResumeError) -> JSONResponse:
        return _error(415, str(exc))

    @app.exception_handler(ValueError)
    async def _bad_value(_request: Request, exc: ValueError) -> JSONResponse:
        return _error(400, str(exc))

    @app.exception_handler(RuntimeError)
    async def _runtime(_request: Request, exc: RuntimeError) -> JSONResponse:
        # e.g. PDF export when Chromium isn't installed — the service can't fulfil it right now.
        return _error(503, str(exc))


def register_request_logging(app: FastAPI) -> None:
    """Log method, path, status, and duration for every request."""

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
