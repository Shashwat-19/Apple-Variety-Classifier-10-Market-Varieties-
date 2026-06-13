"""
Security middleware – response headers, request logging, and error handlers.

Registered once during app creation via ``register_security_middleware(app)``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from flask import Flask, Response, g, request

logger = logging.getLogger(__name__)


def register_security_middleware(app: Flask) -> None:
    """Attach ``before_request`` / ``after_request`` hooks and error handlers.

    Args:
        app: The Flask application instance to instrument.
    """

    # ── Request Timing ─────────────────────────────────────────────────
    @app.before_request
    def _start_timer() -> None:
        """Capture request start time for latency measurement."""
        g.start_time = time.perf_counter()

    # ── Request Logging ────────────────────────────────────────────────
    @app.after_request
    def _log_request(response: Response) -> Response:
        """Log every completed HTTP request with timing information."""
        elapsed_ms = 0.0
        if hasattr(g, "start_time"):
            elapsed_ms = round((time.perf_counter() - g.start_time) * 1000, 2)

        logger.info(
            "%s %s %s — %d (%.2f ms)",
            request.remote_addr,
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ── Security Headers ───────────────────────────────────────────────
    @app.after_request
    def _add_security_headers(response: Response) -> Response:
        """Inject standard security headers into every response."""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "SAMEORIGIN",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "camera=(), microphone=(), geolocation=(), payment=()"
            ),
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        }
        for key, value in headers.items():
            response.headers.setdefault(key, value)
        return response

    # ── Centralised Error Handlers ─────────────────────────────────────

    @app.errorhandler(400)
    def _bad_request(error: Any) -> tuple[dict[str, Any], int]:
        return {
            "success": False,
            "error": "Bad Request",
            "message": str(error.description) if hasattr(error, "description") else str(error),
        }, 400

    @app.errorhandler(404)
    def _not_found(error: Any) -> tuple[dict[str, Any], int]:
        return {
            "success": False,
            "error": "Not Found",
            "message": "The requested resource was not found.",
        }, 404

    @app.errorhandler(405)
    def _method_not_allowed(error: Any) -> tuple[dict[str, Any], int]:
        return {
            "success": False,
            "error": "Method Not Allowed",
            "message": "This HTTP method is not allowed for this endpoint.",
        }, 405

    @app.errorhandler(413)
    def _payload_too_large(error: Any) -> tuple[dict[str, Any], int]:
        return {
            "success": False,
            "error": "Payload Too Large",
            "message": "The uploaded file exceeds the maximum allowed size.",
        }, 413

    @app.errorhandler(429)
    def _rate_limit_exceeded(error: Any) -> tuple[dict[str, Any], int]:
        return {
            "success": False,
            "error": "Too Many Requests",
            "message": "Rate limit exceeded. Please slow down.",
        }, 429

    @app.errorhandler(500)
    def _internal_error(error: Any) -> tuple[dict[str, Any], int]:
        logger.exception("Internal server error: %s", error)
        return {
            "success": False,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
        }, 500
