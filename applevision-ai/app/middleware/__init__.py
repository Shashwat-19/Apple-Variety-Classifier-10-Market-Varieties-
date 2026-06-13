"""Middleware package for AppleVision AI."""

from app.middleware.security import register_security_middleware

__all__ = ["register_security_middleware"]
