"""Routes package for AppleVision AI."""

from app.routes.main_routes import main_bp
from app.routes.api_routes import api_bp

__all__ = ["main_bp", "api_bp"]
