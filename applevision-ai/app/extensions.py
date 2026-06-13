"""
Flask extension instances.

Extensions are instantiated here (without binding to an app) so that they
can be imported throughout the application and initialised lazily inside
the app factory via ``init_app()``.
"""

from __future__ import annotations

from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy

# ---------------------------------------------------------------------------
# SQLAlchemy ORM
# ---------------------------------------------------------------------------
db = SQLAlchemy()

# ---------------------------------------------------------------------------
# Rate Limiter – keyed by remote IP address
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per hour"],
    storage_uri="memory://",
)

# ---------------------------------------------------------------------------
# Cross-Origin Resource Sharing
# ---------------------------------------------------------------------------
cors = CORS()
