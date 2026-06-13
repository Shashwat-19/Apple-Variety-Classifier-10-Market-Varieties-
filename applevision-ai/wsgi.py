"""
WSGI entry point for Gunicorn / production WSGI servers.

Usage::

    gunicorn wsgi:app -w 4 -b 0.0.0.0:8000

The ``app`` object exposed here is the fully-initialised Flask
application created by the factory.
"""

from __future__ import annotations

import os

from app import create_app

app = create_app(os.getenv("FLASK_ENV", "production"))
