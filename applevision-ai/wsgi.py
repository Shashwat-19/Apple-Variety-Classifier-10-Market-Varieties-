"""
WSGI entry point for Gunicorn / Vercel / production WSGI servers.

Usage::

    gunicorn wsgi:app -w 4 -b 0.0.0.0:8000

The ``app`` object exposed here is the fully-initialised Flask
application created by the factory.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so ``from app import …`` works
# regardless of how the process is started (Vercel, gunicorn, etc.).
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from app import create_app  # noqa: E402

app = create_app(os.getenv("FLASK_ENV", "production"))
