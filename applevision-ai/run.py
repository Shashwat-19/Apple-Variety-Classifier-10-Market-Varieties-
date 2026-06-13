#!/usr/bin/env python3
"""
AppleVision AI – Development entry point.

Usage::

    python run.py                 # defaults to development mode
    FLASK_ENV=production python run.py
"""

from __future__ import annotations

import os

from app import create_app

config_name: str = os.getenv("FLASK_ENV", "development")
app = create_app(config_name)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    host = os.getenv("HOST", "127.0.0.1")

    app.run(
        host=host,
        port=port,
        debug=app.config.get("DEBUG", True),
    )
