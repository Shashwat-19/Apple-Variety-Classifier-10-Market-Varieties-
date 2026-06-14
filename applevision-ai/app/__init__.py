"""
AppleVision AI – Flask Application Factory.

``create_app()`` wires together configuration, extensions, blueprints,
middleware, and the ML model so the application is ready to serve
requests.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

# Load .env at the earliest opportunity (before config reads os.getenv)
load_dotenv()


def create_app(config_name: str | None = None) -> Flask:
    """Construct and configure the Flask application.

    Args:
        config_name: One of ``"development"``, ``"production"``, or
            ``"testing"``.  Falls back to the ``FLASK_ENV`` env var,
            then ``"development"``.

    Returns:
        A fully-initialised Flask application instance.
    """
    # ── Resolve environment ────────────────────────────────────────────
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development")

    # ── Create Flask instance ──────────────────────────────────────────
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder="templates",
        static_folder="static",
    )

    # Ensure the instance directory exists (for SQLite DBs, etc.)
    _ensure_instance_dir(app)

    # ── Load configuration ─────────────────────────────────────────────
    from app.config import config_by_name  # noqa: WPS433 – local import

    config_class = config_by_name.get(config_name)
    if config_class is None:
        raise ValueError(
            f"Unknown config_name={config_name!r}.  "
            f"Expected one of {list(config_by_name.keys())}."
        )
    app.config.from_object(config_class)

    # ── Ensure database directory exists ───────────────────────────────
    _ensure_db_directory(app)

    # ── Max upload size (16 MB) ────────────────────────────────────────
    app.config.setdefault("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)

    # ── Logging ────────────────────────────────────────────────────────
    _configure_logging(app)

    # ── Extensions ─────────────────────────────────────────────────────
    _init_extensions(app)

    # ── Blueprints ─────────────────────────────────────────────────────
    _register_blueprints(app)

    # ── Middleware ──────────────────────────────────────────────────────
    _register_middleware(app)

    # ── Database tables ────────────────────────────────────────────────
    with app.app_context():
        from app.models import PredictionHistory  # noqa: F401 – import for side-effect

        from app.extensions import db

        db.create_all()

    # ── ML Model ───────────────────────────────────────────────────────
    _load_ml_model(app)

    app.logger.info(
        "AppleVision AI started  [env=%s, debug=%s]",
        config_name,
        app.debug,
    )
    return app


# ── Private helpers ────────────────────────────────────────────────────────


def _ensure_instance_dir(app: Flask) -> None:
    """Create Flask's ``instance/`` folder if it doesn't exist."""
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass


def _ensure_db_directory(app: Flask) -> None:
    """Create the parent directory for the SQLite database file.

    The SQLite URI in config may point to a directory that doesn't
    exist yet (e.g. ``applevision-ai/instance/``).  This helper
    extracts the path from the URI and ensures the directory exists
    before SQLAlchemy tries to connect.
    """
    db_uri: str = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if db_uri.startswith("sqlite:///"):
        db_file_path = Path(db_uri.replace("sqlite:///", "", 1))
        try:
            os.makedirs(db_file_path.parent, exist_ok=True)
        except OSError:
            pass


def _configure_logging(app: Flask) -> None:
    """Set up basic logging with the configured level."""
    log_level = app.config.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _init_extensions(app: Flask) -> None:
    """Bind lazy extension singletons to the app."""
    from app.extensions import cors, db, limiter  # noqa: WPS433

    db.init_app(app)

    limiter.init_app(app)

    # Parse CORS origins from config (comma-separated string or "*")
    origins = app.config.get("CORS_ORIGINS", "*")
    origin_list = [o.strip() for o in origins.split(",")] if origins != "*" else "*"
    cors.init_app(
        app,
        resources={r"/api/*": {"origins": origin_list}},
        supports_credentials=True,
    )


def _register_blueprints(app: Flask) -> None:
    """Import and register all route blueprints."""
    from app.routes import api_bp, main_bp  # noqa: WPS433

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)


def _register_middleware(app: Flask) -> None:
    """Register security middleware hooks."""
    from app.middleware import register_security_middleware  # noqa: WPS433

    register_security_middleware(app)


def _load_ml_model(app: Flask) -> None:
    """Initialise and load the ML model singleton."""
    from app.services.ml_service import MLService  # noqa: WPS433

    model_path = app.config.get("MODEL_PATH", "")
    labels_path = app.config.get("LABELS_PATH", "")

    ml_service = MLService.get_instance(
        model_path=model_path,
        labels_path=labels_path,
    )

    with app.app_context():
        ml_service.load()

    if ml_service.is_ready:
        app.logger.info(
            "ML model ready – %d classes loaded.", ml_service.num_classes
        )
    else:
        app.logger.warning(
            "ML model NOT ready.  Prediction endpoints will return 503."
        )
