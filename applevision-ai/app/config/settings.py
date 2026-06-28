"""
Application configuration settings.

Provides environment-specific configuration classes for development,
production, and testing environments. All sensitive values are loaded
from environment variables with safe defaults for local development.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve key directories once at import time
# ---------------------------------------------------------------------------
# applevision-ai/  (this project's root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Repo root (one level above applevision-ai/)
_REPO_ROOT = _PROJECT_ROOT.parent

# Default paths to ML artefacts.
# Priority: ONNX (~30 MB runtime) > TFLite > Keras (fallback for local dev).
_ONNX_MODEL_PATH = _PROJECT_ROOT / "ml" / "model" / "apple_classifier.onnx"
_TFLITE_MODEL_PATH = _PROJECT_ROOT / "ml" / "model" / "apple_classifier.tflite"
_KERAS_MODEL_PATH = _REPO_ROOT / "apple-variety-streamlit" / "model" / "apple_classifier_final.keras"


def _resolve_model_path() -> str:
    """Pick the first model that exists on disk."""
    for p in (_ONNX_MODEL_PATH, _TFLITE_MODEL_PATH, _KERAS_MODEL_PATH):
        if p.exists():
            return str(p)
    return str(_ONNX_MODEL_PATH)  # will warn at load time


_DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", _resolve_model_path())
_DEFAULT_LABELS_PATH = str(
    _PROJECT_ROOT / "ml" / "labels.json"
    if (_PROJECT_ROOT / "ml" / "labels.json").exists()
    else _REPO_ROOT / "apple-variety-streamlit" / "labels.json"
)


class BaseConfig:
    """Shared configuration applicable to all environments."""

    # ── Flask Core ─────────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
    JSON_SORT_KEYS: bool = False

    # ── Database ───────────────────────────────────────────────────────
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    # ── ML Model ───────────────────────────────────────────────────────
    MODEL_PATH: str = os.getenv("MODEL_PATH", _DEFAULT_MODEL_PATH)
    LABELS_PATH: str = os.getenv("LABELS_PATH", _DEFAULT_LABELS_PATH)
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    IMAGE_SIZE: tuple[int, int] = (224, 224)
    TOP_K_PREDICTIONS: int = int(os.getenv("TOP_K_PREDICTIONS", "5"))

    # ── Rate Limiting ──────────────────────────────────────────────────
    RATELIMIT_DEFAULT: str = os.getenv("RATELIMIT_DEFAULT", "200/hour")
    RATELIMIT_STORAGE_URI: str = os.getenv("RATELIMIT_STORAGE_URI", "memory://")

    # ── CORS ───────────────────────────────────────────────────────────
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

    # ── Application Meta ───────────────────────────────────────────────
    APP_NAME: str = "AppleVision AI"
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")

    # ── Logging ────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


class DevelopmentConfig(BaseConfig):
    """Local development configuration – verbose logging, SQLite DB."""

    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"

    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{_PROJECT_ROOT / 'instance' / 'applevision_dev.db'}",
    )


class ProductionConfig(BaseConfig):
    """Production configuration – requires real secrets & PostgreSQL."""

    DEBUG: bool = False
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "WARNING")

    # In production a real SECRET_KEY **must** be set via env var.
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "CHANGE-ME-NOW")

    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{_PROJECT_ROOT / 'instance' / 'applevision_prod.db'}",
    )

    # Stricter rate limits for production
    RATELIMIT_DEFAULT: str = os.getenv("RATELIMIT_DEFAULT", "100/hour")


class TestingConfig(BaseConfig):
    """Test configuration – in-memory SQLite, testing flag on."""

    TESTING: bool = True
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"

    SQLALCHEMY_DATABASE_URI: str = "sqlite:///:memory:"
    RATELIMIT_ENABLED: bool = False
