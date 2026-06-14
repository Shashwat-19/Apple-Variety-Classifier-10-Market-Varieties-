"""
Shared pytest fixtures for AppleVision AI test suite.

Provides a test Flask application, client, and database session.
The ML model is mocked so tests run without the actual .keras file.
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app import create_app
from app.extensions import db as _db


# ── Fake ML predictions ──────────────────────────────────────────────────────

MOCK_LABELS = {
    "0": "Apple 10",
    "1": "Apple 13",
    "2": "Apple 18",
    "3": "Apple 19",
    "4": "Apple 7",
    "5": "Apple 8",
    "6": "Apple 9",
    "7": "Apple Red Yellow 2",
    "8": "Apple hit 1",
    "9": "Apple worm 1",
}

MOCK_PREDICTION_RESULT = {
    "top_class": "Apple 10",
    "confidence": 0.9732,
    "inference_time_ms": 42.5,
    "is_high_confidence": True,
    "top_predictions": [
        {"class_name": "Apple 10", "score": 0.9732},
        {"class_name": "Apple 13", "score": 0.0121},
        {"class_name": "Apple 18", "score": 0.0058},
        {"class_name": "Apple 19", "score": 0.0041},
        {"class_name": "Apple 7", "score": 0.0023},
    ],
    "threshold": 0.75,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def _mock_ml():
    """Patch the MLService for the entire test session."""
    mock_service = MagicMock()
    mock_service.is_ready = True
    mock_service.model_loaded = True
    mock_service.labels = MOCK_LABELS
    mock_service.num_classes = 10
    mock_service.predict.return_value = MOCK_PREDICTION_RESULT

    with patch("app.services.ml_service.MLService.get_instance", return_value=mock_service):
        with patch("app.services.ml_service.MLService.reset"):
            yield mock_service


@pytest.fixture(scope="session")
def app(_mock_ml) -> Generator:
    """Create a test Flask application with an in-memory database."""
    application = create_app("testing")
    application.config["TESTING"] = True

    yield application


@pytest.fixture(scope="function")
def client(app):
    """Provide a Flask test client, with a fresh database per test."""
    with app.app_context():
        _db.create_all()

    with app.test_client() as test_client:
        yield test_client

    with app.app_context():
        _db.session.remove()
        _db.drop_all()


@pytest.fixture(scope="function")
def db_session(app):
    """Provide a raw database session for service-level tests."""
    with app.app_context():
        _db.create_all()
        yield _db.session
        _db.session.remove()
        _db.drop_all()


# ── Helper Factories ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_image() -> BytesIO:
    """Create a minimal valid JPEG image in memory."""
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    buffer.name = "test_apple.jpg"
    return buffer


@pytest.fixture
def sample_prediction_data() -> dict[str, Any]:
    """Return sample data for creating a PredictionHistory record."""
    return {
        "image_name": "test_apple.jpg",
        "prediction": "Apple 10",
        "confidence": 0.9732,
        "inference_time_ms": 42.5,
        "top_predictions": json.dumps(MOCK_PREDICTION_RESULT["top_predictions"]),
        "ip_address": "127.0.0.1",
    }
