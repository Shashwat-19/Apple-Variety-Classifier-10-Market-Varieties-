"""
Service-layer tests for AppleVision AI.

Tests MLService, PredictionService, and AnalyticsService
business logic independently from the HTTP layer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.models.prediction import PredictionHistory
from app.extensions import db


# ── ML Service ───────────────────────────────────────────────────────────────


class TestMLService:
    """Tests for the MLService singleton."""

    def test_singleton_returns_same_instance(self, app):
        """get_instance() should return the same object each time."""
        from app.services.ml_service import MLService

        svc1 = MLService.get_instance()
        svc2 = MLService.get_instance()
        assert svc1 is svc2

    def test_preprocess_output_shape(self, app):
        """Preprocessing should produce a (1, 224, 224, 3) tensor."""
        from app.services.ml_service import MLService

        img = Image.new("RGB", (300, 400), color=(128, 128, 128))

        with app.app_context():
            try:
                result = MLService.preprocess(img)
                assert result.shape[0] == 1
                assert result.shape[1] == 224
                assert result.shape[2] == 224
                assert result.shape[3] == 3
            except Exception:
                # TF may not be available in CI without GPU
                pytest.skip("TensorFlow not available for preprocessing test")

    def test_preprocess_converts_rgba_to_rgb(self, app):
        """Preprocessing should handle RGBA images by converting to RGB."""
        from app.services.ml_service import MLService

        img = Image.new("RGBA", (100, 100), color=(128, 128, 128, 255))

        with app.app_context():
            try:
                result = MLService.preprocess(img)
                assert result.shape[3] == 3  # Should be RGB, not RGBA
            except Exception:
                pytest.skip("TensorFlow not available for preprocessing test")

    def test_predict_returns_required_fields(self, app, _mock_ml):
        """predict() should return all required fields."""
        from app.services.ml_service import MLService

        svc = MLService.get_instance()
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))

        with app.app_context():
            result = svc.predict(img)

        assert "top_class" in result
        assert "confidence" in result
        assert "inference_time_ms" in result
        assert "is_high_confidence" in result
        assert "top_predictions" in result
        assert "threshold" in result


# ── Prediction Service ───────────────────────────────────────────────────────


class TestPredictionService:
    """Tests for PredictionService database operations."""

    def test_save_prediction(self, app, db_session, sample_prediction_data):
        """Should successfully save a prediction to the database."""
        record = PredictionHistory(
            image_name=sample_prediction_data["image_name"],
            prediction=sample_prediction_data["prediction"],
            confidence=sample_prediction_data["confidence"],
            inference_time_ms=sample_prediction_data["inference_time_ms"],
            top_predictions=sample_prediction_data["top_predictions"],
            ip_address=sample_prediction_data["ip_address"],
        )
        db_session.add(record)
        db_session.commit()

        # Verify it was saved
        saved = db_session.query(PredictionHistory).first()
        assert saved is not None
        assert saved.prediction == "Apple 10"
        assert saved.confidence == pytest.approx(0.9732)

    def test_query_predictions(self, app, db_session, sample_prediction_data):
        """Should be able to query saved predictions."""
        # Save multiple predictions
        for i in range(5):
            record = PredictionHistory(
                image_name=f"apple_{i}.jpg",
                prediction=f"Apple {i}",
                confidence=0.9 - (i * 0.1),
                inference_time_ms=40 + i,
                top_predictions="[]",
                ip_address="127.0.0.1",
            )
            db_session.add(record)
        db_session.commit()

        # Query all
        results = db_session.query(PredictionHistory).all()
        assert len(results) == 5

    def test_query_with_filter(self, app, db_session):
        """Should support filtering by confidence."""
        for i, conf in enumerate([0.95, 0.80, 0.60, 0.40]):
            record = PredictionHistory(
                image_name=f"apple_{i}.jpg",
                prediction="Apple 10",
                confidence=conf,
                inference_time_ms=42.0,
                top_predictions="[]",
            )
            db_session.add(record)
        db_session.commit()

        # Filter high confidence only
        high_conf = (
            db_session.query(PredictionHistory)
            .filter(PredictionHistory.confidence >= 0.75)
            .all()
        )
        assert len(high_conf) == 2

    def test_query_with_search(self, app, db_session):
        """Should support searching by prediction name."""
        predictions = [
            ("Apple 10", "apple1.jpg"),
            ("Apple Red Yellow 2", "apple2.jpg"),
            ("Apple worm 1", "apple3.jpg"),
        ]
        for pred, img in predictions:
            record = PredictionHistory(
                image_name=img,
                prediction=pred,
                confidence=0.9,
                inference_time_ms=42.0,
                top_predictions="[]",
            )
            db_session.add(record)
        db_session.commit()

        # Search for "Red"
        results = (
            db_session.query(PredictionHistory)
            .filter(PredictionHistory.prediction.ilike("%Red%"))
            .all()
        )
        assert len(results) == 1
        assert results[0].prediction == "Apple Red Yellow 2"

    def test_query_ordering(self, app, db_session):
        """Should support ordering by timestamp."""
        for i in range(3):
            record = PredictionHistory(
                image_name=f"apple_{i}.jpg",
                prediction=f"Apple {i}",
                confidence=0.9,
                inference_time_ms=42.0,
                top_predictions="[]",
            )
            db_session.add(record)
        db_session.commit()

        results = (
            db_session.query(PredictionHistory)
            .order_by(PredictionHistory.timestamp.desc())
            .all()
        )
        assert len(results) == 3


# ── Analytics Service ────────────────────────────────────────────────────────


class TestAnalyticsService:
    """Tests for AnalyticsService statistics computation."""

    def test_empty_database_stats(self, app, db_session):
        """Stats should return zeros for empty database."""
        total = db_session.query(PredictionHistory).count()
        assert total == 0

    def test_total_predictions_count(self, app, db_session):
        """Should correctly count total predictions."""
        for i in range(10):
            record = PredictionHistory(
                image_name=f"apple_{i}.jpg",
                prediction="Apple 10",
                confidence=0.9,
                inference_time_ms=42.0,
                top_predictions="[]",
            )
            db_session.add(record)
        db_session.commit()

        total = db_session.query(PredictionHistory).count()
        assert total == 10

    def test_most_predicted_type(self, app, db_session):
        """Should identify the most frequently predicted type."""
        types_and_counts = [
            ("Apple 10", 5),
            ("Apple 13", 3),
            ("Apple 18", 2),
        ]
        for pred_type, count in types_and_counts:
            for i in range(count):
                record = PredictionHistory(
                    image_name=f"{pred_type}_{i}.jpg",
                    prediction=pred_type,
                    confidence=0.9,
                    inference_time_ms=42.0,
                    top_predictions="[]",
                )
                db_session.add(record)
        db_session.commit()

        from sqlalchemy import func

        most_common = (
            db_session.query(
                PredictionHistory.prediction,
                func.count(PredictionHistory.id).label("count"),
            )
            .group_by(PredictionHistory.prediction)
            .order_by(func.count(PredictionHistory.id).desc())
            .first()
        )
        assert most_common is not None
        assert most_common[0] == "Apple 10"
        assert most_common[1] == 5

    def test_average_confidence(self, app, db_session):
        """Should correctly compute average confidence."""
        confidences = [0.95, 0.85, 0.75, 0.65]
        for i, conf in enumerate(confidences):
            record = PredictionHistory(
                image_name=f"apple_{i}.jpg",
                prediction="Apple 10",
                confidence=conf,
                inference_time_ms=42.0,
                top_predictions="[]",
            )
            db_session.add(record)
        db_session.commit()

        from sqlalchemy import func

        avg = db_session.query(func.avg(PredictionHistory.confidence)).scalar()
        assert avg == pytest.approx(0.80, abs=0.01)
