"""
Database model tests for AppleVision AI.

Tests the PredictionHistory ORM model: creation, serialisation,
timestamps, and JSON helper methods.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.extensions import db
from app.models.prediction import PredictionHistory


class TestPredictionHistoryModel:
    """Tests for the PredictionHistory database model."""

    def test_create_prediction(self, app, db_session):
        """Should successfully create a PredictionHistory record."""
        record = PredictionHistory(
            image_name="fuji_apple.jpg",
            prediction="Apple 10",
            confidence=0.9543,
            inference_time_ms=38.2,
            top_predictions='[{"class_name": "Apple 10", "score": 0.9543}]',
            ip_address="192.168.1.1",
        )
        db_session.add(record)
        db_session.commit()

        assert record.id is not None
        assert record.image_name == "fuji_apple.jpg"
        assert record.prediction == "Apple 10"
        assert record.confidence == pytest.approx(0.9543)

    def test_to_dict(self, app, db_session):
        """to_dict() should return a JSON-serialisable dictionary."""
        record = PredictionHistory(
            image_name="test.jpg",
            prediction="Apple 13",
            confidence=0.8765,
            inference_time_ms=45.1,
            top_predictions='[{"class_name": "Apple 13", "score": 0.8765}]',
            ip_address="10.0.0.1",
        )
        db_session.add(record)
        db_session.commit()

        result = record.to_dict()

        assert isinstance(result, dict)
        assert result["id"] == record.id
        assert result["image_name"] == "test.jpg"
        assert result["prediction"] == "Apple 13"
        assert result["confidence"] == pytest.approx(0.8765, abs=0.001)
        assert result["inference_time_ms"] == pytest.approx(45.1)
        assert isinstance(result["top_predictions"], list)
        assert result["ip_address"] == "10.0.0.1"
        assert result["timestamp"] is not None

    def test_to_dict_is_json_serialisable(self, app, db_session):
        """to_dict() output should be fully JSON-serialisable."""
        record = PredictionHistory(
            image_name="test.jpg",
            prediction="Apple 18",
            confidence=0.7,
            inference_time_ms=50.0,
            top_predictions="[]",
        )
        db_session.add(record)
        db_session.commit()

        # This should not raise
        json_str = json.dumps(record.to_dict())
        assert isinstance(json_str, str)

    def test_timestamp_auto_set(self, app, db_session):
        """Timestamp should be automatically set on creation."""
        record = PredictionHistory(
            image_name="auto_ts.jpg",
            prediction="Apple 7",
            confidence=0.88,
            inference_time_ms=33.0,
            top_predictions="[]",
        )
        db_session.add(record)
        db_session.commit()

        assert record.timestamp is not None
        assert isinstance(record.timestamp, datetime)

    def test_set_top_predictions(self, app, db_session):
        """set_top_predictions() should serialise predictions to JSON."""
        record = PredictionHistory(
            image_name="top_pred.jpg",
            prediction="Apple 8",
            confidence=0.91,
            inference_time_ms=41.0,
            top_predictions="[]",
        )

        predictions = [
            {"class_name": "Apple 8", "score": 0.91},
            {"class_name": "Apple 9", "score": 0.04},
            {"class_name": "Apple 10", "score": 0.03},
        ]
        record.set_top_predictions(predictions)
        db_session.add(record)
        db_session.commit()

        # Verify serialisation
        stored = json.loads(record.top_predictions)
        assert len(stored) == 3
        assert stored[0]["class_name"] == "Apple 8"

    def test_get_top_predictions(self, app, db_session):
        """get_top_predictions() should deserialise JSON back to Python."""
        predictions = [
            {"class_name": "Apple 9", "score": 0.85},
            {"class_name": "Apple 7", "score": 0.10},
        ]
        record = PredictionHistory(
            image_name="get_pred.jpg",
            prediction="Apple 9",
            confidence=0.85,
            inference_time_ms=39.0,
            top_predictions=json.dumps(predictions),
        )
        db_session.add(record)
        db_session.commit()

        result = record.get_top_predictions()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["class_name"] == "Apple 9"

    def test_get_top_predictions_handles_invalid_json(self, app, db_session):
        """get_top_predictions() should gracefully handle invalid JSON."""
        record = PredictionHistory(
            image_name="bad_json.jpg",
            prediction="Apple 10",
            confidence=0.8,
            inference_time_ms=42.0,
            top_predictions="not valid json {{{",
        )
        db_session.add(record)
        db_session.commit()

        result = record.get_top_predictions()
        assert result == []

    def test_repr(self, app, db_session):
        """__repr__ should return a readable string."""
        record = PredictionHistory(
            image_name="repr_test.jpg",
            prediction="Apple Red Yellow 2",
            confidence=0.9234,
            inference_time_ms=44.0,
            top_predictions="[]",
        )
        db_session.add(record)
        db_session.commit()

        repr_str = repr(record)
        assert "PredictionHistory" in repr_str
        assert "Apple Red Yellow 2" in repr_str

    def test_nullable_ip_address(self, app, db_session):
        """IP address should be nullable."""
        record = PredictionHistory(
            image_name="no_ip.jpg",
            prediction="Apple worm 1",
            confidence=0.72,
            inference_time_ms=50.0,
            top_predictions="[]",
            ip_address=None,
        )
        db_session.add(record)
        db_session.commit()

        assert record.ip_address is None

    def test_multiple_records(self, app, db_session):
        """Should handle multiple records with auto-incrementing IDs."""
        ids = []
        for i in range(5):
            record = PredictionHistory(
                image_name=f"multi_{i}.jpg",
                prediction=f"Apple {i}",
                confidence=0.9 - (i * 0.05),
                inference_time_ms=40 + i,
                top_predictions="[]",
            )
            db_session.add(record)
        db_session.commit()

        all_records = db_session.query(PredictionHistory).all()
        assert len(all_records) == 5

        # IDs should be unique
        record_ids = [r.id for r in all_records]
        assert len(set(record_ids)) == 5
