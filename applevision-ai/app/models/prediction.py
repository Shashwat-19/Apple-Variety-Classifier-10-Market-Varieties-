"""
PredictionHistory ORM model.

Stores every classification prediction made by the API, including the
top-k results and request metadata for analytics and auditing purposes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.extensions import db


class PredictionHistory(db.Model):  # type: ignore[name-defined]
    """Persistent record of a single apple-classification inference call."""

    __tablename__ = "prediction_history"

    # ── Primary Key ────────────────────────────────────────────────────
    id: int = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # ── Prediction Data ────────────────────────────────────────────────
    image_name: str = db.Column(db.String(255), nullable=False, index=True)
    prediction: str = db.Column(db.String(100), nullable=False, index=True)
    confidence: float = db.Column(db.Float, nullable=False)
    inference_time_ms: float = db.Column(db.Float, nullable=False)
    top_predictions: str = db.Column(db.Text, nullable=False, default="[]")

    # ── Request Metadata ───────────────────────────────────────────────
    ip_address: str = db.Column(db.String(45), nullable=True)  # IPv6 max 45
    timestamp: datetime = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<PredictionHistory id={self.id} "
            f"prediction={self.prediction!r} "
            f"confidence={self.confidence:.2f}>"
        )

    # ── Convenience Helpers ────────────────────────────────────────────

    def set_top_predictions(self, predictions: list[dict[str, Any]]) -> None:
        """Serialise a list of prediction dicts to the JSON text column."""
        self.top_predictions = json.dumps(predictions, ensure_ascii=False)

    def get_top_predictions(self) -> list[dict[str, Any]]:
        """Deserialise the stored JSON text back to Python objects."""
        try:
            return json.loads(self.top_predictions)  # type: ignore[arg-type]
        except (json.JSONDecodeError, TypeError):
            return []

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of this record."""
        return {
            "id": self.id,
            "image_name": self.image_name,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "top_predictions": self.get_top_predictions(),
            "ip_address": self.ip_address,
            "timestamp": (
                self.timestamp.isoformat() if self.timestamp else None
            ),
        }
