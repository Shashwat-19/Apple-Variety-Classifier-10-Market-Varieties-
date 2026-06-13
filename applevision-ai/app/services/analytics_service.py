"""
Analytics service – aggregate statistics over prediction history.

All methods are static helpers that run SQL aggregation queries through
SQLAlchemy and return plain dicts ready for JSON serialisation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func

from app.extensions import db
from app.models.prediction import PredictionHistory

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Compute read-only aggregate statistics for the analytics dashboard."""

    # ── Public API ─────────────────────────────────────────────────────

    @classmethod
    def get_full_stats(cls) -> dict[str, Any]:
        """Return a comprehensive stats payload used by ``GET /api/stats``.

        Returns:
            A dict with keys ``total_predictions``, ``most_predicted_type``,
            ``daily_predictions``, ``weekly_predictions``,
            ``monthly_predictions``, ``predictions_by_type``,
            ``avg_confidence``, and ``recent_predictions``.
        """
        return {
            "total_predictions": cls._total_predictions(),
            "most_predicted_type": cls._most_predicted_type(),
            "daily_predictions": cls._count_since(hours=24),
            "weekly_predictions": cls._count_since(days=7),
            "monthly_predictions": cls._count_since(days=30),
            "predictions_by_type": cls._predictions_by_type(),
            "avg_confidence": cls._avg_confidence(),
            "recent_predictions": cls._recent_predictions(limit=10),
        }

    # ── Scalar Aggregates ──────────────────────────────────────────────

    @staticmethod
    def _total_predictions() -> int:
        """Total number of prediction records."""
        return db.session.query(func.count(PredictionHistory.id)).scalar() or 0

    @staticmethod
    def _most_predicted_type() -> str | None:
        """The class label that appears most frequently."""
        result = (
            db.session.query(
                PredictionHistory.prediction,
                func.count(PredictionHistory.id).label("cnt"),
            )
            .group_by(PredictionHistory.prediction)
            .order_by(func.count(PredictionHistory.id).desc())
            .first()
        )
        return result[0] if result else None

    @staticmethod
    def _avg_confidence() -> float:
        """Average confidence across all predictions, rounded to 4 d.p."""
        val = db.session.query(
            func.avg(PredictionHistory.confidence)
        ).scalar()
        return round(float(val), 4) if val else 0.0

    # ── Time-window Counts ─────────────────────────────────────────────

    @staticmethod
    def _count_since(
        *,
        hours: int = 0,
        days: int = 0,
    ) -> int:
        """Count predictions made since ``now - delta``."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours, days=days)
        return (
            db.session.query(func.count(PredictionHistory.id))
            .filter(PredictionHistory.timestamp >= cutoff)
            .scalar()
            or 0
        )

    # ── Group-by Aggregates ────────────────────────────────────────────

    @staticmethod
    def _predictions_by_type() -> list[dict[str, Any]]:
        """Return ``[{type, count}, …]`` for every class label."""
        rows = (
            db.session.query(
                PredictionHistory.prediction,
                func.count(PredictionHistory.id).label("count"),
            )
            .group_by(PredictionHistory.prediction)
            .order_by(func.count(PredictionHistory.id).desc())
            .all()
        )
        return [{"type": row[0], "count": row[1]} for row in rows]

    # ── Recent History ─────────────────────────────────────────────────

    @staticmethod
    def _recent_predictions(limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent *limit* predictions as dicts."""
        rows = (
            PredictionHistory.query
            .order_by(PredictionHistory.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [row.to_dict() for row in rows]
