"""
Analytics service – aggregate statistics over prediction history.

All methods are static helpers that run SQL aggregation queries through
SQLAlchemy and return plain dicts ready for JSON serialisation.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
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
            "avg_confidence": cls._avg_confidence_raw(),
            "recent_predictions": cls._recent_predictions(limit=10),
        }

    @classmethod
    def get_analytics_payload(cls) -> dict[str, Any]:
        """Return the exact JSON shape consumed by the analytics dashboard.

        This is the payload served by ``GET /api/analytics``.

        Returns:
            {
                total_predictions: int,
                today_predictions: int,
                average_confidence: float (percentage, 1 decimal),
                most_predicted: str | None,
                variety_distribution: {label: count, …},
                daily_predictions: [{date: "YYYY-MM-DD", count: int}, …]
            }
        """
        total = cls._total_predictions()
        most = cls._most_predicted_type()
        today_count = cls._today_count()
        avg_conf = cls._avg_confidence_pct()
        variety_dist = cls._variety_distribution_dict()
        daily_trend = cls._daily_trend_30d()

        return {
            "total_predictions": total,
            "today_predictions": today_count,
            "average_confidence": avg_conf,
            "most_predicted": most,
            "variety_distribution": variety_dist,
            "daily_predictions": daily_trend,
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
    def _avg_confidence_raw() -> float:
        """Average confidence (0-1 range), rounded to 4 decimal places."""
        val = db.session.query(
            func.avg(PredictionHistory.confidence)
        ).scalar()
        return round(float(val), 4) if val else 0.0

    @staticmethod
    def _avg_confidence_pct() -> float:
        """Average confidence as a percentage (0-100), 1 decimal place."""
        val = db.session.query(
            func.avg(PredictionHistory.confidence)
        ).scalar()
        return round(float(val) * 100, 1) if val else 0.0

    @staticmethod
    def _today_count() -> int:
        """Count predictions whose timestamp falls on today (UTC date)."""
        today_utc = date.today()  # server local date used as proxy for UTC date
        # Build UTC midnight boundaries for today
        start = datetime(
            today_utc.year, today_utc.month, today_utc.day,
            tzinfo=timezone.utc
        )
        end = start + timedelta(days=1)
        return (
            db.session.query(func.count(PredictionHistory.id))
            .filter(
                PredictionHistory.timestamp >= start,
                PredictionHistory.timestamp < end,
            )
            .scalar()
            or 0
        )

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

    @staticmethod
    def _variety_distribution_dict() -> dict[str, int]:
        """Return a dict mapping each apple variety label to its count.

        Uses a single SQL GROUP BY query; no Python-side aggregation loop.
        """
        rows = (
            db.session.query(
                PredictionHistory.prediction,
                func.count(PredictionHistory.id).label("count"),
            )
            .group_by(PredictionHistory.prediction)
            .order_by(func.count(PredictionHistory.id).desc())
            .all()
        )
        return {row[0]: row[1] for row in rows}

    @staticmethod
    def _daily_trend_30d() -> list[dict[str, Any]]:
        """Return per-day prediction counts for the last 30 days.

        Missing days are included with ``count=0`` so the line chart
        always shows a complete 30-day window.

        Uses a single SQL GROUP BY on the date portion of ``timestamp``;
        zero-filling is done in Python but only over the 30-element range.
        """
        # Compute the inclusive 30-day window
        today_utc = date.today()
        window_start = today_utc - timedelta(days=29)  # 30 days inclusive
        cutoff_dt = datetime(
            window_start.year, window_start.month, window_start.day,
            tzinfo=timezone.utc,
        )

        # One SQL query – group by date string (SQLite: DATE(), Postgres: DATE_TRUNC)
        rows = (
            db.session.query(
                func.date(PredictionHistory.timestamp).label("day"),
                func.count(PredictionHistory.id).label("count"),
            )
            .filter(PredictionHistory.timestamp >= cutoff_dt)
            .group_by(func.date(PredictionHistory.timestamp))
            .all()
        )

        # Build a lookup: {"YYYY-MM-DD": count}
        counts_by_day: dict[str, int] = {str(row[0]): row[1] for row in rows}

        # Emit one entry per calendar day in the window (zeros for missing days)
        trend: list[dict[str, Any]] = []
        for offset in range(30):
            day = window_start + timedelta(days=offset)
            day_str = day.isoformat()  # "YYYY-MM-DD"
            trend.append({"date": day_str, "count": counts_by_day.get(day_str, 0)})

        return trend

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
