"""
Prediction persistence service.

Encapsulates all database operations for ``PredictionHistory`` records –
creating new entries and querying existing ones with filtering, search,
sorting, and pagination.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_

from app.extensions import db
from app.models.prediction import PredictionHistory

logger = logging.getLogger(__name__)


class PredictionService:
    """Data-access layer for prediction history records."""

    # ── Create ─────────────────────────────────────────────────────────

    @staticmethod
    def save_prediction(
        *,
        image_name: str,
        prediction: str,
        confidence: float,
        inference_time_ms: float,
        top_predictions: list[dict[str, Any]],
        ip_address: str | None = None,
    ) -> PredictionHistory:
        """Persist a new prediction record to the database.

        Args:
            image_name: Original filename uploaded by the client.
            prediction: Winning class label.
            confidence: Confidence score of the top prediction.
            inference_time_ms: How long inference took in milliseconds.
            top_predictions: List of ``{class_name, score}`` dicts.
            ip_address: Request originator IP (may be ``None``).

        Returns:
            The newly-created ``PredictionHistory`` row.
        """
        record = PredictionHistory(
            image_name=image_name,
            prediction=prediction,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            ip_address=ip_address,
            timestamp=datetime.now(timezone.utc),
        )
        record.set_top_predictions(top_predictions)

        try:
            db.session.add(record)
            db.session.commit()
            logger.debug("Saved prediction #%d for '%s'", record.id, image_name)
        except Exception:
            db.session.rollback()
            logger.exception("Failed to save prediction for '%s'", image_name)
            raise

        return record

    # ── Read (paginated, filtered, sorted) ─────────────────────────────

    @staticmethod
    def get_history(
        *,
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        prediction_type: str | None = None,
    ) -> dict[str, Any]:
        """Query prediction history with full filtering support.

        Args:
            page: 1-indexed page number.
            per_page: Number of results per page.
            search: Free-text search across ``image_name`` / ``prediction``.
            sort_by: Column name to order by.
            sort_order: ``"asc"`` or ``"desc"``.
            min_confidence: Lower-bound confidence filter (inclusive).
            max_confidence: Upper-bound confidence filter (inclusive).
            prediction_type: Exact-match filter on prediction class.

        Returns:
            A dict containing ``data`` (list of row dicts) and
            ``pagination`` metadata.
        """
        query = PredictionHistory.query

        # ── Filters ────────────────────────────────────────────────────
        if search:
            like_term = f"%{search}%"
            query = query.filter(
                or_(
                    PredictionHistory.image_name.ilike(like_term),
                    PredictionHistory.prediction.ilike(like_term),
                )
            )

        if min_confidence is not None:
            query = query.filter(PredictionHistory.confidence >= min_confidence)

        if max_confidence is not None:
            query = query.filter(PredictionHistory.confidence <= max_confidence)

        if prediction_type:
            query = query.filter(PredictionHistory.prediction == prediction_type)

        # ── Sorting ────────────────────────────────────────────────────
        _allowed_sorts = {
            "timestamp": PredictionHistory.timestamp,
            "confidence": PredictionHistory.confidence,
            "inference_time_ms": PredictionHistory.inference_time_ms,
            "prediction": PredictionHistory.prediction,
        }
        sort_column = _allowed_sorts.get(sort_by, PredictionHistory.timestamp)
        order = sort_column.desc() if sort_order == "desc" else sort_column.asc()
        query = query.order_by(order)

        # ── Pagination ─────────────────────────────────────────────────
        paginated = query.paginate(page=page, per_page=per_page, error_out=False)

        return {
            "data": [row.to_dict() for row in paginated.items],
            "pagination": {
                "page": paginated.page,
                "per_page": paginated.per_page,
                "total": paginated.total,
                "total_pages": paginated.pages,
                "has_next": paginated.has_next,
                "has_prev": paginated.has_prev,
            },
        }

    # ── Single Record ──────────────────────────────────────────────────

    @staticmethod
    def get_prediction_by_id(prediction_id: int) -> PredictionHistory | None:
        """Fetch a single prediction by its primary key."""
        return db.session.get(PredictionHistory, prediction_id)

    # ── Bulk ───────────────────────────────────────────────────────────

    @staticmethod
    def get_recent(limit: int = 10) -> list[dict[str, Any]]:
        """Return the *N* most recent predictions."""
        rows = (
            PredictionHistory.query
            .order_by(PredictionHistory.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]
