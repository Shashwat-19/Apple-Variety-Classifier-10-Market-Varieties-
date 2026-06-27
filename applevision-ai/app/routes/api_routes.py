"""
REST API routes.

All endpoints live under the ``/api`` prefix.  They return JSON and use
Pydantic schemas for input validation.

Endpoints
---------
POST /api/predict   – classify an uploaded apple image
GET  /api/history   – paginated, searchable prediction history
GET  /api/stats     – aggregate analytics
GET  /api/health    – service health check
"""

from __future__ import annotations

import logging
import time
from typing import Any

from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError
from PIL import Image

from app.extensions import db, limiter
from app.schemas.prediction import HistoryQuerySchema
from app.services.ml_service import MLService
from app.services.prediction_service import PredictionService
from app.services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")

# ── Module-level start time for uptime reporting ───────────────────────────
_BOOT_TIME: float = time.time()


# ===========================================================================
# POST /api/predict
# ===========================================================================
@api_bp.route("/predict", methods=["POST"])
@limiter.limit("100/hour")
def predict() -> tuple[dict[str, Any], int]:
    """Classify an uploaded apple image.

    Expects a ``multipart/form-data`` request with a ``file`` field
    containing the image.  Optional form fields ``confidence_threshold``
    and ``top_k`` are also accepted.

    Returns:
        JSON payload with the classification result and a 200 status, or
        an error payload with the appropriate 4xx/5xx status.
    """
    ml_service = MLService.get_instance()

    # ── Guard: model readiness ─────────────────────────────────────────
    if not ml_service.is_ready:
        return jsonify({
            "success": False,
            "error": "Model not loaded",
            "message": "The ML model or labels could not be loaded on the server.",
        }), 503

    # ── Guard: file presence ───────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file provided",
            "message": "Include an image file in the 'file' form field.",
        }), 400

    file = request.files["file"]
    if file.filename is None or file.filename.strip() == "":
        return jsonify({
            "success": False,
            "error": "Empty filename",
            "message": "The uploaded file has no filename.",
        }), 400

    # ── Optional parameters ────────────────────────────────────────────
    confidence_threshold = float(
        request.form.get(
            "confidence_threshold",
            current_app.config.get("CONFIDENCE_THRESHOLD", 0.75),
        )
    )
    top_k = int(
        request.form.get(
            "top_k",
            current_app.config.get("TOP_K_PREDICTIONS", 5),
        )
    )

    # ── Inference ──────────────────────────────────────────────────────
    try:
        image = Image.open(file.stream)
        result = ml_service.predict(
            image,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )
    except Exception as exc:
        logger.exception("Prediction failed for '%s'", file.filename)
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": str(exc),
        }), 500

    # ── Persist to DB (best-effort) ────────────────────────────────────
    try:
        PredictionService.save_prediction(
            image_name=file.filename,
            prediction=result["top_class"],
            confidence=result["confidence"],
            inference_time_ms=result["inference_time_ms"],
            top_predictions=result["top_predictions"],
            ip_address=request.remote_addr,
        )
    except Exception:
        # Logging already happened inside the service; don't fail the
        # response just because the DB write failed.
        logger.warning("Prediction succeeded but DB save failed.")

    # ── Response ───────────────────────────────────────────────────────
    return jsonify({
        "success": True,
        "image_name": file.filename,
        **result,
    }), 200


# ===========================================================================
# GET /api/history
# ===========================================================================
@api_bp.route("/history", methods=["GET"])
@limiter.limit("200/hour")
def history() -> tuple[dict[str, Any], int]:
    """Return paginated prediction history with search & filters.

    Query Parameters:
        page, per_page, search, sort_by, sort_order,
        min_confidence, max_confidence, prediction_type

    Returns:
        JSON with ``data`` (list of prediction dicts) and ``pagination``
        metadata.
    """
    try:
        params = HistoryQuerySchema(**request.args.to_dict())
    except ValidationError as exc:
        return jsonify({
            "success": False,
            "error": "Invalid query parameters",
            "details": exc.errors(),
        }), 400

    try:
        result = PredictionService.get_history(
            page=params.page,
            per_page=params.per_page,
            search=params.search,
            sort_by=params.sort_by,
            sort_order=params.sort_order,
            min_confidence=params.min_confidence,
            max_confidence=params.max_confidence,
            prediction_type=params.prediction_type,
        )
    except Exception as exc:
        logger.exception("Failed to query prediction history")
        return jsonify({
            "success": False,
            "error": "Database query failed",
            "message": str(exc),
        }), 500

    return jsonify({"success": True, **result}), 200


# ===========================================================================
# GET /api/stats
# ===========================================================================
@api_bp.route("/stats", methods=["GET"])
@limiter.limit("200/hour")
def stats() -> tuple[dict[str, Any], int]:
    """Return aggregate prediction statistics for the analytics dashboard.

    Returns:
        JSON with total_predictions, most_predicted_type,
        daily/weekly/monthly counts, predictions_by_type,
        avg_confidence, and recent_predictions.
    """
    try:
        data = AnalyticsService.get_full_stats()
    except Exception as exc:
        logger.exception("Failed to compute stats")
        return jsonify({
            "success": False,
            "error": "Stats computation failed",
            "message": str(exc),
        }), 500

    return jsonify({"success": True, **data}), 200


# ===========================================================================
# GET /api/analytics
# ===========================================================================
@api_bp.route("/analytics", methods=["GET"])
@limiter.limit("200/hour")
def analytics() -> tuple[dict[str, Any], int]:
    """Return analytics data for the dashboard in the exact shape the JS expects.

    The response contains:
        - total_predictions: int
        - today_predictions: int
        - average_confidence: float (percentage with 1 decimal, e.g. 96.4)
        - most_predicted: str | null
        - variety_distribution: {label: count, …}
        - daily_predictions: [{date: "YYYY-MM-DD", count: int}, …] (30 days)
    """
    try:
        data = AnalyticsService.get_analytics_payload()
    except Exception as exc:
        logger.exception("Failed to compute analytics payload")
        return jsonify({
            "success": False,
            "error": "Analytics computation failed",
            "message": str(exc),
        }), 500

    return jsonify({"success": True, **data}), 200


# ===========================================================================
# GET /api/health
# ===========================================================================
@api_bp.route("/health", methods=["GET"])
@limiter.exempt
def health() -> tuple[dict[str, Any], int]:
    """Service health / readiness check.

    Returns:
        JSON containing ``status``, ``model_loaded``, ``db_connected``,
        ``uptime``, and ``version``.
    """
    # ── Model status ───────────────────────────────────────────────────
    ml_service = MLService.get_instance()
    model_loaded = ml_service.model_loaded

    # ── DB connectivity ────────────────────────────────────────────────
    db_connected = False
    try:
        db.session.execute(db.text("SELECT 1"))
        db_connected = True
    except Exception:
        logger.warning("Database health-check failed.")

    # ── Uptime ─────────────────────────────────────────────────────────
    uptime_seconds = round(time.time() - _BOOT_TIME, 2)

    # ── Overall status ─────────────────────────────────────────────────
    is_healthy = model_loaded and db_connected
    status_text = "healthy" if is_healthy else "degraded"

    payload: dict[str, Any] = {
        "status": status_text,
        "model_loaded": model_loaded,
        "db_connected": db_connected,
        "uptime": uptime_seconds,
        "version": current_app.config.get("APP_VERSION", "1.0.0"),
    }

    return jsonify(payload), 200 if is_healthy else 503
