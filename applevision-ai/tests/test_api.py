"""
API endpoint tests for AppleVision AI.

Tests all REST API endpoints: /api/health, /api/predict,
/api/history, and /api/stats.
"""

from __future__ import annotations

import json
from io import BytesIO

import pytest
from PIL import Image


# ── Health Endpoint ──────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 with status field."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_health_contains_version(self, client):
        """Health response should include version info."""
        response = client.get("/api/health")
        data = response.get_json()
        assert "version" in data

    def test_health_contains_uptime(self, client):
        """Health response should include uptime."""
        response = client.get("/api/health")
        data = response.get_json()
        assert "uptime" in data

    def test_health_contains_model_status(self, client):
        """Health response should indicate model loading state."""
        response = client.get("/api/health")
        data = response.get_json()
        assert "model_loaded" in data


# ── Predict Endpoint ─────────────────────────────────────────────────────────


class TestPredictEndpoint:
    """Tests for POST /api/predict."""

    def test_predict_no_file(self, client):
        """Should return 400 when no file is provided."""
        response = client.post("/api/predict")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_predict_empty_filename(self, client):
        """Should return 400 when file has empty filename."""
        data = {"file": (BytesIO(b""), "")}
        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_predict_invalid_file_type(self, client):
        """Should return 400 when file is not an image."""
        data = {"file": (BytesIO(b"not an image"), "test.txt")}
        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_predict_success(self, client, sample_image):
        """Should return 200 with prediction result for valid image."""
        data = {"file": (sample_image, "test_apple.jpg")}
        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        result = response.get_json()
        assert "top_class" in result
        assert "confidence" in result
        assert "inference_time_ms" in result
        assert "top_predictions" in result
        assert isinstance(result["top_predictions"], list)

    def test_predict_returns_confidence_threshold(self, client, sample_image):
        """Prediction should include the confidence threshold used."""
        data = {"file": (sample_image, "test_apple.jpg")}
        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data",
        )
        result = response.get_json()
        assert "threshold" in result
        assert "is_high_confidence" in result

    def test_predict_webp_accepted(self, client):
        """WEBP format images should be accepted."""
        img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        buffer = BytesIO()
        img.save(buffer, format="WEBP")
        buffer.seek(0)

        data = {"file": (buffer, "apple.webp")}
        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert response.status_code == 200

    def test_predict_png_accepted(self, client):
        """PNG format images should be accepted."""
        img = Image.new("RGB", (100, 100), color=(0, 0, 255))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        data = {"file": (buffer, "apple.png")}
        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert response.status_code == 200


# ── History Endpoint ─────────────────────────────────────────────────────────


class TestHistoryEndpoint:
    """Tests for GET /api/history."""

    def test_history_empty(self, client):
        """Should return empty list when no predictions exist."""
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.get_json()
        assert "predictions" in data
        assert len(data["predictions"]) == 0
        assert data["total"] == 0

    def test_history_after_prediction(self, client, sample_image):
        """History should contain entry after a prediction is made."""
        # Make a prediction first
        pred_data = {"file": (sample_image, "test_apple.jpg")}
        client.post(
            "/api/predict",
            data=pred_data,
            content_type="multipart/form-data",
        )

        # Check history
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.get_json()
        assert data["total"] >= 1

    def test_history_pagination(self, client, sample_image):
        """History should support pagination parameters."""
        # Create a prediction
        pred_data = {"file": (sample_image, "test_apple.jpg")}
        client.post(
            "/api/predict",
            data=pred_data,
            content_type="multipart/form-data",
        )

        response = client.get("/api/history?page=1&per_page=5")
        assert response.status_code == 200
        data = response.get_json()
        assert "page" in data
        assert "per_page" in data
        assert "total" in data

    def test_history_search(self, client):
        """History should support search parameter."""
        response = client.get("/api/history?search=Apple")
        assert response.status_code == 200

    def test_history_sort(self, client):
        """History should support sort parameters."""
        response = client.get("/api/history?sort_by=timestamp&sort_order=desc")
        assert response.status_code == 200


# ── Stats Endpoint ───────────────────────────────────────────────────────────


class TestStatsEndpoint:
    """Tests for GET /api/stats."""

    def test_stats_returns_200(self, client):
        """Stats endpoint should return 200."""
        response = client.get("/api/stats")
        assert response.status_code == 200

    def test_stats_contains_required_keys(self, client):
        """Stats should contain all required analytics keys."""
        response = client.get("/api/stats")
        data = response.get_json()
        expected_keys = [
            "total_predictions",
            "predictions_by_type",
            "avg_confidence",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_stats_total_starts_at_zero(self, client):
        """Total predictions should be 0 with empty database."""
        response = client.get("/api/stats")
        data = response.get_json()
        assert data["total_predictions"] == 0

    def test_stats_after_predictions(self, client, sample_image):
        """Stats should reflect predictions that have been made."""
        # Make a prediction
        pred_data = {"file": (sample_image, "test_apple.jpg")}
        client.post(
            "/api/predict",
            data=pred_data,
            content_type="multipart/form-data",
        )

        response = client.get("/api/stats")
        data = response.get_json()
        assert data["total_predictions"] >= 1


# ── Page Routes ──────────────────────────────────────────────────────────────


class TestPageRoutes:
    """Tests for HTML page routes."""

    @pytest.mark.parametrize(
        "route",
        ["/", "/about", "/predict", "/analytics", "/api-docs", "/contact"],
    )
    def test_page_returns_200(self, client, route):
        """All page routes should return 200."""
        response = client.get(route)
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "route",
        ["/", "/about", "/predict", "/analytics", "/api-docs", "/contact"],
    )
    def test_page_returns_html(self, client, route):
        """All page routes should return HTML content."""
        response = client.get(route)
        assert "text/html" in response.content_type
