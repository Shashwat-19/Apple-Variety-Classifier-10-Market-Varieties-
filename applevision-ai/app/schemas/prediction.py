"""
Pydantic schemas for request / response validation.

Using Pydantic v2 for strict runtime validation of incoming query
parameters and outgoing JSON payloads.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request Schemas ────────────────────────────────────────────────────────


class PredictionRequestSchema(BaseModel):
    """Validates constraints on an incoming prediction request.

    Note: the actual image is sent as multipart form-data and validated
    in the route handler; this schema captures optional knobs.
    """

    confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to flag as high-confidence.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of top predictions to return.",
    )


class HistoryQuerySchema(BaseModel):
    """Validates query parameters for GET /api/history."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed).")
    per_page: int = Field(
        default=20, ge=1, le=100, description="Results per page."
    )
    search: Optional[str] = Field(
        default=None, max_length=200, description="Search in image name or prediction."
    )
    sort_by: Literal["timestamp", "confidence", "inference_time_ms", "prediction"] = Field(
        default="timestamp",
        description="Column to sort results by.",
    )
    sort_order: Literal["asc", "desc"] = Field(
        default="desc",
        description="Sort direction.",
    )
    min_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence filter."
    )
    max_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Maximum confidence filter."
    )
    prediction_type: Optional[str] = Field(
        default=None, max_length=100, description="Filter by prediction class name."
    )

    @field_validator("search", mode="before")
    @classmethod
    def strip_search(cls, v: str | None) -> str | None:
        """Strip leading/trailing whitespace from search queries."""
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v


# ── Response Schemas ───────────────────────────────────────────────────────


class TopPredictionSchema(BaseModel):
    """A single class in the top-k prediction list."""

    class_name: str
    score: float


class PredictionResponseSchema(BaseModel):
    """JSON body returned by POST /api/predict."""

    success: bool = True
    top_class: str
    confidence: float
    inference_time_ms: float
    is_high_confidence: bool
    top_predictions: list[TopPredictionSchema]
    threshold: float
    image_name: str


class HistoryItemSchema(BaseModel):
    """Serialised representation of a single PredictionHistory row."""

    id: int
    image_name: str
    prediction: str
    confidence: float
    inference_time_ms: float
    top_predictions: list[dict[str, Any]]
    ip_address: Optional[str] = None
    timestamp: Optional[datetime] = None


class PaginatedHistorySchema(BaseModel):
    """Paginated wrapper around a list of history items."""

    success: bool = True
    data: list[dict[str, Any]]
    pagination: dict[str, Any]
