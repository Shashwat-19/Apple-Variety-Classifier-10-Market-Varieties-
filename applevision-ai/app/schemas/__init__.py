"""Schemas package for AppleVision AI."""

from app.schemas.prediction import (
    PredictionRequestSchema,
    PredictionResponseSchema,
    HistoryQuerySchema,
)

__all__ = [
    "PredictionRequestSchema",
    "PredictionResponseSchema",
    "HistoryQuerySchema",
]
