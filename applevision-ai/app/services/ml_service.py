"""
Machine Learning service – singleton model loader, preprocessing, and inference.

The ``MLService`` class follows the singleton pattern so the heavy Keras
model is loaded exactly once per process.  It is safe to call
``MLService.get_instance()`` from any request handler.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-import TensorFlow – it's slow and noisy
_tf = None  # type: ignore[assignment]


def _get_tf():  # noqa: ANN202
    """Import TensorFlow lazily to speed up module-level imports."""
    global _tf
    if _tf is None:
        # Suppress TF info / warning logs unless user wants them
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf  # noqa: WPS433

        _tf = tf
    return _tf


class MLService:
    """Thread-safe singleton that owns the Keras model and label map.

    Usage::

        svc = MLService.get_instance(model_path, labels_path)
        result = svc.predict(pil_image)
    """

    _instance: MLService | None = None
    _lock: threading.Lock = threading.Lock()

    # ── Singleton Access ───────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        model_path: str | None = None,
        labels_path: str | None = None,
    ) -> MLService:
        """Return the singleton instance, creating it on first call.

        Args:
            model_path: Filesystem path to the ``.keras`` model file.
            labels_path: Filesystem path to the ``labels.json`` file.

        Returns:
            The shared ``MLService`` instance.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    instance = cls.__new__(cls)
                    instance._model = None
                    instance._labels: dict[str, str] = {}
                    instance._model_path = model_path or ""
                    instance._labels_path = labels_path or ""
                    instance._model_loaded = False
                    instance._load_time: float | None = None
                    cls._instance = instance
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton (mainly useful in tests)."""
        with cls._lock:
            cls._instance = None

    # ── Loading ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the Keras model and class labels from disk.

        Called once during application startup.  Logs warnings rather
        than raising if files are missing so the app can still serve
        non-ML routes.
        """
        self._load_labels()
        self._load_model()

    def _load_labels(self) -> None:
        """Read ``labels.json`` into ``self._labels``."""
        path = Path(self._labels_path)
        if not path.exists():
            logger.warning("Labels file not found: %s", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._labels = json.load(fh)
            logger.info(
                "Loaded %d class labels from %s", len(self._labels), path
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load labels: %s", exc)

    def _load_model(self) -> None:
        """Load the Keras model weights into memory."""
        path = Path(self._model_path)
        if not path.exists():
            logger.warning("Model file not found: %s", path)
            return
        try:
            tf = _get_tf()
            t0 = time.perf_counter()
            self._model = tf.keras.models.load_model(str(path))
            self._load_time = round((time.perf_counter() - t0) * 1000, 2)
            self._model_loaded = True
            logger.info(
                "Model loaded in %.2f ms from %s", self._load_time, path
            )
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            self._model_loaded = False

    # ── Status ─────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """Return ``True`` when the model **and** labels are available."""
        return self._model_loaded and bool(self._labels)

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def num_classes(self) -> int:
        return len(self._labels)

    # ── Preprocessing ──────────────────────────────────────────────────

    @staticmethod
    def preprocess(image: Image.Image, target_size: tuple[int, int] = (224, 224)) -> Any:
        """Convert a PIL Image to a preprocessed TF tensor.

        Steps:
        1. Convert to RGB if necessary.
        2. Resize to ``target_size`` (default 224×224).
        3. Convert to float32 array.
        4. Expand batch dimension.
        5. Apply EfficientNetV2 channel-wise preprocessing.

        Args:
            image: A PIL Image.
            target_size: ``(height, width)`` to resize to.

        Returns:
            A TF tensor of shape ``(1, H, W, 3)`` ready for model input.
        """
        tf = _get_tf()

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, axis=0)
        return tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    # ── Inference ──────────────────────────────────────────────────────

    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        confidence_threshold: float = 0.75,
    ) -> dict[str, Any]:
        """Run classification inference on a single image.

        Args:
            image: A PIL Image to classify.
            top_k: How many top predictions to return.
            confidence_threshold: Cut-off for ``is_high_confidence``.

        Returns:
            A dict ready for JSON serialisation::

                {
                    "top_class": "Apple 10",
                    "confidence": 0.9812,
                    "inference_time_ms": 47.3,
                    "is_high_confidence": True,
                    "top_predictions": [...],
                    "threshold": 0.75,
                }

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.is_ready:
            raise RuntimeError("ML model or labels are not loaded.")

        tf = _get_tf()

        processed = self.preprocess(image)

        # ── Timed inference ────────────────────────────────────────────
        t0 = time.perf_counter()
        raw_predictions = self._model.predict(processed, verbose=0)
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        # ── Post-process scores ────────────────────────────────────────
        scores = tf.nn.softmax(raw_predictions[0]).numpy()

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_class_idx = int(top_indices[0])
        top_score = float(scores[top_class_idx])
        top_class_name = self._labels.get(
            str(top_class_idx), f"Class {top_class_idx}"
        )

        top_predictions = [
            {
                "class_name": self._labels.get(str(int(idx)), f"Class {idx}"),
                "score": round(float(scores[idx]), 4),
            }
            for idx in top_indices
        ]

        return {
            "top_class": top_class_name,
            "confidence": round(top_score, 4),
            "inference_time_ms": inference_ms,
            "is_high_confidence": top_score >= confidence_threshold,
            "top_predictions": top_predictions,
            "threshold": confidence_threshold,
        }
