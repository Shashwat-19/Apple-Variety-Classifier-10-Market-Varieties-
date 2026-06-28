"""
Machine Learning service – singleton model loader, preprocessing, and inference.

Uses ONNX Runtime for lightweight deployment on platforms with strict
bundle size limits (e.g. Vercel's 250 MB serverless function limit).
ONNX Runtime is ~30 MB installed vs TensorFlow's ~1.8 GB.

Falls back to full TensorFlow/Keras if no ONNX model is found.

The ``MLService`` class follows the singleton pattern so the model
is loaded exactly once per process.  It is safe to call
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


class MLService:
    """Thread-safe singleton that owns the ML model and label map.

    Supports two runtime backends:
    - **ONNX Runtime** (preferred): Lightweight, fast, Vercel-compatible.
    - **TensorFlow/Keras** (fallback): Full framework, for local development.

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
            model_path: Path to ``.onnx`` or ``.keras`` model file.
            labels_path: Path to ``labels.json``.

        Returns:
            The shared ``MLService`` instance.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    instance = cls.__new__(cls)
                    instance._session = None       # ONNX session
                    instance._tf_model = None      # TF/Keras model (fallback)
                    instance._labels: dict[str, str] = {}
                    instance._model_path = model_path or ""
                    instance._labels_path = labels_path or ""
                    instance._model_loaded = False
                    instance._backend = "none"     # "onnx" | "tensorflow" | "none"
                    instance._load_time: float | None = None
                    instance._input_name: str | None = None
                    cls._instance = instance
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton (mainly useful in tests)."""
        with cls._lock:
            cls._instance = None

    # ── Loading ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the model and class labels from disk.

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
        """Load the model using the appropriate backend based on file extension."""
        path = Path(self._model_path)
        if not path.exists():
            logger.warning("Model file not found: %s", path)
            return

        if path.suffix == ".onnx":
            self._load_onnx(path)
        elif path.suffix in (".keras", ".h5"):
            self._load_tensorflow(path)
        else:
            logger.error("Unsupported model format: %s", path.suffix)

    def _load_onnx(self, path: Path) -> None:
        """Load an ONNX model using onnxruntime."""
        try:
            import onnxruntime as ort

            t0 = time.perf_counter()
            # Use CPU execution provider (Vercel has no GPU)
            self._session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._load_time = round((time.perf_counter() - t0) * 1000, 2)
            self._model_loaded = True
            self._backend = "onnx"
            logger.info(
                "ONNX model loaded in %.2f ms from %s", self._load_time, path
            )
        except Exception as exc:
            logger.error("Failed to load ONNX model: %s", exc)
            self._model_loaded = False

    def _load_tensorflow(self, path: Path) -> None:
        """Load a Keras model using TensorFlow (fallback for local dev)."""
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf

            t0 = time.perf_counter()
            self._tf_model = tf.keras.models.load_model(str(path))
            self._load_time = round((time.perf_counter() - t0) * 1000, 2)
            self._model_loaded = True
            self._backend = "tensorflow"
            logger.info(
                "TF/Keras model loaded in %.2f ms from %s", self._load_time, path
            )
        except Exception as exc:
            logger.error("Failed to load TF model: %s", exc)
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
    def preprocess(image: Image.Image, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
        """Convert a PIL Image to a preprocessed NumPy array.

        Applies the same EfficientNetV2 preprocessing as the training
        pipeline:
        1. Convert to RGB if necessary.
        2. Resize to ``target_size`` (default 224×224).
        3. Convert to float32 array.
        4. Scale pixel values to [-1, 1] range (EfficientNetV2 preprocessing).
        5. Expand batch dimension.

        Args:
            image: A PIL Image.
            target_size: ``(height, width)`` to resize to.

        Returns:
            A NumPy array of shape ``(1, H, W, 3)`` ready for model input.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize(target_size)
        img_array = np.array(image, dtype=np.float32)

        # EfficientNetV2 preprocessing: scale from [0, 255] to [-1, 1]
        img_array = (img_array / 128.0) - 1.0

        # Add batch dimension: (H, W, 3) → (1, H, W, 3)
        return np.expand_dims(img_array, axis=0)

    # ── Inference ──────────────────────────────────────────────────────

    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        confidence_threshold: float = 0.75,
    ) -> dict[str, Any]:
        """Run classification inference on a single image.

        Dispatches to the ONNX or TF backend depending on which one
        was loaded.

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

        processed = self.preprocess(image)

        if self._backend == "onnx":
            scores, inference_ms = self._infer_onnx(processed)
        elif self._backend == "tensorflow":
            scores, inference_ms = self._infer_tensorflow(processed)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

        return self._postprocess(scores, inference_ms, top_k, confidence_threshold)

    def _infer_onnx(self, processed: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference through ONNX Runtime."""
        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: processed})
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)
        return outputs[0][0].astype(np.float32), inference_ms

    def _infer_tensorflow(self, processed: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference through TensorFlow/Keras."""
        import tensorflow as tf

        t0 = time.perf_counter()
        raw = self._tf_model.predict(processed, verbose=0)
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)
        scores = tf.nn.softmax(raw[0]).numpy()
        return scores, inference_ms

    def _postprocess(
        self,
        scores: np.ndarray,
        inference_ms: float,
        top_k: int,
        confidence_threshold: float,
    ) -> dict[str, Any]:
        """Apply softmax (if needed) and build the result dict."""
        # If scores don't sum to ~1, apply softmax (ONNX model may output
        # logits or already-softmaxed values depending on the export)
        score_sum = float(np.sum(scores))
        if score_sum < 0.99 or score_sum > 1.01:
            exp_scores = np.exp(scores - np.max(scores))
            scores = exp_scores / np.sum(exp_scores)

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
