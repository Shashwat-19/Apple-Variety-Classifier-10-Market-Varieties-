#!/usr/bin/env python3
"""
Convert the EfficientNetV2S .keras model to ONNX format.

ONNX Runtime is ~30 MB installed (vs TensorFlow's ~1.8 GB), making
it suitable for deployment on platforms with strict bundle size limits
like Vercel's 250 MB serverless function limit.

Usage:
    pip install tf2onnx
    python convert_to_onnx.py
"""

import os
import sys
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
KERAS_MODEL = os.path.join(REPO_ROOT, "apple-variety-streamlit", "model", "apple_classifier_final.keras")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ml", "model")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "apple_classifier.onnx")


def main():
    if not os.path.exists(KERAS_MODEL):
        print(f"ERROR: Keras model not found at {KERAS_MODEL}")
        sys.exit(1)

    print(f"Loading Keras model from: {KERAS_MODEL}")
    model = tf.keras.models.load_model(KERAS_MODEL)
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Parameters:   {model.count_params():,}")

    # Save as SavedModel first (tf2onnx needs SavedModel or .pb)
    saved_model_dir = os.path.join(OUTPUT_DIR, "_temp_saved_model")
    print(f"\nExporting to SavedModel: {saved_model_dir}")
    model.export(saved_model_dir)

    # Convert SavedModel → ONNX using tf2onnx CLI
    print(f"\nConverting to ONNX: {OUTPUT_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", saved_model_dir,
            "--output", OUTPUT_FILE,
            "--opset", "17",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"ERROR: tf2onnx conversion failed:\n{result.stderr}")
        sys.exit(1)

    # Cleanup temp SavedModel
    import shutil
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    original_mb = os.path.getsize(KERAS_MODEL) / (1024 * 1024)
    onnx_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    reduction = (1 - onnx_mb / original_mb) * 100

    print(f"\n✅ Conversion complete!")
    print(f"   Original:  {original_mb:.1f} MB  ({KERAS_MODEL})")
    print(f"   ONNX:      {onnx_mb:.1f} MB  ({OUTPUT_FILE})")
    print(f"   Reduction: {reduction:.1f}%")
    print(f"\n   Runtime: onnxruntime (~30 MB) vs tensorflow (~1,800 MB)")


if __name__ == "__main__":
    main()
