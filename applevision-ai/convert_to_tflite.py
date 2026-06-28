#!/usr/bin/env python3
"""
Convert the EfficientNetV2S .keras model to TFLite format.

Converts to pure TFLite built-in ops only (no SELECT_TF_OPS) so that
the model can run with the lightweight tflite-runtime package (~5 MB)
without needing full TensorFlow at runtime.

Usage:
    python convert_to_tflite.py
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
KERAS_MODEL = os.path.join(REPO_ROOT, "apple-variety-streamlit", "model", "apple_classifier_final.keras")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ml", "model")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "apple_classifier.tflite")


def main():
    if not os.path.exists(KERAS_MODEL):
        print(f"ERROR: Keras model not found at {KERAS_MODEL}")
        sys.exit(1)

    print(f"Loading Keras model from: {KERAS_MODEL}")
    model = tf.keras.models.load_model(KERAS_MODEL)
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Parameters:   {model.count_params():,}")

    # Get a concrete function with a fixed input shape for TFLite conversion
    # This avoids dynamic shape issues and produces a cleaner TFLite graph
    input_shape = model.input_shape  # (None, 224, 224, 3)
    concrete_func = tf.function(lambda x: model(x, training=False))
    concrete_func = concrete_func.get_concrete_function(
        tf.TensorSpec([1, input_shape[1], input_shape[2], input_shape[3]], tf.float32)
    )

    # Convert using the concrete function — TFLite built-in ops only
    print("\nConverting to TFLite (built-in ops only, dynamic range quantization)...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Only use TFLite built-in ops — no SELECT_TF_OPS dependency
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"\nBuilt-in ops only failed: {e}")
        print("Retrying with SELECT_TF_OPS fallback...")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        print("⚠️  Model uses SELECT_TF_OPS — requires tensorflow at runtime.")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        f.write(tflite_model)

    original_mb = os.path.getsize(KERAS_MODEL) / (1024 * 1024)
    tflite_mb = len(tflite_model) / (1024 * 1024)
    reduction = (1 - tflite_mb / original_mb) * 100

    print(f"\n✅ Conversion complete!")
    print(f"   Original:  {original_mb:.1f} MB  ({KERAS_MODEL})")
    print(f"   TFLite:    {tflite_mb:.1f} MB  ({OUTPUT_FILE})")
    print(f"   Reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
