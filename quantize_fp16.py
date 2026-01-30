"""
Float16 post-training quantization for a TensorFlow SavedModel using
MediaPipe Model Maker's QuantizationConfig helper.

Example:
    python quantize_fp16.py --saved_model path/to/saved_model \
        --out pose_landmarker_fp16.tflite
"""

import argparse
from pathlib import Path

import tensorflow as tf
from mediapipe_model_maker import quantization


def quantize(saved_model_dir: Path, output_path: Path) -> None:
    """Convert SavedModel to float16 TFLite."""
    quant_config = quantization.QuantizationConfig.for_float16()
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    quant_config.set_converter_with_quantization(converter)

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"Saved {output_path} ({len(tflite_model)/1024/1024:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Float16 post-training quantization.")
    parser.add_argument(
        "--saved_model",
        required=True,
        type=Path,
        help="Path to the TensorFlow SavedModel directory (not a .task file).",
    )
    parser.add_argument(
        "--out",
        default=Path("pose_landmarker_fp16.tflite"),
        type=Path,
        help="Where to write the float16 TFLite model.",
    )
    args = parser.parse_args()
    quantize(args.saved_model, args.out)


if __name__ == "__main__":
    main()
