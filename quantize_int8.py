"""
Full-integer (int8) post-training quantization for a TensorFlow SavedModel.
Uses MediaPipe Model Maker's QuantizationConfig with a small representative
image set to calibrate ranges.

Example:
    python quantize_int8.py --saved_model path/to/saved_model \\
        --rep_data data/representative_frames \\
        --out pose_landmarker_int8.tflite
"""

import argparse
from pathlib import Path
from typing import Iterable

import tensorflow as tf
from mediapipe_model_maker import quantization

# Pose landmarker expects 256x256 input; override with --input_size if needed.
DEFAULT_INPUT_SIZE = 256


def _make_representative_tf_dataset(
    image_dir: Path, input_size: int
) -> tf.data.Dataset:
    """Build a tf.data.Dataset of preprocessed images for calibration."""
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No .jpg/.jpeg/.png images found in {image_dir} for calibration"
        )

    def _gen() -> Iterable[tf.Tensor]:
        for path in image_paths:
            img = tf.io.read_file(str(path))
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, (input_size, input_size))
            img = tf.cast(img, tf.float32) / 255.0
            yield img

    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=tf.TensorSpec(
            shape=(input_size, input_size, 3), dtype=tf.float32
        ),
    )


class RepresentativeDataAdapter:
    """
    Minimal adapter that provides the `gen_tf_dataset` method expected by
    QuantizationConfig. It wraps a tf.data.Dataset.
    """

    def __init__(self, tf_dataset: tf.data.Dataset):
        self._ds = tf_dataset

    def gen_tf_dataset(
        self,
        batch_size: int = 1,
        is_training: bool = False,
        shuffle: bool = False,
        preprocess=None,
        drop_remainder: bool = False,
    ) -> tf.data.Dataset:
        ds = self._ds
        if preprocess:
            ds = ds.map(
                lambda x: preprocess(x, label=None, is_training=is_training),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        if shuffle:
            ds = ds.shuffle(buffer_size=32)
        return ds.batch(batch_size, drop_remainder=drop_remainder)


def quantize(
    saved_model_dir: Path,
    representative_dir: Path,
    output_path: Path,
    input_size: int = DEFAULT_INPUT_SIZE,
    quant_steps: int | None = None,
) -> None:
    """Convert SavedModel to int8 TFLite using representative calibration data."""
    rep_ds = _make_representative_tf_dataset(representative_dir, input_size)
    rep_adapter = RepresentativeDataAdapter(rep_ds)

    int8_kwargs = dict(
        representative_data=rep_adapter,
        inference_input_type=tf.uint8,
        inference_output_type=tf.uint8,
    )
    if quant_steps is not None:
        int8_kwargs["quantization_steps"] = quant_steps

    quant_config = quantization.QuantizationConfig.for_int8(**int8_kwargs)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    quant_config.set_converter_with_quantization(converter)

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"Saved {output_path} ({len(tflite_model)/1024/1024:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Int8 post-training quantization.")
    parser.add_argument(
        "--saved_model",
        required=True,
        type=Path,
        help="Path to the TensorFlow SavedModel directory (not a .task file).",
    )
    parser.add_argument(
        "--rep_data",
        required=True,
        type=Path,
        help="Directory of representative .jpg/.jpeg/.png frames for calibration.",
    )
    parser.add_argument(
        "--out",
        default=Path("pose_landmarker_int8.tflite"),
        type=Path,
        help="Where to write the int8 TFLite model.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Square size to resize images to (default 256 for pose landmarker).",
    )
    parser.add_argument(
        "--quant_steps",
        type=int,
        default=None,
        help="Optional number of calibration steps (defaults to library constant).",
    )
    args = parser.parse_args()
    quantize(
        saved_model_dir=args.saved_model,
        representative_dir=args.rep_data,
        output_path=args.out,
        input_size=args.input_size,
        quant_steps=args.quant_steps,
    )


if __name__ == "__main__":
    main()
