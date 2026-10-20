"""TFLite bird-presence and species inference with durable result records."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

try:  # Bundles libjpeg-turbo and exposes the DCT/upsampling flags TensorFlow uses.
    import simplejpeg as _simplejpeg
except ImportError:  # Pillow fallback: accurate DCT, so pixels differ slightly from training.
    _simplejpeg = None


INFERENCE_SCHEMA_VERSION = 2
DEFAULT_IMAGE_SIZE = 224
CONTRACT_FILENAME = "inference_contract.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interpreter_class():
    """Return an available TFLite Interpreter without requiring full TensorFlow on Pi."""
    errors: list[str] = []
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter, "ai_edge_litert"
    except ImportError as error:
        errors.append(f"ai-edge-litert: {error}")
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter, "tflite_runtime"
    except ImportError as error:
        errors.append(f"tflite-runtime: {error}")
    try:
        from tensorflow.lite import Interpreter

        return Interpreter, "tensorflow"
    except (ImportError, OSError) as error:
        errors.append(f"tensorflow: {error}")
    raise RuntimeError(
        "No usable TFLite interpreter is installed. Install ai-edge-litert or "
        "tflite-runtime for the Raspberry Pi. Details: " + "; ".join(errors)
    )


def jpeg_decoder_name() -> str:
    """Describe the JPEG decoder used for model inputs (recorded with each inference)."""
    if _simplejpeg is not None:
        return "simplejpeg libjpeg-turbo (fast integer DCT, fancy upsampling; matches tf.io.decode_jpeg)"
    return "Pillow (accurate integer DCT; install simplejpeg to match tf.io.decode_jpeg exactly)"


def decode_rgb(image_path: Path) -> np.ndarray:
    """Decode an image to an HxWx3 uint8 RGB array the way training did.

    Training used ``tf.io.decode_jpeg(..., channels=3)`` with its defaults: libjpeg's
    fast integer IDCT with fancy chroma upsampling, and no EXIF orientation. When
    ``simplejpeg`` is installed JPEGs are decoded with exactly those settings
    (bit-identical to TensorFlow); otherwise Pillow is used, whose accurate IDCT
    shifts pixels by about one level on average. Grayscale input is expanded to RGB.
    """
    image_path = Path(image_path)
    data = image_path.read_bytes()
    rgb: np.ndarray | None = None
    if _simplejpeg is not None and data[:2] == b"\xff\xd8":
        try:
            rgb = _simplejpeg.decode_jpeg(
                data, colorspace="RGB", fastdct=True, fastupsample=False
            )
        except ValueError:
            rgb = None  # e.g. CMYK JPEG; let Pillow handle it.
    if rgb is None:
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("Image has invalid dimensions.")
    return rgb


@lru_cache(maxsize=32)
def _antialiased_bilinear_spans(input_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Spans of TensorFlow's ``scale_and_translate`` triangle (antialiased bilinear) kernel.

    This is what ``tf.image.resize(..., method="bilinear", antialias=True)`` runs
    (scale = output/input, no translation). Arithmetic is float32 and sequential,
    as in the TensorFlow CPU kernel. Returns (starts[out], weights[out, span]),
    where unused trailing weights are zero.
    """
    f32 = np.float32
    scale = f32(output_size) / f32(input_size)
    inv_scale = f32(1.0) / scale
    kernel_scale = max(inv_scale, f32(1.0))
    one_over_kernel_scale = f32(1.0) / kernel_scale
    radius = f32(1.0)
    span_size = min(2 * int(np.ceil(radius * kernel_scale)) + 1, input_size)
    starts = np.zeros(output_size, dtype=np.intp)
    weights = np.zeros((output_size, span_size), dtype=np.float32)
    for x in range(output_size):
        sample = (f32(x) + f32(0.5)) * inv_scale
        span_start = int(np.ceil(sample - radius * kernel_scale - f32(0.5)))
        span_end = int(np.floor(sample + radius * kernel_scale - f32(0.5)))
        span_start = min(max(span_start, 0), input_size - 1)
        span_end = min(max(span_end, 0), input_size - 1) + 1
        row = []
        total = f32(0.0)
        for source in range(span_start, span_end):
            position = (f32(source) + f32(0.5) - sample) * one_over_kernel_scale
            distance = abs(position)
            weight = f32(1.0) - distance if distance < f32(1.0) else f32(0.0)
            total = f32(total + weight)
            row.append(f32(weight))
        if abs(float(total)) >= 1000.0 * np.finfo(np.float32).tiny:
            inverse = f32(1.0) / total
            row = [f32(weight * inverse) for weight in row]
        starts[x] = span_start
        weights[x, : len(row)] = row
    return starts, weights


def _resize_axis0(image: np.ndarray, output_size: int) -> np.ndarray:
    """Resize along axis 0, accumulating ``out += w * in`` in TensorFlow's order."""
    input_size = image.shape[0]
    starts, weights = _antialiased_bilinear_spans(input_size, output_size)
    output = np.zeros((output_size,) + image.shape[1:], dtype=np.float32)
    extra = (1,) * (image.ndim - 1)
    for offset in range(weights.shape[1]):
        index = np.minimum(starts + offset, input_size - 1)
        # Zero weights beyond a span's real end contribute exactly 0.
        output += weights[:, offset].reshape((output_size,) + extra) * image[index]
    return output


def tf_resize_with_pad_geometry(height: int, width: int, size: int) -> tuple[int, int, int, int]:
    """Return (resized_h, resized_w, pad_top, pad_left) as ``tf.image.resize_with_pad``.

    TensorFlow computes these in float32 and floors both the resized size and the
    padding offset; reproducing that exactly keeps the image on the same pixel grid.
    """
    f32 = np.float32
    f_height, f_width, f_target = f32(height), f32(width), f32(size)
    ratio = max(f_width / f_target, f_height / f_target)
    resized_height_f = f_height / ratio
    resized_width_f = f_width / ratio
    resized_height = max(1, int(np.floor(resized_height_f)))
    resized_width = max(1, int(np.floor(resized_width_f)))
    pad_top = max(0, int(np.floor((f_target - resized_height_f) / f32(2.0))))
    pad_left = max(0, int(np.floor((f_target - resized_width_f) / f32(2.0))))
    return resized_height, resized_width, pad_top, pad_left


def letterbox_array(rgb: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Letterbox an RGB array exactly as the training notebooks do.

    Equivalent to ``clip(tf.image.resize_with_pad(float32(rgb), size, size,
    antialias=True), 0, 255)``. Returns float32 pixels in [0, 255]; any
    quantization is applied afterwards from the model's own input tensor.
    """
    if image_size < 1:
        raise ValueError("image_size must be positive")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB array, got shape {rgb.shape}.")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    resized_height, resized_width, pad_top, pad_left = tf_resize_with_pad_geometry(
        height, width, image_size
    )
    image = rgb.astype(np.float32)
    # TensorFlow resizes height first (into an output_h x input_w buffer), then width.
    resized = _resize_axis0(image, resized_height)
    resized = np.swapaxes(_resize_axis0(np.swapaxes(resized, 0, 1), resized_width), 0, 1)
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = resized
    return np.clip(canvas, 0.0, 255.0)


def letterbox_rgb(image_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Decode an image and letterbox it on black with TensorFlow-identical geometry."""
    return letterbox_array(decode_rgb(Path(image_path)), image_size)


def _quantize(values: np.ndarray, dtype: np.dtype, quantization: tuple[float, int]) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"Unsupported TFLite input dtype: {dtype}")
    scale, zero_point = quantization
    if scale <= 0:
        raise ValueError(f"Integer TFLite input requires a positive quantization scale, got {scale}.")
    limits = np.iinfo(dtype)
    quantized = np.rint(values / scale + zero_point)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def _dequantize(values: np.ndarray, quantization: tuple[float, int]) -> np.ndarray:
    if np.issubdtype(values.dtype, np.floating):
        return values.astype(np.float32)
    scale, zero_point = quantization
    if scale <= 0:
        raise ValueError(f"Integer TFLite output requires a positive quantization scale, got {scale}.")
    return (values.astype(np.float32) - zero_point) * scale


class TFLiteModel:
    """Validated, thread-safe wrapper around one fixed-shape image classifier."""

    def __init__(
        self,
        model_path: Path,
        *,
        output_size: int,
        image_size: int = DEFAULT_IMAGE_SIZE,
        interpreter_factory: Callable[..., Any] | None = None,
        num_threads: int | None = None,
    ) -> None:
        self.path = Path(model_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {self.path}")
        if interpreter_factory is None:
            interpreter_factory, self.runtime = _interpreter_class()
        else:
            self.runtime = "injected"
        if num_threads is not None:
            self.interpreter = interpreter_factory(model_path=str(self.path), num_threads=num_threads)
        else:
            self.interpreter = interpreter_factory(model_path=str(self.path))
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"{self.path.name} must have exactly one input and one output tensor.")
        self.input = inputs[0]
        self.output = outputs[0]
        expected_input = (1, image_size, image_size, 3)
        actual_input = tuple(int(value) for value in self.input["shape"])
        if actual_input != expected_input:
            raise ValueError(
                f"{self.path.name} input shape is {actual_input}; expected {expected_input}."
            )
        actual_output = int(np.prod(self.output["shape"]))
        if actual_output != output_size:
            raise ValueError(
                f"{self.path.name} has {actual_output} output values; expected {output_size}."
            )
        self.image_size = image_size
        self.output_size = output_size
        self.sha256 = _sha256(self.path)
        self._lock = threading.Lock()

    @property
    def input_quantization(self) -> tuple[float, int]:
        scale, zero_point = self.input.get("quantization", (0.0, 0))
        return float(scale), int(zero_point)

    @property
    def output_quantization(self) -> tuple[float, int]:
        scale, zero_point = self.output.get("quantization", (0.0, 0))
        return float(scale), int(zero_point)

    @property
    def output_is_quantized(self) -> bool:
        return bool(np.issubdtype(np.dtype(self.output["dtype"]), np.integer))

    def predict_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Return dequantized output values for float [0, 255] RGB pixels."""
        return self.predict(pixels)[1]

    def predict(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (raw output tensor values, dequantized float32 values)."""
        expected = (self.image_size, self.image_size, 3)
        if pixels.shape != expected:
            raise ValueError(f"Preprocessed image shape is {pixels.shape}; expected {expected}.")
        tensor = _quantize(
            pixels[np.newaxis, ...].astype(np.float32),
            self.input["dtype"],
            tuple(self.input.get("quantization", (0.0, 0))),
        )
        with self._lock:
            self.interpreter.set_tensor(self.input["index"], tensor)
            self.interpreter.invoke()
            raw = np.asarray(self.interpreter.get_tensor(self.output["index"]))
        raw = raw.reshape(-1)
        values = _dequantize(raw, tuple(self.output.get("quantization", (0.0, 0)))).reshape(-1)
        if values.size != self.output_size or not np.all(np.isfinite(values)):
            raise ValueError(f"{self.path.name} returned invalid output values.")
        return raw, values

    def describe(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "runtime": self.runtime,
            "input_dtype": np.dtype(self.input["dtype"]).name,
            "input_quantization": list(self.input_quantization),
            "output_dtype": np.dtype(self.output["dtype"]).name,
            "output_quantization": list(self.output_quantization),
            "image_size": self.image_size,
        }


class InferenceContract:
    """The deployment contract a training notebook writes beside an is-bird model.

    ``inference_contract.json`` is the authority for preprocessing and the decision
    rule. Loading validates it against what this receiver implements, and
    :meth:`validate_model` checks it against the loaded TFLite tensors, so a
    mismatched model/contract pair fails at startup instead of mis-scoring birds.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{self.path} must contain a JSON object.")
        self.data = data
        self.sha256 = _sha256(self.path)
        problems: list[str] = []

        def expect(key: str, allowed: Callable[[Any], bool], description: str) -> None:
            if key in data and not allowed(data[key]):
                problems.append(f"{key}={data[key]!r} (receiver supports {description})")

        expect("geometry", lambda v: v == "letterbox", "'letterbox' only")
        expect("geometry_location", lambda v: v == "outside the model", "'outside the model'")
        expect("color_order", lambda v: v == "RGB", "'RGB'")
        expect("jpeg_decode_channels", lambda v: v == 3, "3")
        expect("letterbox_padding_value", lambda v: v == 0, "0 (black)")
        expect("resize_method", lambda v: "bilinear" in str(v).lower(), "TensorFlow bilinear")
        expect("resize_antialias", lambda v: v is True, "true")
        expect(
            "numeric_scaling",
            lambda v: "inside the model" in str(v),
            "scaling embedded in the model",
        )
        labels = data.get("labels")
        if labels is not None and str((labels or {}).get("1", "")).lower() != "bird":
            problems.append(f"labels={labels!r} (output index 1 must be 'bird')")
        if problems:
            raise ValueError(f"Unsupported inference contract {self.path}: " + "; ".join(problems))

        shape = data.get("input_shape")
        self.image_size = int(shape[1]) if isinstance(shape, list) and len(shape) == 4 else DEFAULT_IMAGE_SIZE
        threshold = data.get("bird_probability_threshold")
        self.bird_probability_threshold = None if threshold is None else float(threshold)
        raw = data.get("uint8_output_threshold")
        self.raw_output_threshold = None if raw is None else int(raw)

    @classmethod
    def beside(cls, model_path: Path) -> "InferenceContract | None":
        candidate = Path(model_path).expanduser().resolve().with_name(CONTRACT_FILENAME)
        return cls(candidate) if candidate.is_file() else None

    def validate_model(self, model: TFLiteModel) -> None:
        data = self.data
        named = data.get("model_file", "the contract's model")
        problems: list[str] = []
        if "input_shape" in data:
            actual = [int(value) for value in model.input["shape"]]
            if actual != [int(value) for value in data["input_shape"]]:
                problems.append(f"input shape {actual} != {data['input_shape']}")
        for side, detail, quantization in (
            ("input", model.input, model.input_quantization),
            ("output", model.output, model.output_quantization),
        ):
            dtype_key, quant_key = f"{side}_dtype", f"{side}_quantization"
            actual_dtype = np.dtype(detail["dtype"]).name
            if dtype_key in data and actual_dtype != data[dtype_key]:
                problems.append(f"{side} dtype {actual_dtype} != {data[dtype_key]}")
            if quant_key in data and np.issubdtype(np.dtype(detail["dtype"]), np.integer):
                scale, zero_point = (float(data[quant_key][0]), int(data[quant_key][1]))
                if not (np.isclose(quantization[0], scale, rtol=1e-6, atol=0) and quantization[1] == zero_point):
                    problems.append(f"{side} quantization {list(quantization)} != {[scale, zero_point]}")
        if problems:
            raise ValueError(
                f"{model.path.name} does not match {self.path.name} (which describes {named}): "
                + "; ".join(problems)
            )
        if self.raw_output_threshold is not None and model.output_is_quantized:
            limits = np.iinfo(np.dtype(model.output["dtype"]))
            if not limits.min <= self.raw_output_threshold <= limits.max:
                raise ValueError(f"uint8_output_threshold {self.raw_output_threshold} is out of range.")
            if self.bird_probability_threshold is not None:
                scale, zero_point = model.output_quantization
                implied = (self.raw_output_threshold - zero_point) * scale
                if abs(implied - self.bird_probability_threshold) > scale / 2 + 1e-9:
                    raise ValueError(
                        f"Contract thresholds disagree: uint8 {self.raw_output_threshold} implies "
                        f"{implied:.6f}, not {self.bird_probability_threshold}."
                    )

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "model_file": self.data.get("model_file"),
            "geometry": self.data.get("geometry"),
            "bird_probability_threshold": self.bird_probability_threshold,
            "uint8_output_threshold": self.raw_output_threshold,
        }


class BirdInferenceEngine:
    """Run bird-presence and species models over the same letterboxed pixels.

    The is-bird decision follows the model's ``inference_contract.json`` when one
    is present: for a quantized output the exact integer rule
    ``raw_output >= uint8_output_threshold`` (the rule the threshold was selected
    with), otherwise ``bird_probability >= bird_probability_threshold``. Older
    runs without a contract fall back to ``threshold.json``. ``bird_threshold``
    overrides both and always uses the probability rule.
    """

    def __init__(
        self,
        is_bird_model: Path,
        species_model: Path,
        *,
        species_labels: Path | None = None,
        threshold_file: Path | None = None,
        contract_file: Path | None = None,
        bird_threshold: float | None = None,
        top_k: int = 5,
        num_threads: int | None = None,
        interpreter_factory: Callable[..., Any] | None = None,
    ) -> None:
        is_bird_model = Path(is_bird_model).expanduser().resolve()
        species_model = Path(species_model).expanduser().resolve()
        species_labels = (
            Path(species_labels).expanduser().resolve()
            if species_labels is not None
            else species_model.with_name("labels.txt")
        )
        threshold_file = (
            Path(threshold_file).expanduser().resolve()
            if threshold_file is not None
            else is_bird_model.with_name("threshold.json")
        )
        if contract_file is not None:
            contract_path = Path(contract_file).expanduser().resolve()
            if not contract_path.is_file():
                raise FileNotFoundError(f"Inference contract not found: {contract_path}")
            self.contract: InferenceContract | None = InferenceContract(contract_path)
        else:
            self.contract = InferenceContract.beside(is_bird_model)

        self.labels = [line.strip() for line in species_labels.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.labels:
            raise ValueError(f"Species label file is empty: {species_labels}")
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        self.top_k = min(top_k, len(self.labels))

        image_size = self.contract.image_size if self.contract is not None else DEFAULT_IMAGE_SIZE
        self.is_bird = TFLiteModel(
            is_bird_model,
            output_size=1,
            image_size=image_size,
            interpreter_factory=interpreter_factory,
            num_threads=num_threads,
        )
        if self.contract is not None:
            self.contract.validate_model(self.is_bird)
        self.species = TFLiteModel(
            species_model,
            output_size=len(self.labels),
            image_size=image_size,
            interpreter_factory=interpreter_factory,
            num_threads=num_threads,
        )

        threshold_data: dict[str, Any] = {}
        if threshold_file.is_file():
            loaded = json.loads(threshold_file.read_text(encoding="utf-8"))
            threshold_data = loaded if isinstance(loaded, dict) else {}
        self.raw_output_threshold: int | None = None
        if bird_threshold is not None:
            self.threshold_source = "override"
        elif self.contract is not None and self.contract.bird_probability_threshold is not None:
            bird_threshold = self.contract.bird_probability_threshold
            self.raw_output_threshold = self.contract.raw_output_threshold
            self.threshold_source = str(self.contract.path)
            file_threshold = threshold_data.get("bird_probability_threshold")
            if file_threshold is not None and not np.isclose(float(file_threshold), bird_threshold):
                raise ValueError(
                    f"{threshold_file.name} threshold {file_threshold} disagrees with "
                    f"{self.contract.path.name} threshold {bird_threshold}."
                )
        elif "bird_probability_threshold" in threshold_data:
            bird_threshold = float(threshold_data["bird_probability_threshold"])
            if threshold_data.get("uint8_output_threshold") is not None:
                self.raw_output_threshold = int(threshold_data["uint8_output_threshold"])
            self.threshold_source = str(threshold_file)
        else:
            raise FileNotFoundError(
                f"No bird threshold: expected {CONTRACT_FILENAME} or {threshold_file.name} beside "
                f"{is_bird_model.name}, or pass bird_threshold."
            )
        if not 0.0 <= bird_threshold <= 1.0:
            raise ValueError("bird_threshold must be between 0 and 1.")
        if not self.is_bird.output_is_quantized:
            self.raw_output_threshold = None  # Integer rule only applies to quantized outputs.
        self.bird_threshold = float(bird_threshold)
        self.species_labels_path = species_labels
        self.threshold_path = threshold_file

    @property
    def decision_rule(self) -> str:
        if self.raw_output_threshold is not None:
            return f"raw_output >= {self.raw_output_threshold}"
        return f"bird_probability >= {self.bird_threshold}"

    def infer(self, image_path: Path) -> dict[str, Any]:
        image_path = Path(image_path)
        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        pixels = letterbox_rgb(image_path, self.is_bird.image_size)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        bird_started = time.perf_counter()
        bird_raw, bird_values = self.is_bird.predict(pixels)
        bird_ms = (time.perf_counter() - bird_started) * 1000
        bird_probability = min(1.0, max(0.0, float(bird_values[0])))
        raw_output = int(bird_raw[0]) if self.is_bird.output_is_quantized else None
        if self.raw_output_threshold is not None:
            is_bird = raw_output >= self.raw_output_threshold
        else:
            is_bird = bird_probability >= self.bird_threshold

        species_started = time.perf_counter()
        species_scores = self.species.predict_pixels(pixels)
        species_ms = (time.perf_counter() - species_started) * 1000
        order = np.argsort(-species_scores, kind="stable")[: self.top_k]
        top = [
            {"label": self.labels[int(index)], "probability": float(species_scores[int(index)])}
            for index in order
        ]
        result = {
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "status": "complete",
            "completed_at": _utc_now(),
            "image": {
                "filename": image_path.name,
                "sha256": _sha256(image_path),
            },
            "is_bird": bool(is_bird),
            "bird_probability": bird_probability,
            "bird_threshold": self.bird_threshold,
            "bird_decision": {
                "rule": self.decision_rule,
                "raw_output": raw_output,
                "raw_output_threshold": self.raw_output_threshold,
                "threshold_source": self.threshold_source,
            },
            "guess": top[0]["label"],
            "species_probability": top[0]["probability"],
            "species_top_k": top,
            "models": {
                "is_bird": self.is_bird.describe(),
                "species": self.species.describe(),
            },
            "contract": self.contract.describe() if self.contract is not None else None,
            "preprocessing": {
                "method": "letterbox",
                "implementation": "TensorFlow resize_with_pad semantics, antialiased bilinear",
                "size": self.is_bird.image_size,
                "color": "RGB",
                "padding": "black",
                "exif_orientation": "ignored (as tf.io.decode_jpeg)",
                "jpeg_decoder": jpeg_decoder_name(),
                "input_range": "float 0-255, quantized per model input tensor; model performs scaling",
            },
            "timing_ms": {
                "preprocess": preprocess_ms,
                "is_bird": bird_ms,
                "species": species_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        }
        result["classification"] = {
            "is_bird": result["is_bird"],
            "confidence": bird_probability,
            "label": result["guess"],
        }
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "ready": True,
            "bird_threshold": self.bird_threshold,
            "bird_decision_rule": self.decision_rule,
            "threshold_source": self.threshold_source,
            "top_k": self.top_k,
            "species_count": len(self.labels),
            "jpeg_decoder": jpeg_decoder_name(),
            "contract": self.contract.describe() if self.contract is not None else None,
            "is_bird": self.is_bird.describe(),
            "species": self.species.describe(),
        }


def inference_path(image_path: Path) -> Path:
    return image_path.with_suffix(".inference.json")


def write_inference_record(image_path: Path, record: dict[str, Any]) -> Path:
    """Atomically write derived inference without mutating acquisition metadata."""
    destination = inference_path(image_path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_inference_record(image_path: Path) -> dict[str, Any] | None:
    path = inference_path(image_path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None