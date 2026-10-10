"""TFLite bird-presence and species inference with durable result records."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps


INFERENCE_SCHEMA_VERSION = 1
DEFAULT_IMAGE_SIZE = 224


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


def letterbox_rgb(image_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Decode an image, honor EXIF orientation, and letterbox it on a black square."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    with Image.open(image_path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        width, height = rgb.size
        if width < 1 or height < 1:
            raise ValueError("Image has invalid dimensions.")
        scale = min(image_size / width, image_size / height)
        resized_width = max(1, min(image_size, int(round(width * scale))))
        resized_height = max(1, min(image_size, int(round(height * scale))))
        resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        canvas.paste(
            resized,
            ((image_size - resized_width) // 2, (image_size - resized_height) // 2),
        )
        return np.asarray(canvas, dtype=np.uint8)


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
    ) -> None:
        self.path = Path(model_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {self.path}")
        if interpreter_factory is None:
            interpreter_factory, self.runtime = _interpreter_class()
        else:
            self.runtime = "injected"
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

    def predict_pixels(self, pixels: np.ndarray) -> np.ndarray:
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
        values = _dequantize(raw, tuple(self.output.get("quantization", (0.0, 0)))).reshape(-1)
        if values.size != self.output_size or not np.all(np.isfinite(values)):
            raise ValueError(f"{self.path.name} returned invalid output values.")
        return values

    def describe(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "runtime": self.runtime,
            "input_dtype": np.dtype(self.input["dtype"]).name,
            "output_dtype": np.dtype(self.output["dtype"]).name,
            "image_size": self.image_size,
        }


class BirdInferenceEngine:
    """Run bird-presence and species models over the same letterboxed pixels."""

    def __init__(
        self,
        is_bird_model: Path,
        species_model: Path,
        *,
        species_labels: Path | None = None,
        threshold_file: Path | None = None,
        bird_threshold: float | None = None,
        top_k: int = 5,
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
        self.labels = [line.strip() for line in species_labels.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.labels:
            raise ValueError(f"Species label file is empty: {species_labels}")
        if bird_threshold is None:
            threshold_data = json.loads(threshold_file.read_text(encoding="utf-8"))
            bird_threshold = float(threshold_data["bird_probability_threshold"])
        if not 0.0 <= bird_threshold <= 1.0:
            raise ValueError("bird_threshold must be between 0 and 1.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        self.bird_threshold = float(bird_threshold)
        self.top_k = min(top_k, len(self.labels))
        self.is_bird = TFLiteModel(
            is_bird_model,
            output_size=1,
            interpreter_factory=interpreter_factory,
        )
        self.species = TFLiteModel(
            species_model,
            output_size=len(self.labels),
            interpreter_factory=interpreter_factory,
        )
        self.species_labels_path = species_labels
        self.threshold_path = threshold_file

    def infer(self, image_path: Path) -> dict[str, Any]:
        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        pixels = letterbox_rgb(image_path, self.is_bird.image_size)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        bird_started = time.perf_counter()
        bird_probability = float(self.is_bird.predict_pixels(pixels)[0])
        bird_ms = (time.perf_counter() - bird_started) * 1000
        bird_probability = min(1.0, max(0.0, bird_probability))

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
            "is_bird": bird_probability >= self.bird_threshold,
            "bird_probability": bird_probability,
            "bird_threshold": self.bird_threshold,
            "guess": top[0]["label"],
            "species_probability": top[0]["probability"],
            "species_top_k": top,
            "models": {
                "is_bird": self.is_bird.describe(),
                "species": self.species.describe(),
            },
            "preprocessing": {
                "method": "letterbox",
                "size": self.is_bird.image_size,
                "color": "RGB",
                "padding": "black",
                "input_range": "0-255; model performs scaling",
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
            "top_k": self.top_k,
            "species_count": len(self.labels),
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