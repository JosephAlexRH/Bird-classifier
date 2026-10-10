"""Headless ESP32-CAM image receiver intended for Raspberry Pi deployment."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file

try:
    from .capture_storage import METADATA_HEADER, CaptureStore, is_jpeg, parse_camera_metadata
except ImportError:  # Direct execution: python server/raspberry_pi_server.py
    from capture_storage import METADATA_HEADER, CaptureStore, is_jpeg, parse_camera_metadata


DEFAULT_SAVE_DIR = Path.home() / "bird_images"
MAX_JPEG_BYTES = 10 * 1024 * 1024


def classify(image_path: Path) -> dict[str, Any] | None:
    """Run inference and return is_bird, confidence (0-1), and an optional label.

    Replace this stub with the deployed model. Returning ``None`` means that no
    classification decision is available, so the ESP32 will not save the frame.
    """
    del image_path
    return None


def validate_classification(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize classifier output into the contract consumed by the ESP32."""
    if value is None:
        return None, None
    if not isinstance(value, dict):
        return None, "Classifier result must be an object or null."
    if type(value.get("is_bird")) is not bool:
        return None, "Classifier result must contain a boolean is_bird field."
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None, "Classifier result must contain a numeric confidence field."
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        return None, "Classifier confidence must be between 0 and 1."
    result: dict[str, Any] = {
        "is_bird": value["is_bird"],
        "confidence": confidence,
    }
    if "label" in value:
        if not isinstance(value["label"], str):
            return None, "Classifier label must be a string when supplied."
        result["label"] = value["label"][:120]
    return result, None


def create_app(save_dir: Path = DEFAULT_SAVE_DIR) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_JPEG_BYTES
    store = CaptureStore(save_dir)
    state_lock = threading.Lock()
    existing = sorted(store.root.glob("*.jpg"), key=lambda path: path.stat().st_mtime)
    latest: dict[str, Any] = {
        "path": existing[-1] if existing else None,
        "received_at": None,
        "classification": None,
    }

    @app.post("/upload")
    def upload() -> tuple[Response, int] | Response:
        image = request.get_data(cache=False)
        if not image:
            return jsonify(error="Request body is empty; expected raw JPEG bytes."), 400
        if not is_jpeg(image):
            return jsonify(error="Request body is not a complete JPEG image."), 415

        camera_metadata, metadata_error = parse_camera_metadata(request.headers.get(METADATA_HEADER))
        classification: dict[str, Any] | None = None
        classification_error: str | None = None
        # Save first so classification always receives a durable image path.
        record = store.save_capture(
            image,
            source=(camera_metadata or {}).get("source", "motion"),
            camera_metadata=camera_metadata,
            metadata_error=metadata_error,
        )
        destination = store.root / record["filename"]
        try:
            classification, classification_error = validate_classification(classify(destination))
        except Exception as error:  # Inference failure must not discard a valid field image.
            classification_error = str(error)

        # Classification is returned in live state/API. The immutable capture sidecar
        # records acquisition facts; inference systems can write their own versioned output.
        with state_lock:
            latest.update(
                path=destination,
                received_at=record["received_at"],
                classification=classification,
            )

        print(f"Saved: {destination}", flush=True)
        return jsonify(
            status="ok",
            classification=classification,
            classification_error=classification_error,
            **record,
        )

    @app.get("/latest.jpg")
    def latest_image() -> Response | tuple[Response, int]:
        with state_lock:
            path = latest["path"]
        if path is None or not path.exists():
            return jsonify(error="No image has been uploaded yet."), 404
        return send_file(path, mimetype="image/jpeg", max_age=0)

    @app.get("/health")
    def health() -> Response:
        with state_lock:
            path = latest["path"]
            received_at = latest["received_at"]
            classification = latest["classification"]
        return jsonify(
            status="ok",
            save_dir=str(store.root),
            latest_image=path.name if path else None,
            received_at=received_at,
            classification=classification,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: all interfaces)")
    parser.add_argument("--port", type=int, default=5000, help="Listening port (default: 5000)")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="Dataset folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    application = create_app(args.save_dir)
    application.run(host=args.host, port=args.port, debug=False, threaded=True)