"""Headless ESP32-CAM image receiver intended for Raspberry Pi deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, render_template_string, request, send_file

try:
    from .bird_inference import (
        BirdInferenceEngine,
        inference_path,
        read_inference_record,
        write_inference_record,
    )
    from .capture_storage import METADATA_HEADER, CaptureStore, is_jpeg, parse_camera_metadata
    from .laptop_test_server import camera_url_argument, normalize_camera_url
    from .roi_control import ROI_CONTROL_JS
except ImportError:  # Direct execution: python server/raspberry_pi_server.py
    from bird_inference import (
        BirdInferenceEngine,
        inference_path,
        read_inference_record,
        write_inference_record,
    )
    from capture_storage import METADATA_HEADER, CaptureStore, is_jpeg, parse_camera_metadata
    from laptop_test_server import camera_url_argument, normalize_camera_url
    from roi_control import ROI_CONTROL_JS


DEFAULT_SAVE_DIR = Path.home() / "bird_images"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IS_BIRD_MODEL = (
    PROJECT_ROOT / "Models" / "runs" / "isbird_letterbox_224_20261010_173627" / "model_float32.tflite"
)
DEFAULT_SPECIES_MODEL = (
    PROJECT_ROOT / "Models" / "runs" / "letterbox_224_20261009_131631" / "model_float32.tflite"
)
MAX_JPEG_BYTES = 10 * 1024 * 1024
# Sent by the ESP32 when wait_for_classification is false in its config.
CLASSIFICATION_HEADER = "X-ESP32-Classification"
PROXY_TIMEOUT_SECONDS = 8


PI_DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32-CAM Raspberry Pi dashboard</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #101418; color: #edf2f7; }
    header, main { width: min(1000px, calc(100% - 2rem)); margin: auto; }
    header { padding: 1.5rem 0 .5rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; padding-bottom: 2rem; }
    section { background: #192128; border: 1px solid #2d3943; border-radius: 12px; padding: 1rem; }
    img { display: block; width: 100%; min-height: 240px; object-fit: contain; background: #080a0c; border-radius: 8px; }
    button { padding: .6rem .75rem; margin: .25rem .25rem .25rem 0; cursor: pointer; }
    pre { min-height: 5rem; max-height: 22rem; overflow: auto; white-space: pre-wrap; }
    .muted { color: #aab5c0; } .ok { color: #73d99f; } .error { color: #ff8b8b; }
    .roi-stage { position: relative; width: 100%; background: #080a0c; border-radius: 8px; overflow: hidden; }
    .roi-stage img { min-height: 0; height: auto; object-fit: fill; }
    .roi-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
  </style>
</head>
<body>
  <header>
    <h1>ESP32-CAM Raspberry Pi dashboard</h1>
    <p class="muted">Camera: {{ camera_url or 'not configured (uploads remain available)' }}</p>
  </header>
  <main class="grid">
    <section>
      <h2>Motion region of interest</h2>
      <div class="roi-stage">
        <img id="roi-preview" data-roi-preview alt="Refresh the camera preview, then drag to select a motion ROI">
        <canvas id="roi-overlay" aria-label="Drag to select a rectangular motion region"></canvas>
      </div>
      <p id="roi-values" class="muted">x 0.000, y 0.000, width 1.000, height 1.000</p>
      <button onclick="refreshRoiPreview()">Refresh ROI preview</button>
      <button onclick="loadMotionRoi()">Load current ROI</button>
      <button onclick="resetMotionRoi()">Reset full frame</button>
      <button onclick="saveMotionRoi()">Save ROI</button>
      <p class="muted">Drag with a mouse or touch. Save sends only the nested <code>motion_roi</code> update.</p>
    </section>
    <section>
      <h2>Camera controls</h2>
      <button onclick="setMode('motion')">Arm motion</button>
      <button onclick="setMode('idle')">Idle</button>
      <button onclick="cameraStatus()">Refresh status</button>
      <pre id="result">Choose an action.</pre>
    </section>
    <section>
      <h2>Latest classified upload</h2>
      <img id="latest-image" src="/latest.jpg" alt="No image has been uploaded yet">
      <pre id="inference">Waiting for a classified upload.</pre>
    </section>
  </main>
  <script>
    const result = document.getElementById('result');
    window.showCameraResult = (text, failed = false) => {
      result.className = failed ? 'error' : 'ok';
      try { result.textContent = JSON.stringify(JSON.parse(text), null, 2); }
      catch (_) { result.textContent = text; }
    };
    async function cameraStatus() {
      try {
        const response = await fetch('/camera/status', {cache: 'no-store'});
        const text = await response.text();
        if (!response.ok) throw new Error(text);
        showCameraResult(text);
      } catch (error) { showCameraResult(String(error), true); }
    }
    async function setMode(mode) {
      try {
        const response = await fetch('/camera/mode', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mode})
        });
        const text = await response.text();
        if (!response.ok) throw new Error(text);
        showCameraResult(text);
      } catch (error) { showCameraResult(String(error), true); }
    }
    async function refreshInference() {
      try {
        const response = await fetch('/health', {cache: 'no-store'});
        const health = await response.json();
        document.getElementById('inference').textContent = JSON.stringify({
          latest_image: health.latest_image,
          inference: health.latest_inference,
          queue_depth: health.classification_queue_depth,
          models: health.models
        }, null, 2);
        if (health.latest_image) {
          document.getElementById('latest-image').src = `/latest.jpg?t=${Date.now()}`;
        }
      } catch (error) {
        document.getElementById('inference').textContent = String(error);
      }
    }
    refreshInference();
    setInterval(refreshInference, 3000);
  </script>
  <script>{{ roi_control_js|safe }}</script>
</body>
</html>
"""


def classify(image_path: Path) -> dict[str, Any] | None:
    """Legacy injectable classifier hook used when no inference engine is supplied.

    Command-line deployment always constructs :class:`BirdInferenceEngine`; this
    safe hook remains for embedding and lightweight server-contract tests.
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


def create_app(
    save_dir: Path = DEFAULT_SAVE_DIR,
    camera_url: str | None = None,
    inference_engine: BirdInferenceEngine | None = None,
) -> Flask:
    """Create the receiver/classifier app, with optional ESP32 camera controls."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_JPEG_BYTES
    store = CaptureStore(save_dir)
    camera_url = normalize_camera_url(camera_url)
    camera_lock = threading.Lock()
    state_lock = threading.Lock()
    existing = sorted(store.root.glob("*.jpg"), key=lambda path: path.stat().st_mtime)
    latest: dict[str, Any] = {
        "path": existing[-1] if existing else None,
        "received_at": None,
        "classification": None,
        "inference": read_inference_record(existing[-1]) if existing else None,
    }
    pending: queue.Queue[Path] = queue.Queue()
    app.config["CLASSIFICATION_QUEUE"] = pending

    def failed_record(destination: Path, error: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "error",
            "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "capture_id": destination.stem,
            "image": {
                "filename": destination.name,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            },
            "error": error,
        }

    def run_classifier(
        destination: Path,
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
        try:
            if inference_engine is not None:
                inference = inference_engine.infer(destination)
                inference["capture_id"] = destination.stem
                classification, error = validate_classification(inference.get("classification"))
            else:
                classification, error = validate_classification(classify(destination))
                inference = {
                    "schema_version": 1,
                    "status": "complete" if error is None else "error",
                    "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "capture_id": destination.stem,
                    "image": {
                        "filename": destination.name,
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    },
                    "classification": classification,
                    "error": error,
                }
            if error:
                inference["status"] = "error"
                inference["error"] = error
            write_inference_record(destination, inference)
            return classification, error, inference
        except Exception as error:  # Inference failure must not discard a valid field image.
            inference = failed_record(destination, str(error))
            write_inference_record(destination, inference)
            return None, str(error), inference

    def classification_worker() -> None:
        # Deferred uploads are classified here, one at a time, so the camera never
        # waits on inference and a burst simply lengthens this queue.
        while True:
            destination = pending.get()
            try:
                classification, classification_error, inference = run_classifier(destination)
                with state_lock:
                    if latest["path"] == destination:
                        latest["classification"] = classification
                        latest["inference"] = inference
                print(
                    f"Classified (deferred): {destination.name} -> "
                    f"{classification if classification_error is None else classification_error}",
                    flush=True,
                )
            finally:
                pending.task_done()

    threading.Thread(target=classification_worker, name="classifier", daemon=True).start()

    # A process restart must not strand accepted captures whose in-memory queue was lost.
    if inference_engine is not None:
        for destination in existing:
            if not inference_path(destination).exists():
                pending.put(destination)

    def camera_proxy(endpoint: str, method: str = "GET", payload: dict[str, Any] | None = None):
        if not camera_url:
            return jsonify(error="Camera URL is not configured. Start the server with --camera-url."), 503
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        outgoing = Request(
            f"{camera_url}/{endpoint.lstrip('/')}", data=body, headers=headers, method=method
        )
        if not camera_lock.acquire(blocking=False):
            return jsonify(error="Camera is busy. Try again after the current request finishes."), 409
        try:
            with urlopen(outgoing, timeout=PROXY_TIMEOUT_SECONDS) as upstream:
                content = upstream.read(MAX_JPEG_BYTES + 1)
                if len(content) > MAX_JPEG_BYTES:
                    return jsonify(error="Camera response exceeded the 10 MiB limit."), 502
                response = Response(
                    content,
                    status=upstream.status,
                    content_type=upstream.headers.get_content_type(),
                )
                metadata = upstream.headers.get(METADATA_HEADER)
                if metadata:
                    response.headers[METADATA_HEADER] = metadata
                response.headers["Cache-Control"] = "no-store"
                return response
        except HTTPError as error:
            return Response(error.read(), status=error.code, content_type="text/plain")
        except (URLError, TimeoutError, OSError, ValueError) as error:
            return jsonify(error=f"Camera request to {outgoing.full_url} failed: {error}"), 502
        finally:
            camera_lock.release()

    @app.get("/")
    def dashboard() -> str:
        return render_template_string(
            PI_DASHBOARD_HTML, camera_url=camera_url, roi_control_js=ROI_CONTROL_JS
        )

    @app.get("/camera/capture")
    def camera_capture():
        return camera_proxy("capture")

    @app.get("/camera/status")
    def camera_status():
        return camera_proxy("status")

    @app.get("/camera/config")
    def camera_config_get():
        return camera_proxy("config")

    @app.post("/camera/config")
    def camera_config_post():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a valid JSON object."), 400
        return camera_proxy("config", "POST", payload)

    @app.post("/camera/mode")
    def camera_mode():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or payload.get("mode") not in {"motion", "idle"}:
            return jsonify(error="mode must be either motion or idle"), 422
        return camera_proxy("mode", "POST", payload)

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
        deferred = request.headers.get(CLASSIFICATION_HEADER, "").strip().lower() == "deferred"
        if deferred:
            with state_lock:
                latest.update(
                    path=destination,
                    received_at=record["received_at"],
                    classification=None,
                    inference=None,
                )
            pending.put(destination)
            print(f"Saved (classification queued): {destination}", flush=True)
            return (
                jsonify(
                    status="queued",
                    classification=None,
                    classification_error=None,
                    classification_deferred=True,
                    **record,
                ),
                202,
            )
        classification, classification_error, inference = run_classifier(destination)

        # Classification is returned in live state/API. The immutable capture sidecar
        # records acquisition facts; inference systems can write their own versioned output.
        with state_lock:
            latest.update(
                path=destination,
                received_at=record["received_at"],
                classification=classification,
                inference=inference,
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

    def capture_summary(path: Path) -> dict[str, Any]:
        sidecar_path = path.with_suffix(".json")
        sidecar: dict[str, Any] = {}
        try:
            value = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                sidecar = value
        except (OSError, json.JSONDecodeError):
            pass
        inference = read_inference_record(path)
        return {
            "capture_id": sidecar.get("capture_id", path.stem),
            "filename": path.name,
            "received_at": sidecar.get("received_at"),
            "source": sidecar.get("source"),
            "bytes": path.stat().st_size,
            "sha256": sidecar.get("sha256"),
            "inference_status": (inference or {}).get("status", "pending"),
            "inference": inference,
            "image_url": f"/api/captures/{path.stem}/image",
            "inference_url": f"/api/captures/{path.stem}/inference",
        }

    def capture_path(capture_id: str) -> Path | None:
        if not capture_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in capture_id):
            return None
        path = store.root / f"{capture_id}.jpg"
        return path if path.is_file() else None

    @app.get("/api/captures")
    def captures() -> Response:
        after = request.args.get("after", "")
        try:
            limit = max(1, min(200, int(request.args.get("limit", "50"))))
        except ValueError:
            return jsonify(error="limit must be an integer"), 400
        paths = sorted(store.root.glob("capture_*.jpg"), key=lambda item: item.name)
        if after:
            paths = [path for path in paths if path.stem > after]
        selected = paths[:limit]
        return jsonify(
            captures=[capture_summary(path) for path in selected],
            next_after=selected[-1].stem if selected else after or None,
            has_more=len(paths) > len(selected),
        )

    @app.get("/api/captures/<capture_id>")
    def capture_detail(capture_id: str):
        path = capture_path(capture_id)
        if path is None:
            return jsonify(error="Capture not found."), 404
        return jsonify(capture_summary(path))

    @app.get("/api/captures/<capture_id>/image")
    def capture_image(capture_id: str):
        path = capture_path(capture_id)
        if path is None:
            return jsonify(error="Capture not found."), 404
        return send_file(path, mimetype="image/jpeg", max_age=0)

    @app.get("/api/captures/<capture_id>/inference")
    def capture_inference(capture_id: str):
        path = capture_path(capture_id)
        if path is None:
            return jsonify(error="Capture not found."), 404
        inference = read_inference_record(path)
        if inference is None:
            return jsonify(status="pending", capture_id=capture_id), 202
        return jsonify(inference)

    @app.get("/health")
    def health() -> Response:
        with state_lock:
            path = latest["path"]
            received_at = latest["received_at"]
            classification = latest["classification"]
            latest_inference = latest["inference"]
        return jsonify(
            status="ok",
            save_dir=str(store.root),
            camera_url=camera_url,
            camera_configured=bool(camera_url),
            latest_image=path.name if path else None,
            received_at=received_at,
            classification=classification,
            latest_inference=latest_inference,
            classification_queue_depth=pending.unfinished_tasks,
            models=(inference_engine.describe() if inference_engine is not None else {"ready": False}),
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: all interfaces)")
    parser.add_argument("--port", type=int, default=5000, help="Listening port (default: 5000)")
    parser.add_argument(
        "--camera-url",
        type=camera_url_argument,
        help="ESP32-CAM HTTP(S) base URL, for example http://192.168.1.84",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="Dataset folder")
    parser.add_argument(
        "--is-bird-model",
        type=Path,
        default=DEFAULT_IS_BIRD_MODEL,
        help="Binary bird/not-bird TFLite model",
    )
    parser.add_argument(
        "--species-model",
        type=Path,
        default=DEFAULT_SPECIES_MODEL,
        help="Species TFLite model",
    )
    parser.add_argument("--species-labels", type=Path, help="Species labels (default: beside model)")
    parser.add_argument("--threshold-file", type=Path, help="Bird threshold JSON (default: beside model)")
    parser.add_argument("--bird-threshold", type=float, help="Override the model threshold")
    parser.add_argument("--top-k", type=int, default=5, help="Number of species guesses to retain")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        engine = BirdInferenceEngine(
            args.is_bird_model,
            args.species_model,
            species_labels=args.species_labels,
            threshold_file=args.threshold_file,
            bird_threshold=args.bird_threshold,
            top_k=args.top_k,
        )
    except Exception as error:
        raise SystemExit(f"Model startup error: {error}") from error
    print(f"Models ready: {json.dumps(engine.describe(), sort_keys=True)}", flush=True)
    application = create_app(args.save_dir, args.camera_url, engine)
    application.run(host=args.host, port=args.port, debug=False, threaded=True)