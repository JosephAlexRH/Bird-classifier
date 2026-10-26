"""Headless ESP32-CAM image receiver intended for Raspberry Pi deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        INFERENCE_SCHEMA_VERSION,
        SELECTED_CROP_GEOMETRY,
        SELECTED_CROP_SOURCE,
        decode_rgb,
        inference_path,
        jpeg_decoder_name,
        read_inference_record,
        write_inference_record,
    )
    from .camera_registry import CameraRegistry, DEFAULT_REGISTRY_FILE, normalize_camera_address
    from .capture_storage import METADATA_HEADER, CaptureStore, is_jpeg, parse_camera_metadata
    from .laptop_test_server import camera_url_argument, normalize_camera_url
    from .pi_motion import MotionService
    from .motion_crop import rgb_to_gray, select_motion_crop
    from .roi_control import ROI_CONTROL_JS
except ImportError:  # Direct execution: python server/raspberry_pi_server.py
    from bird_inference import (
        BirdInferenceEngine,
        INFERENCE_SCHEMA_VERSION,
        SELECTED_CROP_GEOMETRY,
        SELECTED_CROP_SOURCE,
        decode_rgb,
        inference_path,
        jpeg_decoder_name,
        read_inference_record,
        write_inference_record,
    )
    from camera_registry import CameraRegistry, DEFAULT_REGISTRY_FILE, normalize_camera_address
    from capture_storage import METADATA_HEADER, CaptureStore, is_jpeg, parse_camera_metadata
    from laptop_test_server import camera_url_argument, normalize_camera_url
    from pi_motion import MotionService
    from motion_crop import rgb_to_gray, select_motion_crop
    from roi_control import ROI_CONTROL_JS


DEFAULT_SAVE_DIR = Path.home() / "bird_images"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Full-integer model: decision uses the exact uint8 rule from its inference_contract.json.
DEFAULT_IS_BIRD_MODEL = (
    PROJECT_ROOT
    / "Models"
    / "runs"
    / "isbird_manifests_letterbox_224_20261020_004327"
    / "model_uint8.tflite"
)
DEFAULT_SPECIES_MODEL = (
    PROJECT_ROOT / "Models" / "runs" / "letterbox_224_20261009_131631" / "model_float32.tflite"
)
MAX_JPEG_BYTES = 10 * 1024 * 1024
# Sent by the ESP32 when wait_for_classification is false in its config.
CLASSIFICATION_HEADER = "X-ESP32-Classification"
PROXY_TIMEOUT_SECONDS = 8
CAPTURE_ROI_SOURCE = "capture_sidecar.camera.requested.motion.roi"
DEFAULT_MOTION_POLL_SECONDS = 0.25
DEFAULT_MOTION_TIMEOUT_SECONDS = 2.0


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
    <section>
      <h2>Camera fleet</h2>
      <p class="muted">Add cameras by IP address. Enabled cameras receive an independent Pi motion worker.</p>
      <input id="fleet-name" placeholder="Camera name (optional)">
      <input id="fleet-address" placeholder="192.168.1.84">
      <button onclick="addFleetCamera()">Add camera</button>
      <button onclick="loadFleet()">Refresh cameras</button>
      <div id="fleet-list" class="muted">Loading cameras...</div>
    </section>
  </main>
  <script>
    const result = document.getElementById('result');
    function fleetSummary(camera) {
      const sensor = camera.sensor?.model || camera.status?.camera_sensor_model || 'unknown sensor';
      const service = camera.motion_service || {running: false};
      return `<p><strong>${camera.name}</strong>${camera.default ? ' (default)' : ''}<br>${camera.url}<br>` +
        `${sensor} | ${camera.reachable === true ? 'reachable' : camera.reachable === false ? 'offline' : 'not checked'} | ` +
        `motion ${service.running ? 'running' : 'stopped'}<br>` +
        `<button onclick="updateFleetCamera('${camera.id}', {default:true})">Set default</button> ` +
        `<button onclick="updateFleetCamera('${camera.id}', {enabled:${!camera.enabled}})">${camera.enabled ? 'Disable' : 'Enable'}</button> ` +
        `<button onclick="pollFleetCamera('${camera.id}')">Poll</button> ` +
        `<button onclick="removeFleetCamera('${camera.id}')">Remove</button></p>`;
    }
    async function loadFleet() {
      const response = await fetch('/api/cameras', {cache:'no-store'});
      const payload = await response.json();
      document.getElementById('fleet-list').innerHTML = payload.cameras.length
        ? payload.cameras.map(fleetSummary).join('') : 'No cameras registered.';
    }
    async function addFleetCamera() {
      const address = document.getElementById('fleet-address').value.trim();
      const name = document.getElementById('fleet-name').value.trim() || null;
      const response = await fetch('/api/cameras', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({address,name})});
      const payload = await response.json();
      if (!response.ok) { showCameraResult(payload.error || 'Could not add camera.', true); return; }
      document.getElementById('fleet-address').value = '';
      await loadFleet();
    }
    async function updateFleetCamera(id, values) {
      const response = await fetch(`/api/cameras/${id}`, {method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});
      const payload = await response.json();
      if (!response.ok) showCameraResult(payload.error || 'Could not update camera.', true);
      await loadFleet();
    }
    async function pollFleetCamera(id) { await fetch(`/api/cameras/${id}/poll`, {method:'POST'}); await loadFleet(); }
    async function removeFleetCamera(id) { await fetch(`/api/cameras/${id}`, {method:'DELETE'}); await loadFleet(); }
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
    loadFleet();
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


def capture_roi(image_path: Path) -> tuple[Any, str]:
    """Read the acquisition-time ROI from the immutable matching sidecar.

    Values are deliberately returned without coercion or clamping. A crop-aware
    inference contract must reject missing or malformed provenance rather than
    silently using the current camera setting or the full frame.
    """
    image_path = Path(image_path)
    sidecar_path = image_path.with_suffix(".json")
    source = CAPTURE_ROI_SOURCE
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, source
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, source
    if not isinstance(sidecar, dict):
        return None, source
    if sidecar.get("capture_id") != image_path.stem or sidecar.get("filename") != image_path.name:
        return None, source
    camera = sidecar.get("camera")
    requested = camera.get("requested") if isinstance(camera, dict) else None
    motion = requested.get("motion") if isinstance(requested, dict) else None
    return (motion.get("roi") if isinstance(motion, dict) else None), source


def create_app(
    save_dir: Path = DEFAULT_SAVE_DIR,
    camera_url: str | None = None,
    inference_engine: BirdInferenceEngine | None = None,
    *,
    start_motion_service: bool = False,
    motion_poll_seconds: float = DEFAULT_MOTION_POLL_SECONDS,
    motion_timeout_seconds: float = DEFAULT_MOTION_TIMEOUT_SECONDS,
    motion_service_factory: Callable[..., Any] = MotionService,
    registry_file: Path | None = None,
    camera_registry: CameraRegistry | None = None,
    start_registry_poller: bool = False,
    manage_motion_services: bool | None = None,
) -> Flask:
    """Create the receiver/classifier app, with optional ESP32 camera controls."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_JPEG_BYTES
    store = CaptureStore(save_dir)
    camera_url = normalize_camera_url(camera_url)
    registry = camera_registry or CameraRegistry(
        registry_file or (store.root / "cameras.json"),
        timeout_seconds=motion_timeout_seconds,
        opener=lambda *args, **kwargs: urlopen(*args, **kwargs),
    )
    if camera_url:
        matching = next((camera for camera in registry.snapshot() if camera["url"] == camera_url), None)
        if matching is None:
            registry.add("Default camera", camera_url, make_default=True)
        else:
            registry.update(matching["id"], default=True)
    if start_registry_poller:
        registry.start()
    app.config["CAMERA_REGISTRY"] = registry
    default_camera = registry.resolve()
    if camera_url is None and default_camera is not None:
        camera_url = default_camera["url"]
    inference_lock = threading.Lock()
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
    app.config["MOTION_SERVICE"] = None
    app.config["MOTION_SERVICES"] = {}
    app.config["CLEANUP"] = registry.stop

    def camera_provenance(destination: Path) -> dict[str, Any]:
        sidecar_path = destination.with_suffix(".json")
        sidecar: dict[str, Any] = {}
        try:
            value = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                sidecar = value
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        device = sidecar.get("device") if isinstance(sidecar.get("device"), dict) else {}
        camera = sidecar.get("camera") if isinstance(sidecar.get("camera"), dict) else {}
        sensor = camera.get("sensor") if isinstance(camera.get("sensor"), dict) else {}
        return {
            "model": sidecar.get("camera_model"),
            "model_source": "capture_sidecar.camera.sensor.model",
            "sensor_pid": sensor.get("pid"),
            "sensor_pid_hex": sensor.get("pid_hex"),
            "camera_id": sidecar.get("camera_id") or device.get("camera_id"),
            "capture_sidecar": sidecar_path.name,
            "capture_sidecar_sha256": (
                hashlib.sha256(sidecar_path.read_bytes()).hexdigest() if sidecar_path.is_file() else None
            ),
        }

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
            "camera_provenance": camera_provenance(destination),
            "error": error,
        }

    def run_classifier(
        destination: Path,
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
        try:
            if inference_engine is not None:
                roi, roi_source = capture_roi(destination)
                kwargs: dict[str, Any] = {"roi": roi, "roi_source": roi_source}
                geometry = getattr(getattr(inference_engine, "contract", None), "geometry", None)
                if geometry == SELECTED_CROP_GEOMETRY:
                    rgb = decode_rgb(destination)
                    motion_path = destination.with_suffix(".motion.json")
                    motion = None
                    if motion_path.is_file():
                        loaded = json.loads(motion_path.read_text(encoding="utf-8"))
                        capture = loaded.get("capture") if isinstance(loaded, dict) else None
                        if (
                            isinstance(capture, dict)
                            and capture.get("capture_id") == destination.stem
                            and capture.get("image_sha256") == hashlib.sha256(destination.read_bytes()).hexdigest()
                        ):
                            motion = loaded
                        else:
                            raise ValueError("Existing motion record does not match the stored capture")
                    if motion is None:
                        motion = select_motion_crop(rgb_to_gray(rgb), None, roi)
                    bounds = motion["selection"]["final_bounds_px"]
                    sidecar_path = destination.with_suffix(".json")
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    if not motion_path.is_file():
                        motion["capture"] = {
                            "capture_id": destination.stem,
                            "filename": destination.name,
                            "image_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                            "width": int(rgb.shape[1]),
                            "height": int(rgb.shape[0]),
                            "camera_id": sidecar.get("camera_id") or ((sidecar.get("device") or {}).get("camera_id")),
                            "boot_id": (sidecar.get("device") or {}).get("boot_id"),
                        }
                        motion["static_roi"]["capture_sidecar"] = sidecar_path.name
                        motion["static_roi"]["capture_sidecar_sha256"] = hashlib.sha256(
                            sidecar_path.read_bytes()
                        ).hexdigest()
                        motion["camera_provenance"] = camera_provenance(destination)
                        store.save_derived_record(destination, "motion", motion)
                    kwargs = {
                        "selected_crop": {key: bounds[key] for key in ("left", "top", "right", "bottom")},
                        "crop_source": SELECTED_CROP_SOURCE,
                    }
                with inference_lock:
                    inference = inference_engine.infer(destination, **kwargs)
                inference["capture_id"] = destination.stem
                inference["camera_provenance"] = camera_provenance(destination)
                classification, error = validate_classification(inference.get("classification"))
            else:
                classification, error = validate_classification(classify(destination))
                inference = {
                    "schema_version": INFERENCE_SCHEMA_VERSION,
                    "status": "complete" if error is None else "error",
                    "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "capture_id": destination.stem,
                    "image": {
                        "filename": destination.name,
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    },
                    "camera_provenance": camera_provenance(destination),
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

    motion_services: dict[str, Any] = {}
    motion_management_enabled = (
        start_motion_service if manage_motion_services is None else manage_motion_services
    )

    def motion_persisted(
        destination: Path,
        capture: dict[str, Any],
        inference: dict[str, Any],
    ) -> None:
        classification, _ = validate_classification(inference.get("classification"))
        with state_lock:
            latest.update(
                path=destination,
                received_at=capture.get("received_at"),
                classification=classification,
                inference=inference,
            )

    def start_camera_motion(camera: dict[str, Any]) -> Any:
        existing_service = motion_services.get(camera["id"])
        if existing_service is not None:
            expected = getattr(existing_service, "expected_camera_id", camera.get("hardware_camera_id"))
            if expected == camera.get("hardware_camera_id"):
                if hasattr(existing_service, "camera_name"):
                    existing_service.camera_name = camera["name"]
                return existing_service
            stop_camera_motion(camera["id"])
        if inference_engine is None:
            raise ValueError("Pi motion polling requires both camera_url and inference_engine.")
        operation_lock = registry.operation_lock(camera["id"])
        assert operation_lock is not None
        service = motion_service_factory(
            camera["url"],
            store,
            inference_engine,
            poll_seconds=motion_poll_seconds,
            timeout_seconds=motion_timeout_seconds,
            camera_lock=operation_lock,
            inference_lock=inference_lock,
            on_persist=motion_persisted,
            expected_camera_id=camera.get("hardware_camera_id"),
            camera_registry_id=camera["id"],
            camera_name=camera["name"],
        )
        service.start()
        motion_services[camera["id"]] = service
        return service

    def stop_camera_motion(camera_id: str) -> None:
        service = motion_services.pop(camera_id, None)
        if service is not None:
            service.stop()

    def sync_motion_services() -> None:
        cameras = {camera["id"]: camera for camera in registry.snapshot()}
        for camera_id in list(motion_services):
            camera = cameras.get(camera_id)
            if camera is None or not camera["enabled"]:
                stop_camera_motion(camera_id)
        if motion_management_enabled:
            for camera in cameras.values():
                if camera["enabled"]:
                    start_camera_motion(camera)
        selected = registry.resolve()
        app.config["MOTION_SERVICE"] = (
            motion_services.get(selected["id"]) if selected is not None else None
        )
        app.config["MOTION_SERVICES"] = motion_services

    def cleanup() -> None:
        for camera_id in list(motion_services):
            stop_camera_motion(camera_id)
        registry.stop()

    if start_motion_service:
        if not registry.snapshot() or inference_engine is None:
            raise ValueError("Pi motion polling requires both camera_url and inference_engine.")
    if motion_management_enabled and inference_engine is None:
        raise ValueError("Pi motion polling requires both camera_url and inference_engine.")
    if motion_management_enabled:
        sync_motion_services()
    app.config["CLEANUP"] = cleanup

    # A process restart must not strand accepted captures whose in-memory queue was lost.
    if inference_engine is not None:
        for destination in existing:
            if not inference_path(destination).exists():
                pending.put(destination)

    def camera_proxy(
        endpoint: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        camera_id: str | None = None,
    ):
        camera = registry.resolve(camera_id)
        if camera is None:
            if camera_id:
                return jsonify(error="Camera not found."), 404
            status = 409 if registry.snapshot() else 503
            return jsonify(error="No default camera is configured."), status
        if not camera["enabled"]:
            return jsonify(error="Camera is disabled."), 503
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        outgoing = Request(
            f"{camera['url']}/{endpoint.lstrip('/')}", data=body, headers=headers, method=method
        )
        operation_lock = registry.operation_lock(camera["id"])
        assert operation_lock is not None
        if not operation_lock.acquire(blocking=False):
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
            operation_lock.release()

    @app.get("/")
    def dashboard() -> str:
        selected = registry.resolve()
        return render_template_string(
            PI_DASHBOARD_HTML,
            camera_url=(selected or {}).get("url", camera_url),
            roi_control_js=ROI_CONTROL_JS,
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

    @app.get("/api/cameras")
    def cameras() -> Response:
        values = registry.snapshot()
        for camera in values:
            service = motion_services.get(camera["id"])
            camera["motion_service"] = service.describe() if service is not None else {"running": False}
        return jsonify(cameras=values)

    @app.post("/api/cameras")
    def add_camera() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a valid JSON object."), 400
        try:
            camera = registry.add(
                payload.get("name"),
                normalize_camera_address(payload.get("address", payload.get("url"))),
                enabled=payload.get("enabled", True),
                make_default=payload.get("default"),
            )
        except ValueError as error:
            return jsonify(error=str(error)), 422
        discovered = registry.poll(camera["id"])
        duplicate_error = ((discovered or {}).get("errors") or {}).get("metadata")
        if duplicate_error:
            registry.remove(camera["id"])
            return jsonify(error=duplicate_error), 409
        try:
            sync_motion_services()
        except ValueError as error:
            return jsonify(error=str(error), camera=registry.get(camera["id"])), 409
        return jsonify(camera=registry.get(camera["id"])), 201

    @app.patch("/api/cameras/<camera_id>")
    def update_camera(camera_id: str) -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a valid JSON object."), 400
        try:
            camera = registry.update(camera_id, **payload)
        except ValueError as error:
            return jsonify(error=str(error)), 422
        if camera is None:
            return jsonify(error="Camera not found."), 404
        sync_motion_services()
        return jsonify(camera=camera)

    @app.delete("/api/cameras/<camera_id>")
    def delete_camera(camera_id: str) -> tuple[Response, int] | Response:
        stop_camera_motion(camera_id)
        camera = registry.remove(camera_id)
        if camera is None:
            return jsonify(error="Camera not found."), 404
        sync_motion_services()
        return jsonify(status="removed", camera=camera)

    @app.post("/api/cameras/<camera_id>/poll")
    def poll_camera(camera_id: str) -> tuple[Response, int] | Response:
        camera = registry.poll(camera_id)
        if camera is None:
            return jsonify(error="Camera not found."), 404
        sync_motion_services()
        return jsonify(camera=camera)

    @app.get("/api/cameras/<camera_id>/capture")
    def camera_capture_scoped(camera_id: str):
        return camera_proxy("capture", camera_id=camera_id)

    @app.get("/api/cameras/<camera_id>/<endpoint>")
    def camera_get_scoped(camera_id: str, endpoint: str):
        if endpoint not in {"status", "config", "metadata", "motion"}:
            return jsonify(error="Unsupported camera endpoint."), 404
        return camera_proxy(endpoint, camera_id=camera_id)

    @app.post("/api/cameras/<camera_id>/config")
    def camera_config_scoped(camera_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a valid JSON object."), 400
        return camera_proxy("config", "POST", payload, camera_id=camera_id)

    @app.post("/api/cameras/<camera_id>/mode")
    def camera_mode_scoped(camera_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or payload.get("mode") not in {"motion", "idle"}:
            return jsonify(error="mode must be either motion or idle"), 422
        return camera_proxy("mode", "POST", payload, camera_id=camera_id)

    @app.post("/upload")
    def upload() -> tuple[Response, int] | Response:
        image = request.get_data(cache=False)
        if not image:
            return jsonify(error="Request body is empty; expected raw JPEG bytes."), 400
        if not is_jpeg(image):
            return jsonify(error="Request body is not a complete JPEG image."), 415

        camera_metadata, metadata_error = parse_camera_metadata(request.headers.get(METADATA_HEADER))
        device = (camera_metadata or {}).get("device")
        hardware_camera_id = device.get("camera_id") if isinstance(device, dict) else None
        registered = registry.get_by_hardware_id(hardware_camera_id)
        classification: dict[str, Any] | None = None
        classification_error: str | None = None
        # Save first so classification always receives a durable image path.
        record = store.save_capture(
            image,
            source=(camera_metadata or {}).get("source", "motion"),
            camera_metadata=camera_metadata,
            metadata_error=metadata_error,
            extra={
                "camera_registry_id": registered.get("id") if registered else None,
                "camera_name": registered.get("name") if registered else None,
            },
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
            print(
                f"Saved (classification queued, camera={record.get('camera_model') or 'unknown'}): "
                f"{destination}",
                flush=True,
            )
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

        print(f"Saved (camera={record.get('camera_model') or 'unknown'}): {destination}", flush=True)
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
        motion_path = path.with_suffix(".motion.json")
        motion = None
        try:
            value = json.loads(motion_path.read_text(encoding="utf-8"))
            motion = value if isinstance(value, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        return {
            "capture_id": sidecar.get("capture_id", path.stem),
            "filename": path.name,
            "received_at": sidecar.get("received_at"),
            "source": sidecar.get("source"),
            "camera_id": sidecar.get("camera_id"),
            "camera_registry_id": sidecar.get("camera_registry_id"),
            "camera_name": sidecar.get("camera_name"),
            "camera_model": sidecar.get("camera_model"),
            "bytes": path.stat().st_size,
            "sha256": sidecar.get("sha256"),
            "inference_status": (inference or {}).get("status", "pending"),
            "inference": inference,
            "motion": motion,
            "image_url": f"/api/captures/{path.stem}/image",
            "inference_url": f"/api/captures/{path.stem}/inference",
            "motion_url": f"/api/captures/{path.stem}/motion",
        }

    def capture_path(capture_id: str) -> Path | None:
        if not capture_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in capture_id):
            return None
        path = store.root / f"{capture_id}.jpg"
        return path if path.is_file() else None

    @app.get("/api/captures")
    def captures() -> Response:
        after = request.args.get("after", "")
        camera_id = request.args.get("camera_id")
        camera_registry_id = request.args.get("camera_registry_id")
        try:
            limit = max(1, min(200, int(request.args.get("limit", "50"))))
        except ValueError:
            return jsonify(error="limit must be an integer"), 400
        paths = sorted(store.root.glob("capture_*.jpg"), key=lambda item: item.name)
        if camera_id or camera_registry_id:
            filtered: list[Path] = []
            for path in paths:
                try:
                    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
                if camera_id and sidecar.get("camera_id") != camera_id:
                    continue
                if camera_registry_id and sidecar.get("camera_registry_id") != camera_registry_id:
                    continue
                filtered.append(path)
            paths = filtered
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

    @app.get("/api/captures/<capture_id>/motion")
    def capture_motion(capture_id: str):
        path = capture_path(capture_id)
        if path is None:
            return jsonify(error="Capture not found."), 404
        motion_path = path.with_suffix(".motion.json")
        try:
            value = json.loads(motion_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return jsonify(error="Motion record not found."), 404
        return jsonify(value) if isinstance(value, dict) else (jsonify(error="Invalid motion record."), 500)

    @app.get("/health")
    def health() -> Response:
        with state_lock:
            path = latest["path"]
            received_at = latest["received_at"]
            classification = latest["classification"]
            latest_inference = latest["inference"]
        services = {camera_id: service.describe() for camera_id, service in motion_services.items()}
        selected = registry.resolve()
        selected_service = motion_services.get(selected["id"]) if selected is not None else None
        return jsonify(
            status="ok",
            save_dir=str(store.root),
            camera_url=(selected or {}).get("url", camera_url),
            camera_configured=bool(registry.snapshot()),
            cameras=registry.snapshot(),
            latest_image=path.name if path else None,
            received_at=received_at,
            classification=classification,
            latest_inference=latest_inference,
            classification_queue_depth=pending.unfinished_tasks,
            motion_service=(selected_service.describe() if selected_service is not None else {"running": False}),
            motion_services=services,
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
        "--registry-file",
        type=Path,
        default=DEFAULT_REGISTRY_FILE,
        help=f"Persistent multi-camera registry (default: {DEFAULT_REGISTRY_FILE})",
    )
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
    parser.add_argument(
        "--contract-file",
        type=Path,
        help="is-bird inference_contract.json (default: beside the is-bird model)",
    )
    parser.add_argument(
        "--threshold-file",
        type=Path,
        help="Legacy bird threshold JSON, used when no contract is present (default: beside model)",
    )
    parser.add_argument(
        "--bird-threshold",
        type=float,
        help="Override the contract threshold (uses the probability rule, not the uint8 rule)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of species guesses to retain")
    parser.add_argument(
        "--motion-poll-seconds",
        type=float,
        default=DEFAULT_MOTION_POLL_SECONDS,
        help="Pi-owned full-resolution motion polling interval (default: 0.25 seconds)",
    )
    parser.add_argument(
        "--motion-timeout-seconds",
        type=float,
        default=DEFAULT_MOTION_TIMEOUT_SECONDS,
        help="Per-frame ESP32 monitor request timeout (default: 2 seconds)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="TFLite interpreter threads per model (default: up to 4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        engine = BirdInferenceEngine(
            args.is_bird_model,
            args.species_model,
            species_labels=args.species_labels,
            threshold_file=args.threshold_file,
            contract_file=args.contract_file,
            bird_threshold=args.bird_threshold,
            top_k=args.top_k,
            num_threads=args.num_threads,
        )
    except Exception as error:
        raise SystemExit(f"Model startup error: {error}") from error
    print(f"Models ready: {json.dumps(engine.describe(), sort_keys=True)}", flush=True)
    print(f"Bird decision: {engine.decision_rule} (from {engine.threshold_source})", flush=True)
    if engine.contract is None:
        print("WARNING: no inference_contract.json beside the is-bird model; using legacy threshold.", flush=True)
    if not jpeg_decoder_name().startswith("simplejpeg"):
        print(
            "WARNING: simplejpeg is not installed; JPEGs are decoded with Pillow, whose pixels "
            "differ slightly from training. Run: pip install simplejpeg",
            flush=True,
        )
    if args.motion_poll_seconds <= 0 or args.motion_timeout_seconds <= 0:
        raise SystemExit("Motion poll interval and timeout must be greater than zero.")
    application = create_app(
        args.save_dir,
        args.camera_url,
        engine,
        start_motion_service=bool(args.camera_url),
        manage_motion_services=True,
        motion_poll_seconds=args.motion_poll_seconds,
        motion_timeout_seconds=args.motion_timeout_seconds,
        registry_file=args.registry_file,
        start_registry_poller=True,
    )
    try:
        application.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        application.config["CLEANUP"]()