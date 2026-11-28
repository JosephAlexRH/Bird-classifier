"""Headless ESP32-CAM image receiver intended for Raspberry Pi deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import os
import queue
import threading
from urllib.parse import urlsplit
import time
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
    from .camera_registry import (
        CameraRegistry, DEFAULT_REGISTRY_FILE, STALE_HOST_SUFFIX, normalize_camera_address,
        sync_registry_from_heartbeat,
    )
    from .camera_dashboard_templates import PI_TABBED_DASHBOARD_HTML
    from .capture_storage import METADATA_HEADER, CaptureStore, capture_files, is_jpeg, parse_camera_metadata
    from .laptop_test_server import camera_url_argument, normalize_camera_url
    from .pi_motion import MotionService
    from .motion_crop import MotionConfig, rgb_to_gray, select_motion_crop
    from .roi_control import ROI_CONTROL_JS
    from .fleet_monitor import FleetMonitor, register_fleet_routes
    from .hub_clock import HubClock, step_system_clock
    from .run_log import HubStatusLogger, RunLog
    from .fleet_stats import KEEP_SECONDS as FLEET_KEEP_SECONDS, FleetStats, visit_totals
    from .hub_specs import camera_specs, hub_specs
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
    from camera_registry import (
        CameraRegistry, DEFAULT_REGISTRY_FILE, STALE_HOST_SUFFIX, normalize_camera_address,
        sync_registry_from_heartbeat,
    )
    from camera_dashboard_templates import PI_TABBED_DASHBOARD_HTML
    from capture_storage import METADATA_HEADER, CaptureStore, capture_files, is_jpeg, parse_camera_metadata
    from laptop_test_server import camera_url_argument, normalize_camera_url
    from pi_motion import MotionService
    from motion_crop import MotionConfig, rgb_to_gray, select_motion_crop
    from roi_control import ROI_CONTROL_JS
    from fleet_monitor import FleetMonitor, register_fleet_routes
    from hub_clock import HubClock, step_system_clock
    from run_log import HubStatusLogger, RunLog
    from fleet_stats import KEEP_SECONDS as FLEET_KEEP_SECONDS, FleetStats, visit_totals
    from hub_specs import camera_specs, hub_specs


DEFAULT_SAVE_DIR = Path.home() / "bird_images"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETECTOR_MODEL = (
    PROJECT_ROOT
    / "Models"
    / "pi_detector_float_stride16_224"
    / "bird_detector_motion_crop_stride16_float32.tflite"
)
DEFAULT_SPECIES_MODEL = (
    PROJECT_ROOT / "Models" / "runs" / "letterbox_224_20261009_131631" / "model_float32.tflite"
)
MAX_JPEG_BYTES = 10 * 1024 * 1024
# Sent by the ESP32 when wait_for_classification is false in its config.
CLASSIFICATION_HEADER = "X-ESP32-Classification"
PROXY_TIMEOUT_SECONDS = 8
# A dashboard request (capture, status, config...) waits this long for the
# camera's background worker to finish its current frame instead of answering
# "busy" and making the browser retry. A framing stream holds the camera until
# it is closed, so requests during a stream still get an immediate 409.
CAMERA_WAIT_SECONDS = 6


def is_parked(camera: dict[str, Any]) -> bool:
    """Its address was taken over by another camera; waiting for its own heartbeat."""
    return (urlsplit(camera.get("url") or "").hostname or "").endswith(STALE_HOST_SUFFIX)
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
      <p class="muted">Cameras add themselves when they check in and follow their own IP changes; just give new ones a name. Adding by IP address remains as a fallback. Enabled cameras receive an independent Pi motion worker.</p>
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
        `<button onclick="renameFleetCamera('${camera.id}')">${camera.needs_name ? 'Name this camera' : 'Rename'}</button> ` +
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
    async function renameFleetCamera(id) {
      const name = (window.prompt('Name for this camera (for example: Back garden feeder)') || '').trim();
      if (name) await updateFleetCamera(id, {name});
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
    auto_register_cameras: bool = True,
    manage_system_clock: bool = False,
    start_hub_status: bool = False,
    manage_motion_services: bool | None = None,
    collection_mode: bool = False,
    deployment_id: str = "",
    visit_tracker: Any | None = None,
    static_box_filter: bool = True,
) -> Flask:
    """Create the receiver/classifier app, with optional ESP32 camera controls."""
    app = Flask(__name__)
    try:
        from .collection_dashboard import collection_page
    except ImportError:
        from collection_dashboard import collection_page
    app.config["UNIFIED_DASHBOARD"] = lambda: collection_page()
    app.config["MAX_CONTENT_LENGTH"] = MAX_JPEG_BYTES
    store = CaptureStore(save_dir)
    hub_started = time.time()
    fleet_stats = FleetStats(store.root / "fleet_stats.json")
    last_capture_record: dict[str, dict[str, Any]] = {}  # per camera, for the specs view

    def count_frame(destination: Path, capture: dict[str, Any] | None, inference: dict[str, Any] | None) -> None:
        """Tally a saved frame for the Fleet tab; never let counting cost a capture."""
        try:
            record = capture
            if record is None:
                record = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
            camera = record.get("camera_registry_id") or record.get("camera_id") or "unregistered"
            if capture is not None and capture.get("source") != "background_calibration":
                last_capture_record[str(camera)] = capture
            fleet_stats.observe(str(camera), record.get("capture_id") or destination.stem, inference)
        except Exception:  # noqa: BLE001
            pass

    try:
        from .hub_settings import HubSettings
    except ImportError:
        from hub_settings import HubSettings
    hub_settings = HubSettings(store.root / "hub_settings.json", {
        "collection_mode": collection_mode, "poll_seconds": motion_poll_seconds,
        "species_enabled": getattr(inference_engine, "species", None) is not None,
        "detector_enabled": inference_engine is not None,
        "detector_threshold": float(getattr(inference_engine, "bird_threshold", .5)),
    })
    collection_mode = hub_settings.value["collection_mode"]
    motion_poll_seconds = hub_settings.value["poll_seconds"]
    if inference_engine is not None:
        if callable(getattr(inference_engine, "set_species_enabled", None)):
            inference_engine.set_species_enabled(hub_settings.value["species_enabled"])
        if getattr(inference_engine, "bird_threshold", .5) != hub_settings.value["detector_threshold"]:
            inference_engine.threshold_source = "saved hub settings"
        inference_engine.bird_threshold = hub_settings.value["detector_threshold"]
    app.config["HUB_SETTINGS"] = hub_settings
    app.config["COLLECTION_MODE"] = collection_mode
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
    # Permanent run log + hub time keeping for unattended deployments.
    run_log = RunLog(store.root / "runlog")
    hub_clock = HubClock(
        store.root / "hub_clock.json",
        set_system_time=step_system_clock if manage_system_clock else None,
        journal=run_log.write,
    )
    run_log.context = lambda: {"hub_boot": hub_clock.boot_id, "clock": hub_clock.state()["source"]}
    hub_status_logger = HubStatusLogger(run_log, store.root, hub_clock)
    app.config["RUN_LOG"], app.config["HUB_CLOCK"] = run_log, hub_clock
    app.config["HUB_STATUS_LOGGER"] = hub_status_logger
    if start_hub_status:
        run_log.write("hub_start", pid=os.getpid(), save_dir=str(store.root), clock_state=hub_clock.state())
        hub_status_logger.start()

    def registry_log(message: str) -> None:
        print(message, flush=True)
        run_log.write("registry", message=message)
    fleet = FleetMonitor(
        store.root / "fleet.json",
        name_lookup=lambda hardware_id: (registry.get_by_hardware_id(hardware_id) or {}).get("name"),
    )
    app.config["FLEET_MONITOR"] = fleet

    def heartbeat_seen(payload: dict[str, Any], remote_addr: str | None) -> None:
        # Cameras announce themselves: new ones are added, moved ones re-addressed,
        # and the per-camera motion workers follow the registry.
        _, change = sync_registry_from_heartbeat(registry, payload, remote_addr, log=registry_log)
        if change in {"added", "replaced", "moved", "bound"}:
            sync_motion_services()

    register_fleet_routes(
        app,
        fleet,
        capture_dir=store.root,
        on_heartbeat=heartbeat_seen if auto_register_cameras else None,
        run_log=run_log,
        hub_clock=hub_clock,
    )
    default_camera = registry.resolve()
    if camera_url is None and default_camera is not None:
        camera_url = default_camera["url"]
    inference_lock = threading.Lock()
    state_lock = threading.Lock()
    existing = sorted(capture_files(store.root, "*.jpg"), key=lambda path: path.stat().st_mtime)
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
        record = {
            "schema_version": INFERENCE_SCHEMA_VERSION,
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
        if inference_engine is not None:
            detector_record = getattr(inference_engine, "detector_record", lambda: None)()
            if detector_record is not None:
                record["detector"] = detector_record
        return record

    def run_classifier(
        destination: Path,
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
        def disabled_result():
            record = {"schema_version": INFERENCE_SCHEMA_VERSION, "status": "skipped",
                      "reason": "analysis_disabled", "is_bird": None, "classification": None,
                      "completed_at": datetime.now(timezone.utc).isoformat(), "capture_id": destination.stem,
                      "image": {"filename": destination.name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()},
                      "hub_pipeline": hub_settings.snapshot(), "camera_provenance": camera_provenance(destination)}
            write_inference_record(destination, record)
            return None, None, record
        if inference_engine is not None and not hub_settings.value["detector_enabled"]:
            return disabled_result()
        try:
            if inference_engine is not None:
                roi, roi_source = capture_roi(destination)
                kwargs: dict[str, Any] = {"roi": roi, "roi_source": roi_source}
                crop_contract = getattr(inference_engine, "detector_contract", None) or getattr(
                    inference_engine, "contract", None
                )
                geometry = getattr(crop_contract, "geometry", None)
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
                    if not hub_settings.value["detector_enabled"]:
                        return disabled_result()
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
            if collection_mode:
                try:
                    from .collection import export_bird_crops
                except ImportError:
                    from collection import export_bird_crops
                inference["collection"] = export_bird_crops(destination, inference)
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
                count_frame(destination, None, inference)
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
    runtime_lock = threading.RLock()
    motion_management_enabled = (
        start_motion_service if manage_motion_services is None else manage_motion_services
    )

    def motion_persisted(
        destination: Path,
        capture: dict[str, Any],
        inference: dict[str, Any],
    ) -> None:
        count_frame(destination, capture, inference)
        classification, _ = validate_classification(inference.get("classification"))
        with state_lock:
            if latest["path"] is not None and destination.stem < latest["path"].stem:
                return
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
            # A heartbeat can re-address a camera; a worker on the old URL must restart.
            same_url = getattr(existing_service, "camera_url", camera["url"]) == camera["url"]
            if expected == camera.get("hardware_camera_id") and same_url:
                if hasattr(existing_service, "camera_name"):
                    existing_service.camera_name = camera["name"]
                threads = (getattr(existing_service, "_thread", None), getattr(existing_service, "_analysis_thread", None))
                if getattr(existing_service, "running", True) or any(t is not None and t.is_alive() for t in threads):
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
            **({"collection_mode": True} if collection_mode else {}),
            **({"deployment_id": deployment_id, "collection_run_id": hub_clock.boot_id}
               if deployment_id or collection_mode else {}),
            **({"detector_enabled": False} if not hub_settings.value["detector_enabled"] else {}),
            **({"motion_localization_enabled": False} if not hub_settings.value["motion_localization_enabled"] else {}),
            **({"settings_revision": hub_settings.value["revision"]} if hub_settings.value["revision"] else {}),
            config=MotionConfig(minimum_threshold=int(hub_settings.value['motion_threshold']),
                maximum_threshold=max(48, int(hub_settings.value['motion_threshold'])),
                minimum_component_pixels=int(hub_settings.value['motion_minimum_area'])),
            motion_triggered=hub_settings.value['motion_triggered'],
            empty_sample_seconds=hub_settings.value['empty_sample_seconds'],
            **({"visit_tracker": visit_tracker} if visit_tracker is not None else {}),
            **({"static_box_filter": False} if not static_box_filter else {}),
        )
        service.start()
        motion_services[camera["id"]] = service
        return service

    def stop_camera_motion(camera_id: str) -> None:
        service = motion_services.pop(camera_id, None)
        if service is not None:
            service.stop()

    def sync_motion_services() -> None:
        with runtime_lock:
            _sync_motion_services()

    def _sync_motion_services() -> None:
        cameras = {camera["id"]: camera for camera in registry.snapshot()}
        for camera_id in list(motion_services):
            camera = cameras.get(camera_id)
            if (camera is None or not camera["enabled"] or is_parked(camera)
                    or not hub_settings.camera_running(camera_id)):
                stop_camera_motion(camera_id)
        if motion_management_enabled:
            for camera in cameras.values():
                if camera["enabled"] and not is_parked(camera) and hub_settings.camera_running(camera["id"]):
                    start_camera_motion(camera)
        selected = registry.resolve()
        app.config["MOTION_SERVICE"] = (
            motion_services.get(selected["id"]) if selected is not None else None
        )
        app.config["MOTION_SERVICES"] = motion_services

    def cleanup() -> None:
        app.config["SHUTTING_DOWN"] = True
        for camera_id in list(motion_services):
            stop_camera_motion(camera_id)
        if visit_tracker is not None:
            visit_tracker.close_all()
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

    def acquire_camera(camera: dict[str, Any], operation_lock: threading.Lock) -> bool:
        """Take the camera for a dashboard request, queuing behind the poller."""
        if camera.get("streaming"):
            return operation_lock.acquire(blocking=False)
        return operation_lock.acquire(timeout=CAMERA_WAIT_SECONDS)

    app.config["ACQUIRE_CAMERA"] = acquire_camera

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
        if not acquire_camera(camera, operation_lock):
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
        try:
            from .collection_dashboard import collection_page
        except ImportError:
            from collection_dashboard import collection_page
        return collection_page()

    def settings_response():
        models = inference_engine.describe() if inference_engine is not None else {}
        return {"settings": hub_settings.snapshot(), "capabilities": {
            "detector_available": inference_engine is not None,
            "species_available": bool(models.get("species_available", models.get("species") is not None)),
            "camera_settings_scope": "selected_camera", "hub_settings_scope": "all_cameras"}}

    def apply_hub_settings(proposed, camera_id=None):
        nonlocal collection_mode, motion_poll_seconds
        previous = hub_settings.snapshot()
        previous_threshold_source = getattr(inference_engine, "threshold_source", None)
        if proposed["detector_enabled"] and inference_engine is None:
            raise ValueError("No inference model is loaded on this hub.")
        setter = getattr(inference_engine, "set_species_enabled", None)
        if proposed["species_enabled"] != previous["species_enabled"] and not callable(setter):
            raise ValueError("Species switching is not available for the loaded engine.")
        if proposed["species_enabled"] and not settings_response()["capabilities"]["species_available"]:
            raise ValueError("Species model or labels are missing on the Pi. Detection and collection remain available.")
        targets = [camera_id] if camera_id else list(motion_services)
        for identifier in targets:
            camera = registry.get(identifier)
            if camera and camera.get("streaming"):
                raise RuntimeError("Stop the framing stream before changing capture settings.")
        committed = False
        try:
            for identifier in targets:
                service = motion_services.get(identifier)
                if isinstance(service, MotionService):
                    service._stop.set()
            for identifier in targets:
                service = motion_services.get(identifier)
                if service is not None:
                    service.stop(timeout=30) if isinstance(service, MotionService) else service.stop()
                    if any(t is not None and t.is_alive() for t in
                           (getattr(service, "_thread", None), getattr(service, "_analysis_thread", None))):
                        raise RuntimeError("A camera is still finishing its current operation. Retry shortly.")
                    motion_services.pop(identifier, None)
            if camera_id is None and inference_engine is not None:
                with inference_lock:
                    if callable(setter):
                        setter(proposed["species_enabled"])
                    inference_engine.bird_threshold = proposed["detector_threshold"]
                    if proposed["detector_threshold"] != previous["detector_threshold"]:
                        inference_engine.threshold_source = "saved hub settings"
            hub_settings.save(proposed)
            committed = True
            collection_mode = proposed["collection_mode"]
            motion_poll_seconds = proposed["poll_seconds"]
            app.config["COLLECTION_MODE"] = collection_mode
            sync_motion_services()
            run_log.write("pipeline_settings", camera_registry_id=camera_id, previous=previous, settings=proposed)
        except Exception:
            if camera_id is None and inference_engine is not None:
                with inference_lock:
                    if callable(setter):
                        setter(previous["species_enabled"])
                    inference_engine.bird_threshold = previous["detector_threshold"]
                    if previous_threshold_source is not None:
                        inference_engine.threshold_source = previous_threshold_source
            if committed:
                hub_settings.save(previous)
            collection_mode = previous["collection_mode"]
            motion_poll_seconds = previous["poll_seconds"]
            app.config["COLLECTION_MODE"] = collection_mode
            sync_motion_services()
            # If a long-running interpreter has not returned yet, resume the
            # previous capture policy once it finishes, without a second click.
            for identifier, service in list(motion_services.items()):
                if isinstance(service, MotionService) and service._stop.is_set():
                    def recover(identifier=identifier, service=service):
                        for thread in (service._thread, service._analysis_thread):
                            if thread is not None:
                                thread.join()
                        with runtime_lock:
                            if not app.config.get("SHUTTING_DOWN") and motion_services.get(identifier) is service:
                                motion_services.pop(identifier, None)
                                sync_motion_services()
                    threading.Thread(target=recover, name="hub-capture-recovery", daemon=True).start()
            raise

    @app.route("/api/hub/settings", methods=["GET", "POST"])
    def hub_settings_route():
        if request.method == "POST":
            try:
                with runtime_lock:
                    apply_hub_settings(hub_settings.propose(request.get_json(silent=True)))
            except (ValueError, OSError) as error:
                return jsonify(error=str(error)), 400
            except RuntimeError as error:
                return jsonify(error=str(error)), 409
        return jsonify(settings_response())

    @app.post("/api/cameras/<camera_id>/pipeline")
    def camera_pipeline(camera_id):
        if registry.get(camera_id) is None:
            return jsonify(error="Camera not found."), 404
        try:
            with runtime_lock:
                apply_hub_settings(hub_settings.propose(request.get_json(silent=True), camera_id=camera_id), camera_id)
        except (ValueError, OSError) as error:
            return jsonify(error=str(error)), 400
        except RuntimeError as error:
            return jsonify(error=str(error)), 409
        return jsonify(settings_response())

    @app.route('/api/cameras/<camera_id>/background', methods=['GET', 'POST'])
    def camera_background(camera_id):
        if registry.get(camera_id) is None:
            return jsonify(error='Camera not found.'), 404
        with runtime_lock:
            service = motion_services.get(camera_id)
            if service is None or not hasattr(service, 'background_command'):
                return jsonify(error='Enable automatic capture for this camera before learning a background.'), 409
            if request.method == 'GET':
                return jsonify(service.describe().get('background_calibration'))
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return jsonify(error='Supply a background action.'), 400
            try:
                result = service.background_command(payload.get('action'),
                    confirmed_empty=payload.get('confirmed_empty') is True)
            except ValueError as error:
                return jsonify(error=str(error)), 400
            except RuntimeError as error:
                return jsonify(error=str(error)), 409
            run_log.write('background_action', camera_registry_id=camera_id,
                          action=payload.get('action'), state=result)
            return jsonify(result), 202

    @app.get("/legacy")
    def legacy_dashboard() -> str:
        try:
            from .collection_dashboard import collection_page
        except ImportError:
            from collection_dashboard import collection_page
        return collection_page(initial_tab="settings")

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

    def manual_capture_saved(record):
        destination = store.path_for(record["filename"])
        normal = record["calibration_type"] == "normal"
        analyze = normal and (inference_engine is None or hub_settings.value["detector_enabled"])
        status = "pending" if analyze else "skipped"
        inference = {"status": status, "capture_id": record["capture_id"],
                     "reason": "manual_capture" if analyze else "analysis_disabled" if normal else "calibration_frame"}
        write_inference_record(destination, inference)
        motion_persisted(destination, record, inference)
        if status == "pending":
            pending.put(destination)

    try:
        from .camera_tools import register_camera_tools
    except ImportError:
        from camera_tools import register_camera_tools
    register_camera_tools(app, registry, store, camera_proxy, manual_capture_saved)

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
        destination = store.path_for(record["filename"])
        count_frame(destination, record, None)
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
        count_frame(destination, record, inference)

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

    visit_index: dict[str, Any] = {"key": None, "visits": {}}
    visit_index_lock = threading.Lock()
    visit_fields = ("visit_id", "captures", "species_frames", "start", "end", "duration_seconds", "result")

    def closed_visits() -> dict[str, dict[str, Any]]:
        """Closed visits from visits.jsonl, keyed by visit_id (re-read only when the file changes)."""
        path = store.root / "visits.jsonl"
        try:
            stat = path.stat()
        except OSError:
            return {}
        key = (stat.st_mtime_ns, stat.st_size)
        with visit_index_lock:
            if visit_index["key"] != key:
                visits: dict[str, dict[str, Any]] = {}
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            try:
                                record = json.loads(line)
                            except ValueError:
                                continue
                            if isinstance(record, dict) and isinstance(record.get("visit_id"), str):
                                visits[record["visit_id"]] = {k: record.get(k) for k in visit_fields}
                except OSError:
                    return {}
                visit_index.update(key=key, visits=visits)
            return visit_index["visits"]

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
        review = None
        try:
            review = json.loads(path.with_suffix(".review.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
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
            "review": review,
            "motion": motion,
            "visit_result": closed_visits().get(
                ((inference or {}).get("visit") or {}).get("visit_id") or ""),
            "image_url": f"/api/captures/{path.stem}/image",
            "metadata_url": f"/api/captures/{path.stem}/metadata",
            "inference_url": f"/api/captures/{path.stem}/inference",
            "motion_url": f"/api/captures/{path.stem}/motion",
        }

    def capture_path(capture_id: str) -> Path | None:
        if not capture_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in capture_id):
            return None
        path = store.path_for(f"{capture_id}.jpg")
        return path if path.is_file() else None

    def latest_camera_capture(camera_id: str) -> Path | None:
        if registry.get(camera_id) is None:
            return None
        paths = sorted(capture_files(store.root), key=lambda item: item.name, reverse=True)
        for path in paths:
            try:
                sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(sidecar, dict) and sidecar.get("camera_registry_id") == camera_id:
                return path
        return None

    @app.get("/api/cameras/<camera_id>/latest")
    def latest_camera_detail(camera_id: str):
        if registry.get(camera_id) is None:
            return jsonify(error="Camera not found."), 404
        path = latest_camera_capture(camera_id)
        if path is None:
            return jsonify(error="No capture has been saved for this camera yet."), 404
        return jsonify(capture_summary(path))

    @app.get("/api/cameras/<camera_id>/latest.jpg")
    def latest_camera_image(camera_id: str):
        if registry.get(camera_id) is None:
            return jsonify(error="Camera not found."), 404
        path = latest_camera_capture(camera_id)
        if path is None:
            return jsonify(error="No image has been saved for this camera yet."), 404
        return send_file(path, mimetype="image/jpeg", max_age=0)

    @app.get("/api/captures")
    def captures() -> Response:
        after = request.args.get("after", "")
        camera_id = request.args.get("camera_id")
        camera_registry_id = request.args.get("camera_registry_id")
        try:
            limit = max(1, min(200, int(request.args.get("limit", "50"))))
        except ValueError:
            return jsonify(error="limit must be an integer"), 400
        paths = sorted(capture_files(store.root), key=lambda item: item.name)
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
        if request.args.get("order") == "desc":
            before = request.args.get("before", "")
            if before:
                paths = [path for path in paths if path.stem < before]
            paths.reverse()
        selected = paths[:limit]
        return jsonify(
            captures=[capture_summary(path) for path in selected],
            next_after=selected[-1].stem if selected else after or None,
            next_before=selected[-1].stem if selected else None,
            has_more=len(paths) > len(selected),
        )

    @app.get("/api/collection/export.csv")
    def collection_export():
        try:
            from .collection import manifest_csv
        except ImportError:
            from collection import manifest_csv
        return Response(manifest_csv(store.root), mimetype="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="deployment_collection.csv"'})

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

    @app.get("/api/captures/<capture_id>/metadata")
    def capture_metadata(capture_id: str):
        path = capture_path(capture_id)
        if path is None or not path.with_suffix(".json").is_file():
            return jsonify(error="Capture metadata not found."), 404
        return send_file(path.with_suffix(".json"), mimetype="application/json", max_age=0)

    @app.post("/api/captures/<capture_id>/review")
    def review_capture(capture_id: str):
        path = capture_path(capture_id)
        if path is None:
            return jsonify(error="Capture not found."), 404
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or payload.get("decision") not in {"boxed", "bird", "no_bird", "unusable", "unreviewed"}:
            return jsonify(error="Choose boxed, bird, no_bird, unusable, or unreviewed."), 400
        note = payload.get("note", "")
        if not isinstance(note, str) or len(note) > 1000:
            return jsonify(error="Note must contain at most 1000 characters."), 400
        boxes = payload.get("boxes", [])
        species_name = payload.get("species_name") or None
        if species_name is not None and (not isinstance(species_name, str) or len(species_name) > 120):
            return jsonify(error="Species must be blank or a name of at most 120 characters."), 400
        if payload["decision"] == "boxed":
            if not isinstance(boxes, list) or not 1 <= len(boxes) <= 50:
                return jsonify(error="Confirm between 1 and 50 tight bird boxes."), 400
            from PIL import Image
            with Image.open(path) as image:
                width, height = image.size
            cleaned = []
            for box in boxes:
                if not isinstance(box, dict):
                    return jsonify(error="Each box must contain left, top, right, bottom."), 400
                values = [box.get(k) for k in ("left", "top", "right", "bottom")]
                if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
                    return jsonify(error="Box coordinates must be finite numbers."), 400
                left, top, right, bottom = values
                if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                    return jsonify(error="Boxes must have positive area within the original image."), 400
                cleaned.append(dict(zip(("left", "top", "right", "bottom"), values)))
            boxes = cleaned
        else:
            boxes, species_name = [], None
        with state_lock:
            history = []
            try:
                history = json.loads(path.with_suffix(".review.json").read_text(encoding="utf-8")).get("history", [])
            except (OSError, ValueError):
                pass
            entry = {"decision": payload["decision"], "note": note,
                     "boxes": boxes, "species_name": species_name,
                     "reviewed_at": datetime.now(timezone.utc).isoformat()}
            record = {"schema_version": 1, "capture_id": capture_id,
                      "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "scope": "human_boxes" if boxes else "frame_review",
                      **entry, "history": history + [entry]}
            if boxes:
                try:
                    from .collection import export_bird_crops
                except ImportError:
                    from collection import export_bird_crops
                record["collection"] = export_bird_crops(path, {"status": "complete", "detector": {
                    "threshold": 0, "detections": [{"score": 1.0, "box": box} for box in boxes]}}, reviewed=True)
            try:
                from .collection import write_review_record
            except ImportError:
                from collection import write_review_record
            write_review_record(path.with_suffix(".review.json"), record)
        return jsonify(record)

    @app.get("/api/captures/<capture_id>/review-crops/<int:index>/image")
    def reviewed_crop_image(capture_id: str, index: int):
        original = capture_path(capture_id)
        if original is None:
            return jsonify(error="Capture not found."), 404
        try:
            review = json.loads(original.with_suffix(".review.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return jsonify(error="Review not found."), 404
        path = original.parent / "review_crops" / f"{capture_id}_bird_{index:02d}.png"
        if review.get("decision") != "boxed" or index >= len(review.get("boxes", [])) or not path.is_file():
            return jsonify(error="Reviewed crop not found."), 404
        return send_file(path, mimetype="image/png", max_age=0)

    @app.get("/api/captures/<capture_id>/crops/<int:index>/image")
    def crop_image(capture_id: str, index: int):
        original = capture_path(capture_id)
        if original is None:
            return jsonify(error="Capture not found."), 404
        inference = read_inference_record(original) or {}
        crops = (inference.get("collection") or {}).get("crops", [])
        crop = next((item for item in crops if item.get("detection_index") == index), None)
        path = original.parent / "crops" / f"{capture_id}_bird_{index:02d}.png"
        if crop is None or not path.is_file():
            return jsonify(error="Crop not found."), 404
        return send_file(path, mimetype="image/png", max_age=3600)

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

    def hub_local_midnight(now: float) -> float:
        local = time.localtime(now)
        return time.mktime((local.tm_year, local.tm_mon, local.tm_mday, 0, 0, 0, 0, 0, -1))

    def cpu_temperature() -> float | None:
        try:
            return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000, 1)
        except (OSError, ValueError):
            return None

    @app.get("/api/hub/specs")
    def specs() -> Response:
        """Hub and camera hardware/software facts for the Fleet tab's Specs section."""
        heartbeats = {item.get("camera_id"): item.get("last_heartbeat") for item in fleet.summary()}
        cameras = []
        for camera in registry.snapshot():
            sidecar = last_capture_record.get(camera["id"])
            if sidecar is None:
                path = latest_camera_capture(camera["id"])
                if path is not None:
                    try:
                        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                        last_capture_record[camera["id"]] = sidecar
                    except (OSError, ValueError):
                        sidecar = None
            heartbeat = heartbeats.get(camera.get("hardware_camera_id") or (sidecar or {}).get("camera_id"))
            cameras.append(camera_specs(camera, heartbeat, sidecar))
        return jsonify(hub=hub_specs(store.root, models=inference_engine.describe() if inference_engine else None,
                                     clock=hub_clock.state(), started_at=hub_started), cameras=cameras)

    @app.get("/api/hub/fleet")
    def fleet_status() -> Response:
        """Per-camera health and today's counts for the Fleet tab (no capture-folder scan)."""
        now = time.time()
        try:
            since = float(request.args.get("since", ""))
            if not math.isfinite(since) or since > now:
                raise ValueError
        except ValueError:
            since = hub_local_midnight(now)
        since = max(since, now - FLEET_KEEP_SECONDS)
        cameras = registry.snapshot()
        with runtime_lock:
            services = {camera_id: service.describe() for camera_id, service in motion_services.items()}
        heartbeats = {item.get("camera_id"): item for item in fleet.summary()}
        hardware_to_camera = {s.get("camera_id"): camera_id for camera_id, s in services.items() if s.get("camera_id")}
        hardware_to_camera.update({c["hardware_camera_id"]: c["id"] for c in cameras if c.get("hardware_camera_id")})
        totals = fleet_stats.totals(since)
        visits = visit_totals(closed_visits().values(), since, lambda hardware: hardware_to_camera.get(hardware, hardware))
        settings = hub_settings.snapshot()

        def capture_state(camera: dict[str, Any], service: dict[str, Any]) -> str:
            if camera.get("streaming"):
                return "framing"
            if service.get("last_capture_error") or service.get("last_analysis_error") or service.get("last_error"):
                return "error"
            if service.get("running"):
                return "capturing"
            if settings.get("capture_enabled") is False:
                return "hub_paused"
            return "paused"

        def row(camera_id: str, camera: dict[str, Any] | None) -> dict[str, Any]:
            camera = camera or {}
            service = services.get(camera_id, {})
            heartbeat = heartbeats.get(camera.get("hardware_camera_id") or service.get("camera_id"))
            counts = totals.get(camera_id, {})
            visit = visits.get(camera_id, {})
            return {
                "id": camera_id,
                "name": camera.get("name") or camera_id,
                "color": camera.get("color"),
                "parked": is_parked(camera) if camera else False,
                "registered": bool(camera),
                "url": camera.get("url"),
                "sensor": (camera.get("sensor") or {}).get("model") or (camera.get("status") or {}).get("camera_model"),
                "heartbeat": ({key: heartbeat.get(key) for key in (
                    "status", "seconds_since_seen", "restarts_24h", "bad_restarts_24h", "crashes_24h", "ip")}
                    if heartbeat else None),
                "capture": capture_state(camera, service) if camera else "removed",
                "last_capture_at": service.get("last_capture_at") or service.get("last_completed_at"),
                "frames": counts.get("frames", 0),
                "analysed": counts.get("analysed", 0),
                "bird_frames": counts.get("bird_frames", 0),
                "analysis_errors": counts.get("errors", 0),
                "top_species": [{"label": label, "frames": n} for label, n in counts.get("labels", [])],
                "mean_processing_ms": counts.get("mean_processing_ms"),
                "visits": visit.get("visits", 0),
                "visits_auto": visit.get("auto", 0),
                "visits_review": visit.get("review", 0),
                "failures_since_start": service.get("failures", 0),
                "waiting": service.get("inference_waiting", 0),
                "last_error": (service.get("last_capture_error") or service.get("last_analysis_error")
                               or service.get("last_error")),
            }

        rows = [row(camera["id"], camera) for camera in cameras]
        known = {camera["id"] for camera in cameras}
        rows += [row(camera_id, None) for camera_id in sorted(set(totals) | set(visits)) if camera_id not in known]
        models = inference_engine.describe() if inference_engine is not None else {}
        disk = shutil.disk_usage(store.root)
        try:
            load = round(os.getloadavg()[0], 2)
        except (AttributeError, OSError):
            load = None
        return jsonify(
            since=since,
            generated_at=now,
            counted_since=fleet_stats.earliest(),
            cameras=rows,
            hub={
                "storage_free_bytes": disk.free,
                "storage_total_bytes": disk.total,
                "uptime_seconds": round(now - hub_started),
                "queue": pending.unfinished_tasks + sum(s.get("inference_waiting", 0) for s in services.values()),
                "models": {"bird": bool(models.get("detector") or models.get("is_bird")),
                           "localizer": bool(models.get("localizer")),
                           "species": bool(models.get("species"))},
                "capture_enabled": settings.get("capture_enabled", True),
                "collection_mode": settings.get("collection_mode"),
                "clock": hub_clock.state(),
                "load_1m": load,
                "cpu_temperature_c": cpu_temperature(),
            },
        )

    @app.get("/health")
    def health() -> Response:
        with state_lock:
            path = latest["path"]
            received_at = latest["received_at"]
            classification = latest["classification"]
            latest_inference = latest["inference"]
        with runtime_lock:
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
            collection_mode=collection_mode,
            pipeline=settings_response(),
            storage_free_bytes=shutil.disk_usage(store.root).free,
            hub_clock=hub_clock.state(),
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
        "--detector-model",
        type=Path,
        default=DEFAULT_DETECTOR_MODEL,
        help="Float motion-crop detector TFLite model",
    )
    parser.add_argument(
        "--detector-contract-file",
        type=Path,
        help="detector_contract.json (default: beside the detector model)",
    )
    parser.add_argument(
        "--detector-threshold",
        type=float,
        help="Override detector top-box score threshold (default: contract value 0.5)",
    )
    parser.add_argument(
        "--is-bird-model",
        type=Path,
        help="Binary bird/not-bird TFLite model (disables the default detector)",
    )
    parser.add_argument(
        "--localizer-model",
        type=Path,
        help="ROI detector that only localises birds for species crops (use with --is-bird-model), "
        "e.g. the v3 esp32_detector_int8.tflite with esp32_detector_contract.json beside it",
    )
    parser.add_argument("--localizer-contract", type=Path, help="Localizer contract (default: beside the model)")
    parser.add_argument(
        "--species-contract",
        type=Path,
        help="species_contract.json for a bird-crop species model (default: beside the species model; "
        "without one the species model gets the whole frame, as before)",
    )
    parser.add_argument("--visit-gap-seconds", type=float, default=120.0,
                        help="Captures from one camera closer than this form one visit (default: 120)")
    parser.add_argument("--no-visits", action="store_true", help="Do not track visits or write visits.jsonl")
    parser.add_argument("--no-static-box-filter", action="store_true",
                        help="Classify every localiser box, even ones that also appear on empty frames "
                        "(scenery such as a shrub or the feeder tube); see server/static_boxes.py")
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
    parser.add_argument("--collection-mode", action="store_true",
                        help="Detector only; retain every processed frame and export unlabelled candidate bird crops")
    parser.add_argument("--deployment-id", default="",
                        help="Scene/deployment identifier recorded for grouping data; change when cameras move")
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
    parser.add_argument(
        "--hub-name",
        default="birdhub",
        help="Name cameras look this server up by, as <name>.local (default: birdhub)",
    )
    parser.add_argument(
        "--no-advertise",
        action="store_true",
        help="Do not answer for <hub-name>.local; cameras then need server_ip",
    )
    parser.add_argument(
        "--no-auto-register",
        action="store_true",
        help="Do not add or re-address cameras from their heartbeats; manage the registry by hand",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        detector_model = None if args.is_bird_model is not None else args.detector_model
        engine = BirdInferenceEngine(
            args.is_bird_model,
            args.species_model,
            detector_model=detector_model,
            detector_contract_file=args.detector_contract_file,
            detector_threshold=args.detector_threshold,
            species_labels=args.species_labels,
            threshold_file=args.threshold_file,
            contract_file=args.contract_file,
            bird_threshold=args.bird_threshold,
            top_k=args.top_k,
            num_threads=args.num_threads,
            species_enabled=not args.collection_mode,
            localizer_model=args.localizer_model,
            localizer_contract_file=args.localizer_contract,
            species_contract_file=args.species_contract,
        )
    except Exception as error:
        raise SystemExit(f"Model startup error: {error}") from error
    visit_tracker = None
    if not args.no_visits:
        try:
            from .visits import VisitTracker
        except ImportError:
            from visits import VisitTracker
        species_contract = getattr(engine, "species_contract", None)
        visit_tracker = VisitTracker(args.save_dir / "visits.jsonl", gap_seconds=args.visit_gap_seconds,
                                     automation=species_contract.automation if species_contract is not None else None)
    print(f"Models ready: {json.dumps(engine.describe(), sort_keys=True)}", flush=True)
    print(f"Bird decision: {engine.decision_rule} (from {engine.threshold_source})", flush=True)
    if engine.contract is None and engine.detector_contract is None:
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
        auto_register_cameras=not args.no_auto_register,
        manage_system_clock=True,
        start_hub_status=True,
        collection_mode=args.collection_mode,
        deployment_id=args.deployment_id,
        visit_tracker=visit_tracker,
        static_box_filter=not args.no_static_box_filter,
    )
    try:
        from .hub_advertiser import start_hub_advertiser
    except ImportError:
        from hub_advertiser import start_hub_advertiser
    advertiser = (
        None if args.no_advertise else start_hub_advertiser(args.hub_name, args.port, bind_host=args.host)
    )
    try:
        application.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        if advertiser is not None:
            advertiser.stop()
        application.config["HUB_STATUS_LOGGER"].stop()
        application.config["CLEANUP"]()
