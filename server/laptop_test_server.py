"""Laptop receiver and browser dashboard for the ESP32-CAM."""

from __future__ import annotations

import argparse
import json
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, render_template_string, request, send_file

try:
    from .capture_storage import (
        METADATA_HEADER,
        CalibrationType,
        CaptureStore,
        calibration_type_values,
        is_jpeg,
        parse_calibration_type,
        parse_camera_metadata,
    )
    from .roi_control import ROI_CONTROL_JS
except ImportError:  # Direct execution: python server/laptop_test_server.py
    from capture_storage import (
        METADATA_HEADER,
        CalibrationType,
        CaptureStore,
        calibration_type_values,
        is_jpeg,
        parse_calibration_type,
        parse_camera_metadata,
    )
    from roi_control import ROI_CONTROL_JS


DEFAULT_SAVE_DIR = Path.home() / "Pictures" / "ESP32Cam"
MAX_JPEG_BYTES = 10 * 1024 * 1024
PROXY_TIMEOUT_SECONDS = 8
CAMERA_PROBE_TIMEOUT_SECONDS = 3
STREAM_READ_BYTES = 16 * 1024


class CameraBusyError(RuntimeError):
    """Raised when an exclusive camera operation is already active."""


def normalize_camera_url(value: str | None) -> str | None:
    """Validate and normalize an ESP32 HTTP base URL."""
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        raise ValueError("Camera URL must not be empty.")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port  # Force validation of malformed/out-of-range ports.
    except ValueError as error:
        raise ValueError(f"Invalid camera URL: {error}.") from error
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Camera URL must use http:// or https://.")
    if not parsed.hostname:
        raise ValueError("Camera URL must include a host.")
    if any(character.isspace() for character in parsed.hostname):
        raise ValueError("Camera URL host must not contain whitespace.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Camera URL must not include credentials.")
    if parsed.path not in {"", "/"}:
        raise ValueError("Camera URL must be a base URL without a path.")
    if parsed.query or parsed.fragment:
        raise ValueError("Camera URL must not include a query string or fragment.")

    host = parsed.hostname.lower()
    if ":" in host:  # urlsplit removes IPv6 brackets from hostname.
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, "", "", ""))


def camera_url_argument(value: str) -> str:
    """Argparse adapter that preserves the actionable URL validation message."""
    try:
        normalized = normalize_camera_url(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    assert normalized is not None
    return normalized


def local_source_ip(camera_url: str | None) -> str | None:
    """Return the local address selected by the OS route to the camera, if known."""
    if not camera_url:
        return None
    parsed = urlsplit(camera_url)
    target_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = socket.getaddrinfo(parsed.hostname, target_port, type=socket.SOCK_DGRAM)
    except OSError:
        return None
    for family, socktype, protocol, _, address in addresses:
        try:
            with socket.socket(family, socktype, protocol) as route_socket:
                route_socket.connect(address)
                selected = route_socket.getsockname()[0]
                if selected and selected not in {"0.0.0.0", "::"}:
                    return selected
        except OSError:
            continue
    return None


def preflight_port(host: str, port: int) -> None:
    """Verify that Flask's requested listening address can be bound, then release it."""
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535.")
    try:
        addresses = socket.getaddrinfo(
            host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )
    except OSError as error:
        raise RuntimeError(f"Cannot resolve listening host {host!r}: {error}") from error

    last_error: OSError | None = None
    for family, socktype, protocol, _, address in addresses:
        try:
            with socket.socket(family, socktype, protocol) as listener:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                listener.bind(address)
            return
        except OSError as error:
            last_error = error
    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(
        f"Cannot listen on {host}:{port}; the address or port is unavailable{detail}. "
        "On Windows, identify the owner with: "
        f"Get-NetTCPConnection -LocalPort {port} -State Listen"
    )


def print_startup_targets(host: str, port: int, camera_url: str | None) -> str | None:
    """Print both directions of the camera/laptop connection and return the LAN IP."""
    source_ip = local_source_ip(camera_url)
    print(f"Local dashboard: http://127.0.0.1:{port}", flush=True)
    if source_ip:
        print(f"LAN dashboard: http://{source_ip}:{port}", flush=True)
        print(f"ESP32 upload target: http://{source_ip}:{port}/upload", flush=True)
    else:
        print(f"LAN dashboard: unavailable (server bind: {host}:{port})", flush=True)
        print("ESP32 upload target: unavailable; use this laptop's LAN IP with /upload", flush=True)
    if camera_url:
        print(f"Laptop camera target: {camera_url}", flush=True)
    else:
        print("Laptop camera target: not configured (--camera-url)", flush=True)
    return source_ip

DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32-CAM dashboard</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #101418; color: #edf2f7; }
    header, main { width: min(1150px, calc(100% - 2rem)); margin: auto; }
    header { padding: 1.5rem 0 .5rem; }
    h1, h2 { margin: 0 0 .75rem; }
    .muted { color: #aab5c0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 1rem; padding-bottom: 2rem; }
    section { background: #192128; border: 1px solid #2d3943; border-radius: 12px; padding: 1rem; }
    img { display: block; width: 100%; min-height: 240px; object-fit: contain; background: #080a0c; border-radius: 8px; }
    button, select, input { padding: .6rem .75rem; margin: .25rem .25rem .25rem 0; }
    button { cursor: pointer; }
    button:disabled { cursor: not-allowed; opacity: .55; }
    label { display: block; margin-top: .65rem; }
    label input, label select { display: block; width: 100%; box-sizing: border-box; }
    textarea { width: 100%; min-height: 180px; box-sizing: border-box; font: 13px monospace; }
    pre { min-height: 5rem; max-height: 24rem; overflow: auto; white-space: pre-wrap; }
    .controls { margin: .6rem 0; }
    .badge { display: inline-block; padding: .25rem .55rem; border-radius: 999px; background: #303b44; }
    .badge.ok { background: #174b31; } .badge.running { background: #173f61; }
    .badge.error { background: #632828; }
    .ok { color: #73d99f; } .error { color: #ff8b8b; } .running { color: #78bfff; }
    .roi-stage { position: relative; width: 100%; background: #080a0c; border-radius: 8px; overflow: hidden; }
    .roi-stage img { min-height: 0; height: auto; object-fit: fill; }
    .roi-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
  </style>
</head>
<body>
  <header>
    <h1>ESP32-CAM laptop dashboard</h1>
    <p id="connection" class="muted">Loading server information...</p>
    <p>Camera state: <strong id="camera-state" class="badge">Not checked</strong></p>
  </header>
  <main class="grid">
    <section>
      <h2>Camera preview</h2>
      <img id="live" alt="ESP32-CAM preview is stopped">
      <div class="controls">
        <button id="start" onclick="startLive()">Start stream</button>
        <button id="stop" onclick="stopLive()" disabled>Stop stream</button>
        <button id="arm-motion" class="camera-action" onclick="setCameraMode('motion')">Arm motion</button>
        <button id="set-idle" class="camera-action" onclick="setCameraMode('idle')">Idle</button>
      </div>
      <p id="live-message" class="muted">Preview stopped.</p>
    </section>
    <section>
      <h2>Latest saved capture</h2>
      <img id="latest" alt="No saved capture received yet">
      <p id="upload-message" class="muted">Waiting for a saved capture...</p>
    </section>
    <section>
      <h2>Manual / calibration capture</h2>
      <label>Capture type
        <select id="calibration-type">
          <option value="normal">Normal</option>
          <option value="dark_frame">Dark frame (cover the lens)</option>
          <option value="flat_field">Flat field</option>
          <option value="gray_card">Gray card</option>
          <option value="color_chart">Color chart</option>
          <option value="resolution_target">Resolution target</option>
        </select>
      </label>
      <label>Label<input id="capture-label" maxlength="120" placeholder="Optional dataset label"></label>
      <label>Note<input id="capture-note" maxlength="500" placeholder="Lighting, target, distance, etc."></label>
      <button id="capture" class="camera-action" onclick="saveCapture()">Capture and save</button>
      <button class="camera-action" onclick="saveProfile()">Save camera profile</button>
      <p class="muted">Each saved JPEG gets a JSON sidecar and a manifest.jsonl entry.</p>
    </section>
    <section>
      <h2>Motion region of interest</h2>
      <div class="roi-stage">
        <img id="roi-preview" data-roi-preview alt="Refresh the camera preview, then drag to select a motion ROI">
        <canvas id="roi-overlay" aria-label="Drag to select a rectangular motion region"></canvas>
      </div>
      <p id="roi-values" class="muted">x 0.000, y 0.000, width 1.000, height 1.000</p>
      <div class="controls">
        <button class="camera-action" onclick="refreshRoiPreview()">Refresh ROI preview</button>
        <button class="camera-action" onclick="loadMotionRoi()">Load current ROI</button>
        <button class="camera-action" onclick="resetMotionRoi()">Reset full frame</button>
        <button class="camera-action" onclick="saveMotionRoi()">Save ROI</button>
      </div>
      <p class="muted">Drag with a mouse or touch. Save sends only the nested <code>motion_roi</code> update.</p>
    </section>
    <section>
      <h2>Camera checks</h2>
      <button class="camera-action" onclick="cameraGet('status')">Status details</button>
      <button class="camera-action" onclick="cameraGet('motion')">Consume motion event</button>
      <button class="camera-action" onclick="cameraGet('metadata')">Metadata</button>
      <button class="camera-action" onclick="loadConfig()">Load config</button>
      <pre id="result">Choose an action.</pre>
    </section>
    <section>
      <h2>Camera configuration</h2>
      <textarea id="config">{
  "motion_threshold": 20,
  "motion_sensitivity": 0.05,
  "motion_check_ms": 500,
  "motion_enabled": true,
  "jpeg_quality": 10,
  "framesize": 8
}</textarea>
      <button class="camera-action" onclick="saveConfig()">Send config to camera</button>
    </section>
  </main>
  <script>
    let cameraConfigured = false;
    let liveRunning = false;
    const result = document.getElementById('result');
    const liveImage = document.getElementById('live');

    function setState(label, style = 'muted') {
      const badge = document.getElementById('camera-state');
      badge.textContent = label.replaceAll('_', ' ');
      badge.className = `badge ${style}`;
    }

    function updateControls() {
      document.getElementById('start').disabled = liveRunning || !cameraConfigured;
      document.getElementById('stop').disabled = !liveRunning;
      document.querySelectorAll('.camera-action').forEach(button => {
        button.disabled = liveRunning || !cameraConfigured;
      });
    }

    async function refreshInfo() {
      const response = await fetch('/api/info', {cache: 'no-store'});
      const info = await response.json();
      cameraConfigured = Boolean(info.camera_url);
      const cameraState = info.camera_reachable === true ? 'reachable'
        : info.camera_reachable === false ? `unreachable (${info.camera_error || 'request failed'})`
        : 'not checked yet';
      document.getElementById('connection').textContent = cameraConfigured
        ? `Camera: ${info.camera_url} | ${cameraState} | Dataset: ${info.save_dir}`
        : `No camera URL set. Motion uploads can still be received in ${info.save_dir}`;
      if (info.latest_upload) {
        document.getElementById('latest').src = `/latest.jpg?t=${Date.now()}`;
        document.getElementById('upload-message').textContent =
          `${info.latest_upload.name} | ${info.latest_upload.source || 'legacy'} | ${info.latest_upload.bytes} bytes`;
      }
      if (!liveRunning && info.camera_streaming) {
        setState('streaming', 'running');
      }
      updateControls();
    }

    async function refreshCameraStatus() {
      if (!cameraConfigured || liveRunning) return;
      try {
        const response = await fetch('/camera/status', {cache: 'no-store'});
        const text = await response.text();
        if (!response.ok) throw new Error(text);
        const status = JSON.parse(text);
        const style = status.runtime_state === 'error' ? 'error'
          : status.runtime_state === 'motion_armed' ? 'ok' : 'muted';
        setState(status.runtime_state || status.selected_mode || 'unknown', style);
      } catch (error) {
        setState('unreachable', 'error');
      }
    }

    function startLive() {
      if (!cameraConfigured || liveRunning) return;
      liveRunning = true;
      updateControls();
      setState('streaming', 'running');
      const message = document.getElementById('live-message');
      message.className = 'running';
      message.textContent = 'Connecting to MJPEG stream...';
      liveImage.src = `/camera/stream?t=${Date.now()}`;
    }

    function stopLive() {
      liveRunning = false;
      liveImage.removeAttribute('src');
      updateControls();
      const message = document.getElementById('live-message');
      message.className = 'muted';
      message.textContent = 'Stream stopped.';
      setState('checking');
      setTimeout(() => refreshCameraStatus(), 250);
    }

    liveImage.addEventListener('load', () => {
      if (!liveRunning) return;
      const message = document.getElementById('live-message');
      message.className = 'running';
      message.textContent = 'MJPEG stream running. Other camera functions are paused.';
    });
    liveImage.addEventListener('error', () => {
      if (!liveRunning) return;
      liveRunning = false;
      updateControls();
      setState('stream error', 'error');
      const message = document.getElementById('live-message');
      message.className = 'error';
      message.textContent = 'Stream ended or could not be opened.';
    });

    async function setCameraMode(mode) {
      if (liveRunning) stopLive();
      await new Promise(resolve => setTimeout(resolve, 250));
      try {
        const response = await fetch('/camera/mode', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mode})
        });
        const text = await response.text();
        if (!response.ok) throw new Error(text);
        const status = JSON.parse(text);
        setState(status.runtime_state || mode, mode === 'motion' ? 'running' : 'muted');
        showResult(text);
        setTimeout(() => refreshCameraStatus(), 750);
      } catch (error) { showError(error); setState('error', 'error'); }
    }

    async function saveCapture() {
      const button = document.getElementById('capture');
      button.disabled = true;
      try {
        const payload = {
          calibration_type: document.getElementById('calibration-type').value,
          label: document.getElementById('capture-label').value,
          note: document.getElementById('capture-note').value
        };
        const response = await fetch('/api/captures', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
        });
        const text = await response.text();
        if (!response.ok) throw new Error(text);
        showResult(text);
        await refreshInfo();
      } catch (error) { showError(error); }
      finally { button.disabled = false; }
    }

    async function saveProfile() {
      try {
        const response = await fetch('/api/profiles', {method: 'POST'});
        const text = await response.text();
        if (!response.ok) throw new Error(text);
        showResult(text);
      } catch (error) { showError(error); }
    }

    async function cameraGet(endpoint) {
      const response = await fetch(`/camera/${endpoint}`, {cache: 'no-store'});
      const text = await response.text();
      if (!response.ok) throw new Error(text);
      showResult(text);
      return text;
    }

    async function loadConfig() {
      try {
        const text = await cameraGet('config');
        document.getElementById('config').value = JSON.stringify(JSON.parse(text), null, 2);
      } catch (error) { showError(error); }
    }

    async function saveConfig() {
      try {
        const config = JSON.parse(document.getElementById('config').value);
        const response = await fetch('/camera/config', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(config)
        });
        const text = await response.text();
        if (!response.ok) throw new Error(text);
        showResult(text);
      } catch (error) { showError(error); }
    }

    function pretty(text) {
      try { return JSON.stringify(JSON.parse(text), null, 2); } catch (_) { return text; }
    }
    function showResult(text) { result.className = 'ok'; result.textContent = pretty(text); }
    function showError(error) { result.className = 'error'; result.textContent = String(error); }
    window.showCameraResult = (text, failed = false) => failed ? showError(text) : showResult(text);

    window.addEventListener('beforeunload', () => {
      stopLive();
    });
    Promise.all([refreshInfo(), refreshCameraStatus()]).catch(showError);
    setInterval(() => {
      refreshInfo().catch(showError);
      refreshCameraStatus().catch(showError);
    }, 3000);
  </script>
  <script>{{ roi_control_js|safe }}</script>
</body>
</html>
"""


def create_app(
    save_dir: Path = DEFAULT_SAVE_DIR,
    camera_url: str | None = None,
    probe_camera: bool = False,
) -> Flask:
    """Create the Flask application. Parameters are injectable for tests."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_JPEG_BYTES
    store = CaptureStore(save_dir)
    camera_url = normalize_camera_url(camera_url)
    camera_lock = threading.Lock()
    state_lock = threading.Lock()
    camera_state: dict[str, Any] = {
        "reachable": None,
        "error": None,
        "checked_at": None,
        "streaming": False,
    }

    existing = sorted(store.root.glob("*.jpg"), key=lambda path: path.stat().st_mtime)
    latest: dict[str, Any] = {"path": existing[-1] if existing else None, "record": None}

    def record_camera_state(reachable: bool, error: str | None = None) -> None:
        with state_lock:
            camera_state["reachable"] = reachable
            camera_state["error"] = error
            camera_state["checked_at"] = datetime.now(timezone.utc).isoformat()

    def camera_error_message(
        error: BaseException,
        target: str,
        timeout: int = PROXY_TIMEOUT_SECONDS,
    ) -> str:
        reason = error.reason if isinstance(error, URLError) else error
        if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower():
            return (
                f"Camera request to {target} timed out after {timeout} seconds. "
                "Confirm the ESP32 IP shown in its serial monitor, power, and Wi-Fi connection."
            )
        return f"Camera request to {target} failed: {reason}"

    def upstream_request(
        endpoint: str,
        method: str = "GET",
        body: bytes | None = None,
        timeout: int = PROXY_TIMEOUT_SECONDS,
    ) -> tuple[bytes, int, str, str | None]:
        if not camera_url:
            raise RuntimeError("Camera URL is not configured. Start the server with --camera-url.")
        target = f"{camera_url}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"} if body is not None else {}
        outgoing = Request(target, data=body, headers=headers, method=method)
        if not camera_lock.acquire(blocking=False):
            raise CameraBusyError("Camera is busy streaming. Stop the stream and try again.")
        try:
            with urlopen(outgoing, timeout=timeout) as upstream:
                content = upstream.read(MAX_JPEG_BYTES + 1)
                if len(content) > MAX_JPEG_BYTES:
                    raise ValueError("Camera response exceeded the 10 MiB limit.")
                result = (
                    content,
                    upstream.status,
                    upstream.headers.get_content_type(),
                    upstream.headers.get(METADATA_HEADER),
                )
            record_camera_state(True)
            return result
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            record_camera_state(False, camera_error_message(error, target, timeout))
            raise
        finally:
            camera_lock.release()

    def stream_response() -> Response | tuple[Response, int]:
        if not camera_url:
            return jsonify(error="Camera URL is not configured. Start the server with --camera-url."), 503
        if not camera_lock.acquire(blocking=False):
            return jsonify(error="Camera is busy. Stop the active operation and try again."), 409

        target = f"{camera_url}/stream"
        upstream = None
        try:
            upstream = urlopen(Request(target, method="GET"), timeout=PROXY_TIMEOUT_SECONDS)
            content_type = upstream.headers.get("Content-Type", "")
            if upstream.status != 200 or not content_type.lower().startswith(
                "multipart/x-mixed-replace"
            ):
                upstream.close()
                camera_lock.release()
                return jsonify(error="Camera did not return an MJPEG stream."), 502
            with state_lock:
                camera_state["streaming"] = True
            record_camera_state(True)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            if upstream is not None:
                upstream.close()
            camera_lock.release()
            record_camera_state(False, camera_error_message(error, target))
            return jsonify(error=camera_error_message(error, target)), 502

        def generate():
            try:
                while True:
                    chunk = upstream.read(STREAM_READ_BYTES)
                    if not chunk:
                        break
                    yield chunk
            except (URLError, TimeoutError, OSError, ValueError) as error:
                record_camera_state(False, camera_error_message(error, target))
            finally:
                upstream.close()
                with state_lock:
                    camera_state["streaming"] = False
                camera_lock.release()

        response = Response(generate(), status=upstream.status, content_type=content_type)
        response.headers["Cache-Control"] = "no-store"
        return response

    def probe_camera_status() -> None:
        if not camera_url:
            return
        target = f"{camera_url}/status"
        try:
            _, status, _, _ = upstream_request("status", timeout=CAMERA_PROBE_TIMEOUT_SECONDS)
            if 200 <= status < 300:
                print(f"Camera status probe succeeded: {target} (HTTP {status})", flush=True)
            else:
                print(f"WARNING: Camera status probe returned HTTP {status}: {target}", flush=True)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            warning = camera_state["error"] or camera_error_message(
                error, target, CAMERA_PROBE_TIMEOUT_SECONDS
            )
            print(
                f"WARNING: {warning} Upload reception will remain available; "
                "check the ESP32 serial monitor IP and ensure both devices are on the same network.",
                flush=True,
            )

    if probe_camera and camera_url:
        probe_camera_status()

    def proxy_response(endpoint: str, method: str = "GET", body: bytes | None = None) -> Response:
        try:
            content, status, content_type, metadata_header = upstream_request(endpoint, method, body)
            response = Response(content, status=status, content_type=content_type)
            if metadata_header:
                response.headers[METADATA_HEADER] = metadata_header
            response.headers["Cache-Control"] = "no-store"
            return response
        except CameraBusyError as error:
            return jsonify(error=str(error)), 409
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        except HTTPError as error:
            return Response(error.read(), status=error.code, content_type="text/plain")
        except (URLError, TimeoutError, OSError, ValueError) as error:
            target = f"{camera_url}/{endpoint.lstrip('/')}"
            return jsonify(error=camera_error_message(error, target)), 502

    def update_latest(record: dict[str, Any]) -> None:
        with state_lock:
            latest["path"] = store.root / record["filename"]
            latest["record"] = record

    @app.get("/")
    def dashboard() -> str:
        return render_template_string(DASHBOARD_HTML, roi_control_js=ROI_CONTROL_JS)

    @app.post("/upload")
    def upload() -> tuple[Response, int] | Response:
        image = request.get_data(cache=False)
        if not image:
            return jsonify(error="Request body is empty; expected raw JPEG bytes."), 400
        if not is_jpeg(image):
            return jsonify(error="Request body is not a complete JPEG image."), 415
        metadata, metadata_error = parse_camera_metadata(request.headers.get(METADATA_HEADER))
        record = store.save_capture(
            image,
            source=(metadata or {}).get("source", "motion"),
            camera_metadata=metadata,
            metadata_error=metadata_error,
        )
        update_latest(record)
        print(f"Received {len(image)} bytes: {store.root / record['filename']}", flush=True)
        return jsonify(status="ok", **record)

    @app.post("/api/captures")
    def manual_capture() -> tuple[Response, int] | Response:
        options = request.get_json(silent=True) or {}
        if not isinstance(options, dict):
            return jsonify(error="Expected a JSON object."), 400
        try:
            calibration_type = parse_calibration_type(
                options.get("calibration_type", CalibrationType.NORMAL.value)
            )
        except ValueError:
            allowed = list(calibration_type_values())
            return jsonify(
                error=(
                    f"Unsupported calibration_type {options.get('calibration_type')!r}; "
                    f"allowed values: {', '.join(allowed)}."
                ),
                allowed_values=allowed,
            ), 422
        try:
            image, status, content_type, metadata_header = upstream_request("capture")
        except CameraBusyError as error:
            return jsonify(error=str(error)), 409
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        except HTTPError as error:
            return jsonify(error=f"Camera capture failed with HTTP {error.code}."), 502
        except (URLError, TimeoutError, OSError, ValueError) as error:
            target = f"{camera_url}/capture"
            return jsonify(error=camera_error_message(error, target)), 502
        if status != 200 or content_type != "image/jpeg" or not is_jpeg(image):
            return jsonify(error="Camera did not return a complete JPEG."), 502
        metadata, metadata_error = parse_camera_metadata(metadata_header)
        record = store.save_capture(
            image,
            source="manual",
            camera_metadata=metadata,
            metadata_error=metadata_error,
            calibration_type=calibration_type,
            label=str(options.get("label", ""))[:120],
            note=str(options.get("note", ""))[:500],
        )
        update_latest(record)
        return jsonify(status="ok", **record)

    @app.post("/api/profiles")
    def save_profile() -> tuple[Response, int] | Response:
        profile: dict[str, Any] = {}
        try:
            for endpoint in ("metadata", "config", "status"):
                content, status, _, _ = upstream_request(endpoint)
                if status != 200:
                    return jsonify(error=f"Camera {endpoint} request returned HTTP {status}."), 502
                profile[endpoint] = json.loads(content)
        except CameraBusyError as error:
            return jsonify(error=str(error)), 409
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as error:
            return jsonify(error=f"Could not save camera profile: {error}"), 502
        return jsonify(result="ok", **store.save_profile(profile))

    @app.get("/latest.jpg")
    def latest_image() -> Response | tuple[Response, int]:
        with state_lock:
            path = latest["path"]
        if path is None or not path.exists():
            return jsonify(error="No image has been saved yet."), 404
        return send_file(path, mimetype="image/jpeg", max_age=0)

    @app.get("/api/info")
    def server_info() -> Response:
        with state_lock:
            path = latest["path"]
            record = latest["record"]
        upload = None
        if path is not None and path.exists():
            upload = {
                "name": path.name,
                "bytes": path.stat().st_size,
                "received_at": (record or {}).get("received_at"),
                "source": (record or {}).get("source"),
            }
        with state_lock:
            reachable = camera_state["reachable"]
            camera_error = camera_state["error"]
            checked_at = camera_state["checked_at"]
            streaming = camera_state["streaming"]
        return jsonify(
            camera_url=camera_url,
            camera_reachable=reachable,
            camera_error=camera_error,
            camera_checked_at=checked_at,
            camera_streaming=streaming,
            save_dir=str(store.root),
            latest_upload=upload,
        )

    @app.get("/health")
    def health() -> Response:
        return jsonify(status="ok", save_dir=str(store.root), camera_configured=bool(camera_url))

    @app.get("/camera/capture")
    def camera_capture() -> Response:
        return proxy_response("capture")

    @app.get("/camera/stream")
    def camera_stream() -> Response | tuple[Response, int]:
        return stream_response()

    @app.get("/camera/status")
    def camera_status() -> Response:
        return proxy_response("status")

    @app.get("/camera/motion")
    def camera_motion() -> Response:
        return proxy_response("motion")

    @app.get("/camera/metadata")
    def camera_metadata() -> Response:
        return proxy_response("metadata")

    @app.get("/camera/config")
    def camera_config_get() -> Response:
        return proxy_response("config")

    @app.post("/camera/config")
    def camera_config_post() -> Response:
        config = request.get_json(silent=True)
        if not isinstance(config, dict):
            return jsonify(error="Expected a valid JSON object."), 400
        return proxy_response("config", method="POST", body=json.dumps(config).encode("utf-8"))

    @app.post("/camera/mode")
    def camera_mode() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or payload.get("mode") not in {"motion", "idle"}:
            return jsonify(error="mode must be either motion or idle"), 422
        return proxy_response("mode", method="POST", body=json.dumps(payload).encode("utf-8"))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: all interfaces)")
    parser.add_argument("--port", type=int, default=5000, help="Listening port (default: 5000)")
    parser.add_argument(
        "--camera-url",
        type=camera_url_argument,
        help="ESP32-CAM plain HTTP(S) base URL, for example http://192.168.1.84",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="Dataset folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        preflight_port(args.host, args.port)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"Startup error: {error}") from error
    print_startup_targets(args.host, args.port, args.camera_url)
    application = create_app(args.save_dir, args.camera_url, probe_camera=True)
    application.run(host=args.host, port=args.port, debug=False, threaded=True)