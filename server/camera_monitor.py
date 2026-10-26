"""Persistent browser monitor for multiple ESP32-CAM devices."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, render_template_string, request

try:
    from .camera_registry import CameraRegistry as SharedCameraRegistry
    from .laptop_test_server import normalize_camera_url
except ImportError:  # Direct execution: python server/camera_monitor.py
    from camera_registry import CameraRegistry as SharedCameraRegistry
    from laptop_test_server import normalize_camera_url


DEFAULT_REGISTRY_FILE = Path.home() / ".esp32-camera-monitor" / "cameras.json"
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 3.0
MAX_JSON_BYTES = 1024 * 1024
MAX_CAMERA_NAME_LENGTH = 80
REGISTRY_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_camera_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Camera name must be a string.")
    name = " ".join(value.strip().split())
    if not name:
        raise ValueError("Camera name must not be empty.")
    if len(name) > MAX_CAMERA_NAME_LENGTH:
        raise ValueError(f"Camera name must be at most {MAX_CAMERA_NAME_LENGTH} characters.")
    return name


class LegacyCameraRegistry:
    """Persist camera definitions and maintain their latest status/config snapshots."""

    def __init__(
        self,
        registry_file: Path,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("Polling interval must be greater than zero.")
        if timeout_seconds <= 0:
            raise ValueError("Camera timeout must be greater than zero.")
        self.registry_file = Path(registry_file).expanduser().resolve()
        self.poll_seconds = float(poll_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self._lock = threading.RLock()
        self._poll_locks: dict[str, threading.Lock] = {}
        self._cameras: dict[str, dict[str, Any]] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    @staticmethod
    def _runtime_state() -> dict[str, Any]:
        return {
            "reachable": None,
            "healthy": False,
            "polling": False,
            "last_polled_at": None,
            "last_success_at": None,
            "status": None,
            "config": None,
            "errors": {"status": None, "config": None},
        }

    def _load(self) -> None:
        if not self.registry_file.exists():
            return
        try:
            document = json.loads(self.registry_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read camera registry {self.registry_file}: {error}") from error
        if not isinstance(document, dict) or document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ValueError("Camera registry has an unsupported or missing schema_version.")
        cameras = document.get("cameras")
        if not isinstance(cameras, list):
            raise ValueError("Camera registry cameras must be a JSON array.")
        loaded: dict[str, dict[str, Any]] = {}
        urls: set[str] = set()
        for item in cameras:
            if not isinstance(item, dict):
                raise ValueError("Every camera registry entry must be a JSON object.")
            camera_id = item.get("id")
            if not isinstance(camera_id, str) or not camera_id or camera_id in loaded:
                raise ValueError("Every camera registry entry must have a unique non-empty id.")
            name = normalize_camera_name(item.get("name"))
            if not isinstance(item.get("url"), str):
                raise ValueError("Every camera registry entry must have a URL string.")
            url = normalize_camera_url(item["url"])
            assert url is not None
            if url in urls:
                raise ValueError(f"Camera registry contains duplicate URL {url}.")
            urls.add(url)
            loaded[camera_id] = {
                "id": camera_id,
                "name": name,
                "url": url,
                "added_at": item.get("added_at") if isinstance(item.get("added_at"), str) else None,
                **self._runtime_state(),
            }
        self._cameras = loaded
        self._poll_locks = {camera_id: threading.Lock() for camera_id in loaded}

    def _persist_locked(self) -> None:
        document = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "cameras": [
                {
                    "id": camera["id"],
                    "name": camera["name"],
                    "url": camera["url"],
                    "added_at": camera["added_at"],
                }
                for camera in sorted(self._cameras.values(), key=lambda value: value["name"].casefold())
            ],
        }
        encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.registry_file.with_name(
            f".{self.registry_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with temporary.open("wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.registry_file)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _copy_camera(camera: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": camera["id"],
            "name": camera["name"],
            "url": camera["url"],
            "added_at": camera["added_at"],
            "reachable": camera["reachable"],
            "healthy": camera["healthy"],
            "polling": camera["polling"],
            "last_polled_at": camera["last_polled_at"],
            "last_success_at": camera["last_success_at"],
            "status": camera["status"],
            "config": camera["config"],
            "errors": dict(camera["errors"]),
        }

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            cameras = [self._copy_camera(camera) for camera in self._cameras.values()]
        return sorted(cameras, key=lambda value: (value["name"].casefold(), value["id"]))

    def get(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
            return self._copy_camera(camera) if camera is not None else None

    def add(self, name: Any, url: Any) -> dict[str, Any]:
        normalized_name = normalize_camera_name(name)
        if not isinstance(url, str):
            raise ValueError("Camera URL must be a string.")
        normalized_url = normalize_camera_url(url)
        assert normalized_url is not None
        with self._lock:
            if any(camera["url"] == normalized_url for camera in self._cameras.values()):
                raise ValueError("A camera with this URL is already registered.")
            camera_id = uuid.uuid4().hex
            camera = {
                "id": camera_id,
                "name": normalized_name,
                "url": normalized_url,
                "added_at": utc_now(),
                **self._runtime_state(),
            }
            self._cameras[camera_id] = camera
            self._poll_locks[camera_id] = threading.Lock()
            try:
                self._persist_locked()
            except Exception:
                del self._cameras[camera_id]
                del self._poll_locks[camera_id]
                raise
            return self._copy_camera(camera)

    def remove(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            camera = self._cameras.pop(camera_id, None)
            if camera is None:
                return None
            poll_lock = self._poll_locks.pop(camera_id, None)
            try:
                self._persist_locked()
            except Exception:
                self._cameras[camera_id] = camera
                if poll_lock is not None:
                    self._poll_locks[camera_id] = poll_lock
                raise
            return self._copy_camera(camera)

    def _fetch_json(self, camera_url: str, endpoint: str) -> dict[str, Any]:
        outgoing = Request(
            f"{camera_url}/{endpoint}",
            headers={"Accept": "application/json", "Cache-Control": "no-cache"},
            method="GET",
        )
        try:
            with urlopen(outgoing, timeout=self.timeout_seconds) as upstream:
                body = upstream.read(MAX_JSON_BYTES + 1)
                if len(body) > MAX_JSON_BYTES:
                    raise ValueError(f"/{endpoint} response exceeded the 1 MiB limit")
                if not 200 <= upstream.status < 300:
                    raise ValueError(f"/{endpoint} returned HTTP {upstream.status}")
        except HTTPError as error:
            raise ValueError(f"/{endpoint} returned HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise ValueError(f"/{endpoint} request failed: {reason}") from error
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"/{endpoint} did not return valid UTF-8 JSON") from error
        if not isinstance(document, dict):
            raise ValueError(f"/{endpoint} did not return a JSON object")
        return document

    def poll(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
            poll_lock = self._poll_locks.get(camera_id)
            if camera is None or poll_lock is None:
                return None
            camera_url = camera["url"]
        if not poll_lock.acquire(blocking=False):
            return self.get(camera_id)
        try:
            with self._lock:
                current = self._cameras.get(camera_id)
                if current is None:
                    return None
                current["polling"] = True

            values: dict[str, dict[str, Any] | None] = {"status": None, "config": None}
            errors: dict[str, str | None] = {"status": None, "config": None}
            for endpoint in ("status", "config"):
                try:
                    values[endpoint] = self._fetch_json(camera_url, endpoint)
                except ValueError as error:
                    errors[endpoint] = str(error)

            checked_at = utc_now()
            success_count = sum(value is not None for value in values.values())
            with self._lock:
                current = self._cameras.get(camera_id)
                if current is None:
                    return None
                for endpoint, value in values.items():
                    if value is not None:
                        current[endpoint] = value
                current["errors"] = errors
                current["reachable"] = success_count > 0
                current["healthy"] = success_count == len(values)
                current["last_polled_at"] = checked_at
                if success_count == len(values):
                    current["last_success_at"] = checked_at
                current["polling"] = False
                return self._copy_camera(current)
        finally:
            with self._lock:
                current = self._cameras.get(camera_id)
                if current is not None:
                    current["polling"] = False
            poll_lock.release()

    def poll_all(self) -> list[dict[str, Any]]:
        with self._lock:
            camera_ids = list(self._cameras)
        if not camera_ids:
            return []
        with ThreadPoolExecutor(max_workers=min(16, len(camera_ids))) as executor:
            list(executor.map(self.poll, camera_ids))
        return self.snapshot()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            self.poll_all()
            delay = max(0.0, self.poll_seconds - (time.monotonic() - started))
            self._stop_event.wait(delay)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="camera-monitor", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.timeout_seconds * 2 + 1)


class CameraRegistry(SharedCameraRegistry):
    """Shared registry configured through this module's patchable HTTP opener."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("opener", lambda *call_args, **call_kwargs: urlopen(*call_args, **call_kwargs))
        super().__init__(*args, **kwargs)


MONITOR_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ESP32 camera monitor</title>
  <style>
    :root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#0e1419;color:#edf4f8}
    header,main{width:min(1400px,calc(100% - 2rem));margin:auto}header{padding:1.4rem 0 1rem}
    form,.card{background:#182129;border:1px solid #31404b;border-radius:12px;padding:1rem}
    form{display:flex;gap:.7rem;flex-wrap:wrap;margin-bottom:1rem}input{flex:1;min-width:220px}
    input,button{font:inherit;padding:.65rem .8rem;border-radius:8px;border:1px solid #465966}
    button{cursor:pointer;background:#246b8d;color:white}.danger{background:#8e3540}.secondary{background:#374b58}
    #cameras{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}
    .top{display:flex;justify-content:space-between;gap:1rem;align-items:flex-start}.actions{display:flex;gap:.45rem}
    h1,h2{margin:.1rem 0}.url,.muted{color:#aebdc7}.url{word-break:break-all}.ok{color:#71dc9b}.bad{color:#ff8f8f}.unknown{color:#e9c46a}
    details{margin-top:.8rem}pre{background:#0d1216;padding:.75rem;border-radius:8px;overflow:auto;max-height:26rem;white-space:pre-wrap}
    #message{min-height:1.4rem}.empty{padding:2rem;text-align:center;color:#aebdc7}
  </style>
</head>
<body>
<header><h1>ESP32 camera monitor</h1><p class="muted">Polls every {{ poll_seconds }} seconds. Registry: {{ registry_file }}</p></header>
<main>
  <form id="add-form">
    <input id="name" aria-label="Camera name" placeholder="Camera name, e.g. Backyard" maxlength="80" required>
    <input id="url" aria-label="Camera URL" placeholder="http://192.168.1.84" required>
    <button type="submit">Add camera</button>
    <button class="secondary" type="button" onclick="pollAll()">Poll all now</button>
  </form>
  <p id="message"></p><div id="cameras"></div>
</main>
<script>
const cameras=document.getElementById('cameras'),message=document.getElementById('message');
const text=value=>value===null||value===undefined?'Never':String(value);
const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(list){
  if(!list.length){cameras.innerHTML='<div class="empty">No cameras registered. Add one above.</div>';return;}
  cameras.innerHTML=list.map(camera=>{
    const state=camera.reachable===null?['unknown','Not polled']:camera.healthy?['ok','Online']:camera.reachable?['unknown','Partially reachable']:['bad','Offline'];
    const errors=Object.entries(camera.errors||{}).filter(([,v])=>v).map(([k,v])=>`${k}: ${v}`).join('\n');
    return `<section class="card"><div class="top"><div><h2>${esc(camera.name)}</h2><div class="url">${esc(camera.url)}</div></div>
      <div class="actions"><button class="secondary" onclick="pollCamera('${camera.id}')">Poll now</button><button class="danger" onclick="removeCamera('${camera.id}')">Remove</button></div></div>
      <p class="${state[0]}"><strong>${state[1]}</strong>${camera.polling?' · polling…':''}</p>
      <div class="muted">Last poll: ${esc(text(camera.last_polled_at))}<br>Last full success: ${esc(text(camera.last_success_at))}</div>
      ${errors?`<pre class="bad">${esc(errors)}</pre>`:''}
      <details open><summary>Status</summary><pre>${esc(JSON.stringify(camera.status,null,2)||'Not available')}</pre></details>
      <details><summary>Configuration</summary><pre>${esc(JSON.stringify(camera.config,null,2)||'Not available')}</pre></details></section>`;
  }).join('');
}
async function refresh(){const response=await fetch('/api/cameras',{cache:'no-store'});const body=await response.json();if(!response.ok)throw new Error(body.error||response.statusText);render(body.cameras);}
async function pollAll(){message.textContent='Polling all cameras…';const response=await fetch('/api/cameras/poll',{method:'POST'});const body=await response.json();message.textContent=response.ok?'Fleet poll complete.':(body.error||'Fleet poll failed.');render(body.cameras||[]);}
async function pollCamera(id){message.textContent='Polling camera…';const response=await fetch(`/api/cameras/${id}/poll`,{method:'POST'});const body=await response.json();message.textContent=response.ok?'Poll complete.':(body.error||'Poll failed.');await refresh();}
async function removeCamera(id){if(!confirm('Remove this camera?'))return;const response=await fetch(`/api/cameras/${id}`,{method:'DELETE'});const body=await response.json();message.textContent=response.ok?'Camera removed.':(body.error||'Remove failed.');await refresh();}
document.getElementById('add-form').addEventListener('submit',async event=>{event.preventDefault();message.textContent='Adding camera…';const response=await fetch('/api/cameras',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:document.getElementById('name').value,url:document.getElementById('url').value})});const body=await response.json();if(response.ok){event.target.reset();message.textContent='Camera added.';}else message.textContent=body.error||'Add failed.';await refresh();});
refresh().catch(error=>message.textContent=String(error));setInterval(()=>refresh().catch(error=>message.textContent=String(error)),2000);
</script>
</body>
</html>
"""


def create_app(registry: CameraRegistry, *, start_poller: bool = True) -> Flask:
    app = Flask(__name__)
    app.config["CAMERA_REGISTRY"] = registry
    app.config["CLEANUP"] = registry.stop
    if start_poller:
        registry.start()

    @app.get("/")
    def dashboard() -> str:
        return render_template_string(
            MONITOR_HTML,
            poll_seconds=registry.poll_seconds,
            registry_file=registry.registry_file,
        )

    @app.get("/api/cameras")
    def cameras() -> Response:
        return jsonify(cameras=registry.snapshot(), poll_seconds=registry.poll_seconds)

    @app.post("/api/cameras")
    def add_camera() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a JSON object with name and url."), 400
        try:
            camera = registry.add(payload.get("name"), normalize_camera_url(payload.get("url")))
        except ValueError as error:
            return jsonify(error=str(error)), 422
        registry.poll(camera["id"])
        return jsonify(camera=registry.get(camera["id"])), 201

    @app.post("/api/cameras/poll")
    def poll_cameras() -> Response:
        return jsonify(cameras=registry.poll_all())

    @app.delete("/api/cameras/<camera_id>")
    def remove_camera(camera_id: str) -> tuple[Response, int] | Response:
        removed = registry.remove(camera_id)
        if removed is None:
            return jsonify(error="Camera not found."), 404
        return jsonify(status="removed", camera=removed)

    @app.post("/api/cameras/<camera_id>/poll")
    def poll_camera(camera_id: str) -> tuple[Response, int] | Response:
        camera = registry.poll(camera_id)
        if camera is None:
            return jsonify(error="Camera not found."), 404
        return jsonify(camera=camera)

    @app.get("/api/cameras/<camera_id>/status")
    def camera_status(camera_id: str) -> tuple[Response, int] | Response:
        camera = registry.get(camera_id)
        if camera is None:
            return jsonify(error="Camera not found."), 404
        return jsonify(
            camera_id=camera_id,
            value=camera["status"],
            error=camera["errors"]["status"],
            last_polled_at=camera["last_polled_at"],
        )

    @app.get("/api/cameras/<camera_id>/config")
    def camera_config(camera_id: str) -> tuple[Response, int] | Response:
        camera = registry.get(camera_id)
        if camera is None:
            return jsonify(error="Camera not found."), 404
        return jsonify(
            camera_id=camera_id,
            value=camera["config"],
            error=camera["errors"]["config"],
            last_polled_at=camera["last_polled_at"],
        )

    @app.get("/health")
    def health() -> Response:
        cameras = registry.snapshot()
        return jsonify(
            status="ok",
            cameras=len(cameras),
            healthy=sum(1 for camera in cameras if camera["healthy"]),
            reachable=sum(1 for camera in cameras if camera["reachable"]),
            poll_seconds=registry.poll_seconds,
        )

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind")
    parser.add_argument("--port", type=int, default=5051, help="Monitor port (default: 5051)")
    parser.add_argument("--registry-file", type=Path, default=DEFAULT_REGISTRY_FILE)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    try:
        camera_registry = CameraRegistry(
            arguments.registry_file,
            poll_seconds=arguments.poll_seconds,
            timeout_seconds=arguments.timeout_seconds,
        )
    except ValueError as error:
        raise SystemExit(f"Startup error: {error}") from error
    application = create_app(camera_registry)
    print(f"Multi-camera monitor: http://127.0.0.1:{arguments.port}", flush=True)
    print(f"Camera registry: {camera_registry.registry_file}", flush=True)
    try:
        application.run(host=arguments.host, port=arguments.port, debug=False, threaded=True)
    finally:
        camera_registry.stop()