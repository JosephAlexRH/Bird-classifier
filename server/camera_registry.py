"""Shared persistent registry and runtime state for ESP32-CAM servers."""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_REGISTRY_FILE = Path.home() / ".esp32-cameras" / "cameras.json"
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 3.0
MAX_JSON_BYTES = 1024 * 1024
MAX_CAMERA_NAME_LENGTH = 80
REGISTRY_SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_camera_name(value: Any, *, fallback: str | None = None) -> str:
    if value is None and fallback is not None:
        value = fallback
    if not isinstance(value, str):
        raise ValueError("Camera name must be a string.")
    name = " ".join(value.strip().split())
    if not name:
        raise ValueError("Camera name must not be empty.")
    if len(name) > MAX_CAMERA_NAME_LENGTH:
        raise ValueError(f"Camera name must be at most {MAX_CAMERA_NAME_LENGTH} characters.")
    return name


def normalize_camera_url(value: str | None) -> str | None:
    """Validate and normalize an HTTP(S) camera base URL."""
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        raise ValueError("Camera URL must not be empty.")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
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
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, "", "", ""))


def normalize_camera_address(value: Any) -> str:
    """Normalize a camera IP, hostname, or complete base URL."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Camera address must not be empty.")
    candidate = value.strip()
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    normalized = normalize_camera_url(candidate)
    assert normalized is not None
    return normalized


def camera_identity_from_values(
    status: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Extract stable hardware and sensor identity from ESP32 responses."""
    status = status if isinstance(status, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    device = metadata.get("device") if isinstance(metadata.get("device"), dict) else {}
    camera = metadata.get("camera") if isinstance(metadata.get("camera"), dict) else {}
    sensor = camera.get("sensor") if isinstance(camera.get("sensor"), dict) else {}
    hardware_id = status.get("camera_id") or device.get("camera_id")
    if not isinstance(hardware_id, str) or not hardware_id.strip():
        hardware_id = None
    else:
        hardware_id = hardware_id.strip()
    model = sensor.get("model") or status.get("camera_sensor_model")
    pid = sensor.get("pid") or status.get("camera_sensor_pid")
    pid_hex = sensor.get("pid_hex") or status.get("camera_sensor_pid_hex")
    if model is None and pid is None and pid_hex is None:
        return hardware_id, None
    return hardware_id, {"model": model, "pid": pid, "pid_hex": pid_hex}


class CameraRegistry:
    """Persist definitions while maintaining per-camera runtime state and locks."""

    def __init__(
        self,
        registry_file: Path,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("Polling interval must be greater than zero.")
        if timeout_seconds <= 0:
            raise ValueError("Camera timeout must be greater than zero.")
        self.registry_file = Path(registry_file).expanduser().resolve()
        self.poll_seconds = float(poll_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self._opener = opener
        self._lock = threading.RLock()
        self._poll_locks: dict[str, threading.Lock] = {}
        self._operation_locks: dict[str, threading.Lock] = {}
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
            "streaming": False,
            "last_polled_at": None,
            "last_success_at": None,
            "status": None,
            "config": None,
            "metadata": None,
            "errors": {"status": None, "config": None, "metadata": None},
        }

    @staticmethod
    def _definition(camera: dict[str, Any]) -> dict[str, Any]:
        return {
            key: copy.deepcopy(camera.get(key))
            for key in (
                "id", "name", "url", "enabled", "default", "hardware_camera_id",
                "sensor", "added_at",
            )
        }

    def _load(self) -> None:
        if not self.registry_file.exists():
            return
        try:
            document = json.loads(self.registry_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read camera registry {self.registry_file}: {error}") from error
        if not isinstance(document, dict) or document.get("schema_version") not in {1, REGISTRY_SCHEMA_VERSION}:
            raise ValueError("Camera registry has an unsupported or missing schema_version.")
        cameras = document.get("cameras")
        if not isinstance(cameras, list):
            raise ValueError("Camera registry cameras must be a JSON array.")
        loaded: dict[str, dict[str, Any]] = {}
        urls: set[str] = set()
        hardware_ids: set[str] = set()
        for item in cameras:
            if not isinstance(item, dict):
                raise ValueError("Every camera registry entry must be a JSON object.")
            camera_id = item.get("id")
            if not isinstance(camera_id, str) or not camera_id or camera_id in loaded:
                raise ValueError("Every camera registry entry must have a unique non-empty id.")
            url = normalize_camera_address(item.get("url"))
            if url in urls:
                raise ValueError(f"Camera registry contains duplicate URL {url}.")
            hardware_id = item.get("hardware_camera_id")
            if hardware_id is not None and (not isinstance(hardware_id, str) or not hardware_id.strip()):
                hardware_id = None
            if hardware_id in hardware_ids:
                raise ValueError(f"Camera registry contains duplicate hardware ID {hardware_id}.")
            urls.add(url)
            if hardware_id:
                hardware_ids.add(hardware_id)
            loaded[camera_id] = {
                "id": camera_id,
                "name": normalize_camera_name(item.get("name")),
                "url": url,
                "enabled": item.get("enabled", True) is not False,
                "default": item.get("default", False) is True,
                "hardware_camera_id": hardware_id,
                "sensor": copy.deepcopy(item.get("sensor")) if isinstance(item.get("sensor"), dict) else None,
                "added_at": item.get("added_at") if isinstance(item.get("added_at"), str) else None,
                **self._runtime_state(),
            }
        defaults = [camera for camera in loaded.values() if camera["default"]]
        if len(defaults) > 1:
            for camera in defaults[1:]:
                camera["default"] = False
        self._cameras = loaded
        self._poll_locks = {camera_id: threading.Lock() for camera_id in loaded}
        self._operation_locks = {camera_id: threading.Lock() for camera_id in loaded}

    def _persist_locked(self) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "cameras": [self._definition(camera) for camera in self._cameras.values()],
        }
        payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        temporary = self.registry_file.with_name(f".{self.registry_file.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.registry_file)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _copy_camera(camera: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(camera)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._copy_camera(camera) for camera in self._cameras.values()]

    def get(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
            return self._copy_camera(camera) if camera is not None else None

    def get_by_hardware_id(self, hardware_camera_id: str | None) -> dict[str, Any] | None:
        if not hardware_camera_id:
            return None
        with self._lock:
            camera = next(
                (item for item in self._cameras.values() if item["hardware_camera_id"] == hardware_camera_id),
                None,
            )
            return self._copy_camera(camera) if camera is not None else None

    def resolve(self, camera_id: str | None = None) -> dict[str, Any] | None:
        if camera_id:
            return self.get(camera_id)
        with self._lock:
            enabled = [camera for camera in self._cameras.values() if camera["enabled"]]
            selected = next((camera for camera in enabled if camera["default"]), None)
            if selected is None and len(enabled) == 1:
                selected = enabled[0]
            return self._copy_camera(selected) if selected is not None else None

    def operation_lock(self, camera_id: str) -> threading.Lock | None:
        with self._lock:
            return self._operation_locks.get(camera_id)

    def add(
        self,
        name: Any,
        address: Any,
        *,
        enabled: bool = True,
        make_default: bool | None = None,
    ) -> dict[str, Any]:
        url = normalize_camera_address(address)
        fallback = urlsplit(url).hostname or url
        normalized_name = normalize_camera_name(name, fallback=fallback)
        with self._lock:
            if any(camera["url"] == url for camera in self._cameras.values()):
                raise ValueError(f"Camera URL {url} is already registered.")
            camera_id = uuid.uuid4().hex
            if make_default is None:
                make_default = not self._cameras
            if make_default:
                for camera in self._cameras.values():
                    camera["default"] = False
            camera = {
                "id": camera_id,
                "name": normalized_name,
                "url": url,
                "enabled": bool(enabled),
                "default": bool(make_default),
                "hardware_camera_id": None,
                "sensor": None,
                "added_at": utc_now(),
                **self._runtime_state(),
            }
            self._cameras[camera_id] = camera
            self._poll_locks[camera_id] = threading.Lock()
            self._operation_locks[camera_id] = threading.Lock()
            self._persist_locked()
            return self._copy_camera(camera)

    def update(self, camera_id: str, **changes: Any) -> dict[str, Any] | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
            if camera is None:
                return None
            if "name" in changes:
                camera["name"] = normalize_camera_name(changes["name"])
            if "enabled" in changes:
                if not isinstance(changes["enabled"], bool):
                    raise ValueError("enabled must be true or false.")
                camera["enabled"] = changes["enabled"]
            if changes.get("default") is True:
                for item in self._cameras.values():
                    item["default"] = item["id"] == camera_id
            self._persist_locked()
            return self._copy_camera(camera)

    def bind_identity(
        self,
        camera_id: str,
        hardware_camera_id: str,
        sensor: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Persist a discovered hardware identity while rejecting duplicates."""
        hardware_camera_id = hardware_camera_id.strip()
        if not hardware_camera_id:
            raise ValueError("Hardware camera ID must not be empty.")
        with self._lock:
            camera = self._cameras.get(camera_id)
            if camera is None:
                return None
            duplicate = next(
                (
                    item for item in self._cameras.values()
                    if item["id"] != camera_id
                    and item["hardware_camera_id"] == hardware_camera_id
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(
                    f"Hardware camera {hardware_camera_id} is already registered as {duplicate['name']}."
                )
            camera["hardware_camera_id"] = hardware_camera_id
            if sensor is not None:
                if not isinstance(sensor, dict):
                    raise ValueError("Sensor identity must be a JSON object.")
                camera["sensor"] = copy.deepcopy(sensor)
            self._persist_locked()
            return self._copy_camera(camera)

    def remove(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            camera = self._cameras.pop(camera_id, None)
            if camera is None:
                return None
            self._poll_locks.pop(camera_id, None)
            self._operation_locks.pop(camera_id, None)
            if camera["default"] and self._cameras:
                next(iter(self._cameras.values()))["default"] = True
            self._persist_locked()
            return self._copy_camera(camera)

    def set_streaming(self, camera_id: str, streaming: bool) -> None:
        with self._lock:
            camera = self._cameras.get(camera_id)
            if camera is not None:
                camera["streaming"] = bool(streaming)

    def _fetch_json(self, camera_url: str, endpoint: str) -> dict[str, Any]:
        target = f"{camera_url}/{endpoint}"
        opener = self._opener or urlopen
        try:
            with opener(Request(target, headers={"Accept": "application/json"}), timeout=self.timeout_seconds) as response:
                content_type = response.headers.get_content_type()
                payload = response.read(MAX_JSON_BYTES + 1)
                if len(payload) > MAX_JSON_BYTES:
                    raise ValueError(f"{target} exceeded the 1 MiB response limit.")
                if content_type != "application/json":
                    raise ValueError(f"{target} returned Content-Type {content_type!r}, not application/json.")
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise ValueError(f"{target} returned JSON that was not an object.")
                return value
        except HTTPError as error:
            raise ValueError(f"{target} returned HTTP {error.code}.") from error
        except (URLError, TimeoutError, OSError) as error:
            reason = error.reason if isinstance(error, URLError) else error
            raise ValueError(f"{target} is unavailable: {reason}") from error
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{target} returned invalid JSON: {error}") from error

    def poll(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
            poll_lock = self._poll_locks.get(camera_id)
            if camera is None or poll_lock is None:
                return None
            if not camera["enabled"]:
                return self._copy_camera(camera)
            camera_url = camera["url"]
            camera["polling"] = True
        if not poll_lock.acquire(blocking=False):
            return self.get(camera_id)
        try:
            values: dict[str, dict[str, Any] | None] = {
                "status": None,
                "config": None,
                "metadata": None,
            }
            errors: dict[str, str | None] = {"status": None, "config": None, "metadata": None}
            for endpoint in values:
                try:
                    values[endpoint] = self._fetch_json(camera_url, endpoint)
                except ValueError as error:
                    errors[endpoint] = str(error)
            checked_at = utc_now()
            success_count = sum(value is not None for value in values.values())
            hardware_id, sensor = camera_identity_from_values(
                values["status"], values["metadata"]
            )
            with self._lock:
                current = self._cameras.get(camera_id)
                if current is None:
                    return None
                duplicate = next(
                    (
                        item for item in self._cameras.values()
                        if item["id"] != camera_id and hardware_id and item["hardware_camera_id"] == hardware_id
                    ),
                    None,
                )
                if duplicate is not None:
                    errors["metadata"] = (
                        f"Hardware camera {hardware_id} is already registered as {duplicate['name']}."
                    )
                else:
                    if hardware_id:
                        current["hardware_camera_id"] = hardware_id
                    if sensor:
                        current["sensor"] = sensor
                for endpoint, value in values.items():
                    if value is not None:
                        current[endpoint] = value
                current["errors"] = errors
                current["reachable"] = success_count > 0
                current["healthy"] = success_count == len(values) and errors["metadata"] is None
                current["last_polled_at"] = checked_at
                if current["healthy"]:
                    current["last_success_at"] = checked_at
                current["polling"] = False
                self._persist_locked()
                return self._copy_camera(current)
        finally:
            with self._lock:
                current = self._cameras.get(camera_id)
                if current is not None:
                    current["polling"] = False
            poll_lock.release()

    def poll_all(self) -> list[dict[str, Any]]:
        with self._lock:
            camera_ids = [camera_id for camera_id, camera in self._cameras.items() if camera["enabled"]]
        if camera_ids:
            with ThreadPoolExecutor(max_workers=min(16, len(camera_ids))) as executor:
                list(executor.map(self.poll, camera_ids))
        return self.snapshot()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            self.poll_all()
            self._stop_event.wait(max(0.0, self.poll_seconds - (time.monotonic() - started)))

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="camera-registry", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.timeout_seconds * 2 + 1)