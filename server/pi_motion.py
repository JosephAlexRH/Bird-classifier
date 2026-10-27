"""Active Raspberry Pi motion polling, classification, and selective persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

try:
    from .bird_inference import (
        LETTERBOX_GEOMETRY,
        INFERENCE_SCHEMA_VERSION,
        ROI_CROP_GEOMETRY,
        SELECTED_CROP_GEOMETRY,
        SELECTED_CROP_SOURCE,
        decode_rgb,
        write_inference_record,
    )
    from .camera_capture_job import fetch_camera_capture
    from .capture_storage import CaptureStore, parse_camera_metadata
    from .motion_crop import (
        MotionConfig,
        RunningMedianBackground,
        rgb_to_gray,
        select_motion_crop,
        should_admit_background,
    )
except ImportError:  # Direct execution support.
    from bird_inference import (
        LETTERBOX_GEOMETRY,
        INFERENCE_SCHEMA_VERSION,
        ROI_CROP_GEOMETRY,
        SELECTED_CROP_GEOMETRY,
        SELECTED_CROP_SOURCE,
        decode_rgb,
        write_inference_record,
    )
    from camera_capture_job import fetch_camera_capture
    from capture_storage import CaptureStore, parse_camera_metadata
    from motion_crop import (
        MotionConfig,
        RunningMedianBackground,
        rgb_to_gray,
        select_motion_crop,
        should_admit_background,
    )


FetchFrame = Callable[..., tuple[bytes, str | None]]


def _camera_identity(metadata: dict[str, Any]) -> tuple[str, str | None]:
    device = metadata.get("device")
    if not isinstance(device, dict):
        raise ValueError("Monitor frame metadata lacks device identity")
    camera_id = device.get("camera_id")
    boot_id = device.get("boot_id")
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError("Monitor frame metadata lacks stable camera_id")
    return camera_id, boot_id if isinstance(boot_id, str) else None


def _static_roi(metadata: dict[str, Any]) -> Any:
    camera = metadata.get("camera")
    requested = camera.get("requested") if isinstance(camera, dict) else None
    motion = requested.get("motion") if isinstance(requested, dict) else None
    return motion.get("roi") if isinstance(motion, dict) else None


class MotionService:
    """Single-camera Pi-owned motion service, structured around stable camera identity."""

    def __init__(
        self,
        camera_url: str,
        store: CaptureStore,
        inference_engine: Any,
        *,
        poll_seconds: float = 0.25,
        timeout_seconds: float = 2.0,
        config: MotionConfig = MotionConfig(),
        fetcher: FetchFrame = fetch_camera_capture,
        camera_lock: threading.Lock | None = None,
        inference_lock: threading.Lock | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        on_persist: Callable[[Path, dict[str, Any], dict[str, Any]], None] | None = None,
        minimum_persist_seconds: float = 1.0,
        expected_camera_id: str | None = None,
        camera_registry_id: str | None = None,
        camera_name: str | None = None,
    ) -> None:
        if poll_seconds <= 0 or timeout_seconds <= 0:
            raise ValueError("Motion polling interval and timeout must be positive")
        self.camera_url = camera_url
        self.store = store
        self.engine = inference_engine
        self.poll_seconds = float(poll_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.config = config
        self.fetcher = fetcher
        self.camera_lock = camera_lock or threading.Lock()
        self.inference_lock = inference_lock or threading.Lock()
        self.monotonic = monotonic
        self.on_persist = on_persist
        self.minimum_persist_seconds = float(minimum_persist_seconds)
        self.expected_camera_id = expected_camera_id
        self.camera_registry_id = camera_registry_id
        self.camera_name = camera_name
        self.background: RunningMedianBackground | None = None
        self.camera_id: str | None = None
        self.boot_id: str | None = None
        self.scene_signature: tuple[Any, ...] | None = None
        self.last_persisted_at: float | None = None
        self.running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self.frames_polled = 0
        self.frames_persisted = 0
        self.failures = 0
        self.last_error: str | None = None
        self.last_result: dict[str, Any] | None = None

    @property
    def geometry(self) -> str:
        contract = getattr(self.engine, "detector_contract", None) or getattr(
            self.engine, "contract", None
        )
        return getattr(contract, "geometry", LETTERBOX_GEOMETRY)

    def _infer(self, path: Path, static_roi: Any, motion: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.geometry == ROI_CROP_GEOMETRY:
            kwargs = {"roi": static_roi, "roi_source": "capture_sidecar.camera.requested.motion.roi"}
        elif self.geometry == SELECTED_CROP_GEOMETRY:
            bounds = motion["selection"]["final_bounds_px"]
            kwargs = {
                "selected_crop": {key: bounds[key] for key in ("left", "top", "right", "bottom")},
                "crop_source": SELECTED_CROP_SOURCE,
            }
        with self.inference_lock:
            return self.engine.infer(path, **kwargs)

    def process_once(self) -> dict[str, Any]:
        """Fetch, localize, classify, conditionally learn, and selectively persist one frame."""
        with self.camera_lock:
            image, metadata_header = self.fetcher(
                self.camera_url,
                endpoint="monitor-frame",
                timeout_seconds=self.timeout_seconds,
            )
        metadata, metadata_error = parse_camera_metadata(metadata_header)
        if metadata is None:
            raise ValueError(metadata_error or "Monitor frame metadata is required")
        if metadata.get("source") != "pi_motion_poll":
            raise ValueError("Monitor frame metadata source must be pi_motion_poll")
        camera_id, boot_id = _camera_identity(metadata)
        if self.expected_camera_id is not None and camera_id != self.expected_camera_id:
            raise ValueError(
                f"Camera identity mismatch: expected {self.expected_camera_id}, received {camera_id}"
            )
        static_roi = _static_roi(metadata)
        requested = ((metadata.get("camera") or {}).get("requested") or {}).get("motion") or {}
        if requested.get("owner") != "pi":
            raise ValueError("ESP32 motion_owner must be pi while Pi motion polling is active")

        if self.background is None or camera_id != self.camera_id or boot_id != self.boot_id:
            self.background = RunningMedianBackground(self.config, camera_id=camera_id)
            self.camera_id, self.boot_id = camera_id, boot_id
            self.scene_signature = None

        temporary_path: Path | None = None
        try:
            ram_temp = Path("/dev/shm")
            temporary_dir = ram_temp if ram_temp.is_dir() and os.access(ram_temp, os.W_OK) else None
            fd, temporary_name = tempfile.mkstemp(
                prefix=".pi-motion-", suffix=".jpg", dir=temporary_dir
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(fd, "wb") as output:
                output.write(image)
            rgb = decode_rgb(temporary_path)
            gray = rgb_to_gray(rgb)
            config_hash = ((metadata.get("device") or {}).get("config") or {}).get("file_sha256")
            signature = (
                camera_id, boot_id, gray.shape,
                json.dumps(static_roi, sort_keys=True, separators=(",", ":")), config_hash,
            )
            if self.scene_signature is not None and signature != self.scene_signature:
                self.background.reset(camera_id=camera_id)
            self.scene_signature = signature
            background_before = self.background.median()
            motion = select_motion_crop(gray, background_before, static_roi, self.config)
            motion["background"] = {
                "epoch": self.background.epoch,
                "sample_count": self.background.sample_count,
                "status": "available" if background_before is not None else "warming",
            }
            try:
                inference = self._infer(temporary_path, static_roi, motion)
            except Exception as error:
                inference = {
                    "schema_version": INFERENCE_SCHEMA_VERSION,
                    "status": "error",
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "image": {
                        "filename": temporary_path.name,
                        "sha256": hashlib.sha256(image).hexdigest(),
                    },
                    "error": str(error),
                }
                detector_record = getattr(self.engine, "detector_record", lambda: None)()
                if detector_record is not None:
                    inference["detector"] = detector_record
            threshold = float(getattr(self.engine, "bird_threshold", inference.get("bird_threshold", 0.5)))
            now = self.monotonic()
            admitted = should_admit_background(inference, threshold=threshold) and self.background.admit(gray, now=now)
            clear_blob = motion["selection"]["source"] == "dynamic_motion_blob"
            is_bird = inference.get("status") == "complete" and inference.get("is_bird") is True
            persistence_due = self.last_persisted_at is None or now - self.last_persisted_at >= self.minimum_persist_seconds
            persisted = is_bird or (clear_blob and persistence_due)
            capture: dict[str, Any] | None = None
            if persisted:
                event_id = f"event_{uuid.uuid4().hex}"
                capture = self.store.save_capture(
                    image,
                    source="pi_motion_poll",
                    camera_metadata=metadata,
                    metadata_error=metadata_error,
                    extra={
                        "camera_id": camera_id,
                        "camera_registry_id": self.camera_registry_id,
                        "camera_name": self.camera_name,
                        "event_id": event_id,
                        "motion_selection_source": motion["selection"]["source"],
                    },
                )
                destination = self.store.root / capture["filename"]
                motion["capture"] = {
                    "capture_id": capture["capture_id"],
                    "filename": capture["filename"],
                    "image_sha256": capture["sha256"],
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "camera_id": camera_id,
                    "boot_id": boot_id,
                    "event_id": event_id,
                }
                sidecar_path = self.store.root / capture["sidecar"]
                sidecar_sha256 = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
                sensor = ((capture.get("camera") or {}).get("sensor") or {})
                camera_provenance = {
                    "model": capture.get("camera_model"),
                    "model_source": "capture_sidecar.camera.sensor.model",
                    "sensor_pid": sensor.get("pid"),
                    "sensor_pid_hex": sensor.get("pid_hex"),
                    "camera_id": camera_id,
                    "capture_sidecar": capture["sidecar"],
                    "capture_sidecar_sha256": sidecar_sha256,
                }
                motion["camera_provenance"] = camera_provenance
                motion["static_roi"]["capture_sidecar"] = capture["sidecar"]
                motion["static_roi"]["capture_sidecar_sha256"] = sidecar_sha256
                motion["background"]["admitted_after_inference"] = admitted
                self.store.save_derived_record(destination, "motion", motion)
                inference["capture_id"] = capture["capture_id"]
                inference["image"] = {"filename": destination.name, "sha256": capture["sha256"]}
                inference["camera_provenance"] = camera_provenance
                inference["motion_record"] = f"{capture['capture_id']}.motion.json"
                inference["motion_crop_id"] = motion["crop_id"]
                write_inference_record(destination, inference)
                self.frames_persisted += 1
                self.last_persisted_at = now
                if self.on_persist is not None:
                    self.on_persist(destination, capture, inference)
            result = {
                "camera_id": camera_id,
                "persisted": persisted,
                "capture": capture,
                "motion": motion,
                "inference": inference,
                "background_admitted": admitted,
            }
            with self._state_lock:
                self.frames_polled += 1
                self.last_result = result
                self.last_error = None
            return result
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _run(self) -> None:
        next_due = self.monotonic()
        backoff = 1.0
        while not self._stop.is_set():
            delay = next_due - self.monotonic()
            if delay > 0 and self._stop.wait(delay):
                break
            try:
                self.process_once()
                backoff = 1.0
                next_due += self.poll_seconds
                now = self.monotonic()
                if next_due < now:
                    next_due += (int((now - next_due) / self.poll_seconds) + 1) * self.poll_seconds
            except Exception as error:
                with self._state_lock:
                    self.failures += 1
                    self.last_error = str(error)
                next_due = self.monotonic() + backoff
                backoff = min(30.0, backoff * 2.0)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pi-motion", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self.running = False

    def describe(self) -> dict[str, Any]:
        with self._state_lock:
            background = self.background
            return {
                "running": self.running,
                "camera_registry_id": self.camera_registry_id,
                "camera_name": self.camera_name,
                "camera_url": self.camera_url,
                "expected_camera_id": self.expected_camera_id,
                "camera_id": self.camera_id,
                "boot_id": self.boot_id,
                "poll_seconds": self.poll_seconds,
                "timeout_seconds": self.timeout_seconds,
                "frames_polled": self.frames_polled,
                "frames_persisted": self.frames_persisted,
                "failures": self.failures,
                "last_error": self.last_error,
                "background_epoch": background.epoch if background else None,
                "background_samples": background.sample_count if background else 0,
                "background_ready": background.ready if background else False,
                "last_selection": ((self.last_result or {}).get("motion") or {}).get("selection"),
            }