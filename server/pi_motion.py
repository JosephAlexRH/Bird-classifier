"""Active Raspberry Pi motion polling, classification, and selective persistence."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
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
    from .collection import export_bird_crops
    from .scene_background import SceneBackground
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
    from collection import export_bird_crops
    from scene_background import SceneBackground
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
        collection_mode: bool = False,
        deployment_id: str = "",
        collection_run_id: str = "",
        detector_enabled: bool = True,
        motion_localization_enabled: bool = True,
        settings_revision: int = 0,
        motion_triggered: bool = False,
        empty_sample_seconds: float = 0,
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
        self.collection_mode = collection_mode
        self.deployment_id = deployment_id
        self.collection_run_id = collection_run_id
        self.detector_enabled = detector_enabled
        self.motion_localization_enabled = motion_localization_enabled
        self.settings_revision = settings_revision
        self.motion_triggered = motion_triggered
        self.empty_sample_seconds = empty_sample_seconds
        self.last_empty_sample_at = None
        self._process_lock = threading.Lock()
        self.background: RunningMedianBackground | None = None
        self.camera_id: str | None = None
        self.boot_id: str | None = None
        self.scene_signature: tuple[Any, ...] | None = None
        self.last_persisted_at: float | None = None
        self.running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._analysis_thread: threading.Thread | None = None
        self._captured: queue.Queue = queue.Queue(maxsize=1)
        self.frames_captured = 0
        self.inference_skipped = 0
        self.capture_failures = 0
        self.last_capture_at: str | None = None
        self.last_capture_error: str | None = None
        self.last_analysis_error: str | None = None
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
        if not self.detector_enabled:
            return {"status": "skipped", "reason": "analysis_disabled", "is_bird": None,
                    "bird_probability": None, "classification": None,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
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

    def process_once(self, captured: tuple | None = None) -> dict[str, Any]:
        with self._process_lock:
            return self._process_once(captured)

    def background_command(self, action, *, confirmed_empty=False):
        if action not in ('learn', 'reset'):
            raise ValueError('Choose learn or reset.')
        if action == 'learn' and confirmed_empty is not True:
            raise ValueError('Confirm the entire analysis area is empty of birds before learning.')
        if not self._process_lock.acquire(blocking=False):
            raise RuntimeError('Camera analysis is busy. Try again shortly.')
        try:
            if self.background is None:
                raise RuntimeError('Wait for the first camera poll before learning.')
            if action == 'learn':
                self.background.begin_learning()
            else:
                self.background.invalidate('user_reset')
            return self.background.state()
        finally:
            self._process_lock.release()

    def _process_once(self, captured: tuple | None = None) -> dict[str, Any]:
        """Fetch, localize, classify, conditionally learn, and selectively persist one frame."""
        if captured is None:
            with self.camera_lock:
                image, metadata_header = self.fetcher(
                    self.camera_url,
                    endpoint="monitor-frame",
                    timeout_seconds=self.timeout_seconds,
                )
        else:
            image, metadata_header, _ = captured
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
            key = hashlib.sha256((self.camera_registry_id or camera_id).encode()).hexdigest()[:24]
            self.background = SceneBackground(self.config, camera_id=camera_id,
                cache_path=self.store.root / 'backgrounds' / key / 'reference.npz')
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
            self.background.configure_scene(signature)
            self.scene_signature = signature
            if self.background.phase == 'learning':
                if shutil.disk_usage(self.store.root).free < 256 * 1024 * 1024:
                    raise OSError('Calibration paused: less than 256 MiB free storage.')
                record = captured[2] if captured else self.store.save_capture(
                    image, source='background_calibration', camera_metadata=metadata,
                    extra={'camera_id': camera_id, 'camera_registry_id': self.camera_registry_id,
                           'camera_name': self.camera_name, 'deployment_id': self.deployment_id,
                           'background_session_id': self.background.session,
                           'hub_pipeline': self.pipeline_snapshot()})
                self.background.learn(gray, static_roi, record['capture_id'], now=self.monotonic())
                inference = {'status': 'skipped', 'reason': 'background_calibration',
                             'is_bird': None, 'background_calibration': self.background.state(),
                             'completed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                destination = self.store.root / record['filename']
                write_inference_record(destination, inference)
                if self.on_persist:
                    self.on_persist(destination, record, inference)
                result = {'persisted': True, 'capture': record, 'inference': inference,
                          'background_admitted': False, 'background_calibration': self.background.state()}
                with self._state_lock:
                    self.frames_polled += 1
                    self.frames_persisted += 1
                    self.last_error = None
                    self.last_result = result
                return result
            background_before = self.background.median()
            motion = select_motion_crop(gray, background_before if self.motion_localization_enabled else None,
                                        static_roi, self.config)
            if not self.motion_localization_enabled:
                motion["selection"]["fallback_reason"] = "motion_localization_disabled"
                motion["detection"]["reason"] = "motion_localization_disabled"
            motion["background"] = {
                "epoch": self.background.epoch,
                "sample_count": self.background.sample_count,
                "status": "available" if background_before is not None else "warming",
                'calibration': self.background.state(),
            }
            now = self.monotonic()
            if self.last_empty_sample_at is None:
                self.last_empty_sample_at = now
            quiet = background_before is not None and motion['detection'].get('reason') == 'no_changed_blob'
            empty_sample_due = (quiet and self.empty_sample_seconds > 0 and
                                now - self.last_empty_sample_at >= self.empty_sample_seconds)
            gate = self.motion_triggered and self.motion_localization_enabled
            changed = (motion['selection']['source'] == 'dynamic_motion_blob' or
                       (motion['detection'].get('reason') == 'ambiguous_blob' and
                        motion['detection'].get('changed_pixels', 0) >= self.config.minimum_component_pixels))
            if (motion['detection'].get('reason') in ('broad_scene_change', 'excessive_noise') or
                    abs(motion['detection'].get('global_shift', 0)) > 32):
                changed = False
                empty_sample_due = False
            try:
                if gate and background_before is not None and not changed and not empty_sample_due:
                    inference = {'status': 'skipped', 'reason': 'no_motion' if quiet else 'scene_change',
                                 'is_bird': None, 'bird_probability': None}
                else:
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
            inference["hub_pipeline"] = self.pipeline_snapshot()
            if background_before is not None:
                admitted = self.background.adapt(gray, motion, now)
            else:
                admitted = should_admit_background(inference, threshold=threshold) and self.background.admit(gray, now=now)
                if self.background.ready:
                    self.background.reference_frame = self.background.median().copy()
                    self.background.phase, self.background.reason = 'ready', 'confident_empty_frames_learned'
                    self.background.save_reference()
            clear_blob = motion["selection"]["source"] == "dynamic_motion_blob"
            is_bird = inference.get("status") == "complete" and inference.get("is_bird") is True
            persistence_due = self.last_persisted_at is None or now - self.last_persisted_at >= self.minimum_persist_seconds
            event = is_bird or (clear_blob and persistence_due)
            if gate and (background_before is None or not changed):
                event = False
            persisted = self.collection_mode or event or empty_sample_due
            motion['capture_policy'] = {'motion_required': gate, 'motion_candidate': changed,
                'periodic_background_sample': empty_sample_due,
                'training_label': 'unreviewed', 'background_ready_at_capture': background_before is not None}
            capture: dict[str, Any] | None = None
            if persisted:
                if shutil.disk_usage(self.store.root).free < 256 * 1024 * 1024:
                    raise OSError('Collection paused: less than 256 MiB free storage; originals have been kept')
                event_id = captured[2]["event_id"] if captured else f"event_{uuid.uuid4().hex}"
                capture = captured[2] if captured else self.store.save_capture(
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
                        "deployment_id": self.deployment_id,
                        "collection_run_id": self.collection_run_id,
                        "hub_pipeline": self.pipeline_snapshot(),
                        'capture_policy': motion['capture_policy'],
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
                if inference.get('status') == 'complete':
                    try:
                        inference["collection"] = export_bird_crops(destination, inference)
                    except Exception as error:
                        inference["collection"] = {"status": "error", "error": str(error), "crops": []}
                write_inference_record(destination, inference)
                self.frames_persisted += 1
                if empty_sample_due:
                    self.last_empty_sample_at = now
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

    def _collection_capture_once(self) -> None:
        """Persist acquisition before queueing analysis; one waiting job per camera."""
        if shutil.disk_usage(self.store.root).free < 256 * 1024 * 1024:
            raise OSError("Collection paused: less than 256 MiB free storage; originals have been kept")
        with self.camera_lock:
            image, metadata_header = self.fetcher(self.camera_url, endpoint="monitor-frame",
                                                   timeout_seconds=self.timeout_seconds)
        metadata, metadata_error = parse_camera_metadata(metadata_header)
        if metadata is None or metadata.get("source") != "pi_motion_poll":
            raise ValueError(metadata_error or "Monitor frame metadata is required")
        camera_id, _ = _camera_identity(metadata)
        if self.expected_camera_id is not None and camera_id != self.expected_camera_id:
            raise ValueError("Camera identity mismatch during collection capture")
        motion = ((metadata.get("camera") or {}).get("requested") or {}).get("motion") or {}
        if motion.get("owner") != "pi":
            raise ValueError("ESP32 motion_owner must be pi during collection")
        record = self.store.save_capture(image, source="pi_motion_poll", camera_metadata=metadata,
            metadata_error=metadata_error, extra={"camera_id": camera_id,
            "camera_registry_id": self.camera_registry_id, "camera_name": self.camera_name,
            "event_id": f"event_{uuid.uuid4().hex}", "deployment_id": self.deployment_id,
            "collection_run_id": self.collection_run_id, "hub_pipeline": self.pipeline_snapshot()})
        path = self.store.root / record["filename"]
        pending = {"status": "pending", "capture_id": record["capture_id"],
                   "image": {"filename": path.name, "sha256": record["sha256"]}}
        # An absent result already means pending in the API; this explicit record
        # lets reviewers distinguish an unprocessed frame from a negative result.
        write_inference_record(path, pending)
        if self.on_persist is not None:
            self.on_persist(path, record, pending)
        try:
            old = self._captured.get_nowait()
        except queue.Empty:
            old = None
        if old is not None:
            self._skip_captured(old, "newer_frame_waiting")
            self._captured.task_done()
        self._captured.put_nowait((image, metadata_header, record))
        with self._state_lock:
            self.frames_captured += 1
            self.last_capture_at = record["received_at"]
            self.last_capture_error = None

    def _skip_captured(self, captured: tuple, reason: str) -> None:
        record = captured[2]
        path = self.store.root / record["filename"]
        write_inference_record(path, {"status": "skipped", "reason": reason,
            "capture_id": record["capture_id"], "image": {"filename": path.name, "sha256": record["sha256"]},
            "is_bird": None, "collection": {"status": "unprocessed", "crops": [], "review_status": "unreviewed"}})
        with self._state_lock:
            self.inference_skipped += 1

    def _collection_capture_loop(self) -> None:
        backoff = self.poll_seconds
        while not self._stop.is_set():
            started = self.monotonic()
            try:
                self._collection_capture_once()
                backoff = self.poll_seconds
            except Exception as error:
                with self._state_lock:
                    self.capture_failures += 1
                    self.failures += 1
                    self.last_capture_error = str(error)
                backoff = min(30, max(1, backoff * 2))
            self._stop.wait(max(0, backoff - (self.monotonic() - started)))

    def _collection_analysis_loop(self) -> None:
        while not self._stop.is_set():
            try:
                captured = self._captured.get(timeout=.25)
            except queue.Empty:
                continue
            try:
                self.process_once(captured)
                with self._state_lock:
                    self.last_analysis_error = None
            except Exception as error:
                record = captured[2]
                write_inference_record(self.store.root / record["filename"], {
                    "status": "error", "capture_id": record["capture_id"], "error": str(error)})
                with self._state_lock:
                    self.last_analysis_error = str(error)
            finally:
                self._captured.task_done()

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._stop.clear()
        if self.collection_mode:
            self._analysis_thread = threading.Thread(target=self._collection_analysis_loop, name="pi-collection-analysis", daemon=True)
            self._analysis_thread.start()
        self._thread = threading.Thread(target=self._collection_capture_loop if self.collection_mode else self._run,
                                        name="pi-motion", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        if self._analysis_thread is not None:
            self._analysis_thread.join(timeout)
        if self.collection_mode and not (self._thread and self._thread.is_alive()):
            try:
                self._skip_captured(self._captured.get_nowait(), "service_stopped_before_analysis")
                self._captured.task_done()
            except queue.Empty:
                pass
        self.running = False

    def pipeline_snapshot(self) -> dict[str, Any]:
        return {"settings_revision": self.settings_revision, "collection_mode": self.collection_mode,
                "detector_enabled": self.detector_enabled,
                "motion_localization_enabled": self.motion_localization_enabled,
                'motion_triggered': self.motion_triggered,
                'empty_sample_seconds': self.empty_sample_seconds,
                'motion_threshold': self.config.minimum_threshold,
                'motion_minimum_area': self.config.minimum_component_pixels,
                "species_enabled": getattr(self.engine, "species", None) is not None,
                "poll_seconds": self.poll_seconds}

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
                "frames_captured": self.frames_captured,
                "inference_skipped": self.inference_skipped,
                "inference_waiting": self._captured.qsize(),
                "capture_failures": self.capture_failures,
                "last_capture_at": self.last_capture_at,
                "last_capture_error": self.last_capture_error,
                "last_analysis_error": self.last_analysis_error,
                "frames_persisted": self.frames_persisted,
                "failures": self.failures,
                "last_error": self.last_error,
                "collection_mode": self.collection_mode,
                "pipeline": self.pipeline_snapshot(),
                "deployment_id": self.deployment_id,
                "last_completed_at": ((self.last_result or {}).get("inference") or {}).get("completed_at"),
                "last_inference": {key: ((self.last_result or {}).get("inference") or {}).get(key)
                                   for key in ("status", "is_bird", "bird_probability", "timing_ms", "error")},
                "background_epoch": background.epoch if background else None,
                "background_samples": background.sample_count if background else 0,
                "background_ready": background.ready if background else False,
                'background_calibration': background.state() if background else {'phase': 'waiting_for_camera', 'ready': False},
                "last_selection": ((self.last_result or {}).get("motion") or {}).get("selection"),
            }
