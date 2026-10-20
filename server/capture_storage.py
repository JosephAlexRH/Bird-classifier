"""Shared, dependency-free storage helpers for ESP32-CAM captures."""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any


METADATA_HEADER = "X-ESP32-Camera-Metadata"
CAPTURE_SCHEMA_VERSION = 3


class CalibrationType(str, Enum):
    """Supported capture purposes as stable, JSON-compatible identifiers."""

    NORMAL = "normal"
    DARK_FRAME = "dark_frame"
    FLAT_FIELD = "flat_field"
    GRAY_CARD = "gray_card"
    COLOR_CHART = "color_chart"
    RESOLUTION_TARGET = "resolution_target"


def calibration_type_values() -> tuple[str, ...]:
    """Return the public calibration taxonomy in dashboard/API order."""
    return tuple(item.value for item in CalibrationType)


def parse_calibration_type(value: object) -> CalibrationType:
    """Convert an API/storage value to :class:`CalibrationType` or raise clearly."""
    try:
        return CalibrationType(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(calibration_type_values())
        raise ValueError(f"Unsupported calibration type {value!r}; allowed values: {allowed}.") from error


def is_jpeg(data: bytes) -> bool:
    """Return whether *data* has complete JPEG start/end markers."""
    return len(data) >= 4 and data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


def parse_camera_metadata(value: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Parse optional firmware metadata without rejecting legacy uploads."""
    if not value:
        return None, "Camera metadata was not supplied."
    try:
        metadata = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        return None, f"Camera metadata was invalid JSON: {error}"
    if not isinstance(metadata, dict):
        return None, "Camera metadata must be a JSON object."
    dump = ((metadata.get("camera") or {}).get("register_dump") or {})
    if isinstance(dump, dict) and dump.get("payload_included"):
        encoded = dump.get("payload_base64")
        expected_hash = dump.get("sha256")
        if not isinstance(encoded, str) or not isinstance(expected_hash, str):
            return metadata, "Camera register dump payload lacks base64 data or SHA-256."
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            return metadata, f"Camera register dump payload was invalid base64: {error}"
        actual_hash = hashlib.sha256(decoded).hexdigest()
        if actual_hash != expected_hash.lower():
            return metadata, "Camera register dump SHA-256 did not match its decoded payload."
    return metadata, None


def _frame_utc_estimate(metadata: dict[str, Any]) -> str | None:
    """Estimate framebuffer UTC from one wall-clock/monotonic-clock sample pair."""
    device = metadata.get("device")
    camera = metadata.get("camera")
    if not isinstance(device, dict) or not isinstance(camera, dict):
        return None
    clock = device.get("clock")
    timing = camera.get("timing")
    if not isinstance(clock, dict) or not isinstance(timing, dict):
        return None

    device_time_utc = clock.get("device_time_utc")
    device_time_boot_us = clock.get("device_time_boot_us")
    frame_boot_us = timing.get("frame_boot_us")
    if not isinstance(device_time_utc, str):
        return None
    if (
        isinstance(device_time_boot_us, bool)
        or not isinstance(device_time_boot_us, (int, float))
        or isinstance(frame_boot_us, bool)
        or not isinstance(frame_boot_us, (int, float))
    ):
        return None

    try:
        parsed = datetime.fromisoformat(device_time_utc.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            return None
        estimate = parsed.astimezone(timezone.utc) + timedelta(
            microseconds=frame_boot_us - device_time_boot_us
        )
    except (OverflowError, TypeError, ValueError):
        return None
    return estimate.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _flatten_camera_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Map the firmware envelope into the sidecar without a camera.camera layer."""
    if metadata is None:
        return {
            "firmware_metadata_schema_version": None,
            "device": None,
            "camera": None,
        }

    device = copy.deepcopy(metadata.get("device"))
    camera = copy.deepcopy(metadata.get("camera"))
    if isinstance(camera, dict):
        timing = camera.get("timing")
        if not isinstance(timing, dict):
            timing = {}
            camera["timing"] = timing
        timing["frame_utc_estimate"] = _frame_utc_estimate(metadata)
    return {
        "firmware_metadata_schema_version": metadata.get("schema_version"),
        "device": device,
        "camera": camera,
    }


class CaptureStore:
    """Persist JPEGs, JSON sidecars, profiles, and a JSONL manifest."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"
        self._lock = threading.Lock()

    @staticmethod
    def _identifier(prefix: str) -> tuple[str, str]:
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%dT%H%M%S_%fZ")
        return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}", now.isoformat(timespec="milliseconds")

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def save_capture(
        self,
        image: bytes,
        *,
        source: str,
        camera_metadata: dict[str, Any] | None = None,
        metadata_error: str | None = None,
        calibration_type: CalibrationType | str = CalibrationType.NORMAL,
        label: str = "",
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not is_jpeg(image):
            raise ValueError("Image is not a complete JPEG.")
        calibration = parse_calibration_type(calibration_type)

        capture_id, received_at = self._identifier("capture")
        image_path = self.root / f"{capture_id}.jpg"
        sidecar_path = self.root / f"{capture_id}.json"
        record: dict[str, Any] = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "capture_id": capture_id,
            "filename": image_path.name,
            "sidecar": sidecar_path.name,
            "received_at": received_at,
            "source": source,
            "calibration_type": calibration.value,
            "label": label.strip(),
            "note": note.strip(),
            "bytes": len(image),
            "sha256": hashlib.sha256(image).hexdigest(),
            "metadata_error": metadata_error,
            **_flatten_camera_metadata(camera_metadata),
        }
        if extra:
            record.update(extra)
        # Callers may add derived fields but cannot downgrade the storage contract.
        record["schema_version"] = CAPTURE_SCHEMA_VERSION
        record["calibration_type"] = calibration.value

        encoded_record = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        manifest_line = (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")

        with self._lock:
            try:
                self._atomic_write(image_path, image)
                self._atomic_write(sidecar_path, encoded_record)
                with self.manifest_path.open("a+b") as manifest:
                    manifest.seek(0, os.SEEK_END)
                    previous_length = manifest.tell()
                    try:
                        manifest.write(manifest_line)
                        manifest.flush()
                        os.fsync(manifest.fileno())
                    except Exception:
                        manifest.seek(previous_length)
                        manifest.truncate()
                        manifest.flush()
                        raise
            except Exception:
                image_path.unlink(missing_ok=True)
                sidecar_path.unlink(missing_ok=True)
                raise
        return record

    def save_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        profile_id, captured_at = self._identifier("camera_profile")
        path = self.profiles_dir / f"{profile_id}.json"
        record = {
            **profile,
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "profile_id": profile_id,
            "captured_at": captured_at,
        }
        encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with self._lock:
            self._atomic_write(path, encoded)
        return {**record, "filename": str(path.relative_to(self.root))}

    def save_derived_record(
        self,
        image_path: Path,
        suffix: str,
        record: dict[str, Any],
    ) -> Path:
        """Atomically persist an immutable derived record beside a stored image."""
        image_path = Path(image_path).resolve()
        if image_path.parent != self.root or not image_path.is_file():
            raise ValueError("Derived records require an existing image directly inside the capture store.")
        if not suffix or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in suffix):
            raise ValueError("Derived record suffix must contain lowercase letters, digits, or underscores.")
        if not isinstance(record, dict):
            raise ValueError("Derived record must be a JSON object.")
        destination = image_path.with_suffix(f".{suffix}.json")
        encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with self._lock:
            self._atomic_write(destination, encoded)
        return destination