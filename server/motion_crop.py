"""Pi-side background modelling and full-resolution motion crop selection."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    import cv2 as _cv2
except ImportError:  # NumPy fallback is covered by tests and works on minimal Pi installs.
    _cv2 = None


MOTION_SCHEMA_VERSION = 1
FINAL_CROP_SOURCE = "capture_motion_record.selection.final_bounds_px"


@dataclass(frozen=True)
class MotionConfig:
    background_frames: int = 9
    minimum_background_frames: int = 3
    minimum_threshold: int = 20
    maximum_threshold: int = 48
    noise_multiplier: float = 4.5
    margin_fraction: float = 0.20
    minimum_crop_size: int = 160
    minimum_component_pixels: int = 100
    minimum_component_fraction: float = 0.0005
    maximum_component_fraction: float = 0.20
    maximum_foreground_fraction: float = 0.30
    minimum_fill_ratio: float = 0.15
    minimum_dominance: float = 0.50
    minimum_largest_ratio: float = 1.5
    minimum_component_dimension: int = 5
    admission_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.background_frames < 1 or self.background_frames % 2 == 0:
            raise ValueError("background_frames must be a positive odd number")
        if not 1 <= self.minimum_background_frames <= self.background_frames:
            raise ValueError("minimum_background_frames is out of range")
        if self.minimum_crop_size < 1 or not 0 <= self.margin_fraction < 1:
            raise ValueError("crop settings are invalid")

    @property
    def sha256(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class RunningMedianBackground:
    """One restart-scoped grayscale background window for a single camera."""

    def __init__(self, config: MotionConfig = MotionConfig(), *, camera_id: str = "camera") -> None:
        self.config = config
        self.camera_id = camera_id
        self.epoch = uuid.uuid4().hex
        self._frames: deque[np.ndarray] = deque(maxlen=config.background_frames)
        self._median: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self.last_admitted_at: float | None = None

    @property
    def sample_count(self) -> int:
        return len(self._frames)

    @property
    def ready(self) -> bool:
        return self.sample_count >= self.config.minimum_background_frames

    def reset(self, *, camera_id: str | None = None) -> None:
        if camera_id is not None:
            self.camera_id = camera_id
        self.epoch = uuid.uuid4().hex
        self._frames.clear()
        self._median = None
        self._shape = None
        self.last_admitted_at = None

    def admit(self, gray: np.ndarray, *, now: float | None = None, force: bool = False) -> bool:
        gray = validate_gray(gray)
        now = time.monotonic() if now is None else float(now)
        if not force and self.last_admitted_at is not None:
            if now - self.last_admitted_at < self.config.admission_seconds:
                return False
        if self._shape is not None and gray.shape != self._shape:
            self.reset()
        self._shape = gray.shape
        self._frames.append(np.ascontiguousarray(gray.copy()))
        self._median = None
        self.last_admitted_at = now
        return True

    def median(self) -> np.ndarray | None:
        if not self.ready:
            return None
        if self._median is None:
            frames = tuple(self._frames)
            if len(frames) % 2 == 0:
                frames = frames[1:]  # Use the newest odd count during warm-up.
            stack = np.stack(frames, axis=0)
            middle = stack.shape[0] // 2
            self._median = np.partition(stack, middle, axis=0)[middle].astype(np.uint8)
        return self._median


def validate_gray(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray)
    if gray.dtype != np.uint8 or gray.ndim != 2 or min(gray.shape) < 1:
        raise ValueError("Expected a non-empty HxW uint8 grayscale image")
    return gray


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an HxWx3 uint8 RGB image")
    if _cv2 is not None:
        return _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
    values = (
        rgb[..., 0].astype(np.uint16) * 77
        + rgb[..., 1].astype(np.uint16) * 150
        + rgb[..., 2].astype(np.uint16) * 29
        + 128
    ) >> 8
    return values.astype(np.uint8)


def normalized_roi_bounds(roi: Any, width: int, height: int) -> tuple[int, int, int, int]:
    keys = {"x", "y", "width", "height"}
    if not isinstance(roi, dict) or set(roi) != keys:
        raise ValueError("Static ROI must contain exactly x, y, width, and height")
    values = []
    for key in ("x", "y", "width", "height"):
        value = roi[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Static ROI {key} must be finite and numeric")
        values.append(float(value))
    x, y, roi_width, roi_height = values
    if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0 or x + roi_width > 1.000001 or y + roi_height > 1.000001:
        raise ValueError("Static ROI lies outside the normalized frame")
    return (
        max(0, int(math.floor(x * width))),
        max(0, int(math.floor(y * height))),
        min(width, int(math.ceil((x + roi_width) * width))),
        min(height, int(math.ceil((y + roi_height) * height))),
    )


def final_crop_bounds(
    bounds: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    *,
    margin_fraction: float = 0.20,
    minimum_size: int = 160,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    if not (0 <= left < right <= frame_width and 0 <= top < bottom <= frame_height):
        raise ValueError("Crop bounds are outside the image")
    width, height = right - left, bottom - top
    target_width = min(frame_width, max(minimum_size, int(math.ceil(width * (1 + margin_fraction)))))
    target_height = min(frame_height, max(minimum_size, int(math.ceil(height * (1 + margin_fraction)))))
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    crop_left = int(math.floor(center_x - target_width / 2))
    crop_top = int(math.floor(center_y - target_height / 2))
    crop_left = min(max(0, crop_left), frame_width - target_width)
    crop_top = min(max(0, crop_top), frame_height - target_height)
    return crop_left, crop_top, crop_left + target_width, crop_top + target_height


def _blur(gray: np.ndarray) -> np.ndarray:
    if _cv2 is not None:
        return _cv2.GaussianBlur(gray, (5, 5), 0)
    padded = np.pad(gray.astype(np.uint16), 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    return np.rint(windows.mean(axis=(-1, -2))).astype(np.uint8)


def _binary_morphology(mask: np.ndarray) -> np.ndarray:
    if _cv2 is not None:
        opened = _cv2.morphologyEx(mask.astype(np.uint8), _cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return _cv2.morphologyEx(opened, _cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)

    def erode(value: np.ndarray, size: int) -> np.ndarray:
        padded = np.pad(value, size // 2, constant_values=False)
        return np.lib.stride_tricks.sliding_window_view(padded, (size, size)).all(axis=(-1, -2))

    def dilate(value: np.ndarray, size: int) -> np.ndarray:
        padded = np.pad(value, size // 2, constant_values=False)
        return np.lib.stride_tricks.sliding_window_view(padded, (size, size)).any(axis=(-1, -2))

    return erode(dilate(dilate(erode(mask, 3), 3), 7), 7)


def _components(mask: np.ndarray) -> list[dict[str, int]]:
    if _cv2 is not None:
        count, _, stats, _ = _cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        return [
            {"left": int(s[0]), "top": int(s[1]), "right": int(s[0] + s[2]),
             "bottom": int(s[1] + s[3]), "area": int(s[4])}
            for s in stats[1:count]
        ]
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[dict[str, int]] = []
    for y, x in zip(*np.nonzero(mask & ~visited)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        area = 0
        left = right = int(x)
        top = bottom = int(y)
        while stack:
            cy, cx = stack.pop()
            area += 1
            left, right = min(left, cx), max(right, cx)
            top, bottom = min(top, cy), max(bottom, cy)
            for ny in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                for nx in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        components.append({"left": left, "top": top, "right": right + 1, "bottom": bottom + 1, "area": area})
    return components


def select_motion_crop(
    gray: np.ndarray,
    background: np.ndarray | None,
    static_roi: Any,
    config: MotionConfig = MotionConfig(),
) -> dict[str, Any]:
    """Return a dynamic largest-blob crop or a strict static-ROI fallback."""
    gray = validate_gray(gray)
    height, width = gray.shape
    roi_bounds = normalized_roi_bounds(static_roi, width, height)
    fallback_reason = "background_warming" if background is None else None
    detection: dict[str, Any] = {"status": "fallback", "reason": fallback_reason}
    tight_bounds: tuple[int, int, int, int] | None = None
    if background is not None:
        background = validate_gray(background)
        if background.shape != gray.shape:
            fallback_reason = "background_shape_mismatch"
        else:
            current_blur, background_blur = _blur(gray), _blur(background)
            signed = current_blur.astype(np.int16) - background_blur.astype(np.int16)
            left, top, right, bottom = roi_bounds
            roi_signed = signed[top:bottom, left:right]
            global_shift = float(np.median(roi_signed))
            residual = np.abs(signed.astype(np.float32) - global_shift)
            sigma = float(1.4826 * np.median(np.abs(roi_signed.astype(np.float32) - global_shift)))
            raw_threshold = max(config.minimum_threshold, int(math.ceil(config.noise_multiplier * sigma)))
            if raw_threshold > config.maximum_threshold:
                fallback_reason = "excessive_noise"
            else:
                mask = residual >= raw_threshold
                search = np.zeros_like(mask)
                search[top:bottom, left:right] = True
                mask = _binary_morphology(mask & search) & search
                components = sorted(_components(mask), key=lambda item: item["area"], reverse=True)
                foreground = int(mask.sum())
                search_area = (right - left) * (bottom - top)
                detection.update({
                    "threshold": raw_threshold, "noise_sigma": sigma, "global_shift": global_shift,
                    "changed_pixels": foreground, "component_count": len(components),
                })
                if not components or foreground == 0:
                    fallback_reason = "no_changed_blob"
                elif foreground > config.maximum_foreground_fraction * search_area:
                    fallback_reason = "broad_scene_change"
                else:
                    largest = components[0]
                    second_area = components[1]["area"] if len(components) > 1 else 0
                    box_area = (largest["right"] - largest["left"]) * (largest["bottom"] - largest["top"])
                    border_touches = sum((largest["left"] <= left, largest["top"] <= top,
                                          largest["right"] >= right, largest["bottom"] >= bottom))
                    minimum_area = max(config.minimum_component_pixels, int(math.ceil(config.minimum_component_fraction * search_area)))
                    clear = (
                        largest["area"] >= minimum_area
                        and largest["area"] <= config.maximum_component_fraction * search_area
                        and largest["right"] - largest["left"] >= config.minimum_component_dimension
                        and largest["bottom"] - largest["top"] >= config.minimum_component_dimension
                        and largest["area"] / box_area >= config.minimum_fill_ratio
                        and largest["area"] / foreground >= config.minimum_dominance
                        and (second_area == 0 or largest["area"] / second_area >= config.minimum_largest_ratio)
                        and border_touches < 3
                    )
                    if clear:
                        tight_bounds = (largest["left"], largest["top"], largest["right"], largest["bottom"])
                        fallback_reason = None
                        detection.update({"status": "clear_blob", "reason": None, "largest": largest, "second_area": second_area})
                    else:
                        fallback_reason = "ambiguous_blob"
    selected_bounds = tight_bounds or roi_bounds
    final_bounds = final_crop_bounds(
        selected_bounds, width, height,
        margin_fraction=config.margin_fraction, minimum_size=config.minimum_crop_size,
    )
    source = "dynamic_motion_blob" if tight_bounds else "static_roi_fallback"
    detection["status"] = "clear_blob" if tight_bounds else "fallback"
    detection["reason"] = fallback_reason
    record = {
        "schema_version": MOTION_SCHEMA_VERSION,
        "producer": {"name": "pi_motion_blob", "version": 1, "config_sha256": config.sha256},
        "static_roi": {"source": "capture_sidecar.camera.requested.motion.roi", "normalized": dict(static_roi)},
        "detection": {**detection, "tight_blob_bounds_px": _bounds_dict(tight_bounds) if tight_bounds else None},
        "selection": {
            "source": source,
            "fallback_reason": fallback_reason,
            "coordinate_space": "original_decoded_frame_pixels",
            "bounds_semantics": "half_open_left_top_right_bottom",
            "final_bounds_px": _bounds_dict(final_bounds),
            "margin_total_fraction": config.margin_fraction,
            "minimum_size_px": config.minimum_crop_size,
        },
    }
    canonical = json.dumps(record["selection"], sort_keys=True, separators=(",", ":")).encode()
    record["crop_id"] = hashlib.sha256(canonical).hexdigest()
    return record


def _bounds_dict(bounds: tuple[int, int, int, int]) -> dict[str, int]:
    left, top, right, bottom = bounds
    return {"left": left, "top": top, "right": right, "bottom": bottom,
            "width": right - left, "height": bottom - top}


def should_admit_background(inference: dict[str, Any], *, threshold: float) -> bool:
    if inference.get("status") != "complete" or inference.get("is_bird") is not False:
        return False
    probability = inference.get("bird_probability")
    return isinstance(probability, (int, float)) and not isinstance(probability, bool) and math.isfinite(probability) and probability <= min(0.25, 0.5 * threshold)