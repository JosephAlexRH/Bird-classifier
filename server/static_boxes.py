"""Remember where the box localiser fires on confidently empty frames, and flag those boxes.

A fixed camera's localiser can fire on scenery: a dark shrub against a bright sky, the
feeder tube, a shadowed tray. Those boxes then reach the species model and can be
auto-labelled as a bird. The fix here needs no retraining: every so often the hub also
runs the localiser on a frame the isBird model is confident is empty, and remembers the
boxes it finds. In a later bird frame, a box that lands on a remembered empty-scene box
(IoU >= ``match_iou``, or at least ``containment`` of the box inside it) is marked static. Optionally (``score_margin``) a box is kept when its
score beats the strongest empty-scene score at that place by the margin; this is off by
default because on the first new scene (Nov 2026) the localiser's scores on a shrub varied
from 0.29 to 0.68 between frames, as much as a real bird lifted them.

A real bird where the localiser never fires on the empty scene is unaffected. A bird sitting
exactly on a scenery site is skipped too; species then falls back to the selected crop,
which is the right outcome until the localiser is retrained on the scene.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence

STATIC_MATCH_IOU = 0.5
STATIC_CONTAINMENT = 0.8  # a small box mostly inside a scenery box (a weak box within the shrub)
STATIC_SCORE_MARGIN: float | None = None  # None: every match with empty scenery is static
EMPTY_REFERENCE_MAX_PROBABILITY = 0.25
DEFAULT_REFERENCE_FRAMES = 12
DEFAULT_MAX_AGE_SECONDS = 3600.0
DEFAULT_REFERENCE_INTERVAL_SECONDS = 60.0
_BOX_KEYS = ("left", "top", "right", "bottom")


def _area(r: dict[str, float]) -> float:
    return max(0.0, float(r["right"]) - float(r["left"])) * max(0.0, float(r["bottom"]) - float(r["top"]))


def _intersection(a: dict[str, float], b: dict[str, float]) -> float:
    width = min(float(a["right"]), float(b["right"])) - max(float(a["left"]), float(b["left"]))
    height = min(float(a["bottom"]), float(b["bottom"])) - max(float(a["top"]), float(b["top"]))
    return width * height if width > 0 and height > 0 else 0.0


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    """Intersection over union of two left/top/right/bottom pixel boxes."""
    intersection = _intersection(a, b)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def box_containment(inner: dict[str, float], outer: dict[str, float]) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    area = _area(inner)
    return _intersection(inner, outer) / area if area > 0 else 0.0


@dataclass(frozen=True)
class _EmptyFrame:
    at: float
    capture: str | None
    boxes: tuple[tuple[dict[str, float], float], ...]


class StaticBoxMemory:
    """Per-camera memory of localiser boxes found on confidently empty frames."""

    def __init__(
        self,
        *,
        max_frames: int = DEFAULT_REFERENCE_FRAMES,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        match_iou: float = STATIC_MATCH_IOU,
        containment: float = STATIC_CONTAINMENT,
        score_margin: float | None = STATIC_SCORE_MARGIN,
        min_score: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (max_frames < 1 or max_age_seconds <= 0 or not 0 < match_iou <= 1
                or (score_margin is not None and score_margin < 0)):
            raise ValueError("Invalid static-box memory settings")
        self.max_frames = int(max_frames)
        self.max_age_seconds = float(max_age_seconds)
        self.match_iou = float(match_iou)
        self.containment = float(containment)
        self.score_margin = None if score_margin is None else float(score_margin)
        self.min_score = float(min_score)
        self.clock = clock
        self._frames: dict[str, deque[_EmptyFrame]] = {}
        self._last_reference: dict[str, float] = {}
        self._lock = threading.Lock()

    def _fresh(self, camera: str, now: float) -> list[_EmptyFrame]:
        frames = self._frames.get(camera)
        if not frames:
            return []
        while frames and now - frames[0].at > self.max_age_seconds:
            frames.popleft()
        return list(frames)

    def due(self, camera: str, interval_seconds: float = DEFAULT_REFERENCE_INTERVAL_SECONDS,
            now: float | None = None) -> bool:
        """Whether a new empty-scene reference should be taken for ``camera``."""
        now = self.clock() if now is None else now
        with self._lock:
            last = self._last_reference.get(camera)
        return last is None or now - last >= interval_seconds

    def remember(self, camera: str, detections: Sequence[dict[str, Any]], *, capture_id: str | None = None,
                 now: float | None = None) -> int:
        """Store the boxes the localiser found on a confidently empty frame. Returns how many."""
        now = self.clock() if now is None else now
        boxes = tuple(({k: float(d["box"][k]) for k in _BOX_KEYS}, float(d["score"]))
                      for d in detections if float(d.get("score", 0.0)) >= self.min_score)
        with self._lock:
            frames = self._frames.setdefault(camera, deque(maxlen=self.max_frames))
            frames.append(_EmptyFrame(now, capture_id, boxes))
            self._last_reference[camera] = now
        return len(boxes)

    def annotate(self, camera: str, detections: Sequence[dict[str, Any]], now: float | None = None,
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Copies of ``detections`` with ``static`` set, plus a summary for the inference record."""
        now = self.clock() if now is None else now
        with self._lock:
            frames = self._fresh(camera, now)
        summary: dict[str, Any] = {
            "applied": bool(frames), "reference_frames": len(frames), "suppressed": 0,
            "match_iou": self.match_iou, "containment": self.containment,
            "score_margin": self.score_margin, "max_age_seconds": self.max_age_seconds,
        }
        out: list[dict[str, Any]] = []
        for detection in detections:
            item = dict(detection)
            best: tuple[float, float, _EmptyFrame] | None = None  # (empty score, iou, frame)
            for frame in frames:
                for box, score in frame.boxes:
                    overlap = box_iou(detection["box"], box)
                    inside = box_containment(detection["box"], box)
                    if ((overlap >= self.match_iou or inside >= self.containment)
                            and (best is None or score > best[0])):
                        best = (score, overlap, frame)
            if best is None:
                item["static"] = False
            else:
                empty_score, overlap, frame = best
                item["static"] = (self.score_margin is None
                                  or float(detection["score"]) < empty_score + self.score_margin)
                item["static_match"] = {"empty_score": empty_score, "iou": overlap,
                                        "age_seconds": now - frame.at, "empty_capture": frame.capture}
            summary["suppressed"] += int(item["static"])
            out.append(item)
        return out, summary

    def clear(self, camera: str | None = None) -> None:
        with self._lock:
            if camera is None:
                self._frames.clear()
                self._last_reference.clear()
            else:
                self._frames.pop(camera, None)
                self._last_reference.pop(camera, None)

    def describe(self) -> dict[str, Any]:
        with self._lock:
            counts = {camera: len(frames) for camera, frames in self._frames.items()}
        return {"reference_frames": counts, "max_frames": self.max_frames,
                "max_age_seconds": self.max_age_seconds, "match_iou": self.match_iou,
                "score_margin": self.score_margin}
