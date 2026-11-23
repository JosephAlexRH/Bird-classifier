"""Running per-camera tallies for the dashboard's Fleet tab.

The hub saves thousands of frames a day, so the Fleet tab must not rescan the
capture folder. Instead every saved frame is counted once as it is persisted,
into 15-minute buckets per camera. Buckets are kept for a week and written to
``fleet_stats.json`` beside the captures, so counts survive a restart.

15-minute buckets line up with local midnight in every time zone, so the
browser can ask for "today" in its own time zone (``since`` in Unix seconds).
Closed visits are not counted here; they come from ``visits.jsonl``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable

BUCKET_SECONDS = 900
KEEP_SECONDS = 8 * 86400
SAVE_EVERY_SECONDS = 30
RECENT_CAPTURES = 4096
FINAL_STATUSES = {"complete", "skipped", "error"}


def best_label(inference: dict[str, Any]) -> str | None:
    """The top species guess for a bird frame, or None."""
    if not isinstance(inference, dict) or inference.get("is_bird") is not True:
        return None
    species = inference.get("species") if isinstance(inference.get("species"), dict) else {}
    for box in species.get("boxes") or []:
        top = (box or {}).get("top_k") or []
        if top and isinstance(top[0], dict) and top[0].get("label"):
            return str(top[0]["label"])
    top = inference.get("species_top_k") or []
    if top and isinstance(top[0], dict) and top[0].get("label"):
        return str(top[0]["label"])
    return None


def _empty_bucket() -> dict[str, Any]:
    return {"frames": 0, "analysed": 0, "bird_frames": 0, "errors": 0, "ms_total": 0.0, "ms_count": 0, "labels": {}}


class FleetStats:
    def __init__(self, path: Path | None = None, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, dict[int, dict[str, Any]]] = {}
        self._recent: OrderedDict[str, tuple[str, int, bool]] = OrderedDict()
        self._dirty = False
        self._saved_at = 0.0
        self.started_at = clock()
        self._load()

    # ------------------------------------------------------------- recording
    def observe(self, camera: str | None, capture_id: str | None, inference: dict[str, Any] | None) -> None:
        """Count a saved frame. Safe to call again when its analysis finishes."""
        if not camera or not capture_id:
            return
        inference = inference if isinstance(inference, dict) else {}
        status = inference.get("status")
        final = status in FINAL_STATUSES or (status is None and "is_bird" in inference)
        now = self._clock()
        with self._lock:
            known = self._recent.get(capture_id)
            if known is None:
                bucket_key = int(now // BUCKET_SECONDS) * BUCKET_SECONDS
                bucket = self._bucket(camera, bucket_key)
                bucket["frames"] += 1
            else:
                camera, bucket_key, already_final = known
                if already_final or not final:
                    self._recent.move_to_end(capture_id)
                    return
                bucket = self._bucket(camera, bucket_key)
            if final:
                self._finalise(bucket, inference)
            self._recent[capture_id] = (camera, bucket_key, final)
            self._recent.move_to_end(capture_id)
            while len(self._recent) > RECENT_CAPTURES:
                self._recent.popitem(last=False)
            self._dirty = True
            self._prune(now)
        self.save_if_due()

    def _bucket(self, camera: str, key: int) -> dict[str, Any]:
        return self._buckets.setdefault(camera, {}).setdefault(key, _empty_bucket())

    @staticmethod
    def _finalise(bucket: dict[str, Any], inference: dict[str, Any]) -> None:
        status = inference.get("status")
        if status == "error":
            bucket["errors"] += 1
            return
        if status == "skipped":
            return
        bucket["analysed"] += 1
        if inference.get("is_bird") is True:
            bucket["bird_frames"] += 1
            label = best_label(inference)
            if label:
                bucket["labels"][label] = bucket["labels"].get(label, 0) + 1
        timing = inference.get("timing_ms") if isinstance(inference.get("timing_ms"), dict) else {}
        total = timing.get("total")
        if isinstance(total, (int, float)) and total >= 0:
            bucket["ms_total"] += float(total)
            bucket["ms_count"] += 1

    def _prune(self, now: float) -> None:
        oldest = now - KEEP_SECONDS
        for camera in list(self._buckets):
            buckets = self._buckets[camera]
            for key in [k for k in buckets if k < oldest]:
                del buckets[key]
            if not buckets:
                del self._buckets[camera]

    # --------------------------------------------------------------- reading
    def totals(self, since: float, until: float | None = None) -> dict[str, dict[str, Any]]:
        """Per-camera sums of every bucket that starts at or after ``since``."""
        floor = int(since // BUCKET_SECONDS) * BUCKET_SECONDS
        result: dict[str, dict[str, Any]] = {}
        with self._lock:
            for camera, buckets in self._buckets.items():
                total = _empty_bucket()
                labels: Counter[str] = Counter()
                for key, bucket in buckets.items():
                    if key < floor or (until is not None and key >= until):
                        continue
                    for field in ("frames", "analysed", "bird_frames", "errors", "ms_total", "ms_count"):
                        total[field] += bucket[field]
                    labels.update(bucket["labels"])
                total["labels"] = labels.most_common(3)
                total["mean_processing_ms"] = (round(total["ms_total"] / total["ms_count"])
                                               if total["ms_count"] else None)
                for field in ("ms_total", "ms_count"):
                    total.pop(field)
                result[camera] = total
        return result

    def earliest(self) -> int | None:
        """Start of the oldest bucket held, so the page can say when counting began."""
        with self._lock:
            starts = [key for buckets in self._buckets.values() for key in buckets]
        return min(starts) if starts else None

    # ----------------------------------------------------------- persistence
    def _load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        cameras = data.get("cameras") if isinstance(data, dict) else None
        if not isinstance(cameras, dict):
            return
        for camera, buckets in cameras.items():
            if not isinstance(buckets, dict):
                continue
            for key, bucket in buckets.items():
                try:
                    start = int(key)
                except ValueError:
                    continue
                if not isinstance(bucket, dict):
                    continue
                clean = _empty_bucket()
                for field in ("frames", "analysed", "bird_frames", "errors", "ms_count"):
                    clean[field] = int(bucket.get(field) or 0)
                clean["ms_total"] = float(bucket.get("ms_total") or 0)
                labels = bucket.get("labels")
                clean["labels"] = {str(k): int(v) for k, v in labels.items()} if isinstance(labels, dict) else {}
                self._buckets.setdefault(str(camera), {})[start] = clean
        self._prune(self._clock())

    def save_if_due(self, force: bool = False) -> None:
        if self.path is None:
            return
        now = self._clock()
        with self._lock:
            if not self._dirty or (not force and now - self._saved_at < SAVE_EVERY_SECONDS):
                return
            payload = {"version": 1, "bucket_seconds": BUCKET_SECONDS,
                       "cameras": {camera: {str(k): v for k, v in buckets.items()}
                                   for camera, buckets in self._buckets.items()}}
            self._dirty = False
            self._saved_at = now
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".fleet_stats.", suffix=".tmp", dir=self.path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            os.replace(temporary, self.path)
        except OSError:
            with self._lock:
                self._dirty = True  # counts stay in memory; retry on the next frame


def visit_totals(visits: Iterable[dict[str, Any]], since: float,
                 camera_for: Callable[[str], str | None]) -> dict[str, dict[str, int]]:
    """Closed visits per camera that ended at or after ``since``.

    ``camera_for`` maps the hardware camera id at the front of a visit id
    (``esp32cam-a/capture_...``) to the key used for the dashboard's cameras.
    """
    from datetime import datetime

    result: dict[str, dict[str, int]] = {}
    for visit in visits:
        visit_id = str(visit.get("visit_id") or "")
        stamp = visit.get("end") or visit.get("start")
        try:
            ended = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if ended < since:
            continue
        camera = camera_for(visit_id.split("/", 1)[0])
        if camera is None:
            continue
        counts = result.setdefault(camera, {"visits": 0, "auto": 0, "review": 0})
        counts["visits"] += 1
        decision = ((visit.get("result") or {}).get("decision"))
        counts["auto" if decision in {"species", "group"} else "review"] += 1
    return result
