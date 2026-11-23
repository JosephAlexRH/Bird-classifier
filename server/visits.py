"""Group captures into bird visits, live on the hub and in batch over a dataset.

A visit is a run of persisted captures from one camera with no gap longer than
``gap_seconds`` between consecutive captures. A change of deployment or camera boot
always starts a new visit. A visit's ID is ``<camera key>/<first capture id>``, so
rebuilding visits from the same captures gives the same IDs. Training notebooks
and labelling tools must use this module rather than their own copy of the rule.

Visits are *derived* data. They are never written into capture sidecars (which
record acquisition only). The hub writes a provisional ``visit`` block into each
``.inference.json`` and appends one line per closed visit to ``visits.jsonl``. The
batch rebuild (``python -m server.visits rebuild <capture dir>``) is authoritative.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

VISIT_SCHEMA_VERSION = 1
DEFAULT_GAP_SECONDS = 120.0
# A visit needs at least this many species-scored frames before it is auto-labelled; a single frame
# is one look at one box and stays in review. A contract can override it with "min_frames_for_auto".
DEFAULT_MIN_FRAMES_FOR_AUTO = 2
LEGACY_CAMERA_KEY = "legacy"
_CAPTURE_ID = re.compile(r"^capture_(\d{8}T\d{6})_(\d{6})Z_")


def capture_time(capture_id: str) -> datetime:
    """UTC time encoded in a capture ID such as capture_20261117T141007_411624Z_107653df."""
    match = _CAPTURE_ID.match(capture_id)
    if match is None:
        raise ValueError(f"Not a capture ID: {capture_id!r}")
    stamp = datetime.strptime("".join(match.groups()), "%Y%m%dT%H%M%S%f")
    return stamp.replace(tzinfo=timezone.utc)


def camera_key(camera_id: str | None) -> str:
    return camera_id.strip() if isinstance(camera_id, str) and camera_id.strip() else LEGACY_CAMERA_KEY


def visit_id_for(camera: str, first_capture_id: str) -> str:
    return f"{camera}/{first_capture_id}"


@dataclass(frozen=True)
class CaptureRef:
    """The facts visit grouping needs from one capture sidecar."""

    capture_id: str
    camera_id: str | None = None
    deployment_id: str | None = None
    boot_id: str | None = None

    @property
    def camera(self) -> str:
        return camera_key(self.camera_id)

    @property
    def time(self) -> datetime:
        return capture_time(self.capture_id)


def starts_new_visit(previous: CaptureRef | None, current: CaptureRef, gap_seconds: float) -> bool:
    if previous is None or previous.camera != current.camera:
        return True
    if (previous.deployment_id or "") != (current.deployment_id or ""):
        return True
    if previous.boot_id and current.boot_id and previous.boot_id != current.boot_id:
        return True
    return (current.time - previous.time).total_seconds() > gap_seconds


def assign_visits(captures: Iterable[CaptureRef], gap_seconds: float = DEFAULT_GAP_SECONDS) -> dict[str, str]:
    """Batch rule: capture_id -> visit_id for every capture (input order does not matter)."""
    if not gap_seconds > 0:
        raise ValueError("gap_seconds must be positive")
    ordered = sorted(captures, key=lambda c: (c.camera, c.time, c.capture_id))
    result: dict[str, str] = {}
    previous: CaptureRef | None = None
    current_visit = ""
    for capture in ordered:
        if starts_new_visit(previous, capture, gap_seconds):
            current_visit = visit_id_for(capture.camera, capture.capture_id)
        result[capture.capture_id] = current_visit
        previous = capture
    return result


# ---- Species aggregation and the automation decision ----------------------------------


def mean_probabilities(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        return []
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("All probability rows must have the same length")
    return [sum(row[k] for row in rows) / len(rows) for k in range(width)]


def decide(mean_probs: Sequence[float], labels: Sequence[str], automation: dict[str, Any] | None,
           n_frames: int | None = None) -> dict[str, Any]:
    """Apply a species contract's automation block to a visit's mean probabilities.

    Returns decision "species" (auto-labelled), "group" (auto-labelled to a species group)
    or "review", plus the top species and top group with their confidences. With ``n_frames``
    (the number of species-scored frames averaged), a visit with fewer than
    ``min_frames_for_auto`` frames (default DEFAULT_MIN_FRAMES_FOR_AUTO) is never auto-labelled.
    """
    if not mean_probs:
        return {"decision": "review", "reason": "no_species_scores"}
    if len(mean_probs) != len(labels):
        raise ValueError("mean_probs and labels differ in length")
    automation = automation or {}
    top = max(range(len(mean_probs)), key=lambda k: mean_probs[k])
    species, confidence = labels[top], float(mean_probs[top])
    group_of = {member: group for group, members in (automation.get("groups") or {}).items() for member in members}
    group_scores: dict[str, float] = {}
    for label, p in zip(labels, mean_probs):
        group = group_of.get(label, label)
        group_scores[group] = group_scores.get(group, 0.0) + float(p)
    top_group = max(group_scores, key=group_scores.get)
    out = {"species": species, "species_confidence": confidence,
           "group": top_group, "group_confidence": group_scores[top_group]}
    species_min = automation.get("species_min_confidence")
    group_min = automation.get("group_min_confidence")
    auto_species = set(automation.get("auto_species") or [])
    min_frames = automation.get("min_frames_for_auto", DEFAULT_MIN_FRAMES_FOR_AUTO)
    if n_frames is not None and isinstance(min_frames, int) and n_frames < min_frames:
        return {"decision": "review", "label": None, "reason": "too_few_frames", "species_frames": n_frames, **out}
    if isinstance(species_min, (int, float)) and species in auto_species and confidence >= species_min:
        return {"decision": "species", "label": species, **out}
    if isinstance(group_min, (int, float)) and group_scores[top_group] >= group_min:
        return {"decision": "group", "label": top_group, **out}
    return {"decision": "review", "label": None, **out}


# ---- Live tracking on the hub -----------------------------------------------------------


@dataclass
class _OpenVisit:
    visit_id: str
    camera: str
    last: CaptureRef
    first_capture_id: str
    first_time: datetime
    last_time: datetime
    captures: list[str] = field(default_factory=list)
    species_rows: list[list[float]] = field(default_factory=list)
    labels: tuple[str, ...] | None = None


class VisitTracker:
    """Thread-safe live visit assignment shared by every camera's motion service.

    ``observe`` returns the visit block for a persisted capture's inference record.
    Visits close when the next capture from the same camera starts a new visit, or
    when ``close_stale`` sees no capture for longer than the gap; each closed visit
    is appended to ``log_path`` as one JSON line.
    """

    def __init__(self, log_path: Path | None, *, gap_seconds: float = DEFAULT_GAP_SECONDS,
                 automation: dict[str, Any] | None = None) -> None:
        if not gap_seconds > 0:
            raise ValueError("gap_seconds must be positive")
        self.log_path = Path(log_path) if log_path is not None else None
        self.gap_seconds = float(gap_seconds)
        self.automation = automation
        self._open: dict[str, _OpenVisit] = {}
        self._lock = threading.Lock()

    def observe(self, capture: CaptureRef, inference: dict[str, Any] | None = None) -> dict[str, Any]:
        species = _species_row(inference)
        with self._lock:
            current = self._open.get(capture.camera)
            closed = None
            if current is None or starts_new_visit(current.last, capture, self.gap_seconds):
                closed = current
                current = _OpenVisit(visit_id_for(capture.camera, capture.capture_id), capture.camera, capture,
                                     capture.capture_id, capture.time, capture.time)
                self._open[capture.camera] = current
            current.last, current.last_time = capture, capture.time
            current.captures.append(capture.capture_id)
            if species is not None:
                labels, probs = species
                if current.labels is None:
                    current.labels = labels
                if current.labels == labels:
                    current.species_rows.append(probs)
            block = {"visit_id": current.visit_id, "provisional": True, "gap_seconds": self.gap_seconds,
                     "schema_version": VISIT_SCHEMA_VERSION, "frame_index": len(current.captures) - 1}
        if closed is not None:
            self._write(closed)
        return block

    def close_stale(self, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            stale = [v for v in self._open.values() if (now - v.last_time).total_seconds() > self.gap_seconds]
            for visit in stale:
                del self._open[visit.camera]
        return [self._write(visit) for visit in stale]

    def close_all(self) -> list[dict[str, Any]]:
        with self._lock:
            visits, self._open = list(self._open.values()), {}
        return [self._write(visit) for visit in visits]

    def summary(self, visit: _OpenVisit) -> dict[str, Any]:
        mean = mean_probabilities(visit.species_rows)
        return {
            "schema_version": VISIT_SCHEMA_VERSION, "visit_id": visit.visit_id, "camera": visit.camera,
            "deployment_id": visit.last.deployment_id or "", "start": visit.first_time.isoformat(),
            "end": visit.last_time.isoformat(), "duration_seconds": (visit.last_time - visit.first_time).total_seconds(),
            "captures": len(visit.captures), "capture_ids": visit.captures,
            "species_frames": len(visit.species_rows),
            "species_labels": list(visit.labels or ()), "species_mean_probabilities": mean,
            "result": (decide(mean, list(visit.labels or ()), self.automation, n_frames=len(visit.species_rows))
                       if mean else {"decision": "review", "reason": "no_species_scores"}),
            "gap_seconds": self.gap_seconds, "source": "live",
        }

    def _write(self, visit: _OpenVisit) -> dict[str, Any]:
        record = self.summary(visit)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return record


def _species_row(inference: dict[str, Any] | None) -> tuple[tuple[str, ...], list[float]] | None:
    """The best-localised bird's species probabilities from an inference record, if any.

    Only the first box (highest localiser score) is used, so a second bird of another
    species in the same frame does not blur the visit's average.
    """
    if not isinstance(inference, dict) or inference.get("status") != "complete":
        return None
    species = inference.get("species") or {}
    labels = species.get("labels")
    boxes = species.get("boxes") or []
    rows = [b.get("probabilities") for b in boxes if isinstance(b, dict) and isinstance(b.get("probabilities"), list)]
    if not isinstance(labels, list) or not rows or len(rows[0]) != len(labels):
        return None
    return tuple(labels), [float(p) for p in rows[0]]


# ---- Batch rebuild from a capture directory ------------------------------------------------


def read_capture_ref(sidecar_path: Path) -> CaptureRef | None:
    try:
        data = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    capture_id = data.get("capture_id") or Path(sidecar_path).stem
    if not isinstance(capture_id, str) or _CAPTURE_ID.match(capture_id) is None:
        return None
    device = data.get("device") or {}
    return CaptureRef(capture_id, data.get("camera_id") or device.get("camera_id"),
                      data.get("deployment_id"), device.get("boot_id"))


def rebuild(directory: Path, gap_seconds: float = DEFAULT_GAP_SECONDS, automation: dict[str, Any] | None = None,
            ) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Rebuild every visit from the capture sidecars (and any .inference.json) in ``directory``."""
    directory = Path(directory)
    refs = [ref for path in sorted(directory.glob("capture_*.json"))
            if not path.name.endswith((".inference.json", ".motion.json"))
            for ref in [read_capture_ref(path)] if ref is not None]
    mapping = assign_visits(refs, gap_seconds)
    tracker = VisitTracker(None, gap_seconds=gap_seconds, automation=automation)
    visits = []
    for ref in sorted(refs, key=lambda c: (c.camera, c.time, c.capture_id)):
        inference_path = directory / f"{ref.capture_id}.inference.json"
        inference = None
        if inference_path.is_file():
            try:
                inference = json.loads(inference_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                inference = None
        before = tracker._open.get(ref.camera)
        block = tracker.observe(ref, inference)
        if before is not None and before.visit_id != block["visit_id"]:
            visits.append(tracker.summary(before))
        if block["visit_id"] != mapping[ref.capture_id]:
            raise AssertionError("live and batch visit rules disagree")
    visits += [tracker.summary(v) for v in tracker._open.values()]
    for visit in visits:
        visit["source"] = "rebuild"
    return sorted(visits, key=lambda v: (v["camera"], v["start"])), mapping


def write_visit_tables(visits: list[dict[str, Any]], mapping: dict[str, str], out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visits_csv, captures_csv = out_dir / "visits.csv", out_dir / "capture_visits.csv"
    columns = ["visit_id", "camera", "deployment_id", "start", "end", "duration_seconds", "captures",
               "species_frames", "decision", "label", "species", "species_confidence", "group", "group_confidence"]
    with visits_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        for v in visits:
            result = v.get("result") or {}
            writer.writerow({**{k: v.get(k) for k in columns[:8]}, **{k: result.get(k) for k in columns[8:]}})
    with captures_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["capture_id", "visit_id"])
        writer.writerows(sorted(mapping.items()))
    return visits_csv, captures_csv


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild bird visits from a capture directory.")
    sub = parser.add_subparsers(dest="command", required=True)
    rebuild_parser = sub.add_parser("rebuild", help="write visits.csv and capture_visits.csv")
    rebuild_parser.add_argument("directory", type=Path)
    rebuild_parser.add_argument("--gap-seconds", type=float, default=DEFAULT_GAP_SECONDS)
    rebuild_parser.add_argument("--species-contract", type=Path, help="apply its automation thresholds")
    rebuild_parser.add_argument("--out", type=Path, help="output folder (default: the capture directory)")
    args = parser.parse_args(argv)
    automation = None
    if args.species_contract is not None:
        automation = json.loads(args.species_contract.read_text(encoding="utf-8")).get("automation")
    visits, mapping = rebuild(args.directory, args.gap_seconds, automation)
    paths = write_visit_tables(visits, mapping, args.out or args.directory)
    print(f"{len(visits)} visits from {len(mapping)} captures -> {paths[0]} and {paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
