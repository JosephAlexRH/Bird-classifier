"""Append-only run log for unattended deployments.

One JSON object per line in ``<save-dir>/runlog/run-YYYY-MM-DD.jsonl``. The
fleet dashboard only keeps recent history; this keeps everything, so a whole
field run can be analysed afterwards (``python server/run_report.py``).

Every line carries:
  type      what happened (heartbeat, camera_boot, camera_event, camera_crash,
            registry, clock_reference, clock_restored, hub_start, hub_status)
  t / unix  hub wall-clock time (UTC ISO and seconds)
  mono      seconds on the hub's monotonic clock; immune to clock changes, so
            intervals inside one hub power-up are always right
  hub_boot  which hub power-up the line belongs to
  clock     where hub time came from then: ntp, system, reference, offset, unverified
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

FSYNC_SECONDS = 10.0
STATUS_SECONDS = 300.0
CLOCK_TICK_SECONDS = 60.0
HEARTBEAT_OMIT = ("events", "last_crash")  # logged once each as camera_event / camera_crash


class RunLog:
    def __init__(
        self,
        directory: Path,
        *,
        clock: Callable[[], float] = time.time,
        context: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self._clock = clock
        self.context = context
        self._lock = threading.Lock()
        self._handle: Any = None
        self._day: str | None = None
        self._last_fsync = 0.0

    def _open(self, day: str) -> None:
        if self._handle is not None:
            self._handle.close()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handle = (self.directory / f"run-{day}.jsonl").open("a", encoding="utf-8")
        self._day = day

    def write(self, kind: str, **fields: Any) -> None:
        """Never raises: a full disk must not take the server down with it."""
        try:
            now = self._clock()
            record: dict[str, Any] = {
                "type": kind,
                "t": datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z"),
                "unix": round(now, 3),
                "mono": round(time.monotonic(), 3),
            }
            if self.context is not None:
                record.update(self.context())
            record.update(fields)
            line = json.dumps(record, separators=(",", ":"), default=str)
            day = record["t"][:10]
            with self._lock:
                if self._handle is None or day != self._day:
                    self._open(day)
                self._handle.write(line + "\n")
                self._handle.flush()
                # Survive a sudden power cut with at most a few seconds lost.
                if time.monotonic() - self._last_fsync >= FSYNC_SECONDS:
                    os.fsync(self._handle.fileno())
                    self._last_fsync = time.monotonic()
        except Exception:  # noqa: BLE001
            pass

    def heartbeat(self, payload: dict[str, Any], remote_addr: str | None) -> None:
        self.write(
            "heartbeat",
            camera_id=payload.get("camera_id"),
            remote_addr=remote_addr,
            heartbeat={key: value for key, value in payload.items() if key not in HEARTBEAT_OMIT},
        )

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.flush()
                    os.fsync(self._handle.fileno())
                    self._handle.close()
                except OSError:
                    pass
                self._handle = None


def _read_number(path: str, scale: float = 1.0) -> float | None:
    try:
        return round(float(Path(path).read_text(encoding="ascii").split()[0]) / scale, 2)
    except (OSError, ValueError, IndexError):
        return None


def pi_throttled() -> dict[str, Any] | None:
    """Raspberry Pi power flags; under-voltage is the one to watch on a battery."""
    try:
        result = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3, check=False)
        value = int(result.stdout.strip().split("=")[1], 16)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return {
        "raw": hex(value),
        "under_voltage_now": bool(value & 0x1),
        "throttled_now": bool(value & 0x4),
        "under_voltage_since_boot": bool(value & 0x10000),
        "throttled_since_boot": bool(value & 0x40000),
    }


def hub_status(save_dir: Path) -> dict[str, Any]:
    status: dict[str, Any] = {}
    try:
        usage = shutil.disk_usage(save_dir)
        status["disk_free_mb"] = usage.free // (1024 * 1024)
        status["disk_total_mb"] = usage.total // (1024 * 1024)
    except OSError:
        pass
    status["cpu_temp_c"] = _read_number("/sys/class/thermal/thermal_zone0/temp", 1000.0)
    status["uptime_s"] = _read_number("/proc/uptime")
    status["load_1m"] = _read_number("/proc/loadavg")
    status["power"] = pi_throttled()
    return {key: value for key, value in status.items() if value is not None}


class HubStatusLogger:
    """Background thread: periodic hub_status lines and the hub clock's minute tick."""

    def __init__(self, run_log: RunLog, save_dir: Path, hub_clock: Any = None,
                 *, status_seconds: float = STATUS_SECONDS) -> None:
        self.run_log, self.save_dir, self.hub_clock = run_log, Path(save_dir), hub_clock
        self.status_seconds = status_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        last_status = None
        while not self._stop.is_set():
            now = time.monotonic()
            if last_status is None or now - last_status >= self.status_seconds:
                last_status = now
                self.run_log.write("hub_status", **hub_status(self.save_dir))
            if self.hub_clock is not None:
                self.hub_clock.tick()
            self._stop.wait(CLOCK_TICK_SECONDS)

    def start(self) -> "HubStatusLogger":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="hub-status", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        self.run_log.close()
