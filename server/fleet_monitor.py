"""Fleet health for ESP32 cameras: heartbeats, crash reports, event logs, remote reboot.

Each camera POSTs /api/heartbeat every 15 s (see src/device_health.cpp). This
module keeps the latest heartbeat per camera, the camera's event log (which
survives crashes on the device), its restart history and decoded crash reports,
and serves a single status page at /fleet.

Nothing here polls the cameras, so a camera that is crashing, rebooting or has
changed IP still shows up as soon as it manages one heartbeat.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, request, send_file

try:
    from .console_dashboard import CONSOLE_DASHBOARD_HTML
except ImportError:  # Direct execution from the server folder
    from console_dashboard import CONSOLE_DASHBOARD_HTML

HEARTBEAT_INTERVAL_SECONDS = 15
ONLINE_WITHIN_SECONDS = 40
LATE_WITHIN_SECONDS = 120
MAX_EVENTS = 400
MAX_CRASHES = 50
MAX_BOOTS = 200
SAVE_EVERY_SECONDS = 10
CAMERA_REQUEST_TIMEOUT_SECONDS = 4

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIRMWARE_ELF = REPO_ROOT / ".pio" / "build" / "esp32cam" / "firmware.elf"
BAD_RESET_REASONS = {"crash", "interrupt_watchdog", "task_watchdog", "other_watchdog", "brownout"}


def _utc_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def find_addr2line(explicit: str | None = None) -> str | None:
    """Locate the Xtensa addr2line that PlatformIO installed, if any."""
    candidates = [explicit, os.environ.get("ESP32_ADDR2LINE")]
    candidates.append(shutil.which("xtensa-esp32-elf-addr2line"))
    toolchain = Path.home() / ".platformio" / "packages" / "toolchain-xtensa-esp32" / "bin"
    candidates += [
        str(toolchain / "xtensa-esp32-elf-addr2line.exe"),
        str(toolchain / "xtensa-esp32-elf-addr2line"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


class CrashDecoder:
    """Turns crash PCs into function names using the firmware ELF, when it matches."""

    def __init__(
        self,
        firmware_elf: Path | None = DEFAULT_FIRMWARE_ELF,
        addr2line: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.firmware_elf = Path(firmware_elf) if firmware_elf else None
        self.addr2line = find_addr2line(addr2line)
        self._runner = runner
        self._sha_cache: tuple[float, str] | None = None

    def _elf_sha16(self) -> str | None:
        if self.firmware_elf is None or not self.firmware_elf.is_file():
            return None
        mtime = self.firmware_elf.stat().st_mtime
        if self._sha_cache is None or self._sha_cache[0] != mtime:
            digest = hashlib.sha256(self.firmware_elf.read_bytes()).hexdigest()
            self._sha_cache = (mtime, digest[:16])
        return self._sha_cache[1]

    def decode(self, crash: dict[str, Any]) -> dict[str, Any]:
        """Return {"status": ..., "frames": [...]} for a last_crash heartbeat block."""
        if self.addr2line is None:
            return {"status": "no_addr2line", "frames": []}
        local_sha = self._elf_sha16()
        if local_sha is None:
            return {"status": "no_firmware_elf", "frames": []}
        crash_sha = str(crash.get("firmware_elf_sha256") or "")[:16]
        if crash_sha != local_sha:
            return {"status": "firmware_mismatch", "frames": [], "local_elf_sha256": local_sha}
        addresses = [crash.get("pc")] + list(crash.get("backtrace") or [])
        addresses = [str(a) for a in addresses if isinstance(a, str) and a.startswith("0x")]
        if not addresses:
            return {"status": "no_addresses", "frames": []}
        try:
            result = self._runner(
                [self.addr2line, "-pfiaC", "-e", str(self.firmware_elf), *addresses],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return {"status": f"addr2line_failed: {error}", "frames": []}
        frames = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return {"status": "decoded", "frames": frames}


class FleetMonitor:
    def __init__(
        self,
        state_file: Path,
        *,
        decoder: CrashDecoder | None = None,
        clock: Callable[[], float] = time.time,
        opener: Callable[..., Any] = urlopen,
        name_lookup: Callable[[str], str | None] | None = None,
        journal: Callable[..., None] | None = None,
    ) -> None:
        self.state_file = Path(state_file)
        self.decoder = decoder or CrashDecoder()
        self._clock = clock
        self._opener = opener
        self._name_lookup = name_lookup
        # Optional permanent record (see run_log.py): journal(kind, **fields).
        self.journal = journal
        self._lock = threading.Lock()
        self._cameras: dict[str, dict[str, Any]] = {}
        self._last_save = 0.0
        self._load()

    # ---------------------------------------------------------------- storage
    def _load(self) -> None:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        cameras = data.get("cameras") if isinstance(data, dict) else None
        if isinstance(cameras, dict):
            for camera_id, camera in cameras.items():
                if isinstance(camera, dict):
                    camera["pending_reboot"] = False
                    self._cameras[str(camera_id)] = camera

    def _save_locked(self, force: bool = False) -> None:
        now = self._clock()
        if not force and now - self._last_save < SAVE_EVERY_SECONDS:
            return
        self._last_save = now
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({"cameras": self._cameras}, indent=1), encoding="utf-8")
            os.replace(temporary, self.state_file)
        except OSError:
            pass  # Health tracking must never break image uploads.

    # -------------------------------------------------------------- heartbeat
    def record_heartbeat(self, payload: dict[str, Any], remote_addr: str | None) -> dict[str, Any]:
        camera_id = str(payload.get("camera_id") or "").strip()
        if not camera_id:
            raise ValueError("camera_id is required")
        now = self._clock()
        boot = payload.get("boot") if isinstance(payload.get("boot"), dict) else {}
        boot_id = str(payload.get("boot_id") or "")
        events = [e for e in payload.get("events") or [] if isinstance(e, dict)]
        crash = payload.get("last_crash") if isinstance(payload.get("last_crash"), dict) else None

        journal_entries: list[tuple[str, dict[str, Any]]] = []
        with self._lock:
            camera = self._cameras.setdefault(
                camera_id,
                {
                    "camera_id": camera_id,
                    "first_seen": now,
                    "events": [],
                    "boots": [],
                    "crashes": [],
                    "last_event_seq": -1,
                    "pending_reboot": False,
                },
            )
            important = False
            camera["last_seen"] = now
            camera["ip"] = payload.get("ip") or remote_addr
            camera["last_heartbeat"] = {k: v for k, v in payload.items() if k != "events"}

            # Restart history: one entry per boot_id.
            known_boots = {b.get("boot_id") for b in camera["boots"]}
            if boot_id and boot_id not in known_boots:
                camera["boots"].append(
                    {
                        "boot_id": boot_id,
                        "first_seen": now,
                        "reset_reason": boot.get("reset_reason"),
                        "number_since_power_on": boot.get("number_since_power_on"),
                    }
                )
                del camera["boots"][:-MAX_BOOTS]
                important = True
                journal_entries.append(("camera_boot", {
                    "camera_id": camera_id, "boot_id": boot_id,
                    "reset_reason": boot.get("reset_reason"),
                    "number_since_power_on": boot.get("number_since_power_on"),
                }))
                if boot.get("reset_reason") == "power_on":
                    camera["last_event_seq"] = -1  # device sequence numbers restart at power-on

            # Event log: sequence numbers only grow until the next power-on.
            sequences = [int(e.get("seq", -1)) for e in events]
            if sequences and max(sequences) < camera.get("last_event_seq", -1):
                camera["last_event_seq"] = -1
            uptime_ms = int(payload.get("uptime_s") or 0) * 1000
            current_boot = boot.get("number_since_power_on")
            for event in events:
                sequence = int(event.get("seq", -1))
                if sequence <= camera.get("last_event_seq", -1):
                    continue
                entry = {
                    "seq": sequence,
                    "boot": event.get("boot"),
                    "t_ms": event.get("t_ms"),
                    "text": str(event.get("text", ""))[:200],
                    "received": now,
                }
                # Events from the running boot can be placed on the wall clock.
                if event.get("boot") == current_boot and isinstance(event.get("t_ms"), int):
                    entry["approx_time"] = now - max(0, uptime_ms - event["t_ms"]) / 1000
                camera["events"].append(entry)
                camera["last_event_seq"] = sequence
                journal_entries.append(("camera_event", {
                    "camera_id": camera_id, "seq": sequence, "boot": entry["boot"],
                    "t_ms": entry["t_ms"], "text": entry["text"],
                    "approx_unix": entry.get("approx_time"),
                }))
            del camera["events"][:-MAX_EVENTS]

            # Crash reports: the device repeats its last crash until a newer one.
            if crash is not None:
                key = json.dumps(
                    [crash.get(k) for k in ("firmware_elf_sha256", "pc", "cause", "boot", "backtrace")]
                    + [boot_id if crash.get("before_power_on") else None]
                )
                previous = camera["crashes"][-1]["key"] if camera["crashes"] else None
                if key != previous:
                    record = dict(crash)
                    record["key"] = key
                    record["received"] = now
                    record["reported_by_boot_id"] = boot_id
                    record["decoded"] = self.decoder.decode(crash)
                    camera["crashes"].append(record)
                    del camera["crashes"][:-MAX_CRASHES]
                    important = True
                    journal_entries.append(("camera_crash", {
                        "camera_id": camera_id,
                        "crash": {k: v for k, v in record.items() if k != "key"},
                    }))

            reboot = bool(camera.get("pending_reboot"))
            if reboot:
                camera["pending_reboot"] = False
                camera["reboot_delivered_at"] = now
                important = True
            self._save_locked(force=important)
        if self.journal is not None:
            for kind, fields in journal_entries:
                try:
                    self.journal(kind, **fields)
                except Exception:  # noqa: BLE001 - logging must never cost a heartbeat
                    pass
        return {"reboot": reboot, "server_time": _utc_iso(now)}

    # ------------------------------------------------------------------ views
    def _status(self, camera: dict[str, Any], now: float) -> str:
        age = now - camera.get("last_seen", 0)
        if age <= ONLINE_WITHIN_SECONDS:
            return "online"
        if age <= LATE_WITHIN_SECONDS:
            return "late"
        return "offline"

    def summary(self) -> list[dict[str, Any]]:
        now = self._clock()
        with self._lock:
            cameras = json.loads(json.dumps(list(self._cameras.values())))
        result = []
        for camera in cameras:
            day_ago = now - 86400
            recent_boots = [b for b in camera.get("boots", []) if b.get("first_seen", 0) >= day_ago]
            name = self._name_lookup(camera["camera_id"]) if self._name_lookup else None
            camera.update(
                name=name,
                status=self._status(camera, now),
                seconds_since_seen=round(now - camera.get("last_seen", now)),
                # Every boot is a restart except the very first one this server saw.
                restarts_24h=len(recent_boots)
                - (1 if camera.get("boots") and camera["boots"][0].get("first_seen", 0) >= day_ago else 0),
                bad_restarts_24h=sum(1 for b in recent_boots if b.get("reset_reason") in BAD_RESET_REASONS),
                crashes_24h=sum(1 for c in camera.get("crashes", []) if c.get("received", 0) >= day_ago),
                events=camera.get("events", [])[-60:],
                crashes=camera.get("crashes", [])[-5:],
                boots=camera.get("boots", [])[-20:],
            )
            result.append(camera)
        result.sort(key=lambda c: (c["status"] != "online", c.get("name") or c["camera_id"]))
        return result

    def report_text(self, camera_id: str) -> str | None:
        """Plain-text diagnostics, easy to copy into a chat or an issue."""
        camera = next((c for c in self.summary() if c["camera_id"] == camera_id), None)
        if camera is None:
            return None
        heartbeat = camera.get("last_heartbeat") or {}
        lines = [
            f"Camera {camera.get('name') or camera_id} [{camera_id}] ({heartbeat.get('camera_model', '?')})",
            f"Status: {camera['status']}, last seen {camera['seconds_since_seen']} s ago, ip {camera.get('ip')}",
            f"Firmware ELF sha256: {heartbeat.get('firmware_elf_sha256')}",
            f"Restarts in 24 h: {camera['restarts_24h']} ({camera['bad_restarts_24h']} crash/watchdog/brownout)",
            "",
            "Last heartbeat:",
            json.dumps(heartbeat, indent=2),
            "",
            "Restart history (newest last):",
        ]
        for boot in camera.get("boots", []):
            lines.append(
                f"  {_utc_iso(boot['first_seen'])}  boot #{boot.get('number_since_power_on')}  {boot.get('reset_reason')}"
            )
        lines += ["", "Crashes (newest last):"]
        for crash in camera.get("crashes", []):
            lines.append(
                f"  {_utc_iso(crash['received'])}  task {crash.get('task')}  {crash.get('cause_name')}"
                f"  pc {crash.get('pc')}  fault addr {crash.get('fault_address')}"
            )
            lines.append(f"    backtrace: {' '.join(crash.get('backtrace') or [])}")
            decoded = crash.get("decoded") or {}
            lines.append(f"    decode: {decoded.get('status')}")
            lines += [f"      {frame}" for frame in decoded.get("frames", [])]
        lines += ["", "Event log (newest last):"]
        for event in camera.get("events", []):
            lines.append(f"  boot #{event.get('boot')} +{(event.get('t_ms') or 0) / 1000:.1f}s  {event.get('text')}")
        return "\n".join(lines) + "\n"

    # ---------------------------------------------------------------- actions
    def _camera_url(self, camera_id: str) -> str | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
            ip = (camera or {}).get("ip")
        return f"http://{ip}" if ip else None

    def request_reboot(self, camera_id: str) -> dict[str, Any]:
        """Reboot directly if the camera answers; otherwise on its next heartbeat."""
        url = self._camera_url(camera_id)
        if url is None:
            raise LookupError("Unknown camera")
        try:
            with self._opener(Request(f"{url}/reboot", data=b"", method="POST"),
                              timeout=CAMERA_REQUEST_TIMEOUT_SECONDS) as response:
                if 200 <= response.status < 300:
                    return {"result": "rebooting", "via": "direct"}
        except Exception:  # noqa: BLE001 - any failure falls back to the heartbeat channel
            pass
        with self._lock:
            self._cameras[camera_id]["pending_reboot"] = True
            self._cameras[camera_id]["reboot_requested_at"] = self._clock()
        return {"result": "queued", "via": "next_heartbeat"}

    def set_mode(self, camera_id: str, mode: str) -> dict[str, Any]:
        url = self._camera_url(camera_id)
        if url is None:
            raise LookupError("Unknown camera")
        body = json.dumps({"mode": mode}).encode("utf-8")
        req = Request(f"{url}/mode", data=body, method="POST", headers={"Content-Type": "application/json"})
        with self._opener(req, timeout=CAMERA_REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    def forget(self, camera_id: str) -> bool:
        with self._lock:
            removed = self._cameras.pop(camera_id, None) is not None
            self._save_locked(force=True)
        return removed

    def flush(self) -> None:
        with self._lock:
            self._save_locked(force=True)


CAPTURE_SCAN_LIMIT = 600  # newest images inspected when listing one camera's captures


def recent_captures(capture_dir: Path, camera_id: str, limit: int) -> list[dict[str, Any]]:
    """Newest saved captures whose sidecar names this hardware camera id."""
    try:
        images = sorted(capture_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    found: list[dict[str, Any]] = []
    for image in images[:CAPTURE_SCAN_LIMIT]:
        try:
            sidecar = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if not isinstance(sidecar, dict) or sidecar.get("camera_id") != camera_id:
            continue
        classification = sidecar.get("classification")
        found.append(
            {
                "filename": image.name,
                "received_at": sidecar.get("received_at"),
                "source": sidecar.get("source"),
                "classification": classification if isinstance(classification, dict) else None,
            }
        )
        if len(found) >= limit:
            break
    return found


def register_fleet_routes(
    app: Flask,
    fleet: FleetMonitor,
    *,
    capture_dir: Path | None = None,
    dashboard_paths: tuple[str, ...] = ("/fleet",),
    on_heartbeat: Callable[[dict[str, Any], str | None], None] | None = None,
    run_log: Any = None,
    hub_clock: Any = None,
) -> None:
    if run_log is not None and fleet.journal is None:
        fleet.journal = run_log.write

    @app.post("/api/heartbeat")
    def fleet_heartbeat() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a JSON object."), 400
        try:
            reply = fleet.record_heartbeat(payload, request.remote_addr)
        except ValueError as error:
            return jsonify(error=str(error)), 422
        if on_heartbeat is not None:
            # Registry upkeep must never cost a camera its heartbeat reply.
            try:
                on_heartbeat(payload, request.remote_addr)
            except Exception as error:  # noqa: BLE001
                print(f"Heartbeat follow-up failed: {error}", flush=True)
        if run_log is not None:
            run_log.heartbeat(payload, request.remote_addr)
        if hub_clock is not None:
            # Cameras without internet time set their clocks from this.
            reply.update(hub_clock.reply_fields())
        return jsonify(reply)

    @app.get("/api/time")
    def hub_time() -> Response:
        state = hub_clock.state() if hub_clock is not None else None
        return jsonify(server_time=_utc_iso(time.time()), server_time_unix_ms=int(time.time() * 1000), clock=state)

    @app.post("/api/time")
    def offer_time() -> tuple[Response, int] | Response:
        """A phone or laptop viewing the dashboard offers its clock as a reference."""
        if hub_clock is None:
            return jsonify(applied=False, reason="hub clock management is not enabled")
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a JSON object."), 400
        try:
            return jsonify(hub_clock.offer_reference(payload.get("client_unix_ms"), source="browser"))
        except ValueError as error:
            return jsonify(error=str(error)), 422

    @app.get("/api/fleet")
    def fleet_summary() -> Response:
        return jsonify(
            cameras=fleet.summary(),
            server_time=_utc_iso(time.time()),
            heartbeat_interval_s=HEARTBEAT_INTERVAL_SECONDS,
            hub_clock=hub_clock.state() if hub_clock is not None else None,
            crash_decoding={
                "addr2line": fleet.decoder.addr2line,
                "firmware_elf": str(fleet.decoder.firmware_elf) if fleet.decoder.firmware_elf else None,
            },
        )

    @app.get("/api/fleet/<camera_id>/report.txt")
    def fleet_report(camera_id: str) -> tuple[Response, int] | Response:
        text = fleet.report_text(camera_id)
        if text is None:
            return Response("Unknown camera\n", status=404, mimetype="text/plain")
        return Response(text, mimetype="text/plain; charset=utf-8")

    @app.post("/api/fleet/<camera_id>/reboot")
    def fleet_reboot(camera_id: str) -> tuple[Response, int] | Response:
        try:
            return jsonify(fleet.request_reboot(camera_id))
        except LookupError as error:
            return jsonify(error=str(error)), 404

    @app.post("/api/fleet/<camera_id>/mode")
    def fleet_mode(camera_id: str) -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode")
        if mode not in {"motion", "idle"}:
            return jsonify(error="mode must be motion or idle"), 422
        try:
            return jsonify(fleet.set_mode(camera_id, mode))
        except LookupError as error:
            return jsonify(error=str(error)), 404
        except Exception as error:  # noqa: BLE001
            return jsonify(error=f"Camera did not respond: {error}"), 502

    @app.delete("/api/fleet/<camera_id>")
    def fleet_forget(camera_id: str) -> tuple[Response, int] | Response:
        if not fleet.forget(camera_id):
            return jsonify(error="Unknown camera"), 404
        return jsonify(result="forgotten")

    @app.get("/api/fleet/<camera_id>/captures")
    def fleet_captures(camera_id: str) -> Response:
        limit = max(1, min(request.args.get("limit", default=6, type=int) or 6, 48))
        captures = recent_captures(Path(capture_dir), camera_id, limit) if capture_dir else []
        return jsonify(captures=captures)

    @app.get("/api/fleet/captures/<filename>")
    def fleet_capture_image(filename: str) -> tuple[Response, int] | Response:
        if capture_dir is None or "/" in filename or "\\" in filename or not filename.endswith(".jpg"):
            return jsonify(error="Not found."), 404
        path = Path(capture_dir) / filename
        if path.parent.resolve() != Path(capture_dir).resolve() or not path.is_file():
            return jsonify(error="Not found."), 404
        return send_file(path, mimetype="image/jpeg", max_age=3600)

    def fleet_dashboard() -> Response:
        unified = app.config.get("UNIFIED_DASHBOARD")
        if callable(unified):
            page = unified()
            if page is not None:
                return Response(page, mimetype="text/html")
        return Response(CONSOLE_DASHBOARD_HTML, mimetype="text/html")

    for index, path in enumerate(dashboard_paths):
        app.add_url_rule(path, f"fleet_dashboard_{index}", fleet_dashboard, methods=["GET"])
