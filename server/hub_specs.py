"""Hardware and software specs for the hub and its cameras, as label/value rows.

The page renders these rows as they come, so formatting lives here. Values that
are unknown are left out rather than shown as blanks. Camera rows combine the
camera's latest heartbeat (board, firmware, Wi-Fi, memory) with the sidecar of
its latest saved capture (sensor, driver, frame settings).
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import socket
import sys
import time
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

Row = list[str]
Section = dict[str, Any]


def _mb(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    if value < 1048576:
        return f"{value / 1024:.0f} KB"
    return f"{value / 1048576:.1f} MB" if value < 1073741824 else f"{value / 1073741824:.1f} GB"


def _duration(seconds: Any) -> str | None:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return None
    seconds = int(seconds)
    days, hours, minutes = seconds // 86400, seconds % 86400 // 3600, seconds % 3600 // 60
    return f"{days} d {hours} h" if days else f"{hours} h {minutes} min" if hours else f"{minutes} min"


def _section(name: str, rows: Iterable[tuple[str, Any]]) -> Section | None:
    kept = [[label, str(value)] for label, value in rows if value not in (None, "", [])]
    return {"name": name, "rows": kept} if kept else None


def _get(data: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip().strip("\x00") or None
    except OSError:
        return None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _static_hub() -> dict[str, Any]:
    """Facts that cannot change while the hub runs; read once."""
    os_name = None
    release = _read("/etc/os-release")
    if release:
        for line in release.splitlines():
            if line.startswith("PRETTY_NAME="):
                os_name = line.split("=", 1)[1].strip('"')
    memory_total = None
    meminfo = _read("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                memory_total = int(line.split()[1]) * 1024
    server_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(server_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    tflite = next(((name, v) for name in ("tflite-runtime", "ai-edge-litert", "tensorflow")
                   if (v := _version(name))), None)
    return {
        "board": _read("/proc/device-tree/model"),
        "cpu": f"{os.cpu_count() or '?'} × {platform.machine() or 'cpu'}",
        "memory_total": memory_total,
        "os": os_name or platform.platform(terse=True),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "code": digest.hexdigest()[:12],
        "packages": ", ".join(f"{name} {version}" for name, version in (
            ("flask", _version("flask")), ("numpy", _version("numpy")), tflite or (None, None)) if version),
    }


def _primary_ip() -> str | None:
    """The address other devices use to reach the hub (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            return probe.getsockname()[0]
    except OSError:
        return None


def _mount_for(path: Path) -> str | None:
    """Which device and filesystem hold the captures (USB stick or SD card)."""
    mounts = _read("/proc/mounts")
    if not mounts:
        return None
    target = str(path.resolve())
    best = None
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) >= 3 and (target == parts[1] or target.startswith(parts[1].rstrip("/") + "/")):
            if best is None or len(parts[1]) > len(best[1]):
                best = parts
    return f"{best[0]} ({best[2]}) on {best[1]}" if best else None


def _memory_available() -> int | None:
    meminfo = _read("/proc/meminfo")
    for line in (meminfo or "").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _model_rows(models: dict[str, Any]) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, label in (("detector", "Bird detector"), ("is_bird", "Bird classifier"),
                       ("localizer", "Box model"), ("species", "Species model")):
        model = models.get(key)
        if not isinstance(model, dict):
            continue
        size = model.get("image_size")
        size = f"{size[0]}×{size[1]}" if isinstance(size, (list, tuple)) and len(size) == 2 else size
        rows.append((label, " · ".join(str(part) for part in (
            model.get("filename"), size, model.get("input_dtype"), (model.get("sha256") or "")[:12]) if part)))
    rows.append(("Species labels", models.get("species_count") if models.get("species") else None))
    rows.append(("Runtime", next((m.get("runtime") for m in models.values()
                                  if isinstance(m, dict) and m.get("runtime")), None)))
    rows.append(("JPEG decoder", models.get("jpeg_decoder")))
    return rows


def hub_specs(save_dir: Path, *, models: dict[str, Any] | None, clock: dict[str, Any] | None,
              started_at: float) -> dict[str, Any]:
    static = _static_hub()
    disk = shutil.disk_usage(save_dir)
    uptime = _read("/proc/uptime")
    temperature = _read("/sys/class/thermal/thermal_zone0/temp")
    try:
        load = ", ".join(f"{value:.2f}" for value in os.getloadavg())
    except (AttributeError, OSError):
        load = None
    sections = [
        _section("Board", [
            ("Model", static["board"]),
            ("CPU", static["cpu"]),
            ("Memory", f"{_mb(_memory_available())} free of {_mb(static['memory_total'])}"
             if static["memory_total"] else None),
            ("Temperature", f"{int(temperature) / 1000:.1f} °C" if temperature and temperature.isdigit() else None),
            ("Load", load),
        ]),
        _section("System", [
            ("Hostname", static["hostname"]),
            ("IP address", _primary_ip()),
            ("OS", static["os"]),
            ("Kernel", static["kernel"]),
            ("Up", _duration(float(uptime.split()[0])) if uptime else None),
            ("Hub service up", _duration(time.time() - started_at)),
            ("Clock", (clock or {}).get("source")),
        ]),
        _section("Storage", [
            ("Captures", str(save_dir)),
            ("Device", _mount_for(save_dir)),
            ("Free", f"{_mb(disk.free)} of {_mb(disk.total)}"),
        ]),
        _section("Models", _model_rows(models or {})),
        _section("Software", [
            ("Python", static["python"]),
            ("Packages", static["packages"]),
            ("Hub code", static["code"]),
        ]),
    ]
    return {"name": "Hub", "sections": [s for s in sections if s]}


def camera_specs(camera: dict[str, Any], heartbeat: dict[str, Any] | None,
                 sidecar: dict[str, Any] | None) -> dict[str, Any]:
    hb = heartbeat or {}
    hw = hb.get("hardware") if isinstance(hb.get("hardware"), dict) else {}
    sc = sidecar or {}
    sensor = _get(sc, "camera", "sensor") or {}
    capture = _get(sc, "camera", "observed", "capture") or _get(sc, "camera", "requested", "capture") or {}
    driver = _get(sc, "camera", "driver") or {}
    psram_total = hw.get("psram_bytes") or _get(sc, "device", "resources", "psram", "total_bytes")
    chip = None
    if hw.get("chip_model"):
        chip = f"{hw['chip_model']} rev {hw.get('chip_revision', '?')}"
        if hw.get("cpu_cores"):
            chip += f" · {hw['cpu_cores']} cores @ {hw.get('cpu_mhz', '?')} MHz"
    boot = hb.get("boot") if isinstance(hb.get("boot"), dict) else {}
    memory = hb.get("memory") if isinstance(hb.get("memory"), dict) else {}
    wifi = hb.get("wifi") if isinstance(hb.get("wifi"), dict) else {}
    hub = hb.get("hub") if isinstance(hb.get("hub"), dict) else {}
    frame = None
    if capture:
        size = _get(capture, "frame_size", "name")
        dims = f"{capture['width']}×{capture['height']}" if capture.get("width") else None
        frame = " ".join(part for part in (size, f"({dims})" if dims else None) if part)
    sections = [
        _section("Board", [
            ("Board", hw.get("board")),
            ("Chip", chip),
            ("Flash", _mb(hw.get("flash_bytes"))),
            ("PSRAM", _mb(psram_total)),
            ("MAC", hw.get("mac")),
            ("Temperature", f"{hb['chip_temperature_c']:.0f} °C (die)"
             if isinstance(hb.get("chip_temperature_c"), (int, float)) else None),
        ]),
        _section("Camera", [
            ("Sensor", " ".join(p for p in (sensor.get("manufacturer"), sensor.get("model") or hb.get("camera_model"))
                                if p) or None),
            ("Sensor ID", sensor.get("pid_hex")),
            ("Clock", f"{sensor['xclk_hz'] / 1e6:g} MHz" if isinstance(sensor.get("xclk_hz"), (int, float)) else None),
            ("Frame size", frame),
            ("JPEG quality", capture.get("jpeg_quality")),
            ("Frame buffers", _get(driver, "framebuffer", "fb_count")),
            ("Mode", _get(hb, "camera", "mode")),
        ]),
        _section("Firmware", [
            ("Build", (hb.get("firmware_elf_sha256") or "")[:12] or None),
            ("Built", hw.get("firmware_built")),
            ("Arduino core", hw.get("arduino_core") or driver.get("arduino_esp32_version")),
            ("ESP-IDF", hw.get("sdk")),
            ("esp32-camera", _get(driver, "esp32_camera", "bundled_revision")),
            ("Metadata schema", sc.get("firmware_metadata_schema_version")),
        ]),
        _section("Running", [
            ("Up", _duration(hb.get("uptime_s"))),
            ("Last reset", (boot.get("reset_reason") or "").replace("_", " ") or None),
            ("Boots", boot.get("number_since_power_on")),
            ("Faults", "{} crash · {} watchdog · {} brownout".format(
                *(boot.get(k, 0) for k in ("crashes_since_power_on", "watchdog_resets_since_power_on",
                                           "brownouts_since_power_on"))) if boot else None),
            ("Free heap", f"{_mb(memory.get('free_heap'))} (lowest {_mb(memory.get('min_free_heap'))})"
             if memory.get("free_heap") else None),
            ("Free PSRAM", _mb(memory.get("free_psram"))),
        ]),
        _section("Network", [
            ("IP address", hb.get("ip")),
            ("Wi-Fi signal", f"{wifi['rssi_dbm']} dBm" if isinstance(wifi.get("rssi_dbm"), (int, float)) else None),
            ("Wi-Fi drops", wifi.get("disconnects_this_boot")),
            ("Hub", hub.get("host") or hub.get("address")),
            ("Clock", hb.get("time_source")),
        ]),
        _section("Hub record", [
            ("Hardware ID", camera.get("hardware_camera_id") or hb.get("camera_id")),
            ("URL", camera.get("url")),
            ("Last capture", str(sc["received_at"])[:16].replace("T", " ") + " UTC" if sc.get("received_at") else None),
            ("Deployment", sc.get("deployment_id")),
        ]),
    ]
    result: dict[str, Any] = {"id": camera.get("id"), "name": camera.get("name") or hb.get("camera_id"),
                              "sections": [s for s in sections if s]}
    if not hw:
        result["note"] = "Board, chip, flash and MAC appear after the camera firmware update."
    if not heartbeat:
        result["note"] = "No heartbeat received yet."
    return result
