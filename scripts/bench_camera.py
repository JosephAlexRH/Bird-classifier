"""Benchmark and soak-test camera boards over their HTTP API.

Stdlib only, so it runs unchanged on the laptop or the Pi hub. Point it at one
or more cameras to compare boards (e.g. an AI Thinker ESP32-CAM and the
Freenove ESP32-S3) under identical conditions.

  # Capture latency / throughput / JPEG size at several frame sizes
  python scripts/bench_camera.py --camera http://192.168.1.50 --camera http://192.168.1.51 \
      --sizes VGA,SVGA,XGA,UXGA --frames 20 --stream-seconds 10

  # Field-readiness soak: one capture every 10 s for 2 hours
  python scripts/bench_camera.py --camera http://192.168.1.51 --soak-minutes 120 --soak-interval 10

Frame-size runs change the active sensor profile through POST /config, which
the firmware persists to the SD card. The original profile is always restored
at the end (also on Ctrl-C). Stop Pi motion polling for the camera first, or
the hub's requests will be mixed into the timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TIMEOUT_S = 15


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_json(base: str, path: str) -> dict[str, Any]:
    with urlopen(base + path, timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def post_json(base: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    request = Request(base + path, data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Width/height from the first SOF marker, or None if the JPEG is malformed."""
    if len(data) < 4 or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            return None
        marker = data[index + 1]
        length = int.from_bytes(data[index + 2:index + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            height = int.from_bytes(data[index + 5:index + 7], "big")
            width = int.from_bytes(data[index + 7:index + 9], "big")
            return width, height
        index += 2 + length
    return None


def capture(base: str) -> dict[str, Any]:
    """One GET /capture: timings, size, validity and the device metadata header."""
    started = time.perf_counter()
    try:
        with urlopen(base + "/capture", timeout=TIMEOUT_S) as response:
            first_byte = time.perf_counter()
            header = response.headers.get("X-ESP32-Camera-Metadata")
            data = response.read()
    except (HTTPError, URLError, OSError) as error:
        return {"ok": False, "error": str(error), "total_ms": (time.perf_counter() - started) * 1000}
    done = time.perf_counter()
    metadata: dict[str, Any] = {}
    if header:
        try:
            metadata = json.loads(header)
        except ValueError:
            pass
    dims = jpeg_size(data)
    device = metadata.get("device", {})
    resources = device.get("resources", {})
    return {
        "ok": dims is not None,
        "error": None if dims else "malformed JPEG",
        "ttfb_ms": (first_byte - started) * 1000,
        "total_ms": (done - started) * 1000,
        "bytes": len(data),
        "width": dims[0] if dims else None,
        "height": dims[1] if dims else None,
        "boot_id": device.get("boot_id"),
        "temp_c": resources.get("chip_temperature_c"),
        "free_heap": resources.get("heap", {}).get("free_bytes"),
        "min_free_heap": resources.get("heap", {}).get("minimum_free_bytes_since_boot"),
        "free_psram": resources.get("psram", {}).get("free_bytes"),
    }


def stream_fps(base: str, seconds: float) -> dict[str, Any]:
    """Count JPEG frames in the MJPEG stream for ``seconds``."""
    frames, received = 0, 0
    tail = b""
    started = time.perf_counter()
    try:
        with urlopen(base + "/stream", timeout=TIMEOUT_S) as response:
            while time.perf_counter() - started < seconds:
                chunk = response.read1(65536) if hasattr(response, "read1") else response.read(4096)
                if not chunk:
                    break
                received += len(chunk)
                buffer = tail + chunk
                frames += buffer.count(b"\xff\xd9")
                tail = buffer[-1:]
    except (HTTPError, URLError, OSError) as error:
        return {"ok": False, "error": str(error)}
    elapsed = time.perf_counter() - started
    return {"ok": True, "fps": frames / elapsed if elapsed else 0.0,
            "mbit_s": received * 8 / elapsed / 1e6 if elapsed else 0.0, "frames": frames}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    good = [row for row in rows if row["ok"]]
    if not good:
        return {"ok": 0, "failed": len(rows)}
    totals = sorted(row["total_ms"] for row in good)
    p95 = totals[min(len(totals) - 1, int(round(0.95 * (len(totals) - 1))))]
    return {
        "ok": len(good),
        "failed": len(rows) - len(good),
        "resolution": f"{good[-1]['width']}x{good[-1]['height']}",
        "median_ms": statistics.median(totals),
        "p95_ms": p95,
        "median_ttfb_ms": statistics.median(row["ttfb_ms"] for row in good),
        "captures_per_s": 1000 / statistics.mean(totals),
        "median_kb": statistics.median(row["bytes"] for row in good) / 1024,
        "temp_c": good[-1]["temp_c"],
        "min_free_heap": min((r["min_free_heap"] for r in good if r["min_free_heap"] is not None), default=None),
        "min_free_psram": min((r["free_psram"] for r in good if r["free_psram"] is not None), default=None),
    }


def identify(base: str) -> dict[str, Any]:
    status = get_json(base, "/status")
    config = get_json(base, "/config")
    key = config.get("active_sensor_key") or status.get("camera_sensor_config_key")
    return {"status": status, "config": config, "sensor_key": key}


def run_sizes(base: str, info: dict[str, Any], sizes: list[str], frames: int,
              quality: int | None, stream_seconds: float, warmup: int) -> list[dict[str, Any]]:
    key = info["sensor_key"]
    original = dict(info["config"].get("sensors", {}).get(key, {}))
    results = []
    try:
        for size in sizes or [original.get("frame_size")]:
            if sizes:
                profile = dict(original, frame_size=size)
                if quality is not None:
                    profile["jpeg_quality"] = quality
                post_json(base, "/config", {"sensors": {key: profile}})
                time.sleep(1.5)  # driver reinit + auto exposure settle
            for _ in range(warmup):
                capture(base)
            rows = [capture(base) for _ in range(frames)]
            summary = {"camera": base, "sensor": key, "frame_size": size, **summarize(rows)}
            if stream_seconds > 0:
                stream = stream_fps(base, stream_seconds)
                summary["stream_fps"] = round(stream.get("fps", 0.0), 2) if stream["ok"] else None
                summary["stream_mbit_s"] = round(stream.get("mbit_s", 0.0), 2) if stream["ok"] else None
            results.append(summary)
            print_row(summary)
    finally:
        if sizes and original:
            try:
                post_json(base, "/config", {"sensors": {key: original}})
                print(f"  restored {key} profile on {base}: {original}")
            except (HTTPError, URLError, OSError) as error:
                print(f"  WARNING: could not restore {key} profile on {base}: {error}\n"
                      f"  restore it by hand: {json.dumps({'sensors': {key: original}})}", file=sys.stderr)
    return results


def print_row(summary: dict[str, Any]) -> None:
    def fmt(value: Any, spec: str) -> str:
        return format(value, spec) if isinstance(value, (int, float)) else "-"
    print(f"  {summary['frame_size']!s:>6} {summary.get('resolution', '-'):>10} "
          f"ok {summary['ok']:>3} fail {summary['failed']:>2} | "
          f"median {fmt(summary.get('median_ms'), '7.0f')} ms  p95 {fmt(summary.get('p95_ms'), '7.0f')} ms  "
          f"{fmt(summary.get('captures_per_s'), '5.2f')} cap/s  {fmt(summary.get('median_kb'), '6.1f')} KB | "
          f"stream {fmt(summary.get('stream_fps'), '5.1f')} fps | "
          f"temp {fmt(summary.get('temp_c'), '4.0f')} C  min heap {fmt(summary.get('min_free_heap'), ',')}")


def soak(bases: list[str], minutes: float, interval: float, log_path: Path) -> int:
    """Capture from every camera on a fixed cadence; flag failures, reboots and drift."""
    deadline = time.time() + minutes * 60
    boot_ids: dict[str, str | None] = {}
    problems = 0
    with log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utc", "camera", "ok", "error", "total_ms", "bytes", "temp_c",
                         "free_heap", "min_free_heap", "free_psram", "boot_id", "rebooted"])
        while time.time() < deadline:
            tick = time.time()
            for base in bases:
                row = capture(base)
                rebooted = (row.get("boot_id") is not None and base in boot_ids
                            and boot_ids[base] not in (None, row["boot_id"]))
                if row.get("boot_id"):
                    boot_ids[base] = row["boot_id"]
                if not row["ok"] or rebooted:
                    problems += 1
                    print(f"{_now()} {base}: {'REBOOTED' if rebooted else row['error']}")
                writer.writerow([_now(), base, row["ok"], row.get("error"), round(row["total_ms"]),
                                 row.get("bytes"), row.get("temp_c"), row.get("free_heap"),
                                 row.get("min_free_heap"), row.get("free_psram"), row.get("boot_id"),
                                 rebooted])
            handle.flush()
            time.sleep(max(0.0, interval - (time.time() - tick)))
    print(f"soak finished: {problems} problem(s); log at {log_path}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", action="append", required=True, help="camera base URL (repeatable)")
    parser.add_argument("--sizes", default="", help="comma-separated frame sizes, e.g. VGA,SVGA,XGA,UXGA")
    parser.add_argument("--quality", type=int, default=None, help="JPEG quality for the size runs (0-63)")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--stream-seconds", type=float, default=0.0)
    parser.add_argument("--soak-minutes", type=float, default=0.0)
    parser.add_argument("--soak-interval", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("bench_results"))
    args = parser.parse_args(argv)

    bases = [url.rstrip("/") for url in args.camera]
    sizes = [size.strip().upper() for size in args.sizes.split(",") if size.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    if args.soak_minutes > 0:
        return 1 if soak(bases, args.soak_minutes, args.soak_interval, args.out / f"soak_{stamp}.csv") else 0

    report: dict[str, Any] = {"run_at": _now(), "cameras": {}}
    for base in bases:
        info = identify(base)
        status = info["status"]
        print(f"\n{base}  sensor {info['sensor_key']}  "
              f"model {status.get('camera_sensor_model') or status.get('camera_model') or '?'}")
        results = run_sizes(base, info, sizes, args.frames, args.quality, args.stream_seconds, args.warmup)
        report["cameras"][base] = {"status": status, "results": results}

    path = args.out / f"bench_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
