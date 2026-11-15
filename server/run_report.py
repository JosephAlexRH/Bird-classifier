"""Summarise a field run from the run log.

    python server/run_report.py ~/bird_images/runlog
    python server/run_report.py ~/bird_images/runlog --since 2026-11-20

Reads every run-*.jsonl in the folder and prints, per camera: how much of the
run it was reachable, the gaps, restarts by cause, crashes, Wi-Fi quality and
memory lows; and for the hub: restarts, clock corrections, power warnings and
disk space. Intervals use the hub's monotonic clock within each hub power-up,
so they are right even where the wall clock was wrong.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

HEARTBEAT_SECONDS = 15.0
GAP_SECONDS = 60.0  # four missed heartbeats


def read_lines(directory: Path, since: str | None = None) -> Iterable[dict[str, Any]]:
    for path in sorted(Path(directory).glob("run-*.jsonl")):
        if since and path.stem[4:] < since:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # a line cut short by a power loss
                if isinstance(record, dict):
                    yield record


def duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 120:
        return f"{seconds} s"
    if seconds < 7200:
        return f"{seconds // 60} min"
    return f"{seconds / 3600:.1f} h"


def summarise(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    cameras: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "heartbeats": 0, "first": None, "last": None, "online_s": 0.0, "gaps": [],
        "boots": Counter(), "crashes": 0, "rssi": [], "min_free_heap": None,
        "time_sources": Counter(), "last_point": None, "hub_found_by_name": Counter(),
    })
    hub: dict[str, Any] = {
        "starts": 0, "boots": [], "clock": [], "first": None, "last": None,
        "under_voltage": 0, "min_disk_free_mb": None, "max_cpu_temp_c": None,
        "status_lines": 0, "clock_sources": Counter(), "observed_s": 0.0, "last_point": None,
    }
    for record in records:
        kind, boot, mono = record.get("type"), record.get("hub_boot"), record.get("mono")
        hub["first"] = hub["first"] or record.get("t")
        hub["last"] = record.get("t") or hub["last"]
        if boot and boot not in hub["boots"]:
            hub["boots"].append(boot)
        if isinstance(mono, (int, float)) and boot:
            previous = hub["last_point"]
            if previous and previous[0] == boot and mono >= previous[1]:
                hub["observed_s"] += mono - previous[1]
            hub["last_point"] = (boot, mono)
        if kind == "hub_start":
            hub["starts"] += 1
        elif kind in {"clock_reference", "clock_restored"}:
            hub["clock"].append(record)
        elif kind == "hub_status":
            hub["status_lines"] += 1
            power = record.get("power") or {}
            hub["under_voltage"] += bool(power.get("under_voltage_now") or power.get("under_voltage_since_boot"))
            free, temp = record.get("disk_free_mb"), record.get("cpu_temp_c")
            if free is not None:
                hub["min_disk_free_mb"] = free if hub["min_disk_free_mb"] is None else min(free, hub["min_disk_free_mb"])
            if temp is not None:
                hub["max_cpu_temp_c"] = temp if hub["max_cpu_temp_c"] is None else max(temp, hub["max_cpu_temp_c"])
        camera_id = record.get("camera_id")
        if not camera_id:
            continue
        camera = cameras[camera_id]
        if kind == "heartbeat":
            beat = record.get("heartbeat") or {}
            camera["heartbeats"] += 1
            camera["first"] = camera["first"] or record.get("t")
            camera["last"] = record.get("t")
            hub["clock_sources"][record.get("clock") or "?"] += 1
            previous = camera["last_point"]
            if previous and previous[0] == boot and isinstance(mono, (int, float)) and mono >= previous[1]:
                gap = mono - previous[1]
                if gap > GAP_SECONDS:
                    camera["gaps"].append((gap, previous[2], record.get("t")))
                else:
                    camera["online_s"] += gap
            if isinstance(mono, (int, float)):
                camera["last_point"] = (boot, mono, record.get("t"))
            wifi, memory = beat.get("wifi") or {}, beat.get("memory") or {}
            if isinstance(wifi.get("rssi_dbm"), (int, float)):
                camera["rssi"].append(wifi["rssi_dbm"])
            heap = memory.get("min_free_heap", memory.get("free_heap"))
            if isinstance(heap, (int, float)):
                camera["min_free_heap"] = heap if camera["min_free_heap"] is None else min(heap, camera["min_free_heap"])
            camera["time_sources"][beat.get("time_source") or "?"] += 1
            found = (beat.get("hub") or {}).get("found_by_name")
            if found is not None:
                camera["hub_found_by_name"][bool(found)] += 1
        elif kind == "camera_boot":
            camera["boots"][record.get("reset_reason") or "unknown"] += 1
        elif kind == "camera_crash":
            camera["crashes"] += 1
    return {"cameras": dict(cameras), "hub": hub}


def render(summary: dict[str, Any]) -> str:
    hub, lines = summary["hub"], []
    lines.append(f"RUN  {hub['first'] or '-'}  ->  {hub['last'] or '-'}")
    lines.append(
        f"HUB  observed {duration(hub['observed_s'])} | power-ups {len(hub['boots'])} | "
        f"server starts {hub['starts']} | clock corrections {len(hub['clock'])}"
    )
    extras = []
    if hub["status_lines"]:
        extras.append(f"under-voltage in {hub['under_voltage']}/{hub['status_lines']} status checks")
    if hub["min_disk_free_mb"] is not None:
        extras.append(f"lowest free disk {hub['min_disk_free_mb']} MB")
    if hub["max_cpu_temp_c"] is not None:
        extras.append(f"hottest {hub['max_cpu_temp_c']} C")
    if extras:
        lines.append("     " + " | ".join(extras))
    if hub["clock_sources"]:
        total = sum(hub["clock_sources"].values())
        lines.append("     hub time source: " + ", ".join(
            f"{name} {100 * count / total:.0f}%" for name, count in hub["clock_sources"].most_common()))
    for record in hub["clock"]:
        lines.append(f"     {record.get('t')}  {record.get('type')}: offset {record.get('offset_s', '-')} s, "
                     f"{record.get('note', '')}")
    lines.append("")
    for camera_id, camera in sorted(summary["cameras"].items()):
        observed = hub["observed_s"] or 1.0
        lines.append(f"CAMERA {camera_id}")
        lines.append(f"  seen {camera['first'] or '-'} -> {camera['last'] or '-'} | {camera['heartbeats']} heartbeats")
        lines.append(f"  reachable {duration(camera['online_s'])} = {100 * camera['online_s'] / observed:.1f}% of hub-observed time")
        gaps = sorted(camera["gaps"], reverse=True)
        if gaps:
            lines.append(f"  gaps over {int(GAP_SECONDS)} s: {len(gaps)}, longest {duration(gaps[0][0])}")
            for gap, start, end in gaps[:5]:
                lines.append(f"    {duration(gap):>8}  {start} -> {end}")
        boots = camera["boots"]
        lines.append(f"  boots {sum(boots.values())}: " + (", ".join(f"{k} x{v}" for k, v in boots.most_common()) or "none logged")
                     + f" | crashes {camera['crashes']}")
        if camera["rssi"]:
            lines.append(f"  wifi rssi median {statistics.median(camera['rssi']):.0f} dBm, worst {min(camera['rssi'])} dBm")
        if camera["min_free_heap"] is not None:
            lines.append(f"  lowest free heap {int(camera['min_free_heap'])} bytes")
        sources = camera["time_sources"]
        if sources and set(sources) != {"?"}:
            lines.append("  camera time source: " + ", ".join(f"{k} x{v}" for k, v in sources.most_common()))
        found = camera["hub_found_by_name"]
        if found:
            lines.append(f"  hub found by name in {found[True]}/{found[True] + found[False]} heartbeats")
        lines.append("")
    if not summary["cameras"]:
        lines.append("No camera heartbeats in this log.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runlog", type=Path, help="Folder containing run-YYYY-MM-DD.jsonl files")
    parser.add_argument("--since", help="Only read days from this date on (YYYY-MM-DD)")
    args = parser.parse_args()
    print(render(summarise(read_lines(args.runlog, args.since))))


if __name__ == "__main__":
    main()
