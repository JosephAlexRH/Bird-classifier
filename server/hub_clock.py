"""Hub time for a field deployment with no internet and no battery clock.

A Raspberry Pi has no real-time clock: after a power cut it resumes from
whatever time it last saved, and with no network nothing corrects it. This
module keeps the damage small and, above all, recorded:

* The cameras take their time from the hub (the heartbeat reply carries
  ``server_time_unix_ms``), so camera and hub timestamps always agree.
* Any phone or laptop that opens the dashboard offers its own clock
  (``POST /api/time``). If the hub is not network-synchronised, that becomes
  the reference: the system clock is stepped when the service is allowed to,
  and the offset is written to the run log either way, so timestamps taken
  before the visit can be corrected afterwards.
* The last time seen is saved every minute. If the system clock ever starts
  behind it, the clock is moved forward so time never runs backwards.

Once a clock module (RTC) is fitted none of this is needed, and it stays out
of the way: a phone only ever steps the clock when it disagrees by seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

STEP_THRESHOLD_SECONDS = 2.0
BACKWARDS_TOLERANCE_SECONDS = 60.0
NTP_CHECK_SECONDS = 60.0
EARLIEST_PLAUSIBLE_UNIX = 1_700_000_000  # late 2023; anything older is not a real clock
LATEST_PLAUSIBLE_UNIX = 4_102_444_800  # 2100


def system_boot_id() -> str:
    """Identify this power-up of the machine (not of the server process)."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return f"process-{uuid.uuid4().hex}"


def system_ntp_synced() -> bool | None:
    """True/False from systemd on Linux; None when it cannot be told."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        result = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().lower()
    return True if value == "yes" else False if value == "no" else None


def step_system_clock(unix_seconds: float) -> bool:
    """Set the system clock. Needs the sudoers rule added by deploy/install_pi_hub.sh."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        result = subprocess.run(
            ["sudo", "-n", "date", "-u", "-s", f"@{int(round(unix_seconds))}"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


class HubClock:
    def __init__(
        self,
        state_file: Path,
        *,
        clock: Callable[[], float] = time.time,
        ntp_synced: Callable[[], bool | None] = system_ntp_synced,
        set_system_time: Callable[[float], bool] | None = None,
        boot_id: Callable[[], str] = system_boot_id,
        journal: Callable[..., None] | None = None,
    ) -> None:
        self.state_file = Path(state_file)
        self._clock = clock
        self._ntp_synced = ntp_synced
        self._set_system_time = set_system_time
        self.boot_id = boot_id()
        self.journal = journal
        self._lock = threading.Lock()
        self._ntp_cache: tuple[float, bool | None] | None = None
        self._reference: dict[str, Any] | None = None
        self._restore()

    # ------------------------------------------------------------- storage
    def _write(self, kind: str, **fields: Any) -> None:
        if self.journal is not None:
            try:
                self.journal(kind, **fields)
            except Exception:  # noqa: BLE001 - time keeping must never break the server
                pass

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        document = {"last_seen_unix": self._clock(), "boot_id": self.boot_id, "reference": self._reference}
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(document), encoding="utf-8")
            os.replace(temporary, self.state_file)
        except OSError:
            pass

    def _restore(self) -> None:
        saved = self._load()
        if saved.get("boot_id") == self.boot_id and isinstance(saved.get("reference"), dict):
            self._reference = saved["reference"]  # same power-up: the reference still holds
        last_seen = saved.get("last_seen_unix")
        now = self._clock()
        if isinstance(last_seen, (int, float)) and now < last_seen - BACKWARDS_TOLERANCE_SECONDS:
            stepped = bool(self._set_system_time and self._set_system_time(last_seen))
            self._write(
                "clock_restored", hub_unix_before=now, last_seen_unix=last_seen, stepped=stepped,
                note="system clock started behind the last time this hub saw; moved forward" if stepped
                else "system clock started behind the last time this hub saw; could not move it",
            )
        self._save()

    # --------------------------------------------------------------- state
    def _ntp(self) -> bool | None:
        now = time.monotonic()
        if self._ntp_cache is None or now - self._ntp_cache[0] >= NTP_CHECK_SECONDS:
            self._ntp_cache = (now, self._ntp_synced())
        return self._ntp_cache[1]

    def state(self) -> dict[str, Any]:
        """Where the hub's time comes from and whether to believe it.

        source: "ntp" (network time), "system" (a machine with its own clock,
        such as the laptop), "reference" (set from a phone/laptop this
        power-up), "offset" (a reference was seen but the clock could not be
        set; the offset is in the run log) or "unverified".
        """
        with self._lock:
            reference = dict(self._reference) if self._reference else None
        ntp = self._ntp()
        if ntp is True:
            source = "ntp"
        elif ntp is None:
            source = "system"
        elif reference is None:
            source = "unverified"
        else:
            source = "reference" if reference.get("trusted") else "offset"
        return {
            "source": source,
            "trusted": source in {"ntp", "system", "reference"},
            "boot_id": self.boot_id,
            "reference": reference,
        }

    def reply_fields(self) -> dict[str, Any]:
        """Added to every heartbeat reply so cameras can set their clocks."""
        return {
            "server_time_unix_ms": int(self._clock() * 1000),
            "server_time_trusted": self.state()["trusted"],
        }

    def tick(self) -> None:
        """Call about once a minute: remembers the latest time across power cuts."""
        with self._lock:
            self._save()

    # ----------------------------------------------------------- reference
    def offer_reference(self, client_unix_ms: Any, source: str = "browser") -> dict[str, Any]:
        """A visiting device reports its clock; use it if the hub has nothing better."""
        if isinstance(client_unix_ms, bool) or not isinstance(client_unix_ms, (int, float)):
            raise ValueError("client_unix_ms must be a number of milliseconds since 1970.")
        client = float(client_unix_ms) / 1000.0
        if not EARLIEST_PLAUSIBLE_UNIX <= client <= LATEST_PLAUSIBLE_UNIX:
            raise ValueError("client_unix_ms is not a plausible current time.")
        hub = self._clock()
        offset = client - hub
        state = self.state()
        if state["source"] in {"ntp", "system"}:
            return {"applied": False, "offset_s": round(offset, 3), "state": state,
                    "reason": "the hub already has a real clock"}

        log_entry: dict[str, Any] | None = None
        with self._lock:
            previous = self._reference
            needs_step = abs(offset) > STEP_THRESHOLD_SECONDS
            stepped = False
            if needs_step and self._set_system_time is not None:
                stepped = bool(self._set_system_time(client))
            if needs_step or previous is None:
                # Only worth a line when it tells us something new.
                log_entry = dict(
                    source=source, hub_unix_before=hub, reference_unix=client,
                    offset_s=round(offset, 3), stepped=stepped,
                    note=("hub clock corrected" if stepped else
                          "hub clock agrees with the reference" if not needs_step else
                          "hub clock could not be set; add offset_s to hub times from this power-up"),
                )
            self._reference = {
                "source": source,
                "at_unix": self._clock(),
                "offset_s": 0.0 if stepped or not needs_step else round(offset, 3),
                "trusted": stepped or not needs_step,
            }
            self._save()
        # Journal outside the lock: the run log's context calls state(), which
        # takes this same (non-reentrant) lock. Writing while holding it
        # deadlocked the hub the first time a dashboard offered its clock.
        if log_entry is not None:
            self._write("clock_reference", **log_entry)
        return {"applied": True, "stepped": stepped, "offset_s": round(offset, 3), "state": self.state()}
