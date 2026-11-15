import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from server.fleet_monitor import CrashDecoder, FleetMonitor


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


def unreachable_opener(*args, **kwargs):
    raise OSError("camera unreachable")


def heartbeat(boot_id="boot-a", reason="power_on", number=1, events=(), crash=None, uptime=60):
    return {
        "camera_id": "esp32cam-test",
        "boot_id": boot_id,
        "firmware_elf_sha256": "0123456789abcdef",
        "ip": "192.168.1.50",
        "uptime_s": uptime,
        "boot": {"number_since_power_on": number, "reset_reason": reason},
        "camera": {"mode": "motion"},
        "last_crash": crash,
        "events": list(events),
    }


def event(seq, boot, text, t_ms=1000):
    return {"seq": seq, "boot": boot, "t_ms": t_ms, "text": text}


CRASH = {
    "boot": 1,
    "before_power_on": False,
    "task": "ppT",
    "pc": "0x401a2967",
    "cause": 0,
    "cause_name": "IllegalInstruction",
    "firmware_elf_sha256": "0123456789abcdef",
    "backtrace": ["0x401a2967"],
}


class FleetMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.clock = FakeClock()
        self.fleet = FleetMonitor(
            self.directory / "fleet.json",
            decoder=CrashDecoder(firmware_elf=None),
            clock=self.clock,
            opener=unreachable_opener,
        )

    def test_events_are_deduplicated_across_heartbeats(self) -> None:
        first = [event(0, 1, "boot"), event(1, 1, "wifi: connected")]
        self.fleet.record_heartbeat(heartbeat(events=first), "10.0.0.2")
        self.fleet.record_heartbeat(heartbeat(events=first + [event(2, 1, "motion: armed")]), "10.0.0.2")
        texts = [e["text"] for e in self.fleet.summary()[0]["events"]]
        self.assertEqual(texts, ["boot", "wifi: connected", "motion: armed"])

    def test_power_on_restarts_event_sequence(self) -> None:
        self.fleet.record_heartbeat(heartbeat(events=[event(0, 1, "a"), event(1, 1, "b")]), None)
        self.fleet.record_heartbeat(heartbeat(boot_id="boot-b", events=[event(0, 1, "after power cycle")]), None)
        self.assertEqual(self.fleet.summary()[0]["events"][-1]["text"], "after power cycle")

    def test_crash_recorded_once_and_counted_as_bad_restart(self) -> None:
        self.fleet.record_heartbeat(heartbeat(), None)
        self.clock.now += 30
        for _ in range(3):
            self.fleet.record_heartbeat(heartbeat("boot-b", "crash", 2, crash=CRASH), None)
        camera = self.fleet.summary()[0]
        self.assertEqual(len(camera["crashes"]), 1)
        self.assertIn(camera["crashes"][0]["decoded"]["status"], {"no_firmware_elf", "no_addr2line"})
        self.assertEqual(camera["restarts_24h"], 1)
        self.assertEqual(camera["bad_restarts_24h"], 1)

    def test_reboot_is_queued_when_camera_does_not_answer(self) -> None:
        self.fleet.record_heartbeat(heartbeat(), None)
        self.assertEqual(self.fleet.request_reboot("esp32cam-test")["result"], "queued")
        self.assertTrue(self.fleet.record_heartbeat(heartbeat(), None)["reboot"])
        self.assertFalse(self.fleet.record_heartbeat(heartbeat(), None)["reboot"])

    def test_status_goes_offline_and_state_survives_restart(self) -> None:
        self.fleet.record_heartbeat(heartbeat(events=[event(0, 1, "boot")]), None)
        self.clock.now += 600
        self.assertEqual(self.fleet.summary()[0]["status"], "offline")
        self.fleet.flush()
        reloaded = FleetMonitor(self.directory / "fleet.json", decoder=CrashDecoder(None), clock=self.clock)
        self.assertEqual(reloaded.summary()[0]["events"][0]["text"], "boot")

    def test_missing_camera_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.fleet.record_heartbeat({"events": []}, None)

    def test_decoder_refuses_mismatched_firmware(self) -> None:
        elf = self.directory / "firmware.elf"
        elf.write_bytes(b"not the crashing firmware")
        addr2line = self.directory / "addr2line"
        addr2line.write_text("")
        decoder = CrashDecoder(elf, str(addr2line), runner=lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
        self.assertEqual(decoder.decode(CRASH)["status"], "firmware_mismatch")


if __name__ == "__main__":
    unittest.main()


class FleetDashboardRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from server.laptop_test_server import create_app

        self.directory = Path(tempfile.mkdtemp())
        self.app = create_app(self.directory / "captures", None, registry_file=self.directory / "cams.json")
        self.client = self.app.test_client()

    def test_console_dashboard_is_the_home_page_and_legacy_moved(self) -> None:
        home = self.client.get("/").get_data(as_text=True)
        self.assertIn("bird-cameras", home)
        self.assertIn('id="pi"', home)
        self.assertIn("Capture and save", self.client.get("/legacy").get_data(as_text=True))
        self.assertEqual(self.client.get("/fleet").status_code, 200)

    def test_captures_are_listed_per_camera_and_served_safely(self) -> None:
        store = self.directory / "captures"
        jpeg = b"\xff\xd8\xff\xe0" + b"0" * 64 + b"\xff\xd9"
        for name, camera_id in (("a", "esp32cam-one"), ("b", "esp32cam-two")):
            (store / f"{name}.jpg").write_bytes(jpeg)
            (store / f"{name}.json").write_text(
                json.dumps({"camera_id": camera_id, "received_at": "2026-10-28T12:00:00Z", "source": "motion"})
            )
        listed = self.client.get("/api/fleet/esp32cam-one/captures").get_json()["captures"]
        self.assertEqual([c["filename"] for c in listed], ["a.jpg"])
        self.assertEqual(self.client.get("/api/fleet/captures/a.jpg").data, jpeg)
        self.assertEqual(self.client.get("/api/fleet/captures/..%2Fcams.json").status_code, 404)
        self.assertEqual(self.client.get("/api/fleet/captures/missing.jpg").status_code, 404)


class ConsoleDashboardContractTests(unittest.TestCase):
    def test_calibration_choices_match_the_server(self) -> None:
        import re

        from server.capture_storage import calibration_type_values
        from server.console_dashboard import CONSOLE_DASHBOARD_HTML

        listed = re.search(r"const CAL_TYPES = \[(.*?)\]", CONSOLE_DASHBOARD_HTML).group(1)
        self.assertEqual(tuple(re.findall(r'"([a-z_]+)"', listed)), calibration_type_values())
