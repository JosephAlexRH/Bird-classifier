"""Run log, hub time keeping and the run report."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from server.hub_clock import HubClock
from server.laptop_test_server import create_app as create_laptop_app
from server.run_log import RunLog
from server.run_report import read_lines, render, summarise


class FakeTime:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now
        self.steps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def step(self, to: float) -> bool:
        self.steps.append(to)
        self.now = to
        return True


def heartbeat(camera_id="esp32cam-aabbccdd0001", ip="192.168.1.50", events=()):
    return {
        "camera_id": camera_id,
        "boot_id": "boot-a",
        "camera_model": "OV5640",
        "ip": ip,
        "uptime_s": 60,
        "time_source": "hub",
        "boot": {"number_since_power_on": 1, "reset_reason": "power_on"},
        "wifi": {"rssi_dbm": -61},
        "memory": {"free_heap": 150000, "min_free_heap": 120000},
        "events": list(events),
    }


class HubClockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.file = Path(self.temporary.name) / "hub_clock.json"
        self.time = FakeTime()
        self.journal: list[tuple[str, dict]] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make(self, ntp=False, can_step=True, boot="boot-1"):
        return HubClock(
            self.file,
            clock=self.time,
            ntp_synced=lambda: ntp,
            set_system_time=self.time.step if can_step else None,
            boot_id=lambda: boot,
            journal=lambda kind, **fields: self.journal.append((kind, fields)),
        )

    def test_unverified_until_a_phone_offers_the_time_then_stepped(self) -> None:
        clock = self.make()
        self.assertEqual(clock.state()["source"], "unverified")
        self.assertFalse(clock.reply_fields()["server_time_trusted"])
        true_time = self.time.now + 7200  # hub is two hours behind after a power cut
        result = clock.offer_reference(true_time * 1000)
        self.assertTrue(result["applied"] and result["stepped"])
        self.assertAlmostEqual(result["offset_s"], 7200, places=1)
        self.assertEqual(self.time.steps, [true_time])
        self.assertEqual(clock.state()["source"], "reference")
        self.assertEqual(clock.reply_fields()["server_time_unix_ms"], int(true_time * 1000))
        self.assertEqual(self.journal[-1][0], "clock_reference")
        self.assertAlmostEqual(self.journal[-1][1]["offset_s"], 7200, places=1)

    def test_agreeing_reference_is_trusted_without_touching_the_clock(self) -> None:
        clock = self.make()
        clock.offer_reference((self.time.now + 0.4) * 1000)
        clock.offer_reference((self.time.now + 0.5) * 1000)
        self.assertEqual(self.time.steps, [])
        self.assertEqual(clock.state()["source"], "reference")
        self.assertEqual(len(self.journal), 1)  # the second, identical visit adds nothing

    def test_offset_is_recorded_when_the_clock_cannot_be_set(self) -> None:
        clock = self.make(can_step=False)
        clock.offer_reference((self.time.now + 500) * 1000)
        state = clock.state()
        self.assertEqual(state["source"], "offset")
        self.assertFalse(state["trusted"])
        self.assertAlmostEqual(state["reference"]["offset_s"], 500, places=1)

    def test_network_time_wins_over_a_browser(self) -> None:
        clock = self.make(ntp=True)
        result = clock.offer_reference((self.time.now + 500) * 1000)
        self.assertFalse(result["applied"])
        self.assertEqual(self.time.steps, [])
        self.assertEqual(clock.state()["source"], "ntp")

    def test_implausible_times_are_rejected(self) -> None:
        clock = self.make()
        for bad in (0, 1000, "now", None, True):
            with self.assertRaises(ValueError):
                clock.offer_reference(bad)

    def test_reference_survives_a_server_restart_but_not_a_power_cut(self) -> None:
        self.make().offer_reference((self.time.now + 0.1) * 1000)
        self.assertEqual(self.make().state()["source"], "reference")
        self.assertEqual(self.make(boot="boot-2").state()["source"], "unverified")

    def test_clock_that_starts_in_the_past_is_moved_forward(self) -> None:
        self.make().tick()
        last_seen = self.time.now
        self.time.now -= 86400  # power cut; the Pi woke up a day in the past
        self.make(boot="boot-2")
        self.assertEqual(self.time.steps, [last_seen])
        self.assertEqual(self.journal[-1][0], "clock_restored")


class RunLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_lines_rotate_by_day_and_carry_context(self) -> None:
        now = FakeTime(1_800_000_000.0)  # 2027-01-15
        log = RunLog(self.directory, clock=now, context=lambda: {"hub_boot": "b1", "clock": "ntp"})
        log.write("hub_start", pid=1)
        now.now += 86400
        log.heartbeat(heartbeat(events=[{"seq": 1, "text": "x"}]), "192.168.1.50")
        log.close()
        files = sorted(path.name for path in self.directory.glob("*.jsonl"))
        self.assertEqual(files, ["run-2027-01-15.jsonl", "run-2027-01-16.jsonl"])
        records = list(read_lines(self.directory))
        self.assertEqual([record["type"] for record in records], ["hub_start", "heartbeat"])
        self.assertEqual(records[0]["hub_boot"], "b1")
        self.assertIn("mono", records[0])
        self.assertNotIn("events", records[1]["heartbeat"])
        self.assertEqual(records[1]["camera_id"], "esp32cam-aabbccdd0001")

    def test_a_line_cut_short_by_power_loss_is_skipped(self) -> None:
        (self.directory / "run-2027-01-15.jsonl").write_text('{"type":"hub_start"}\n{"type":"heartb', encoding="utf-8")
        self.assertEqual(len(list(read_lines(self.directory))), 1)


class ServerRunLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.save_dir = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_heartbeats_are_logged_and_answered_with_hub_time(self) -> None:
        app = create_laptop_app(self.save_dir)
        client = app.test_client()
        events = [{"seq": 0, "boot": 1, "t_ms": 1000, "text": "boot #1"}]
        reply = client.post("/api/heartbeat", json=heartbeat(events=events)).get_json()
        self.assertGreater(reply["server_time_unix_ms"], 1_700_000_000_000)
        self.assertIn("server_time_trusted", reply)
        client.post("/api/heartbeat", json=heartbeat(events=events))  # same event again: not re-logged
        app.config["RUN_LOG"].close()

        records = list(read_lines(self.save_dir / "runlog"))
        kinds = [record["type"] for record in records]
        self.assertEqual(kinds.count("heartbeat"), 2)
        self.assertEqual(kinds.count("camera_boot"), 1)
        self.assertEqual(kinds.count("camera_event"), 1)
        self.assertEqual(kinds.count("registry"), 1)
        self.assertTrue(all("hub_boot" in record and "clock" in record for record in records))

        report = render(summarise(records))
        self.assertIn("CAMERA esp32cam-aabbccdd0001", report)
        self.assertIn("power_on x1", report)

    def test_time_endpoint_reports_and_accepts_a_reference(self) -> None:
        client = create_laptop_app(self.save_dir).test_client()
        self.assertIn("clock", client.get("/api/time").get_json())
        self.assertIn("hub_clock", client.get("/api/fleet").get_json())
        self.assertEqual(client.post("/api/time", json={"client_unix_ms": "soon"}).status_code, 422)
        import time
        accepted = client.post("/api/time", json={"client_unix_ms": int(time.time() * 1000)})
        self.assertEqual(accepted.status_code, 200)


class RunReportTests(unittest.TestCase):
    def test_gaps_and_uptime_use_the_monotonic_clock(self) -> None:
        records, mono = [], 0.0

        def beat(camera="cam-a"):
            records.append({"type": "heartbeat", "t": f"t{mono:.0f}", "mono": mono, "hub_boot": "b1",
                            "clock": "unverified", "camera_id": camera, "heartbeat": {"wifi": {"rssi_dbm": -70}}})

        for _ in range(40):  # 10 minutes online
            beat()
            mono += 15
        mono += 600  # camera silent for ~10 minutes
        for _ in range(40):
            beat()
            mono += 15
        records.append({"type": "camera_boot", "t": "x", "mono": mono, "hub_boot": "b1",
                        "camera_id": "cam-a", "reset_reason": "brownout"})
        summary = summarise(records)
        camera = summary["cameras"]["cam-a"]
        self.assertEqual(len(camera["gaps"]), 1)
        self.assertAlmostEqual(camera["gaps"][0][0], 615, delta=1)
        self.assertAlmostEqual(camera["online_s"], 78 * 15, delta=1)
        self.assertEqual(camera["boots"]["brownout"], 1)
        text = render(summary)
        self.assertIn("gaps over 60 s: 1", text)
        self.assertIn("unverified 100%", text)


if __name__ == "__main__":
    unittest.main()
