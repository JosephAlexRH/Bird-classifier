"""Specs view: hub and camera hardware/software rows, and the /api/hub/specs endpoint."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from server.hub_specs import camera_specs, hub_specs
from server.raspberry_pi_server import create_app

HEARTBEAT = {
    "camera_id": "esp32cam-a", "ip": "192.168.7.50", "uptime_s": 5000, "firmware_elf_sha256": "3f9a1c0be2d47a55ff",
    "boot": {"reset_reason": "power_on", "number_since_power_on": 3, "crashes_since_power_on": 0,
             "watchdog_resets_since_power_on": 1, "brownouts_since_power_on": 0},
    "wifi": {"rssi_dbm": -61}, "memory": {"free_heap": 132648, "min_free_heap": 64436},
    "hardware": {"board": "AI Thinker ESP32-CAM", "chip_model": "ESP32-D0WDQ6", "chip_revision": 1,
                 "cpu_cores": 2, "cpu_mhz": 240, "flash_bytes": 4194304, "psram_bytes": 4194304,
                 "mac": "B8:D6:1A:5D:50:20"},
}
SIDECAR = {"camera_id": "esp32cam-a", "received_at": "2026-11-20T15:21:28.248+00:00",
           "camera": {"sensor": {"manufacturer": "OmniVision", "model": "OV5640", "pid_hex": "0x5640",
                                 "xclk_hz": 20000000},
                      "observed": {"capture": {"frame_size": {"name": "VGA"}, "width": 640, "height": 480,
                                               "jpeg_quality": 10}},
                      "driver": {"arduino_esp32_version": "2.0.17"}}}


def rows(spec):
    return {label: value for section in spec["sections"] for label, value in section["rows"]}


class CameraSpecsTests(unittest.TestCase):
    def test_heartbeat_and_sidecar_are_combined(self):
        r = rows(camera_specs({"id": "reg-a", "name": "Feeder"}, HEARTBEAT, SIDECAR))
        self.assertEqual(r["Board"], "AI Thinker ESP32-CAM")
        self.assertEqual(r["Chip"], "ESP32-D0WDQ6 rev 1 · 2 cores @ 240 MHz")
        self.assertEqual((r["Flash"], r["PSRAM"]), ("4.0 MB", "4.0 MB"))
        self.assertEqual(r["Sensor"], "OmniVision OV5640")
        self.assertEqual(r["Frame size"], "VGA (640×480)")
        self.assertEqual(r["Free heap"], "130 KB (lowest 63 KB)")
        self.assertEqual(r["Faults"], "0 crash · 1 watchdog · 0 brownout")
        self.assertEqual(r["Last capture"], "2026-11-20 15:21 UTC")

    def test_older_firmware_and_missing_heartbeat_say_so(self):
        old = {k: v for k, v in HEARTBEAT.items() if k != "hardware"}
        spec = camera_specs({"id": "reg-a"}, old, SIDECAR)
        self.assertIn("firmware update", spec["note"])
        self.assertNotIn("Board", rows(spec))
        spec = camera_specs({"id": "reg-b", "url": "http://192.168.7.51"}, None, None)
        self.assertEqual(spec["note"], "No heartbeat received yet.")
        self.assertEqual(rows(spec), {"URL": "http://192.168.7.51"})


class HubSpecsTests(unittest.TestCase):
    def test_hub_rows_include_storage_and_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = {"detector": {"filename": "det.tflite", "image_size": [320, 320], "input_dtype": "float32",
                                   "sha256": "abcdef1234567890"}, "species": {"filename": "sp.tflite"},
                      "species_count": 42}
            spec = hub_specs(Path(temporary), models=models, clock={"source": "ntp"}, started_at=0)
            r = rows(spec)
            self.assertEqual(r["Captures"], temporary)
            self.assertEqual(r["Bird detector"], "det.tflite · 320×320 · float32 · abcdef123456")
            self.assertEqual(r["Species labels"], "42")
            self.assertEqual(r["Clock"], "ntp")
            self.assertRegex(r["Hub code"], r"^[0-9a-f]{12}$")

    def test_endpoint_lists_hub_and_registered_cameras(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root)
            client = app.test_client()
            body = client.get("/api/hub/specs").get_json()
            self.assertEqual(body["hub"]["name"], "Hub")
            self.assertEqual(body["cameras"], [])
            self.assertIn("Storage", [section["name"] for section in body["hub"]["sections"]])


if __name__ == "__main__":
    unittest.main()
