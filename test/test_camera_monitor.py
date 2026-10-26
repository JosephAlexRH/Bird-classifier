"""Tests for the inference-free multi-camera status/config monitor."""

from __future__ import annotations

import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from server.camera_monitor import CameraRegistry, create_app


class FakeJsonResponse:
    def __init__(self, value: dict, status: int = 200) -> None:
        self.body = json.dumps(value).encode("utf-8")
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"

    def __enter__(self) -> "FakeJsonResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


class CameraRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.registry_path = Path(self.temporary.name) / "cameras.json"
        self.registry = CameraRegistry(self.registry_path, poll_seconds=5, timeout_seconds=1)

    def tearDown(self) -> None:
        self.registry.stop()
        self.temporary.cleanup()

    def test_add_remove_and_reload_persist_only_definitions(self) -> None:
        first = self.registry.add("  Backyard   feeder ", "HTTP://Camera-One.Local/")
        second = self.registry.add("Side yard", "http://camera-two.local:8080")
        self.assertEqual(first["name"], "Backyard feeder")
        self.assertEqual(first["url"], "http://camera-one.local")
        self.assertTrue(self.registry_path.is_file())

        reloaded = CameraRegistry(self.registry_path)
        try:
            cameras = reloaded.snapshot()
            self.assertEqual([camera["name"] for camera in cameras], ["Backyard feeder", "Side yard"])
            self.assertIsNone(cameras[0]["reachable"])
            self.assertIsNone(cameras[0]["status"])
            removed = reloaded.remove(second["id"])
            self.assertEqual(removed["name"], "Side yard")
        finally:
            reloaded.stop()

        final = CameraRegistry(self.registry_path)
        try:
            self.assertEqual([camera["id"] for camera in final.snapshot()], [first["id"]])
        finally:
            final.stop()

    def test_duplicate_and_invalid_cameras_are_rejected(self) -> None:
        self.registry.add("One", "http://camera.local")
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.registry.add("Duplicate", "HTTP://CAMERA.LOCAL/")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.registry.add(" ", "http://other.local")
        with self.assertRaisesRegex(ValueError, "base URL"):
            self.registry.add("Bad URL", "http://other.local/status")

    def test_bare_ip_default_and_enabled_state_persist(self) -> None:
        first = self.registry.add(None, "192.168.1.84")
        second = self.registry.add("Side yard", "camera-two.local", make_default=True)
        self.registry.update(first["id"], enabled=False)
        reloaded = CameraRegistry(self.registry_path)
        try:
            values = {camera["id"]: camera for camera in reloaded.snapshot()}
            self.assertEqual(values[first["id"]]["url"], "http://192.168.1.84")
            self.assertEqual(values[first["id"]]["name"], "192.168.1.84")
            self.assertFalse(values[first["id"]]["enabled"])
            self.assertTrue(values[second["id"]]["default"])
            self.assertEqual(reloaded.resolve()["id"], second["id"])
        finally:
            reloaded.stop()

    @patch("server.camera_monitor.urlopen")
    def test_poll_discovers_stable_camera_and_sensor_identity(self, mocked_urlopen) -> None:
        camera = self.registry.add("Backyard", "http://camera.local")
        mocked_urlopen.side_effect = [
            FakeJsonResponse({
                "camera_id": "esp32cam-aabbccddeeff",
                "camera_sensor_model": "OV2640",
                "camera_sensor_pid": 0x26,
                "camera_sensor_pid_hex": "0x0026",
            }),
            FakeJsonResponse({"camera_restart_required": False}),
            FakeJsonResponse({
                "device": {"camera_id": "esp32cam-aabbccddeeff"},
                "camera": {"sensor": {"model": "OV2640", "pid": 0x26, "pid_hex": "0x0026"}},
            }),
        ]
        result = self.registry.poll(camera["id"])
        self.assertEqual(result["hardware_camera_id"], "esp32cam-aabbccddeeff")
        self.assertEqual(result["sensor"]["model"], "OV2640")
        self.assertEqual(result["sensor"]["pid"], 0x26)

    @patch("server.camera_monitor.urlopen")
    def test_poll_all_fetches_status_and_config_for_every_camera(self, mocked_urlopen) -> None:
        first = self.registry.add("One", "http://one.local")
        second = self.registry.add("Two", "http://two.local")

        def response_for(request, **_kwargs):
            endpoint = request.full_url.rsplit("/", 1)[-1]
            host = request.full_url.split("//", 1)[1].split("/", 1)[0]
            return FakeJsonResponse({"endpoint": endpoint, "host": host})

        mocked_urlopen.side_effect = response_for
        cameras = self.registry.poll_all()
        self.assertEqual(len(cameras), 2)
        self.assertTrue(all(camera["healthy"] for camera in cameras))
        self.assertTrue(all(camera["reachable"] for camera in cameras))
        self.assertEqual(self.registry.get(first["id"])["status"]["host"], "one.local")
        self.assertEqual(self.registry.get(second["id"])["config"]["host"], "two.local")
        self.assertEqual(mocked_urlopen.call_count, 6)

    @patch("server.camera_monitor.urlopen")
    def test_partial_failure_keeps_successful_endpoint_and_reports_error(self, mocked_urlopen) -> None:
        camera = self.registry.add("One", "http://one.local")

        def response_for(request, **_kwargs):
            if request.full_url.endswith("/status"):
                return FakeJsonResponse({"uptime_ms": 1234})
            if request.full_url.endswith("/config"):
                raise URLError("config unavailable")
            return FakeJsonResponse({"device": {"camera_id": "esp32-one"}})

        mocked_urlopen.side_effect = response_for
        result = self.registry.poll(camera["id"])
        self.assertTrue(result["reachable"])
        self.assertFalse(result["healthy"])
        self.assertEqual(result["status"]["uptime_ms"], 1234)
        self.assertIsNone(result["config"])
        self.assertIn("config unavailable", result["errors"]["config"])
        self.assertIsNotNone(result["last_polled_at"])
        self.assertIsNone(result["last_success_at"])


class CameraMonitorApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.registry = CameraRegistry(Path(self.temporary.name) / "cameras.json")
        self.app = create_app(self.registry, start_poller=False)
        self.app.testing = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.registry.stop()
        self.temporary.cleanup()

    @patch("server.camera_monitor.urlopen")
    def test_management_poll_and_cached_endpoint_api(self, mocked_urlopen) -> None:
        mocked_urlopen.side_effect = [
            FakeJsonResponse({"camera_sensor_model": "OV5640"}),
            FakeJsonResponse({"camera_restart_required": False}),
            FakeJsonResponse({"device": {"camera_id": "esp32-one"}}),
            FakeJsonResponse({"camera_sensor_model": "OV5640"}),
            FakeJsonResponse({"camera_restart_required": False}),
            FakeJsonResponse({"device": {"camera_id": "esp32-one"}}),
            FakeJsonResponse({"camera_sensor_model": "OV5640"}),
            FakeJsonResponse({"camera_restart_required": False}),
            FakeJsonResponse({"device": {"camera_id": "esp32-one"}}),
        ]
        added = self.client.post(
            "/api/cameras",
            json={"name": "Backyard", "url": "http://camera.local"},
        )
        self.assertEqual(added.status_code, 201)
        camera = added.get_json()["camera"]
        self.assertTrue(camera["healthy"])

        listing = self.client.get("/api/cameras").get_json()
        self.assertEqual(len(listing["cameras"]), 1)
        camera_id = camera["id"]
        self.assertEqual(
            self.client.get(f"/api/cameras/{camera_id}/status").get_json()["value"]["camera_sensor_model"],
            "OV5640",
        )
        self.assertFalse(
            self.client.get(f"/api/cameras/{camera_id}/config").get_json()["value"][
                "camera_restart_required"
            ]
        )
        self.assertEqual(self.client.post(f"/api/cameras/{camera_id}/poll").status_code, 200)
        fleet_poll = self.client.post("/api/cameras/poll")
        self.assertEqual(fleet_poll.status_code, 200)
        self.assertEqual(len(fleet_poll.get_json()["cameras"]), 1)
        self.assertEqual(self.client.get("/health").get_json()["healthy"], 1)
        self.assertEqual(self.client.delete(f"/api/cameras/{camera_id}").status_code, 200)
        self.assertEqual(self.client.get("/api/cameras").get_json()["cameras"], [])

    def test_api_rejects_invalid_payloads_and_unknown_ids(self) -> None:
        self.assertEqual(self.client.post("/api/cameras", data="bad").status_code, 400)
        invalid = self.client.post("/api/cameras", json={"name": "Camera", "url": "not-a-url"})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(self.client.delete("/api/cameras/missing").status_code, 404)
        self.assertEqual(self.client.post("/api/cameras/missing/poll").status_code, 404)

    def test_dashboard_contains_camera_management_controls(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"ESP32 camera monitor", page.data)
        self.assertIn(b"Add camera", page.data)
        self.assertIn(b"Poll all now", page.data)
        self.assertIn(b"Poll now", page.data)


if __name__ == "__main__":
    unittest.main()