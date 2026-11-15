"""Cameras register and re-address themselves from their heartbeats."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from server.camera_registry import (
    CameraRegistry,
    heartbeat_address,
    sync_registry_from_heartbeat,
)
from server.laptop_test_server import create_app as create_laptop_app
from server.raspberry_pi_server import create_app as create_pi_app


def heartbeat(camera_id="esp32cam-aabbccdd0001", ip="192.168.1.50", model="OV5640"):
    return {
        "camera_id": camera_id,
        "boot_id": "boot-a",
        "camera_model": model,
        "ip": ip,
        "uptime_s": 60,
        "boot": {"number_since_power_on": 1, "reset_reason": "power_on"},
        "events": [],
    }


class RegistryHeartbeatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.file = Path(self.temporary.name) / "cameras.json"
        self.registry = CameraRegistry(self.file)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_unknown_camera_is_added_and_asks_for_a_name(self) -> None:
        camera, change = self.registry.observe_heartbeat(
            "esp32cam-aabbccdd0001", "192.168.1.50", model="OV5640"
        )
        self.assertEqual(change, "added")
        self.assertEqual(camera["url"], "http://192.168.1.50")
        self.assertEqual(camera["hardware_camera_id"], "esp32cam-aabbccdd0001")
        self.assertEqual(camera["name"], "New camera 0001")
        self.assertEqual(camera["sensor"]["model"], "OV5640")
        self.assertTrue(camera["needs_name"] and camera["enabled"] and camera["default"])

        reloaded = CameraRegistry(self.file).snapshot()
        self.assertEqual(len(reloaded), 1)
        self.assertTrue(reloaded[0]["needs_name"])

    def test_repeat_heartbeat_changes_nothing(self) -> None:
        first, _ = self.registry.observe_heartbeat("esp32cam-1", "192.168.1.50")
        second, change = self.registry.observe_heartbeat("esp32cam-1", "192.168.1.50")
        self.assertEqual(change, "unchanged")
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(len(self.registry.snapshot()), 1)

    def test_new_ip_rewrites_the_url_and_keeps_the_camera(self) -> None:
        first, _ = self.registry.observe_heartbeat("esp32cam-1", "192.168.1.50")
        self.registry.update(first["id"], name="Feeder")
        moved, change = self.registry.observe_heartbeat("esp32cam-1", "192.168.1.77")
        self.assertEqual(change, "moved")
        self.assertEqual(moved["id"], first["id"])
        self.assertEqual(moved["name"], "Feeder")
        self.assertEqual(moved["url"], "http://192.168.1.77")
        self.assertEqual(CameraRegistry(self.file).snapshot()[0]["url"], "http://192.168.1.77")

    def test_two_cameras_swapping_addresses_sort_themselves_out(self) -> None:
        a, _ = self.registry.observe_heartbeat("esp32cam-a", "192.168.1.10")
        b, _ = self.registry.observe_heartbeat("esp32cam-b", "192.168.1.11")
        self.registry.observe_heartbeat("esp32cam-a", "192.168.1.11")
        self.registry.observe_heartbeat("esp32cam-b", "192.168.1.10")
        self.assertEqual(self.registry.get(a["id"])["url"], "http://192.168.1.11")
        self.assertEqual(self.registry.get(b["id"])["url"], "http://192.168.1.10")
        CameraRegistry(self.file)  # still loads: URLs and hardware IDs stayed unique

    def test_manual_entry_at_that_address_is_bound_not_duplicated(self) -> None:
        manual = self.registry.add("Backyard", "192.168.1.50")
        bound, change = self.registry.observe_heartbeat("esp32cam-1", "192.168.1.50", model="OV2640")
        self.assertEqual(change, "bound")
        self.assertEqual(bound["id"], manual["id"])
        self.assertEqual(bound["name"], "Backyard")
        self.assertEqual(bound["hardware_camera_id"], "esp32cam-1")
        self.assertFalse(bound["needs_name"])
        self.assertEqual(len(self.registry.snapshot()), 1)

    def test_address_held_by_another_known_camera_is_left_alone(self) -> None:
        self.registry.observe_heartbeat("esp32cam-old", "192.168.1.50")
        camera, change = self.registry.observe_heartbeat("esp32cam-new", "192.168.1.50")
        self.assertIsNone(camera)
        self.assertEqual(change, "conflict")
        self.assertEqual(len(self.registry.snapshot()), 1)

    def test_naming_a_camera_clears_the_prompt(self) -> None:
        camera, _ = self.registry.observe_heartbeat("esp32cam-1", "192.168.1.50")
        renamed = self.registry.update(camera["id"], name="Oak tree")
        self.assertFalse(renamed["needs_name"])
        self.assertFalse(CameraRegistry(self.file).snapshot()[0]["needs_name"])

    def test_default_name_never_carries_markup_from_the_camera_id(self) -> None:
        camera, _ = self.registry.observe_heartbeat("cam<b>", "192.168.1.50")
        self.assertEqual(camera["name"], "New camera camb")

    def test_heartbeat_address_only_accepts_real_ip_addresses(self) -> None:
        self.assertEqual(heartbeat_address({"ip": "192.168.1.50"}, "10.0.0.9"), "192.168.1.50")
        self.assertEqual(heartbeat_address({"ip": "0.0.0.0"}, "10.0.0.9"), "10.0.0.9")
        self.assertEqual(heartbeat_address({"ip": "evil.example.com"}, "10.0.0.9"), "10.0.0.9")
        self.assertEqual(heartbeat_address({}, "10.0.0.9"), "10.0.0.9")
        self.assertIsNone(heartbeat_address({"ip": "not-an-ip"}, "127.0.0.1"))

    def test_sync_logs_changes_once_and_ignores_bad_heartbeats(self) -> None:
        lines: list[str] = []
        self.assertEqual(sync_registry_from_heartbeat(self.registry, {"ip": "192.168.1.50"}, None, log=lines.append)[1], "ignored")
        self.assertEqual(sync_registry_from_heartbeat(self.registry, heartbeat(), None, log=lines.append)[1], "added")
        self.assertEqual(sync_registry_from_heartbeat(self.registry, heartbeat(), None, log=lines.append)[1], "unchanged")
        other = heartbeat(camera_id="esp32cam-other")
        for _ in range(3):
            self.assertEqual(sync_registry_from_heartbeat(self.registry, other, None, log=lines.append)[1], "conflict")
        self.assertEqual(len(lines), 2)  # one "added", one conflict notice


class ServerHeartbeatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.save_dir = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_laptop_server_adds_and_follows_a_heartbeating_camera(self) -> None:
        client = create_laptop_app(self.save_dir).test_client()
        self.assertEqual(client.post("/api/heartbeat", json=heartbeat()).status_code, 200)
        cameras = client.get("/api/cameras").get_json()["cameras"]
        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0]["url"], "http://192.168.1.50")
        self.assertTrue(cameras[0]["needs_name"])

        client.post("/api/heartbeat", json=heartbeat(ip="192.168.1.99"))
        cameras = client.get("/api/cameras").get_json()["cameras"]
        self.assertEqual([camera["url"] for camera in cameras], ["http://192.168.1.99"])

        named = client.patch(f"/api/cameras/{cameras[0]['id']}", json={"name": "Feeder"})
        self.assertFalse(named.get_json()["camera"]["needs_name"])
        fleet = client.get("/api/fleet").get_json()["cameras"]
        self.assertEqual(fleet[0]["name"], "Feeder")

    def test_auto_registration_can_be_switched_off(self) -> None:
        client = create_laptop_app(self.save_dir, auto_register_cameras=False).test_client()
        self.assertEqual(client.post("/api/heartbeat", json=heartbeat()).status_code, 200)
        self.assertEqual(client.get("/api/cameras").get_json()["cameras"], [])

    def test_pi_server_starts_and_re_addresses_motion_workers(self) -> None:
        class FakeEngine:
            def describe(self):
                return {"ready": True}

        class FakeMotionService:
            def __init__(self, camera_url, *args, **kwargs):
                self.camera_url, self.kwargs, self.started = camera_url, kwargs, False
                self.expected_camera_id = kwargs.get("expected_camera_id")

            def start(self):
                self.started = True

            def stop(self):
                self.started = False

            def describe(self):
                return {"running": self.started}

        app = create_pi_app(
            self.save_dir,
            inference_engine=FakeEngine(),
            manage_motion_services=True,
            motion_service_factory=FakeMotionService,
        )
        client = app.test_client()
        self.assertEqual(client.post("/api/heartbeat", json=heartbeat()).status_code, 200)
        camera = client.get("/api/cameras").get_json()["cameras"][0]
        first = app.config["MOTION_SERVICES"][camera["id"]]
        self.assertTrue(first.started)
        self.assertEqual(first.camera_url, "http://192.168.1.50")
        self.assertEqual(first.expected_camera_id, "esp32cam-aabbccdd0001")

        client.post("/api/heartbeat", json=heartbeat())
        self.assertIs(app.config["MOTION_SERVICES"][camera["id"]], first)

        client.post("/api/heartbeat", json=heartbeat(ip="192.168.1.99"))
        second = app.config["MOTION_SERVICES"][camera["id"]]
        self.assertIsNot(second, first)
        self.assertFalse(first.started)
        self.assertTrue(second.started)
        self.assertEqual(second.camera_url, "http://192.168.1.99")
        app.config["CLEANUP"]()


if __name__ == "__main__":
    unittest.main()
