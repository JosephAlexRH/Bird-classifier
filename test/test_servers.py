"""Server contract tests requiring only the application's Flask dependency."""

from __future__ import annotations

import hashlib
import json
import re
import socket
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from server.capture_storage import (
    CAPTURE_SCHEMA_VERSION,
    METADATA_HEADER,
    CalibrationType,
    CaptureStore,
    calibration_type_values,
    parse_calibration_type,
)
from server.laptop_test_server import (
    create_app as create_laptop_app,
    local_source_ip,
    normalize_camera_url,
    preflight_port,
)
from server.raspberry_pi_server import create_app as create_pi_app, validate_classification


JPEG = b"\xff\xd8test-jpeg-payload\xff\xd9"
REGISTER_DUMP = b"OVD1"
CAMERA_METADATA = {
    "schema_version": 4,
    "source": "preview",
    "device": {
        "boot_id": "86b20b95-393c-4e4a-91ab-224e78401484",
        "clock": {
            "sync_status": "synchronized",
            "device_time_utc": "2026-09-30T17:15:42.123456Z",
            "device_time_boot_us": 1003000,
        },
        "config": {"file_sha256": "a" * 64},
    },
    "camera": {
        "requested": {"capture": {"frame_size": {"name": "VGA", "value": 8}}},
        "observed": {
            "capture": {"width": 640, "height": 480, "framebuffer_bytes": len(JPEG)}
        },
        "timing": {"frame_boot_us": 1000000, "register_read_boot_us": 1002500},
        "register_dump": {
            "sha256": hashlib.sha256(REGISTER_DUMP).hexdigest(),
            "payload_included": True,
            "encoding": "base64",
            "payload_base64": "T1ZEMQ==",
        },
    },
}


class FakeUpstream:
    def __init__(self, body: bytes, content_type: str = "application/json", metadata: str | None = None):
        self.body = body
        self.status = 200
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if metadata:
            self.headers[METADATA_HEADER] = metadata

    def __enter__(self) -> "FakeUpstream":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


class FakeStreamUpstream(FakeUpstream):
    def __init__(self, chunks: list[bytes]):
        super().__init__(b"", "multipart/x-mixed-replace; boundary=esp32cam-frame")
        self.chunks = iter(chunks)
        self.closed = False

    def read(self, _size: int = -1) -> bytes:
        return next(self.chunks, b"")

    def close(self) -> None:
        self.closed = True


class CaptureStorageTests(unittest.TestCase):
    def test_calibration_enum_has_exact_stable_taxonomy(self) -> None:
        expected = (
            "normal",
            "dark_frame",
            "flat_field",
            "gray_card",
            "color_chart",
            "resolution_target",
        )
        self.assertEqual(calibration_type_values(), expected)
        self.assertTrue(issubclass(CalibrationType, str))
        for value in expected:
            with self.subTest(value=value):
                self.assertIs(parse_calibration_type(value), CalibrationType(value))
                self.assertEqual(json.dumps(CalibrationType(value)), json.dumps(value))

    def test_legacy_and_unknown_calibration_types_are_rejected(self) -> None:
        for value in ("color_reference", "lens_grid", "other", "unknown", None, 4):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "allowed values"):
                    parse_calibration_type(value)

    def test_all_calibration_types_store_as_strings_in_schema_v3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CaptureStore(Path(temporary))
            for calibration_type in CalibrationType:
                with self.subTest(calibration_type=calibration_type):
                    record = store.save_capture(
                        JPEG, source="test", calibration_type=calibration_type
                    )
                    self.assertEqual(record["schema_version"], 3)
                    self.assertEqual(record["calibration_type"], calibration_type.value)
                    self.assertIs(type(record["calibration_type"]), str)
                    sidecar = json.loads(
                        (store.root / record["sidecar"]).read_text(encoding="utf-8")
                    )
                    self.assertEqual(sidecar["calibration_type"], calibration_type.value)
                    self.assertEqual(sidecar["schema_version"], 3)

    def test_new_profile_uses_schema_v3_even_if_input_attempts_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CaptureStore(Path(temporary))
            profile = store.save_profile({"schema_version": 1, "status": {"ok": True}})
            self.assertEqual(CAPTURE_SCHEMA_VERSION, 3)
            self.assertEqual(profile["schema_version"], 3)
            saved = json.loads((store.root / profile["filename"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], 3)

    def test_frame_utc_estimate_is_null_without_a_valid_clock_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CaptureStore(Path(temporary))
            for device_time_utc in (None, "not-a-timestamp"):
                with self.subTest(device_time_utc=device_time_utc):
                    metadata = json.loads(json.dumps(CAMERA_METADATA))
                    metadata["device"]["clock"]["device_time_utc"] = device_time_utc
                    record = store.save_capture(
                        JPEG, source="test", camera_metadata=metadata
                    )
                    self.assertIsNone(record["camera"]["timing"]["frame_utc_estimate"])

    def test_flattening_does_not_mutate_received_camera_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CaptureStore(Path(temporary))
            metadata = json.loads(json.dumps(CAMERA_METADATA))
            record = store.save_capture(JPEG, source="test", camera_metadata=metadata)
            self.assertNotIn("frame_utc_estimate", metadata["camera"]["timing"])
            self.assertIn("frame_utc_estimate", record["camera"]["timing"])
            self.assertNotIn("camera", record["camera"])


class LaptopServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.save_dir = Path(self.temporary.name)
        self.app = create_laptop_app(self.save_dir, "http://camera.local")
        self.app.testing = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_legacy_motion_upload_saves_sidecar_and_manifest(self) -> None:
        response = self.client.post("/upload", data=JPEG, content_type="image/jpeg")
        self.assertEqual(response.status_code, 200)
        record = response.get_json()
        self.assertEqual(record["source"], "motion")
        self.assertIn("not supplied", record["metadata_error"])
        self.assertEqual(record["sha256"], hashlib.sha256(JPEG).hexdigest())
        self.assertEqual((self.save_dir / record["filename"]).read_bytes(), JPEG)
        sidecar = json.loads((self.save_dir / record["sidecar"]).read_text(encoding="utf-8"))
        manifest = json.loads((self.save_dir / "manifest.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["capture_id"], record["capture_id"])
        self.assertEqual(manifest["capture_id"], record["capture_id"])
        latest_response = self.client.get("/latest.jpg")
        try:
            self.assertEqual(latest_response.data, JPEG)
        finally:
            latest_response.close()

    def test_motion_upload_accepts_camera_metadata(self) -> None:
        response = self.client.post(
            "/upload",
            data=JPEG,
            content_type="image/jpeg",
            headers={METADATA_HEADER: json.dumps({**CAMERA_METADATA, "source": "motion"})},
        )
        record = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(record["firmware_metadata_schema_version"], 4)
        self.assertEqual(record["camera"]["observed"]["capture"]["width"], 640)
        self.assertEqual(
            record["camera"]["timing"]["register_read_boot_us"], 1002500
        )
        self.assertEqual(
            record["camera"]["timing"]["frame_utc_estimate"],
            "2026-09-30T17:15:42.120456Z",
        )
        self.assertTrue(
            record["camera"]["register_dump"]["payload_included"]
        )
        self.assertIsNone(record["metadata_error"])

    def test_malformed_metadata_does_not_reject_legacy_compatible_upload(self) -> None:
        response = self.client.post(
            "/upload",
            data=JPEG,
            content_type="image/jpeg",
            headers={METADATA_HEADER: "{not-json"},
        )
        self.assertEqual(response.status_code, 200)
        record = response.get_json()
        self.assertIsNone(record["camera"])
        self.assertIn("invalid JSON", record["metadata_error"])

    def test_register_dump_hash_mismatch_is_retained_with_integrity_error(self) -> None:
        metadata = json.loads(json.dumps(CAMERA_METADATA))
        metadata["camera"]["register_dump"]["sha256"] = "0" * 64
        response = self.client.post(
            "/upload",
            data=JPEG,
            content_type="image/jpeg",
            headers={METADATA_HEADER: json.dumps(metadata)},
        )
        self.assertEqual(response.status_code, 200)
        record = response.get_json()
        self.assertEqual(record["firmware_metadata_schema_version"], 4)
        self.assertIn("SHA-256 did not match", record["metadata_error"])

    def test_upload_rejects_empty_and_incomplete_images(self) -> None:
        self.assertEqual(self.client.post("/upload", data=b"").status_code, 400)
        self.assertEqual(self.client.post("/upload", data=b"not jpeg").status_code, 415)
        self.assertFalse(list(self.save_dir.glob("*.jpg")))

    @patch("server.laptop_test_server.urlopen")
    def test_manual_calibration_capture_saves_user_fields(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = FakeUpstream(
            JPEG, "image/jpeg", json.dumps(CAMERA_METADATA)
        )
        response = self.client.post(
            "/api/captures",
            json={"calibration_type": "dark_frame", "label": "covered", "note": "warm sensor"},
        )
        self.assertEqual(response.status_code, 200)
        record = response.get_json()
        self.assertEqual(record["source"], "manual")
        self.assertEqual(record["calibration_type"], "dark_frame")
        self.assertEqual(record["label"], "covered")
        self.assertEqual(record["note"], "warm sensor")
        requested = mocked_urlopen.call_args.args[0]
        self.assertEqual(requested.full_url, "http://camera.local/capture")

    def test_manual_capture_validates_calibration_type(self) -> None:
        for calibration_type in ("unknown", "color_reference", "lens_grid", "other"):
            with self.subTest(calibration_type=calibration_type):
                response = self.client.post(
                    "/api/captures", json={"calibration_type": calibration_type}
                )
                self.assertEqual(response.status_code, 422)
                result = response.get_json()
                self.assertEqual(result["allowed_values"], list(calibration_type_values()))
                self.assertIn(calibration_type, result["error"])
                self.assertIn("gray_card", result["error"])

    @patch("server.laptop_test_server.urlopen")
    def test_manual_capture_accepts_every_calibration_type(self, mocked_urlopen) -> None:
        mocked_urlopen.side_effect = lambda *_args, **_kwargs: FakeUpstream(
            JPEG, "image/jpeg", json.dumps(CAMERA_METADATA)
        )
        for calibration_type in CalibrationType:
            with self.subTest(calibration_type=calibration_type.value):
                response = self.client.post(
                    "/api/captures", json={"calibration_type": calibration_type.value}
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["calibration_type"], calibration_type.value)
                self.assertEqual(response.get_json()["schema_version"], 3)

    @patch("server.laptop_test_server.urlopen")
    def test_camera_profile_combines_three_endpoints(self, mocked_urlopen) -> None:
        responses = {
            "/metadata": {"sensor": {"pid": 38}},
            "/config": {"jpeg_quality": 10},
            "/status": {"motion_enabled": True},
        }

        def respond(outgoing, timeout):
            del timeout
            endpoint = outgoing.full_url.removeprefix("http://camera.local")
            return FakeUpstream(json.dumps(responses[endpoint]).encode())

        mocked_urlopen.side_effect = respond
        response = self.client.post("/api/profiles")
        self.assertEqual(response.status_code, 200)
        profile = response.get_json()
        self.assertEqual(profile["metadata"]["sensor"]["pid"], 38)
        self.assertEqual(profile["config"]["jpeg_quality"], 10)
        self.assertTrue((self.save_dir / profile["filename"]).exists())

    @patch("server.laptop_test_server.urlopen", side_effect=URLError("offline"))
    def test_proxy_network_failure_is_502(self, _mocked_urlopen) -> None:
        response = self.client.get("/camera/status")
        self.assertEqual(response.status_code, 502)
        self.assertIn("http://camera.local/status", response.get_json()["error"])
        self.assertIn("offline", response.get_json()["error"])
        info = self.client.get("/api/info").get_json()
        self.assertFalse(info["camera_reachable"])
        self.assertIsNotNone(info["camera_checked_at"])

    @patch("server.laptop_test_server.urlopen", side_effect=URLError(socket.timeout("timed out")))
    def test_capture_timeout_has_actionable_error(self, _mocked_urlopen) -> None:
        response = self.client.post("/api/captures", json={})
        self.assertEqual(response.status_code, 502)
        error = response.get_json()["error"]
        self.assertIn("http://camera.local/capture", error)
        self.assertIn("timed out after 8 seconds", error)
        self.assertIn("serial monitor", error)

    @patch("server.laptop_test_server.urlopen")
    def test_successful_camera_request_updates_reachability(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = FakeUpstream(b'{"motion_enabled":true}')
        self.assertEqual(self.client.get("/camera/status").status_code, 200)
        info = self.client.get("/api/info").get_json()
        self.assertTrue(info["camera_reachable"])
        self.assertIsNone(info["camera_error"])

    def test_unconfigured_camera_is_503_and_dashboard_has_controls(self) -> None:
        app = create_laptop_app(self.save_dir / "unconfigured", None)
        app.testing = True
        client = app.test_client()
        self.assertEqual(client.get("/camera/capture").status_code, 503)
        dashboard = client.get("/").get_data(as_text=True)
        self.assertIn("Start stream", dashboard)
        self.assertIn("Arm motion", dashboard)
        self.assertIn("Camera state", dashboard)
        self.assertIn("Capture and save", dashboard)
        self.assertIn("Save camera profile", dashboard)

    @patch("server.laptop_test_server.urlopen")
    def test_mode_control_proxies_valid_json(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = FakeUpstream(
            b'{"selected_mode":"motion","runtime_state":"motion_arming"}'
        )
        response = self.client.post("/camera/mode", json={"mode": "motion"})
        self.assertEqual(response.status_code, 200)
        requested = mocked_urlopen.call_args.args[0]
        self.assertEqual(requested.full_url, "http://camera.local/mode")
        self.assertEqual(requested.method, "POST")
        self.assertEqual(json.loads(requested.data), {"mode": "motion"})

    def test_mode_control_rejects_unknown_mode(self) -> None:
        response = self.client.post("/camera/mode", json={"mode": "stream"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("motion or idle", response.get_json()["error"])

    @patch("server.laptop_test_server.urlopen")
    def test_stream_is_incremental_and_preserves_multipart_boundary(self, mocked_urlopen) -> None:
        upstream = FakeStreamUpstream([b"first-part", b"second-part", b""])
        mocked_urlopen.return_value = upstream
        response = self.client.get("/camera/stream", buffered=False)
        try:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.headers["Content-Type"],
                "multipart/x-mixed-replace; boundary=esp32cam-frame",
            )
            iterator = iter(response.response)
            self.assertEqual(next(iterator), b"first-part")
            self.assertEqual(next(iterator), b"second-part")
        finally:
            response.close()
        self.assertTrue(upstream.closed)

    @patch("server.laptop_test_server.urlopen")
    def test_camera_requests_return_busy_while_stream_is_open(self, mocked_urlopen) -> None:
        upstream = FakeStreamUpstream([b"frame", b""])
        mocked_urlopen.return_value = upstream
        stream = self.client.get("/camera/stream", buffered=False)
        iterator = iter(stream.response)
        self.assertEqual(next(iterator), b"frame")
        try:
            busy = self.client.get("/camera/status")
            self.assertEqual(busy.status_code, 409)
            self.assertIn("busy streaming", busy.get_json()["error"])
            info = self.client.get("/api/info").get_json()
            self.assertTrue(info["camera_streaming"])
        finally:
            stream.close()
        self.assertFalse(self.client.get("/api/info").get_json()["camera_streaming"])

    def test_dashboard_has_exact_six_calibration_choices(self) -> None:
        dashboard = self.client.get("/").get_data(as_text=True)
        choices = re.findall(r'<option value="([^"]+)">', dashboard)
        self.assertEqual(choices, list(calibration_type_values()))

    def test_dashboard_has_mouse_touch_roi_editor(self) -> None:
        dashboard = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="roi-overlay"', dashboard)
        self.assertIn("pointerdown", dashboard)
        self.assertIn("Load current ROI", dashboard)
        self.assertIn("Reset full frame", dashboard)
        self.assertIn("Save ROI", dashboard)
        self.assertIn("motion_roi", dashboard)

    @patch("server.laptop_test_server.urlopen")
    def test_roi_partial_update_is_forwarded_without_other_config(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = FakeUpstream(b'{"status":"ok"}')
        roi = {"x": 0.125, "y": 0.25, "width": 0.5, "height": 0.625}
        response = self.client.post("/camera/config", json={"motion_roi": roi})
        self.assertEqual(response.status_code, 200)
        requested = mocked_urlopen.call_args.args[0]
        self.assertEqual(requested.full_url, "http://camera.local/config")
        self.assertEqual(requested.method, "POST")
        self.assertEqual(json.loads(requested.data), {"motion_roi": roi})

    def test_profile_response_uses_schema_v3(self) -> None:
        responses = [
            FakeUpstream(b'{"sensor":{}}'),
            FakeUpstream(b'{"jpeg_quality":10}'),
            FakeUpstream(b'{"motion_enabled":true}'),
        ]
        with patch("server.laptop_test_server.urlopen", side_effect=responses):
            response = self.client.post("/api/profiles")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["schema_version"], 3)


class LaptopStartupDiagnosticsTests(unittest.TestCase):
    def test_camera_url_normalization(self) -> None:
        self.assertEqual(normalize_camera_url(" HTTP://Camera.Local/ "), "http://camera.local")
        self.assertEqual(
            normalize_camera_url("https://[2001:db8::1]:8443/"),
            "https://[2001:db8::1]:8443",
        )
        self.assertIsNone(normalize_camera_url(None))

    def test_camera_url_rejects_non_base_and_malformed_urls(self) -> None:
        invalid = (
            "camera.local",
            "ftp://camera.local",
            "http:///",
            "http://camera.local/status",
            "http://camera.local?x=1",
            "http://camera.local/#fragment",
            "http://user:secret@camera.local",
            "http://camera local",
            "http://camera.local:99999",
            "  ",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_camera_url(value)

    @patch("server.laptop_test_server.socket.getaddrinfo")
    @patch("server.laptop_test_server.socket.socket")
    def test_local_source_ip_uses_route_selected_address(self, mocked_socket, mocked_getaddrinfo) -> None:
        mocked_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP, "", ("192.0.2.20", 80))
        ]
        route_socket = mocked_socket.return_value.__enter__.return_value
        route_socket.getsockname.return_value = ("192.0.2.10", 54321)
        self.assertEqual(local_source_ip("http://camera.local"), "192.0.2.10")
        route_socket.connect.assert_called_once_with(("192.0.2.20", 80))

    @patch("server.laptop_test_server.socket.getaddrinfo", side_effect=OSError("DNS unavailable"))
    def test_local_source_ip_failure_is_nonfatal(self, _mocked_getaddrinfo) -> None:
        self.assertIsNone(local_source_ip("http://camera.invalid"))
        self.assertIsNone(local_source_ip(None))

    def test_port_preflight_without_fixed_port(self) -> None:
        preflight_port("127.0.0.1", 0)

    def test_port_preflight_has_actionable_duplicate_port_error(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(
                RuntimeError, rf"Get-NetTCPConnection -LocalPort {port} -State Listen"
            ):
                preflight_port("127.0.0.1", port)

    def test_create_app_rejects_invalid_camera_url_without_network_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "query string"):
                create_laptop_app(Path(temporary), "http://camera.local?bad=1")

    @patch("server.laptop_test_server.urlopen")
    def test_opt_in_startup_probe_is_synchronous_and_nonfatal(self, mocked_urlopen) -> None:
        mocked_urlopen.side_effect = URLError("camera is starting")
        with tempfile.TemporaryDirectory() as temporary:
            app = create_laptop_app(
                Path(temporary), "http://camera.local", probe_camera=True
            )
            app.testing = True
            info = app.test_client().get("/api/info").get_json()
        requested = mocked_urlopen.call_args.args[0]
        self.assertEqual(requested.full_url, "http://camera.local/status")
        self.assertEqual(mocked_urlopen.call_args.kwargs["timeout"], 3)
        self.assertFalse(info["camera_reachable"])
        self.assertIn("camera is starting", info["camera_error"])


class RaspberryPiServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.save_dir = Path(self.temporary.name)
        self.app = create_pi_app(self.save_dir)
        self.app.testing = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @patch(
        "server.raspberry_pi_server.classify",
        return_value={"is_bird": True, "label": "sparrow", "confidence": 0.92},
    )
    def test_upload_preserves_classification_contract(self, mocked_classify) -> None:
        response = self.client.post(
            "/upload",
            data=JPEG,
            content_type="image/jpeg",
            headers={METADATA_HEADER: json.dumps({**CAMERA_METADATA, "source": "motion"})},
        )
        self.assertEqual(response.status_code, 200)
        record = response.get_json()
        self.assertTrue(record["classification"]["is_bird"])
        self.assertEqual(record["classification"]["label"], "sparrow")
        mocked_classify.assert_called_once_with(self.save_dir / record["filename"])
        health = self.client.get("/health").get_json()
        self.assertEqual(health["latest_image"], record["filename"])
        self.assertEqual(health["classification"]["confidence"], 0.92)

    @patch(
        "server.raspberry_pi_server.classify",
        return_value={"is_bird": True, "label": "sparrow", "confidence": 0.92},
    )
    def test_deferred_upload_returns_before_classification(self, mocked_classify) -> None:
        response = self.client.post(
            "/upload",
            data=JPEG,
            content_type="image/jpeg",
            headers={"X-ESP32-Classification": "deferred"},
        )
        self.assertEqual(response.status_code, 202)
        record = response.get_json()
        self.assertEqual(record["status"], "queued")
        self.assertIsNone(record["classification"])
        self.assertTrue(record["classification_deferred"])
        self.assertTrue((self.save_dir / record["filename"]).exists())
        self.app.config["CLASSIFICATION_QUEUE"].join()
        mocked_classify.assert_called_once_with(self.save_dir / record["filename"])
        health = self.client.get("/health").get_json()
        self.assertEqual(health["classification"]["label"], "sparrow")
        self.assertEqual(health["classification_queue_depth"], 0)

    @patch(
        "server.raspberry_pi_server.classify",
        return_value={"is_bird": False, "label": "squirrel", "confidence": 0.97},
    )
    def test_upload_returns_explicit_non_bird_decision(self, _mocked_classify) -> None:
        response = self.client.post("/upload", data=JPEG, content_type="image/jpeg")
        self.assertEqual(response.status_code, 200)
        classification = response.get_json()["classification"]
        self.assertFalse(classification["is_bird"])
        self.assertEqual(classification["confidence"], 0.97)

    @patch("server.raspberry_pi_server.classify", return_value={"label": "sparrow", "confidence": 0.92})
    def test_invalid_classifier_result_is_not_actionable(self, _mocked_classify) -> None:
        response = self.client.post("/upload", data=JPEG, content_type="image/jpeg")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIsNone(result["classification"])
        self.assertIn("is_bird", result["classification_error"])

    @patch("server.raspberry_pi_server.classify", side_effect=RuntimeError("model offline"))
    def test_classifier_failure_is_not_actionable(self, _mocked_classify) -> None:
        response = self.client.post("/upload", data=JPEG, content_type="image/jpeg")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIsNone(result["classification"])
        self.assertEqual(result["classification_error"], "model offline")

    def test_classification_validation_rejects_unsafe_values(self) -> None:
        for value in (
            True,
            {"is_bird": 1, "confidence": 0.9},
            {"is_bird": True, "confidence": True},
            {"is_bird": True, "confidence": 1.1},
            {"is_bird": True, "confidence": 0.9, "label": 4},
        ):
            with self.subTest(value=value):
                classification, error = validate_classification(value)
                self.assertIsNone(classification)
                self.assertIsNotNone(error)

    def test_invalid_upload_is_rejected(self) -> None:
        self.assertEqual(self.client.post("/upload", data=b"bad").status_code, 415)
        self.assertEqual(self.client.get("/latest.jpg").status_code, 404)

    def test_dashboard_has_roi_controls_and_unconfigured_proxy_is_503(self) -> None:
        dashboard = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="roi-overlay"', dashboard)
        self.assertIn("pointerdown", dashboard)
        self.assertIn("Load current ROI", dashboard)
        self.assertIn("Reset full frame", dashboard)
        self.assertIn("Save ROI", dashboard)
        self.assertEqual(self.client.get("/camera/capture").status_code, 503)
        health = self.client.get("/health").get_json()
        self.assertFalse(health["camera_configured"])

    @patch("server.raspberry_pi_server.urlopen")
    def test_optional_camera_proxies_preview_config_status_and_mode(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = FakeUpstream(b'{"status":"ok"}')
        app = create_pi_app(self.save_dir / "configured", "HTTP://Camera.Local/")
        app.testing = True
        client = app.test_client()

        self.assertEqual(client.get("/camera/capture").status_code, 200)
        self.assertEqual(client.get("/camera/status").status_code, 200)
        self.assertEqual(client.get("/camera/config").status_code, 200)
        roi = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
        self.assertEqual(client.post("/camera/config", json={"motion_roi": roi}).status_code, 200)
        requested = mocked_urlopen.call_args.args[0]
        self.assertEqual(requested.full_url, "http://camera.local/config")
        self.assertEqual(requested.method, "POST")
        self.assertEqual(json.loads(requested.data), {"motion_roi": roi})
        self.assertEqual(client.post("/camera/mode", json={"mode": "motion"}).status_code, 200)
        requested = mocked_urlopen.call_args.args[0]
        self.assertEqual(requested.full_url, "http://camera.local/mode")
        self.assertEqual(json.loads(requested.data), {"mode": "motion"})
        health = client.get("/health").get_json()
        self.assertTrue(health["camera_configured"])
        self.assertEqual(health["camera_url"], "http://camera.local")

    def test_pi_camera_routes_validate_json_and_mode(self) -> None:
        self.assertEqual(
            self.client.post("/camera/config", data="bad", content_type="application/json").status_code,
            400,
        )
        self.assertEqual(self.client.post("/camera/mode", json={"mode": "stream"}).status_code, 422)

    def test_inference_engine_persists_result_and_exposes_polling_api(self) -> None:
        class FakeEngine:
            def infer(self, image_path: Path) -> dict:
                return {
                    "schema_version": 1,
                    "status": "complete",
                    "image": {"filename": image_path.name, "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()},
                    "is_bird": True,
                    "bird_probability": 0.88,
                    "bird_threshold": 0.8,
                    "guess": "sparrow",
                    "species_probability": 0.7,
                    "species_top_k": [{"label": "sparrow", "probability": 0.7}],
                    "classification": {"is_bird": True, "confidence": 0.88, "label": "sparrow"},
                }

            def describe(self) -> dict:
                return {"ready": True, "bird_threshold": 0.8}

        app = create_pi_app(self.save_dir / "models", inference_engine=FakeEngine())
        app.testing = True
        client = app.test_client()
        response = client.post("/upload", data=JPEG, content_type="image/jpeg")
        self.assertEqual(response.status_code, 200)
        capture_id = response.get_json()["capture_id"]
        inference_path = self.save_dir / "models" / f"{capture_id}.inference.json"
        self.assertTrue(inference_path.is_file())
        inference = json.loads(inference_path.read_text(encoding="utf-8"))
        self.assertEqual(inference["guess"], "sparrow")

        listing = client.get("/api/captures?limit=1").get_json()
        self.assertEqual(listing["captures"][0]["capture_id"], capture_id)
        self.assertEqual(listing["captures"][0]["inference_status"], "complete")
        image_response = client.get(f"/api/captures/{capture_id}/image")
        try:
            self.assertEqual(image_response.data, JPEG)
        finally:
            image_response.close()
        self.assertEqual(
            client.get(f"/api/captures/{capture_id}/inference").get_json()["bird_probability"],
            0.88,
        )
        health = client.get("/health").get_json()
        self.assertTrue(health["models"]["ready"])
        self.assertEqual(health["latest_inference"]["guess"], "sparrow")


if __name__ == "__main__":
    unittest.main()