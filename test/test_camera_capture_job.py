"""Tests for the standalone ESP32 environment capture job."""

from __future__ import annotations

import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from urllib.error import URLError

from server.camera_capture_job import (
    COLLECTION_PURPOSE,
    CameraCaptureError,
    capture_environment_frame,
    fetch_camera_capture,
    normalize_camera_url,
    parse_args,
    run_interval,
)
from server.capture_storage import METADATA_HEADER, CaptureStore


JPEG = b"\xff\xd8environment-frame\xff\xd9"
METADATA = {
    "schema_version": 4,
    "source": "preview",
    "device": {"clock": {"device_time_utc": None}},
    "camera": {"timing": {"frame_boot_us": 1000}},
}


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        content_type: str = "image/jpeg",
        metadata: str | None = None,
        status: int = 200,
        declared_length: int | None = None,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if metadata is not None:
            self.headers[METADATA_HEADER] = metadata
        if declared_length is not None:
            self.headers["Content-Length"] = str(declared_length)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class CameraCaptureJobTests(unittest.TestCase):
    def test_capture_saves_unreviewed_candidate_without_automatic_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CaptureStore(Path(temporary))

            def opener(request, timeout):
                self.assertEqual(request.full_url, "http://camera.local/capture")
                self.assertEqual(timeout, 4.0)
                return FakeResponse(JPEG, metadata=json.dumps(METADATA))

            record = capture_environment_frame(
                "HTTP://CAMERA.Local/", store, timeout_seconds=4.0, opener=opener
            )
            sidecar = json.loads((store.root / record["sidecar"]).read_text(encoding="utf-8"))
            manifest = [json.loads(line) for line in store.manifest_path.read_text().splitlines()]

            self.assertEqual((store.root / record["filename"]).read_bytes(), JPEG)
            self.assertEqual(record["source"], "environment")
            self.assertEqual(record["collection"]["purpose"], COLLECTION_PURPOSE)
            self.assertEqual(record["collection"]["review_status"], "unreviewed")
            self.assertEqual(record["collection"]["mode"], "one_shot")
            self.assertIsNone(record["collection"]["interval_seconds"])
            self.assertNotIn("manual_label", record)
            self.assertNotIn("manual_label", sidecar)
            self.assertEqual(manifest, [sidecar])

    def test_interval_provenance_is_stored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CaptureStore(Path(temporary))
            record = capture_environment_frame(
                "http://camera.local",
                store,
                collection_mode="interval",
                interval_seconds=1800.0,
                opener=lambda *_args, **_kwargs: FakeResponse(JPEG),
            )
            self.assertEqual(record["collection"]["mode"], "interval")
            self.assertEqual(record["collection"]["interval_seconds"], 1800.0)

    def test_malformed_metadata_is_recorded_without_losing_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CaptureStore(Path(temporary))
            record = capture_environment_frame(
                "http://camera.local",
                store,
                opener=lambda *_args, **_kwargs: FakeResponse(JPEG, metadata="not-json"),
            )
            self.assertTrue((store.root / record["filename"]).is_file())
            self.assertIn("invalid JSON", record["metadata_error"])

    def test_fetch_rejects_wrong_content_type_incomplete_and_oversized_images(self) -> None:
        cases = (
            (FakeResponse(JPEG, content_type="application/json"), "Content-Type"),
            (FakeResponse(b"not-a-jpeg"), "complete JPEG"),
            (FakeResponse(JPEG, declared_length=101), "exceeds"),
            (FakeResponse(b"\xff\xd8" + b"x" * 20 + b"\xff\xd9"), "exceeds"),
        )
        for response, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(CameraCaptureError, message):
                fetch_camera_capture(
                    "http://camera.local",
                    max_jpeg_bytes=100 if message != "exceeds" else 10,
                    opener=lambda *_args, response=response, **_kwargs: response,
                )

    def test_network_failure_has_camera_endpoint_in_error(self) -> None:
        def offline(*_args, **_kwargs):
            raise URLError("offline")

        with self.assertRaisesRegex(CameraCaptureError, "camera.local/capture.*offline"):
            fetch_camera_capture("http://camera.local", opener=offline)

    def test_monitor_frame_endpoint_uses_the_shared_bounded_fetcher(self) -> None:
        def opener(request, timeout):
            self.assertEqual(request.full_url, "http://camera.local/monitor-frame")
            self.assertEqual(timeout, 2.0)
            return FakeResponse(JPEG, metadata=json.dumps(METADATA))

        image, metadata = fetch_camera_capture(
            "http://camera.local",
            endpoint="monitor-frame",
            timeout_seconds=2.0,
            opener=opener,
        )
        self.assertEqual(image, JPEG)
        self.assertEqual(json.loads(metadata), METADATA)

    def test_fetch_rejects_unsafe_endpoint_names(self) -> None:
        for endpoint in ("", "/capture", "two/parts", "Capture", "capture?x=1"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                fetch_camera_capture("http://camera.local", endpoint=endpoint)

    def test_fixed_rate_loop_runs_immediately_and_continues_after_error(self) -> None:
        clock = FakeClock()
        attempted_at: list[float] = []
        errors: list[str] = []

        def capture():
            attempted_at.append(clock.now)
            if len(attempted_at) == 1:
                raise CameraCaptureError("temporary")
            return {"capture_id": str(len(attempted_at))}

        run_interval(
            capture,
            30.0,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            on_error=lambda error: errors.append(str(error)),
            max_attempts=3,
        )
        self.assertEqual(attempted_at, [0.0, 30.0, 60.0])
        self.assertEqual(clock.sleeps, [30.0, 30.0])
        self.assertEqual(errors, ["temporary"])

    def test_url_and_positive_cli_values_are_validated(self) -> None:
        self.assertEqual(normalize_camera_url("HTTP://CAMERA.Local/"), "http://camera.local")
        for invalid in ("camera.local", "http://camera.local/capture", "http://user@camera.local"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_camera_url(invalid)
        with self.assertRaises(SystemExit):
            parse_args(["--camera-url", "http://camera.local", "--interval-minutes", "0"])


if __name__ == "__main__":
    unittest.main()