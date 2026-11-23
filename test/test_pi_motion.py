from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from server.bird_inference import SELECTED_CROP_GEOMETRY
from server.capture_storage import CaptureStore
from server.motion_crop import MotionConfig
from server.pi_motion import MotionService


def jpeg(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array).save(output, "JPEG", quality=95)
    return output.getvalue()


def metadata(roi=None):
    return json.dumps({
        "schema_version": 5,
        "source": "pi_motion_poll",
        "device": {"camera_id": "esp32-aabbcc", "boot_id": "boot-1", "clock": {}},
        "camera": {
            "sensor": {"model": "OV5640", "pid": 0x5640, "pid_hex": "0x5640"},
            "requested": {"motion": {"owner": "pi", "roi": roi or {"x": 0, "y": 0, "width": 1, "height": 1}}},
            "observed": {"capture": {"width": 160, "height": 120}},
            "timing": {},
        },
    })


class FakeEngine:
    contract = None
    bird_threshold = 0.8

    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def infer(self, path, **kwargs):
        self.calls.append((Path(path), kwargs))
        return next(self.results)


class PiMotionTests(unittest.TestCase):
    def test_quiet_confident_non_bird_stays_memory_only_and_builds_background(self):
        frame = np.full((120, 160, 3), 50, np.uint8)
        engine = FakeEngine([{"status": "complete", "is_bird": False, "bird_probability": .1, "bird_threshold": .8}] * 3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = MotionService(
                "http://camera", CaptureStore(root), engine,
                config=MotionConfig(minimum_background_frames=3, admission_seconds=0),
                fetcher=lambda *_a, **_k: (jpeg(frame), metadata()),
                monotonic=iter([0, 1, 2, 3]).__next__,
            )
            results = [service.process_once() for _ in range(3)]
            self.assertTrue(all(not result["persisted"] for result in results))
            self.assertTrue(service.background.ready)
            self.assertEqual(list(root.glob("capture_*.jpg")), [])

    def test_clear_blob_is_persisted_with_motion_and_inference_records(self):
        quiet = np.zeros((120, 160, 3), np.uint8)
        changed = quiet.copy()
        changed[40:75, 60:95] = 255
        frames = iter([quiet, quiet, quiet, changed])
        engine = FakeEngine([{"status": "complete", "is_bird": False, "bird_probability": .1, "bird_threshold": .8}] * 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            persisted = []
            service = MotionService(
                "http://camera", CaptureStore(root), engine,
                config=MotionConfig(minimum_background_frames=3, admission_seconds=0, minimum_component_pixels=20),
                fetcher=lambda *_a, **_k: (jpeg(next(frames)), metadata()),
                monotonic=iter([0, 1, 2, 3]).__next__,
                on_persist=lambda path, capture, inference: persisted.append((path, capture, inference)),
            )
            for _ in range(3):
                service.process_once()
            result = service.process_once()
            self.assertTrue(result["persisted"])
            capture = result["capture"]
            self.assertTrue((root / capture["filename"]).is_file())
            self.assertTrue((root / f"{capture['capture_id']}.motion.json").is_file())
            self.assertTrue((root / f"{capture['capture_id']}.inference.json").is_file())
            motion = json.loads((root / f"{capture['capture_id']}.motion.json").read_text())
            inference = json.loads((root / f"{capture['capture_id']}.inference.json").read_text())
            self.assertEqual(capture["camera_model"], "OV5640")
            self.assertEqual(motion["camera_provenance"]["model"], "OV5640")
            self.assertEqual(motion["camera_provenance"]["sensor_pid"], 0x5640)
            self.assertEqual(inference["camera_provenance"]["model"], "OV5640")
            self.assertEqual(persisted[0][0], root / capture["filename"])

    def test_bird_static_fallback_is_persisted_but_not_admitted(self):
        frame = np.full((120, 160, 3), 50, np.uint8)
        engine = FakeEngine([{"status": "complete", "is_bird": True, "bird_probability": .9, "bird_threshold": .8}])
        with tempfile.TemporaryDirectory() as temporary:
            service = MotionService(
                "http://camera", CaptureStore(Path(temporary)), engine,
                fetcher=lambda *_a, **_k: (jpeg(frame), metadata()), monotonic=lambda: 0,
            )
            result = service.process_once()
            self.assertTrue(result["persisted"])
            self.assertFalse(result["background_admitted"])

    def test_birds_bypass_event_persistence_throttle(self):
        frame = np.full((120, 160, 3), 50, np.uint8)
        engine = FakeEngine([
            {"status": "complete", "is_bird": True, "bird_probability": .9, "bird_threshold": .8},
            {"status": "complete", "is_bird": True, "bird_probability": .9, "bird_threshold": .8},
        ])
        with tempfile.TemporaryDirectory() as temporary:
            service = MotionService(
                "http://camera", CaptureStore(Path(temporary)), engine,
                fetcher=lambda *_a, **_k: (jpeg(frame), metadata()),
                monotonic=iter([0, .1]).__next__, minimum_persist_seconds=10,
            )
            self.assertTrue(service.process_once()["persisted"])
            self.assertTrue(service.process_once()["persisted"])

    def test_clear_blob_survives_inference_failure_with_durable_error(self):
        quiet = np.zeros((120, 160, 3), np.uint8)
        changed = quiet.copy(); changed[40:75, 60:95] = 255
        frames = iter([quiet, quiet, quiet, changed])

        class FailingAfterWarmup(FakeEngine):
            def infer(self, path, **kwargs):
                if len(self.calls) == 3:
                    self.calls.append((Path(path), kwargs))
                    raise RuntimeError("model unavailable")
                return super().infer(path, **kwargs)

        engine = FailingAfterWarmup([
            {"status": "complete", "is_bird": False, "bird_probability": .1, "bird_threshold": .8}
        ] * 3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = MotionService(
                "http://camera", CaptureStore(root), engine,
                config=MotionConfig(minimum_background_frames=3, admission_seconds=0, minimum_component_pixels=20),
                fetcher=lambda *_a, **_k: (jpeg(next(frames)), metadata()),
                monotonic=iter([0, 1, 2, 3]).__next__,
            )
            for _ in range(3): service.process_once()
            result = service.process_once()
            self.assertTrue(result["persisted"])
            self.assertEqual(result["inference"]["status"], "error")
            capture = result["capture"]
            saved = json.loads((root / f"{capture['capture_id']}.inference.json").read_text())
            self.assertIn("model unavailable", saved["error"])
            motion = json.loads((root / f"{capture['capture_id']}.motion.json").read_text())
            self.assertEqual(motion["static_roi"]["capture_sidecar"], capture["sidecar"])
            self.assertEqual(len(motion["static_roi"]["capture_sidecar_sha256"]), 64)
            self.assertEqual(len(motion["crop_id"]), 64)

    def test_continuing_event_is_persisted_at_most_once_per_second(self):
        quiet = np.zeros((120, 160, 3), np.uint8)
        changed = quiet.copy(); changed[40:75, 60:95] = 255
        frames = iter([quiet, quiet, quiet, changed, changed])
        engine = FakeEngine([{"status": "complete", "is_bird": False, "bird_probability": .1, "bird_threshold": .8}] * 5)
        with tempfile.TemporaryDirectory() as temporary:
            service = MotionService(
                "http://camera", CaptureStore(Path(temporary)), engine,
                config=MotionConfig(minimum_background_frames=3, admission_seconds=0, minimum_component_pixels=20),
                fetcher=lambda *_a, **_k: (jpeg(next(frames)), metadata()),
                monotonic=iter([0, 1, 2, 3, 3.25]).__next__, minimum_persist_seconds=1,
            )
            results = [service.process_once() for _ in range(5)]
            self.assertTrue(results[3]["persisted"])
            self.assertFalse(results[4]["persisted"])

    def test_restart_identity_resets_background(self):
        frame = np.full((120, 160, 3), 50, np.uint8)
        headers = iter([metadata(), metadata(), json.dumps({**json.loads(metadata()), "device": {"camera_id": "esp32-aabbcc", "boot_id": "boot-2", "clock": {}}})])
        engine = FakeEngine([{"status": "complete", "is_bird": False, "bird_probability": .1, "bird_threshold": .8}] * 3)
        with tempfile.TemporaryDirectory() as temporary:
            service = MotionService(
                "http://camera", CaptureStore(Path(temporary)), engine,
                config=MotionConfig(minimum_background_frames=2, admission_seconds=0),
                fetcher=lambda *_a, **_k: (jpeg(frame), next(headers)),
                monotonic=iter([0, 1, 2]).__next__,
            )
            service.process_once(); service.process_once()
            epoch = service.background.epoch
            service.process_once()
            self.assertNotEqual(service.background.epoch, epoch)
            self.assertEqual(service.background.sample_count, 1)

    def test_firmware_owned_frame_is_rejected(self):
        frame = np.full((120, 160, 3), 50, np.uint8)
        value = json.loads(metadata())
        value["camera"]["requested"]["motion"]["owner"] = "firmware"
        engine = FakeEngine([{"status": "complete", "is_bird": False, "bird_probability": .1}])
        with tempfile.TemporaryDirectory() as temporary:
            service = MotionService(
                "http://camera", CaptureStore(Path(temporary)), engine,
                fetcher=lambda *_a, **_k: (jpeg(frame), json.dumps(value)),
            )
            with self.assertRaisesRegex(ValueError, "motion_owner must be pi"):
                service.process_once()

    def test_expected_hardware_identity_rejects_reassigned_ip(self):
        frame = np.full((120, 160, 3), 50, np.uint8)
        engine = FakeEngine([{"status": "complete", "is_bird": False, "bird_probability": .1}])
        with tempfile.TemporaryDirectory() as temporary:
            service = MotionService(
                "http://camera",
                CaptureStore(Path(temporary)),
                engine,
                expected_camera_id="esp32-other",
                fetcher=lambda *_a, **_k: (jpeg(frame), metadata()),
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                service.process_once()


class StaticBoxEngine(FakeEngine):
    supports_static_boxes = True
    contract = SimpleNamespace(geometry=SELECTED_CROP_GEOMETRY)


def detection(left, top, right, bottom, score):
    return {"box": {"left": left, "top": top, "right": right, "bottom": bottom}, "score": score}


class StaticBoxServiceTests(unittest.TestCase):
    def test_empty_references_feed_the_box_filter_for_later_bird_frames(self):
        frame = np.full((120, 160, 3), 50, np.uint8)
        empty = {"status": "complete", "is_bird": False, "bird_probability": .05, "bird_threshold": .8,
                 "localizer": {"role": "empty_reference", "detections": [detection(10, 10, 40, 40, .6)]}}
        bird = {"status": "complete", "is_bird": True, "bird_probability": .95, "bird_threshold": .8}
        engine = StaticBoxEngine([empty, bird])
        with tempfile.TemporaryDirectory() as temporary:
            service = MotionService(
                "http://camera", CaptureStore(Path(temporary)), engine,
                fetcher=lambda *_a, **_k: (jpeg(frame), metadata()), monotonic=lambda: 0,
            )
            service.process_once()
            service.process_once()
        (_, first), (_, second) = engine.calls
        self.assertEqual(first["empty_reference_crop"], {"left": 0, "top": 0, "right": 160, "bottom": 120})
        self.assertIn("box_filter", first)
        self.assertNotIn("empty_reference_crop", second)  # the next reference is not due for a minute
        boxes, summary = second["box_filter"]([detection(12, 10, 42, 40, .62), detection(100, 60, 130, 90, .9)])
        self.assertEqual([b["static"] for b in boxes], [True, False])
        self.assertEqual(summary["reference_frames"], 1)

    def test_filter_can_be_switched_off_and_plain_engines_are_untouched(self):
        frame = np.full((120, 160, 3), 50, np.uint8)
        result = {"status": "complete", "is_bird": True, "bird_probability": .95, "bird_threshold": .8}
        with tempfile.TemporaryDirectory() as temporary:
            off = StaticBoxEngine([result])
            MotionService("http://camera", CaptureStore(Path(temporary)), off, static_box_filter=False,
                          fetcher=lambda *_a, **_k: (jpeg(frame), metadata()), monotonic=lambda: 0).process_once()
            plain = FakeEngine([result])
            plain.contract = SimpleNamespace(geometry=SELECTED_CROP_GEOMETRY)
            MotionService("http://camera", CaptureStore(Path(temporary)), plain,
                          fetcher=lambda *_a, **_k: (jpeg(frame), metadata()), monotonic=lambda: 0).process_once()
        for engine in (off, plain):
            kwargs = engine.calls[0][1]
            self.assertNotIn("box_filter", kwargs)
            self.assertNotIn("empty_reference_crop", kwargs)


if __name__ == "__main__":
    unittest.main()