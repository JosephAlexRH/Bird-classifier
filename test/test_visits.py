"""Visit grouping: the batch rule, live tracking, the automation decision and the rebuild CLI."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from server.capture_storage import CaptureStore
from server.pi_motion import MotionService
from server.visits import (
    CaptureRef,
    VisitTracker,
    _main,
    assign_visits,
    capture_time,
    decide,
    rebuild,
)

START = datetime(2026, 11, 17, 14, 10, 0, tzinfo=timezone.utc)


def cid(seconds: float, suffix: str = "0000aaaa") -> str:
    when = START + timedelta(seconds=seconds)
    return f"capture_{when:%Y%m%dT%H%M%S}_{when:%f}Z_{suffix}"


def ref(seconds: float, camera="esp32cam-a", deployment="garden", boot="boot-1", suffix="0000aaaa") -> CaptureRef:
    return CaptureRef(cid(seconds, suffix), camera, deployment, boot)


AUTOMATION = {
    "species_min_confidence": 0.8,
    "group_min_confidence": 0.7,
    "auto_species": ["House Sparrow", "Mourning Dove"],
    "groups": {"sparrow-like": ["House Sparrow", "Song Sparrow"], "dove/pigeon": ["Mourning Dove"]},
}
LABELS = ["House Sparrow", "Mourning Dove", "Song Sparrow"]


def species_inference(probabilities):
    return {"status": "complete", "species": {"labels": LABELS, "boxes": [{"probabilities": probabilities}]}}


class CaptureTimeTests(unittest.TestCase):
    def test_parses_capture_id_to_utc(self):
        when = capture_time("capture_20261117T141007_411624Z_107653df")
        self.assertEqual(when, datetime(2026, 11, 17, 14, 10, 7, 411624, tzinfo=timezone.utc))

    def test_rejects_other_names(self):
        with self.assertRaises(ValueError):
            capture_time("IMG_0001")


class AssignVisitsTests(unittest.TestCase):
    def test_gap_splits_visits_and_ids_use_first_capture(self):
        captures = [ref(0), ref(30), ref(140), ref(400)]
        mapping = assign_visits(captures, gap_seconds=120)
        first = f"esp32cam-a/{cid(0)}"
        self.assertEqual(mapping[cid(0)], first)
        self.assertEqual(mapping[cid(30)], first)
        self.assertEqual(mapping[cid(140)], first)  # 110 s after the previous capture
        self.assertEqual(mapping[cid(400)], f"esp32cam-a/{cid(400)}")

    def test_input_order_does_not_matter(self):
        captures = [ref(0), ref(30), ref(500), ref(510)]
        self.assertEqual(assign_visits(captures), assign_visits(list(reversed(captures))))

    def test_cameras_deployments_and_reboots_start_new_visits(self):
        captures = [
            ref(0), ref(10, camera="esp32cam-b", suffix="0000bbbb"),
            ref(20, deployment="garden-2"), ref(30, deployment="garden-2", boot="boot-2"),
        ]
        mapping = assign_visits(captures)
        self.assertEqual(len(set(mapping.values())), 4)
        self.assertTrue(mapping[cid(10, "0000bbbb")].startswith("esp32cam-b/"))

    def test_missing_camera_id_is_grouped_as_legacy(self):
        mapping = assign_visits([CaptureRef(cid(0)), CaptureRef(cid(5))])
        self.assertEqual(set(mapping.values()), {f"legacy/{cid(0)}"})

    def test_gap_must_be_positive(self):
        with self.assertRaises(ValueError):
            assign_visits([ref(0)], gap_seconds=0)


class DecideTests(unittest.TestCase):
    def test_confident_auto_species_is_labelled(self):
        result = decide([0.9, 0.05, 0.05], LABELS, AUTOMATION)
        self.assertEqual((result["decision"], result["label"]), ("species", "House Sparrow"))

    def test_uncertain_species_falls_back_to_group(self):
        result = decide([0.45, 0.1, 0.45], LABELS, AUTOMATION)
        self.assertEqual((result["decision"], result["label"]), ("group", "sparrow-like"))
        self.assertAlmostEqual(result["group_confidence"], 0.9)

    def test_species_outside_auto_list_is_never_species_labelled(self):
        result = decide([0.05, 0.05, 0.9], LABELS, AUTOMATION)
        self.assertEqual(result["decision"], "group")
        self.assertEqual(result["species"], "Song Sparrow")

    def test_unsure_goes_to_review(self):
        result = decide([0.4, 0.35, 0.25], LABELS, AUTOMATION)
        self.assertEqual(result["decision"], "review")
        self.assertIsNone(result["label"])

    def test_without_automation_everything_is_reviewed(self):
        self.assertEqual(decide([1.0, 0.0, 0.0], LABELS, None)["decision"], "review")


class VisitTrackerTests(unittest.TestCase):
    def test_live_ids_match_batch_rule_and_closed_visit_is_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "visits.jsonl"
            tracker = VisitTracker(log, gap_seconds=120, automation=AUTOMATION)
            captures = [ref(0), ref(20), ref(400)]
            blocks = [tracker.observe(c, species_inference([0.9, 0.1, 0.0])) for c in captures]
            self.assertEqual([b["visit_id"] for b in blocks], [assign_visits(captures)[c.capture_id] for c in captures])
            self.assertTrue(all(b["provisional"] for b in blocks))
            self.assertEqual([b["frame_index"] for b in blocks], [0, 1, 0])
            lines = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(lines), 1)  # the first visit closed when the third capture arrived
            self.assertEqual(lines[0]["captures"], 2)
            self.assertEqual(lines[0]["result"]["decision"], "species")
            self.assertEqual(lines[0]["species_mean_probabilities"], [0.9, 0.1, 0.0])

    def test_mean_over_frames_and_close_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "visits.jsonl"
            tracker = VisitTracker(log, automation=AUTOMATION)
            tracker.observe(ref(0), species_inference([1.0, 0.0, 0.0]))
            tracker.observe(ref(10), species_inference([0.6, 0.4, 0.0]))
            tracker.observe(ref(20), {"status": "error"})
            self.assertEqual(tracker.close_stale(now=START + timedelta(seconds=60)), [])
            closed = tracker.close_stale(now=START + timedelta(seconds=200))
            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0]["captures"], 3)
            self.assertEqual(closed[0]["species_frames"], 2)
            np.testing.assert_allclose(closed[0]["species_mean_probabilities"], [0.8, 0.2, 0.0])
            self.assertEqual(len(log.read_text().splitlines()), 1)

    def test_only_the_best_localised_box_feeds_the_visit(self):
        tracker = VisitTracker(None, automation=AUTOMATION)
        tracker.observe(ref(0), {"status": "complete", "species": {"labels": LABELS, "boxes": [
            {"probabilities": [0.9, 0.1, 0.0]}, {"probabilities": [0.0, 1.0, 0.0]}]}})
        self.assertEqual(tracker.close_all()[0]["species_mean_probabilities"], [0.9, 0.1, 0.0])

    def test_visit_without_species_scores_is_reviewed(self):
        tracker = VisitTracker(None)
        tracker.observe(ref(0), None)
        closed = tracker.close_all()
        self.assertEqual(closed[0]["result"]["decision"], "review")


class RebuildTests(unittest.TestCase):
    def write_capture(self, root: Path, capture: CaptureRef, probabilities=None):
        sidecar = {"capture_id": capture.capture_id, "camera_id": capture.camera_id,
                   "deployment_id": capture.deployment_id, "device": {"boot_id": capture.boot_id}}
        (root / f"{capture.capture_id}.json").write_text(json.dumps(sidecar))
        if probabilities is not None:
            (root / f"{capture.capture_id}.inference.json").write_text(json.dumps(species_inference(probabilities)))

    def test_rebuild_matches_batch_rule_and_writes_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures = [ref(0), ref(30), ref(500), ref(5, camera="esp32cam-b", suffix="0000bbbb")]
            for c in captures:
                self.write_capture(root, c, [0.95, 0.05, 0.0])
            (root / "capture_bad.json").write_text("{not json")
            visits, mapping = rebuild(root, automation=AUTOMATION)
            self.assertEqual(mapping, assign_visits(captures))
            self.assertEqual(len(visits), 3)
            self.assertTrue(all(v["source"] == "rebuild" for v in visits))
            self.assertEqual(_main(["rebuild", str(root), "--out", str(root / "out")]), 0)
            with (root / "out" / "visits.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            with (root / "out" / "capture_visits.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)


class MotionServiceVisitTests(unittest.TestCase):
    def test_persisted_birds_carry_one_provisional_visit_id(self):
        frame = np.full((120, 160, 3), 50, np.uint8)
        output = io.BytesIO()
        Image.fromarray(frame).save(output, "JPEG", quality=95)
        image = output.getvalue()
        meta = json.dumps({
            "schema_version": 5, "source": "pi_motion_poll",
            "device": {"camera_id": "esp32-aabbcc", "boot_id": "boot-1", "clock": {}},
            "camera": {"sensor": {"model": "OV5640"},
                       "requested": {"motion": {"owner": "pi", "roi": {"x": 0, "y": 0, "width": 1, "height": 1}}},
                       "observed": {"capture": {"width": 160, "height": 120}}, "timing": {}},
        })

        class Engine:
            contract = None
            bird_threshold = 0.8

            def infer(self, path, **kwargs):
                return {"status": "complete", "is_bird": True, "bird_probability": 0.9, "bird_threshold": 0.8,
                        **species_inference([0.9, 0.1, 0.0])}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracker = VisitTracker(root / "visits.jsonl", automation=AUTOMATION)
            service = MotionService("http://camera", CaptureStore(root), Engine(),
                                    fetcher=lambda *_a, **_k: (image, meta),
                                    monotonic=iter([0, 0.1]).__next__, visit_tracker=tracker)
            first, second = service.process_once(), service.process_once()
            records = [json.loads((root / f"{r['capture']['capture_id']}.inference.json").read_text())
                       for r in (first, second)]
            self.assertEqual(records[0]["visit"]["visit_id"], records[1]["visit"]["visit_id"])
            self.assertTrue(records[0]["visit"]["visit_id"].startswith("esp32-aabbcc/capture_"))
            closed = tracker.close_all()
            self.assertEqual(closed[0]["captures"], 2)
            self.assertEqual(closed[0]["result"]["label"], "House Sparrow")
            sidecar = json.loads((root / f"{first['capture']['capture_id']}.json").read_text())
            self.assertNotIn("visit", sidecar)  # visits never touch acquisition records


if __name__ == "__main__":
    unittest.main()
