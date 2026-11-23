"""Static-box memory: localiser boxes that also appear on confidently empty frames are scenery."""

from __future__ import annotations

import unittest

from server.static_boxes import StaticBoxMemory, box_containment, box_iou


def det(left, top, right, bottom, score):
    return {"box": {"left": left, "top": top, "right": right, "bottom": bottom}, "score": score}


class Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class StaticBoxMemoryTests(unittest.TestCase):
    def test_iou(self):
        a = det(0, 0, 10, 10, 1)["box"]
        self.assertAlmostEqual(box_iou(a, a), 1.0)
        self.assertAlmostEqual(box_iou(a, det(5, 0, 15, 10, 1)["box"]), 50 / 150)
        self.assertEqual(box_iou(a, det(20, 20, 30, 30, 1)["box"]), 0.0)

    def test_containment(self):
        outer = det(0, 0, 100, 100, 1)["box"]
        self.assertEqual(box_containment(det(10, 10, 30, 30, 1)["box"], outer), 1.0)
        self.assertAlmostEqual(box_containment(det(90, 0, 110, 10, 1)["box"], outer), 0.5)

    def test_a_small_box_inside_a_scenery_box_is_static(self):
        memory = StaticBoxMemory()
        memory.remember("cam", [det(398, 311, 493, 388, 0.35)])  # the shrub, Nov 20
        boxes, _ = memory.annotate("cam", [det(426, 334, 486, 392, 0.15), det(300, 300, 340, 340, 0.15)])
        self.assertEqual([b["static"] for b in boxes], [True, False])  # IoU 0.43 but 93% inside

    def test_without_references_nothing_is_flagged(self):
        memory = StaticBoxMemory()
        boxes, summary = memory.annotate("cam", [det(0, 0, 10, 10, 0.9)])
        self.assertFalse(summary["applied"])
        self.assertFalse(boxes[0]["static"])
        self.assertEqual(summary["suppressed"], 0)

    def test_box_on_an_empty_scene_site_is_static_and_others_are_kept(self):
        clock = Clock()
        memory = StaticBoxMemory(clock=clock)
        self.assertEqual(memory.remember("cam", [det(0, 0, 10, 10, 0.6), det(80, 80, 90, 90, 0.01)]), 2)
        clock.now = 30
        boxes, summary = memory.annotate("cam", [det(1, 0, 11, 10, 0.65), det(50, 50, 60, 60, 0.9)])
        self.assertTrue(boxes[0]["static"])
        self.assertAlmostEqual(boxes[0]["static_match"]["empty_score"], 0.6)
        self.assertEqual(boxes[0]["static_match"]["age_seconds"], 30)
        self.assertFalse(boxes[1]["static"])
        self.assertNotIn("static_match", boxes[1])
        self.assertEqual((summary["applied"], summary["reference_frames"], summary["suppressed"]), (True, 1, 1))

    def test_by_default_every_box_on_a_scenery_site_is_static(self):
        memory = StaticBoxMemory()
        memory.remember("cam", [det(0, 0, 10, 10, 0.3)])
        boxes, summary = memory.annotate("cam", [det(0, 0, 10, 10, 0.95)])
        self.assertTrue(boxes[0]["static"])
        self.assertIsNone(summary["score_margin"])

    def test_with_a_margin_a_clearly_stronger_box_on_a_scenery_site_is_kept(self):
        memory = StaticBoxMemory(score_margin=0.10)
        memory.remember("cam", [det(0, 0, 10, 10, 0.6)])
        boxes, _ = memory.annotate("cam", [det(0, 0, 10, 10, 0.75)])
        self.assertFalse(boxes[0]["static"])  # a bird on the site lifts the score past empty + margin
        self.assertIn("static_match", boxes[0])

    def test_matching_uses_the_strongest_overlapping_reference(self):
        memory = StaticBoxMemory(score_margin=0.10)
        memory.remember("cam", [det(0, 0, 10, 10, 0.3)])
        memory.remember("cam", [det(0, 0, 10, 10, 0.7)])
        boxes, _ = memory.annotate("cam", [det(0, 0, 10, 10, 0.75)])
        self.assertTrue(boxes[0]["static"])

    def test_references_expire_and_are_capped(self):
        clock = Clock()
        memory = StaticBoxMemory(max_age_seconds=100, max_frames=2, clock=clock)
        memory.remember("cam", [det(0, 0, 10, 10, 0.6)])
        clock.now = 101
        boxes, summary = memory.annotate("cam", [det(0, 0, 10, 10, 0.5)])
        self.assertFalse(summary["applied"])
        self.assertFalse(boxes[0]["static"])
        for _ in range(3):
            memory.remember("cam", [])
        self.assertEqual(memory.describe()["reference_frames"]["cam"], 2)

    def test_cameras_are_separate_and_due_and_clear(self):
        clock = Clock()
        memory = StaticBoxMemory(clock=clock)
        self.assertTrue(memory.due("a", 60))
        memory.remember("a", [det(0, 0, 10, 10, 0.6)])
        self.assertFalse(memory.due("a", 60))
        self.assertTrue(memory.due("b", 60))
        self.assertFalse(memory.annotate("b", [det(0, 0, 10, 10, 0.5)])[0][0]["static"])
        clock.now = 60
        self.assertTrue(memory.due("a", 60))
        memory.clear("a")
        self.assertFalse(memory.annotate("a", [det(0, 0, 10, 10, 0.5)])[1]["applied"])

    def test_inputs_are_not_mutated(self):
        memory = StaticBoxMemory()
        memory.remember("cam", [det(0, 0, 10, 10, 0.6)])
        detections = [det(0, 0, 10, 10, 0.5)]
        memory.annotate("cam", detections)
        self.assertNotIn("static", detections[0])


if __name__ == "__main__":
    unittest.main()
