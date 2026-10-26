from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from server.motion_crop import (
    MotionConfig,
    RunningMedianBackground,
    final_crop_bounds,
    select_motion_crop,
    should_admit_background,
)


class MotionCropTests(unittest.TestCase):
    def test_running_median_warms_resets_and_rate_limits(self) -> None:
        config = MotionConfig(background_frames=3, minimum_background_frames=3, admission_seconds=2)
        model = RunningMedianBackground(config)
        self.assertTrue(model.admit(np.full((4, 5), 10, np.uint8), now=0))
        self.assertFalse(model.admit(np.full((4, 5), 20, np.uint8), now=1))
        self.assertTrue(model.admit(np.full((4, 5), 20, np.uint8), now=2))
        self.assertTrue(model.admit(np.full((4, 5), 30, np.uint8), now=4))
        np.testing.assert_array_equal(model.median(), 20)
        old_epoch = model.epoch
        model.admit(np.zeros((3, 3), np.uint8), now=6)
        self.assertNotEqual(model.epoch, old_epoch)
        self.assertEqual(model.sample_count, 1)

    def test_even_warmup_uses_newest_odd_samples(self) -> None:
        model = RunningMedianBackground(MotionConfig(background_frames=5, minimum_background_frames=3, admission_seconds=0))
        for index, value in enumerate((0, 10, 20, 200)):
            model.admit(np.full((2, 2), value, np.uint8), now=index)
        np.testing.assert_array_equal(model.median(), 20)

    def test_minimum_crop_shifts_at_edges(self) -> None:
        self.assertEqual(final_crop_bounds((0, 0, 20, 20), 640, 480), (0, 0, 160, 160))
        self.assertEqual(final_crop_bounds((630, 470, 640, 480), 640, 480), (480, 320, 640, 480))
        self.assertEqual(final_crop_bounds((0, 0, 20, 20), 100, 80), (0, 0, 100, 80))

    def test_clear_largest_blob_is_selected_and_padded_once(self) -> None:
        config = MotionConfig(minimum_component_pixels=20, minimum_crop_size=40)
        background = np.zeros((120, 160), np.uint8)
        current = background.copy()
        current[40:70, 60:90] = 255
        result = select_motion_crop(current, background, {"x": 0, "y": 0, "width": 1, "height": 1}, config)
        self.assertEqual(result["selection"]["source"], "dynamic_motion_blob")
        bounds = result["selection"]["final_bounds_px"]
        self.assertGreaterEqual(bounds["width"], 40)
        self.assertGreaterEqual(bounds["height"], 40)
        self.assertLessEqual(bounds["left"], 60)
        self.assertGreaterEqual(bounds["right"], 90)

    def test_uniform_brightness_shift_falls_back_to_static_roi(self) -> None:
        background = np.full((120, 160), 80, np.uint8)
        current = np.full((120, 160), 110, np.uint8)
        result = select_motion_crop(current, background, {"x": .25, "y": .25, "width": .5, "height": .5})
        self.assertEqual(result["selection"]["source"], "static_roi_fallback")
        self.assertEqual(result["selection"]["fallback_reason"], "no_changed_blob")

    def test_ambiguous_equal_blobs_fall_back(self) -> None:
        config = MotionConfig(minimum_component_pixels=10, minimum_crop_size=20)
        background = np.zeros((100, 100), np.uint8)
        current = background.copy()
        current[20:35, 20:35] = 255
        current[60:75, 60:75] = 255
        result = select_motion_crop(current, background, {"x": 0, "y": 0, "width": 1, "height": 1}, config)
        self.assertEqual(result["selection"]["fallback_reason"], "ambiguous_blob")

    def test_numpy_fallback_path(self) -> None:
        background = np.zeros((40, 50), np.uint8)
        with patch("server.motion_crop._cv2", None):
            result = select_motion_crop(background, background, {"x": 0, "y": 0, "width": 1, "height": 1}, MotionConfig(minimum_crop_size=20))
        self.assertEqual(result["selection"]["source"], "static_roi_fallback")

    def test_background_admission_requires_confident_complete_not_bird(self) -> None:
        self.assertTrue(should_admit_background({"status": "complete", "is_bird": False, "bird_probability": .1}, threshold=.8))
        for value in (
            {"status": "complete", "is_bird": True, "bird_probability": .01},
            {"status": "error", "is_bird": False, "bird_probability": .01},
            {"status": "complete", "is_bird": False, "bird_probability": .3},
        ):
            self.assertFalse(should_admit_background(value, threshold=.8))


if __name__ == "__main__":
    unittest.main()