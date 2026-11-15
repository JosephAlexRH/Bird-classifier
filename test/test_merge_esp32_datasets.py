"""Tests for merging the legacy and newer ESP32 annotation CSVs."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_esp32_datasets.py"
SPEC = importlib.util.spec_from_file_location("merge_esp32_datasets", SCRIPT)
assert SPEC and SPEC.loader
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


class MergeEsp32DatasetsTests(unittest.TestCase):
    @staticmethod
    def legacy_rows() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "record_id": "capture_20261020T120000_000000Z_bird",
                    "capture_id": "capture_20261020T120000_000000Z_bird",
                    "image_relative_path": "bird.jpg",
                    "image_path": "/mnt/e/Datasets/ESP32_capture/bird.jpg",
                    "width_px": "160",
                    "height_px": "120",
                    "bird_present": "True",
                    "species": "Blue Jay",
                },
                {
                    "record_id": "capture_20261020T120001_000000Z_empty",
                    "capture_id": "capture_20261020T120001_000000Z_empty",
                    "image_relative_path": "empty.jpg",
                    "image_path": "/mnt/e/Datasets/ESP32_capture/empty.jpg",
                    "width_px": "160",
                    "height_px": "120",
                    "bird_present": "False",
                    "species": None,
                },
            ]
        )

    def test_legacy_positive_needs_box_and_negative_is_trainable(self) -> None:
        converted = merge.convert_capture_manifest(self.legacy_rows())

        bird = converted.iloc[0]
        self.assertEqual(bird["review_state"], "skipped")
        self.assertTrue(bool(bird["is_Bird"]))
        self.assertEqual(bird["species_name"], "Blue Jay")
        self.assertFalse(bird["eligible_for_training"])
        self.assertTrue(pd.isna(bird["xmin"]))

        empty = converted.iloc[1]
        self.assertEqual(empty["review_state"], "no_bird")
        self.assertFalse(bool(empty["is_Bird"]))
        self.assertTrue(empty["eligible_for_training"])
        self.assertEqual(empty["bird_count"], 0)

    def test_newer_sample_replaces_legacy_and_preserves_new_context(self) -> None:
        old = self.legacy_rows()
        new = pd.DataFrame(
            [
                {
                    "annotation_id": "capture_20261020T120000_000000Z_bird:0",
                    "sample_id": "capture_20261020T120000_000000Z_bird",
                    "source_dataset": "image_directory",
                    "source_image_id": "capture_20261020T120000_000000Z_bird",
                    "image_path": r"C:\images\bird.jpg",
                    "relative_path": "bird.jpg",
                    "width_px": "640",
                    "height_px": "480",
                    "review_state": "boxed",
                    "annotation_trust": "human_verified",
                    "annotation_origin": "combined_label_bbox_editor",
                    "eligible_for_training": None,
                    "xmin": "10",
                    "ymin": "20",
                    "xmax": "110",
                    "ymax": "220",
                    "is_Bird": "True",
                    "bird_count": "1",
                    "context_percent": "20",
                    "context_xmin": "0",
                    "context_ymin": "0",
                    "context_xmax": "120",
                    "context_ymax": "240",
                    "split_group": None,
                }
            ]
        )

        combined, replaced = merge.merge_datasets(old, new)

        self.assertEqual(replaced, 1)
        self.assertEqual(len(combined), 2)
        authoritative = combined.loc[
            combined["sample_id"].eq("capture_20261020T120000_000000Z_bird")
        ]
        self.assertEqual(len(authoritative), 1)
        self.assertEqual(authoritative.iloc[0]["source_dataset"], "image_directory")
        self.assertEqual(authoritative.iloc[0]["context_percent"], 20)
        self.assertEqual(authoritative.iloc[0]["context_xmax"], 120)
        self.assertTrue(authoritative.iloc[0]["eligible_for_training"])
        merge.validate_merged(combined)


if __name__ == "__main__":
    unittest.main()