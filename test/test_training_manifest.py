"""Tests for the bridge-to-training-manifest conversion."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "Notebooks" / "build_bird_training_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_bird_training_manifest", SCRIPT)
assert SPEC and SPEC.loader
training_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training_manifest)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TrainingManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.crosswalk = self.root / "crosswalk.csv"
        self.bridge = self.root / "bridge.csv"
        self.output = self.root / "output"
        write_csv(
            self.crosswalk,
            [
                "canonical_taxon_id",
                "match_status",
                "match_method",
                "cub_class_id",
                "inaturalist_category_id",
                "scientific_name",
                "canonical_common_name",
            ],
            [
                {
                    "canonical_taxon_id": "inaturalist:20",
                    "match_status": "shared_taxon",
                    "match_method": "normalized_common_name",
                    "cub_class_id": "2",
                    "inaturalist_category_id": "20",
                    "scientific_name": "Avis secunda",
                    "canonical_common_name": "Second Bird",
                },
                {
                    "canonical_taxon_id": "inaturalist:10",
                    "match_status": "shared_taxon",
                    "match_method": "normalized_common_name",
                    "cub_class_id": "1",
                    "inaturalist_category_id": "10",
                    "scientific_name": "Avis prima",
                    "canonical_common_name": "First Bird",
                },
                {
                    "canonical_taxon_id": "inaturalist:30",
                    "match_status": "inaturalist_only",
                    "match_method": "source_only",
                    "cub_class_id": "",
                    "inaturalist_category_id": "30",
                    "scientific_name": "Avis tertia",
                    "canonical_common_name": "Third Bird",
                },
            ],
        )
        fields = sorted(training_manifest.BRIDGE_REQUIRED_FIELDS | {
            "bbox_x", "bbox_y", "bbox_width", "bbox_height", "license_name", "license_url"
        })
        rows: list[dict[str, str]] = []
        for taxon, cub_class in (("inaturalist:10", "1"), ("inaturalist:20", "2")):
            for index in range(10):
                is_test = index >= 5
                rows.append(
                    self.bridge_row(
                        f"cub:{cub_class}:{index}", taxon, "cub_200_2011", index,
                        "test" if is_test else "train", "0" if is_test else "1",
                    )
                )
            for index in range(10):
                rows.append(
                    self.bridge_row(
                        f"inat:{cub_class}:{index}", taxon, "inaturalist", index, "", ""
                    )
                )
        write_csv(self.bridge, fields, rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def bridge_row(
        image_id: str,
        taxon: str,
        source: str,
        index: int,
        source_split: str,
        is_training: str,
    ) -> dict[str, str]:
        return {
            "bridge_image_id": image_id,
            "source_dataset": source,
            "source_image_id": image_id,
            "canonical_taxon_id": taxon,
            "match_status": "shared_taxon",
            "match_method": "normalized_common_name",
            "photo_relative_path": f"class/image_{index}.jpg",
            "absolute_path": f"Z:/missing/{source}/{taxon}/{index}.jpg",
            "file_exists": "True",
            "width_px": "640",
            "height_px": "480",
            "split": source_split,
            "is_training_image": is_training,
            "scan_error": "",
            "bbox_x": "1" if source == "cub_200_2011" else "",
            "bbox_y": "2" if source == "cub_200_2011" else "",
            "bbox_width": "100" if source == "cub_200_2011" else "",
            "bbox_height": "80" if source == "cub_200_2011" else "",
            "license_name": "CC BY" if source == "inaturalist" else "",
            "license_url": "https://example.test/license" if source == "inaturalist" else "",
        }

    def build(self, **options: object) -> dict[str, object]:
        return training_manifest.build_training_manifest(
            self.bridge, self.crosswalk, self.output, **options
        )

    def read_manifest(self) -> list[dict[str, str]]:
        with (self.output / "bird_training_manifest.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            return list(csv.DictReader(handle))

    def test_shared_policy_builds_stable_class_map_and_source_aware_splits(self) -> None:
        summary = self.build()
        rows = self.read_manifest()
        self.assertEqual(summary["dataset"]["classes"], 2)
        self.assertEqual(summary["dataset"]["images"], 40)
        self.assertEqual(summary["dataset"]["split_counts"], {
            "train": 24, "validation": 4, "test": 12
        })

        with (self.output / "bird_training_classes.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            classes = list(csv.DictReader(handle))
        self.assertEqual(
            [(row["class_index"], row["canonical_taxon_id"]) for row in classes],
            [("0", "inaturalist:10"), ("1", "inaturalist:20")],
        )
        cub_test = [
            row for row in rows
            if row["source_dataset"] == "cub_200_2011" and row["source_split"] == "test"
        ]
        self.assertTrue(cub_test)
        self.assertEqual({row["model_split"] for row in cub_test}, {"test"})
        inaturalist = [row for row in rows if row["source_dataset"] == "inaturalist"]
        self.assertEqual(
            {split: sum(row["model_split"] == split for row in inaturalist)
             for split in training_manifest.MODEL_SPLITS},
            {"train": 16, "validation": 2, "test": 2},
        )

    def test_rebuild_is_deterministic_except_summary_timestamp(self) -> None:
        first = self.build(seed="repeatable")
        first_manifest = (self.output / "bird_training_manifest.csv").read_bytes()
        first_classes = (self.output / "bird_training_classes.csv").read_bytes()
        second = self.build(seed="repeatable")
        self.assertEqual(first_manifest, (self.output / "bird_training_manifest.csv").read_bytes())
        self.assertEqual(first_classes, (self.output / "bird_training_classes.csv").read_bytes())
        self.assertEqual(
            first["outputs"]["training_manifest_sha256"],
            second["outputs"]["training_manifest_sha256"],
        )

    def test_verify_files_fails_for_missing_photos(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.build(verify_files=True)

    def test_invalid_split_fractions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must sum to less than 1"):
            self.build(
                inaturalist_validation_fraction=0.5,
                inaturalist_test_fraction=0.5,
            )

    def test_summary_is_valid_json_and_records_input_hashes(self) -> None:
        summary = self.build()
        saved = json.loads(
            (self.output / "bird_training_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved, summary)
        self.assertEqual(len(saved["inputs"]["bridge_manifest_sha256"]), 64)
        self.assertNotIn(False, saved["validations"].values())


if __name__ == "__main__":
    unittest.main()