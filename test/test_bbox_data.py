"""Tests for the dependency-free shared bounding-box data layer."""

from __future__ import annotations

import csv
import dataclasses
import tempfile
import unittest
from pathlib import Path

import bbox_data


class ParsingTests(unittest.TestCase):
    def test_optional_number_and_boolean_parsing_is_defensive(self) -> None:
        self.assertEqual(bbox_data.parse_optional_float(" 1.25 "), 1.25)
        self.assertIsNone(bbox_data.parse_optional_float("nan"))
        self.assertIsNone(bbox_data.parse_optional_float("Infinity"))
        self.assertIsNone(bbox_data.parse_optional_float("not a number"))
        self.assertEqual(bbox_data.parse_optional_int("53.0"), 53)
        self.assertIsNone(bbox_data.parse_optional_int("53.5"))
        for value in ("TRUE", "yes", "1", True):
            self.assertIs(bbox_data.parse_optional_bool(value), True)
        for value in ("False", "NO", "0", False):
            self.assertIs(bbox_data.parse_optional_bool(value), False)
        for value in ("", None, "maybe", "2"):
            self.assertIsNone(bbox_data.parse_optional_bool(value))

    def test_records_are_compact_immutable_slot_dataclasses(self) -> None:
        box = bbox_data.BBox(1, 2, 3, 4)
        record = bbox_data.ImageRecord("cub:1", "cub", "1", "class/a.jpg", bbox=box)
        self.assertTrue(dataclasses.is_dataclass(record))
        self.assertFalse(hasattr(record, "__dict__"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.image_id = "2"  # type: ignore[misc]


class GeometryTests(unittest.TestCase):
    def test_bbox_normalizes_and_clips(self) -> None:
        reversed_box = bbox_data.BBox(12, 8, -10, -6)
        self.assertEqual(reversed_box.normalized(), bbox_data.BBox(2, 2, 10, 6))
        self.assertEqual(
            bbox_data.BBox(-5, 5, 20, 20).clipped(10, 15),
            bbox_data.BBox(0, 5, 10, 10),
        )
        self.assertIsNone(bbox_data.BBox(20, 20, 2, 2).clipped(10, 10))

    def test_fit_transform_round_trips_and_clips_letterbox_coordinates(self) -> None:
        transform = bbox_data.ImageCanvasTransform.fit(400, 200, 300, 300)
        self.assertEqual(transform.scale, 0.75)
        self.assertEqual((transform.offset_x, transform.offset_y), (0, 75))
        canvas_box = transform.image_bbox_to_canvas(bbox_data.BBox(40, 20, 100, 50))
        self.assertEqual(canvas_box, bbox_data.BBox(30, 90, 75, 37.5))
        self.assertEqual(
            transform.canvas_bbox_to_image(canvas_box),
            bbox_data.BBox(40, 20, 100, 50),
        )
        # A drag may begin in the letterbox margin and end beyond the image.
        self.assertEqual(
            transform.canvas_bbox_to_image(-10, 0, 330, 400),
            bbox_data.BBox(0, 0, 400, 200),
        )


class ManifestLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_csv(self, name: str, fields: list[str], rows: list[list[object]]) -> Path:
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
        return path

    def test_generated_inaturalist_csv_retains_only_compact_metadata(self) -> None:
        path = self.write_csv(
            "inat.csv",
            [
                "bridge_image_id", "source_dataset", "source_image_id", "photos_directory",
                "relative_path", "photo_relative_path", "absolute_path", "width_px", "height_px",
                "common_name", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
                "bbox_confidence", "bird_detection_count", "bbox_detection_status", "bbox_source",
                "irrelevant_huge_column",
            ],
            [[
                "inaturalist:53", "inaturalist", "53", r"D:\old\photos",
                "train_mini/train_mini/00336/a.jpg", "00336/a.jpg", r"D:\old\photos\00336\a.jpg",
                "429", "417", "Plover", "1.5", "2.5", "100.25", "80.5", "0.997", "4",
                "detected", "pseudo_label", "x" * 1000,
            ], [
                "inaturalist:54", "inaturalist", "54", r"D:\old\photos",
                "train_mini/train_mini/00336/b.jpg", "00336/b.jpg", r"D:\old\photos\00336\b.jpg",
                "500", "333", "Plover", "", "", "", "", "NaN", "9",
                "below_threshold", "pseudo_label", "ignored",
            ]],
        )

        dataset = bbox_data.load_dataset(path, image_root=self.directory / "portable")

        self.assertEqual(dataset.format, "inaturalist_bbox_csv")
        self.assertEqual(len(dataset), 2)
        record = dataset[0]
        self.assertEqual(record.source_id, "inaturalist:53")
        self.assertEqual(record.bbox, bbox_data.BBox(1.5, 2.5, 100.25, 80.5))
        self.assertEqual(record.confidence, 0.997)
        self.assertEqual(record.detection_count, 4)
        self.assertEqual(record.class_name, "Plover")
        self.assertFalse(hasattr(record, "irrelevant_huge_column"))
        self.assertEqual(
            dataset.resolve_image_path(record), self.directory / "portable" / "00336" / "a.jpg"
        )
        self.assertIsNone(dataset[1].bbox)
        self.assertIsNone(dataset[1].confidence)

    def test_generated_cub_manifest_and_cross_platform_path_override(self) -> None:
        path = self.write_csv(
            "cub.csv",
            [
                "image_id", "relative_path", "image_path", "class_id", "class_name_friendly",
                "is_training_image", "split", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
                "width_px", "height_px", "attr_present__hundreds_are_ignored",
            ],
            [[
                1, r"001.Bird\image.jpg", r"E:\CUB\images\001.Bird\image.jpg", 1, "Bird",
                0, "test", 60, 27, 325, 304, 500, 335, 1,
            ]],
        )
        root = self.directory / "images"

        dataset = bbox_data.load_manifest_csv(path, image_root=root)
        record = dataset[0]

        self.assertEqual(dataset.format, "cub_manifest_csv")
        self.assertEqual(record.source_id, "cub_200_2011:1")
        self.assertEqual(record.bbox_status, "ground_truth")
        self.assertEqual(record.bbox_source, "cub_annotation")
        self.assertIs(record.is_training, False)
        self.assertEqual(record.width, 500)
        self.assertEqual(record.resolve_image_path(root), root / "001.Bird" / "image.jpg")

    def test_missing_identity_or_path_is_rejected_with_line_number(self) -> None:
        path = self.write_csv(
            "bad.csv",
            ["image_id", "relative_path", "bbox_x", "bbox_y", "bbox_width", "bbox_height"],
            [["", "", 1, 2, 3, 4]],
        )
        with self.assertRaisesRegex(ValueError, r"bad.csv:2: missing stable image identity"):
            bbox_data.load_dataset(path)

    def test_duplicate_identity_and_partial_bbox_are_rejected(self) -> None:
        duplicate = self.write_csv(
            "duplicate.csv",
            ["image_id", "relative_path", "bbox_x", "bbox_y", "bbox_width", "bbox_height"],
            [[1, "a.jpg", 1, 2, 3, 4], [1, "b.jpg", 1, 2, 3, 4]],
        )
        with self.assertRaisesRegex(ValueError, "duplicate source_id"):
            bbox_data.load_dataset(duplicate)
        partial = self.write_csv(
            "partial.csv",
            ["image_id", "relative_path", "bbox_x", "bbox_y", "bbox_width", "bbox_height"],
            [[1, "a.jpg", 1, 2, "", 4]],
        )
        with self.assertRaisesRegex(ValueError, "incomplete bounding box"):
            bbox_data.load_dataset(partial)


class RawCubTests(unittest.TestCase):
    def test_standard_raw_directory_is_joined_by_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CUB_200_2011"
            (root / "images" / "002.Second_Bird").mkdir(parents=True)
            (root / "images.txt").write_text(
                "2 002.Second_Bird/two.jpg\n1 001.First_Bird/one.jpg\n", encoding="utf-8"
            )
            (root / "bounding_boxes.txt").write_text(
                "1 1.5 2.5 30 40\n2 5 6 70 80\n", encoding="utf-8"
            )
            (root / "image_class_labels.txt").write_text("1 1\n2 2\n", encoding="utf-8")
            (root / "train_test_split.txt").write_text("1 1\n2 0\n", encoding="utf-8")
            (root / "classes.txt").write_text(
                "1 001.First_Bird\n2 002.Second_Bird\n", encoding="utf-8"
            )

            dataset = bbox_data.load_bbox_dataset(root)

            self.assertEqual(dataset.format, "cub_raw")
            self.assertEqual([record.image_id for record in dataset], ["2", "1"])
            self.assertEqual(dataset[0].bbox, bbox_data.BBox(5, 6, 70, 80))
            self.assertEqual(dataset[0].class_name, "002.Second Bird")
            self.assertIs(dataset[0].is_training, False)
            self.assertEqual(dataset[0].split, "test")
            self.assertEqual(
                dataset.resolve_image_path(dataset[0]), root / "images" / "002.Second_Bird" / "two.jpg"
            )

    def test_incomplete_standard_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "missing images.txt"):
                bbox_data.load_cub_directory(temporary)


class FilteringTests(unittest.TestCase):
    def test_filter_and_natural_sort_helpers(self) -> None:
        records = (
            bbox_data.ImageRecord("inat:10", "inat", "10", "Robin10.jpg", bbox_status="detected"),
            bbox_data.ImageRecord("inat:2", "inat", "2", "Robin2.jpg", bbox_status="none"),
            bbox_data.ImageRecord(
                "cub:1", "cub", "1", "Sparrow.jpg", bbox=bbox_data.BBox(1, 1, 2, 2),
                bbox_status="ground_truth",
            ),
        )
        self.assertEqual(
            [record.source_id for record in bbox_data.sort_records(records)],
            ["cub:1", "inat:2", "inat:10"],
        )
        self.assertEqual(
            bbox_data.filter_records(records, source_dataset="inat", query="robin", has_bbox=False),
            records[:2],
        )
        self.assertEqual(
            bbox_data.filter_records(records, statuses={"GROUND_TRUTH"}), (records[2],)
        )


class GroundTruthPersistenceTests(unittest.TestCase):
    def record(self, state: str, source_id: str = "inaturalist:53") -> bbox_data.GroundTruthRecord:
        return bbox_data.GroundTruthRecord(
            source_id=source_id,
            source_dataset="inaturalist",
            image_id=source_id.rsplit(":", 1)[-1],
            relative_path="class/image.jpg",
            source_width=429,
            source_height=417,
            review_state=state,
            bbox=bbox_data.BBox(1.25, 2.5, 100, 80) if state == "boxed" else None,
        )

    def test_validation_enforces_review_state_and_box_invariants(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a valid"):
            dataclasses.replace(self.record("boxed"), bboxes=[])
        with self.assertRaisesRegex(ValueError, "must not contain"):
            dataclasses.replace(self.record("boxed"), review_state="no_bird")
        with self.assertRaisesRegex(ValueError, "within source dimensions"):
            dataclasses.replace(
                self.record("boxed"), bboxes=[bbox_data.BBox(400, 400, 100, 100)]
            )
        with self.assertRaisesRegex(ValueError, "invalid review state"):
            dataclasses.replace(self.record("skipped"), review_state="pending")

    def test_atomic_save_round_trip_preserves_identity_dimensions_and_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "ground_truth.csv"
            records = {
                "inat:3": self.record("skipped", "inat:3"),
                "inat:1": self.record("boxed", "inat:1"),
                "inat:2": self.record("no_bird", "inat:2"),
                "inat:4": self.record("unusable", "inat:4"),
            }

            bbox_data.save_ground_truth(path, records)
            loaded = bbox_data.load_ground_truth(path)

            self.assertEqual(loaded, {
                key: value for key, value in sorted(records.items())
            })
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(lines[0].startswith("source_id,source_dataset,image_id"))
            self.assertTrue(lines[1].startswith("inat:1,"))
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_fractional_edge_aligned_box_round_trips_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ground_truth.csv"
            x = 1.2345678901234567
            box = bbox_data.BBox(x, 0.12345678901234567, 429.0 - x, 416.87654321098764)
            record = dataclasses.replace(self.record("boxed"), bboxes=[box])
            bbox_data.save_ground_truth(path, [record])
            self.assertEqual(bbox_data.load_ground_truth(path)[record.source_id], record)

    def test_multiple_boxes_are_kept_as_a_list_and_saved_one_row_per_box(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ground_truth.csv"
            boxes = [
                bbox_data.BBox(1, 2, 30, 40),
                bbox_data.BBox(100, 120, 50, 60),
                bbox_data.BBox(250, 200, 20, 25),
            ]
            record = dataclasses.replace(self.record("boxed"), bboxes=boxes)

            bbox_data.save_ground_truth(path, [record])
            loaded = bbox_data.load_ground_truth(path)[record.source_id]

            self.assertIsInstance(loaded.bboxes, list)
            self.assertEqual(loaded.bboxes, boxes)
            self.assertEqual(loaded.bbox, boxes[0])
            with path.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_repeated_box_rows_must_describe_the_same_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ground_truth.csv"
            record = dataclasses.replace(
                self.record("boxed"),
                bboxes=[bbox_data.BBox(1, 2, 3, 4), bbox_data.BBox(5, 6, 7, 8)],
            )
            bbox_data.save_ground_truth(path, [record])
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("class/image.jpg", "other.jpg", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent rows"):
                bbox_data.load_ground_truth(path)

    def test_save_revalidates_a_mutated_box_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ground_truth.csv"
            record = self.record("boxed")
            record.bboxes.append(bbox_data.BBox(420, 410, 20, 20))
            with self.assertRaisesRegex(ValueError, "within source dimensions"):
                bbox_data.save_ground_truth(path, [record])
            self.assertFalse(path.exists())

    def test_missing_file_is_empty_and_malformed_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.csv"
            self.assertEqual(bbox_data.load_ground_truth(path), {})
            record = self.record("boxed")
            bbox_data.save_ground_truth(path, [record])
            text = path.read_text(encoding="utf-8").replace(",100,80\n", ",,80\n")
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete bounding box"):
                bbox_data.load_ground_truth(path)

    def test_from_image_record_requires_or_preserves_source_dimensions(self) -> None:
        image = bbox_data.ImageRecord(
            "cub:1", "cub", "1", "bird.jpg", width=100, height=80
        )
        review = bbox_data.GroundTruthRecord.from_image_record(
            image, "boxed", bbox_data.BBox(1, 2, 30, 40)
        )
        self.assertEqual((review.source_width, review.source_height), (100, 80))


class RealDataSmokeTests(unittest.TestCase):
    def test_checked_in_generated_manifests_are_compatible(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = (
            root / "Notebooks/Outputs/CUB Outputs/cub_200_2011_image_manifest.csv",
            root / (
                "Notebooks/Outputs/iNaturalist Outputs/Bounding Box Outputs/"
                "inaturalist_bird_bbox_manifest.csv"
            ),
        )
        for path in paths:
            if not path.is_file():
                continue
            with self.subTest(path=path.name):
                dataset = bbox_data.load_dataset(path)
                self.assertGreater(len(dataset), 1)
                self.assertTrue(dataset[0].source_id)
                self.assertTrue(dataset[0].relative_path)
                self.assertGreater(dataset[0].width or 0, 0)
                self.assertGreater(dataset[0].height or 0, 0)
                self.assertTrue(dataset[0].has_bbox)

    def test_local_raw_cub_install_is_compatible_when_available(self) -> None:
        root = Path(r"E:\Datasets\CUB_200\CUB_200_2011")
        if not root.is_dir():
            self.skipTest("local raw CUB dataset is unavailable")
        dataset = bbox_data.load_cub_directory(root)
        self.assertEqual(len(dataset), 11788)
        self.assertEqual(dataset[0].source_id, "cub_200_2011:1")
        self.assertEqual(dataset[0].bbox, bbox_data.BBox(60, 27, 325, 304))
        self.assertTrue(dataset.resolve_image_path(dataset[0]).is_file())


if __name__ == "__main__":
    unittest.main()