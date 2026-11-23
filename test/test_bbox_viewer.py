"""Regression tests for the bounding-box viewer's non-GUI behavior."""

from __future__ import annotations

import tempfile
import tkinter as tk
import unittest
from pathlib import Path

import bbox_viewer
from bbox_data import BBox, BBoxDataset, GroundTruthRecord, ImageRecord, SourceBox


def record(index: int, *, box: bool = True, status: str = "detected") -> ImageRecord:
    return ImageRecord(
        source_id=f"inat:{index}",
        source_dataset="inat",
        image_id=str(index),
        relative_path=f"birds/image_{index}.jpg",
        width=100,
        height=80,
        bbox=BBox(10, 12, 30, 20) if box else None,
        bbox_status=status,
        class_name="Robin" if index % 2 else "Sparrow",
        confidence=index / 10,
    )


class VirtualRowsTests(unittest.TestCase):
    def test_large_collection_materializes_only_viewport_rows(self) -> None:
        rows = bbox_viewer.visible_row_range(
            74300, 40000 * bbox_viewer.ROW_HEIGHT, 850
        )
        self.assertLessEqual(len(rows), 39)
        self.assertLess(rows.start, 40000)
        self.assertGreater(rows.stop, 40000)

    def test_empty_and_edges_are_bounded(self) -> None:
        self.assertEqual(list(bbox_viewer.visible_row_range(0, 0, 500)), [])
        self.assertEqual(bbox_viewer.visible_row_range(10, -20, 100).start, 0)
        self.assertEqual(bbox_viewer.visible_row_range(10, 10000, 100).stop, 10)


class RecentSpeciesTests(unittest.TestCase):
    def test_merge_is_recent_first_case_insensitive_and_limited(self) -> None:
        self.assertEqual(
            bbox_viewer.merge_recent_species(
                ["Robin", "Blue Tit", "Goldfinch"],
                [" sparrow ", "ROBIN", ""],
                limit=3,
            ),
            ["sparrow", "ROBIN", "Blue Tit"],
        )
        self.assertEqual(bbox_viewer.merge_recent_species(["Robin"], [], limit=0), [])

    def test_saved_reviews_seed_shared_and_per_box_species(self) -> None:
        first = record(1)
        second = record(2)
        shared = GroundTruthRecord.from_image_record(
            first, "boxed", bboxes=[BBox(1, 2, 10, 10)], species="Robin"
        )
        mixed = GroundTruthRecord.from_image_record(
            second,
            "boxed",
            bboxes=[BBox(1, 2, 10, 10), BBox(20, 5, 10, 10)],
            mixed_species=True,
            box_species=["Sparrow", "Blue Tit"],
        )
        self.assertEqual(
            bbox_viewer.recent_species_from_reviews([shared, mixed]),
            ["Sparrow", "Blue Tit", "Robin"],
        )


class DirectoryReviewTests(unittest.TestCase):
    def test_only_reviews_for_current_files_are_kept(self) -> None:
        current = ImageRecord(
            "current-id", "image_directory", "bird", "bird.jpg", width=100, height=80
        )
        stale_record = ImageRecord(
            "stale-id", "image_directory", "old", "old.jpg", width=100, height=80
        )
        saved_current = GroundTruthRecord.from_image_record(
            ImageRecord(
                "old-id", "image_directory", "bird", "bird.jpg", width=100, height=80
            ),
            "boxed",
            bboxes=[BBox(1, 2, 30, 20)],
            species="Robin",
        )
        saved_stale = GroundTruthRecord.from_image_record(stale_record, "no_bird")
        dataset = BBoxDataset((current,), Path("images"), "image_directory", Path("images"))

        reviews = bbox_viewer.reviews_for_current_images(
            dataset, [saved_current, saved_stale]
        )

        self.assertEqual(set(reviews), {"current-id"})
        self.assertEqual(reviews["current-id"].relative_path, "bird.jpg")
        self.assertEqual(reviews["current-id"].species, "Robin")
        self.assertEqual(reviews["current-id"].bboxes, [BBox(1, 2, 30, 20)])

    def test_recognized_capture_bundle_excludes_mismatched_sidecars(self) -> None:
        import hashlib
        import json
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image = directory / "capture.jpg"
            Image.new("RGB", (100, 80)).save(image)
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            image.with_suffix(".json").write_text(json.dumps({
                "capture_id": "capture", "filename": "capture.jpg", "sha256": image_hash,
            }), encoding="utf-8")
            image.with_suffix(".inference.json").write_text(json.dumps({
                "capture_id": "other",
                "image": {"filename": "other.jpg", "sha256": "0" * 64},
            }), encoding="utf-8")
            image.with_suffix(".motion.json").write_text(json.dumps({
                "capture": {
                    "capture_id": "capture", "filename": "capture.jpg",
                    "image_sha256": image_hash,
                },
            }), encoding="utf-8")

            paths = bbox_viewer.recognized_capture_bundle_paths(image)

            self.assertEqual(
                [path.name for path in paths],
                ["capture.jpg", "capture.json", "capture.motion.json"],
            )


class CameraDisplayTests(unittest.TestCase):
    def test_camera_details_hide_model_output_until_requested(self) -> None:
        source = ImageRecord(
            "capture:1",
            "esp32_capture",
            "capture:1",
            "capture.jpg",
            width=640,
            height=480,
            captured_at="2026-11-20T15:23:45Z",
            camera_id="esp32cam-test",
            camera_name="Feeder left",
            camera_model="OV5640",
            event_id="event-test",
            deployment_id="garden-test",
            inference_status="complete",
            predicted_is_bird=True,
            predicted_species="Nuthatch",
            prediction_confidence=0.98,
            predicted_bird_count=1,
            source_boxes=(SourceBox(BBox(10, 20, 30, 40), 0.75, "Nuthatch"),),
            motion_bbox=BBox(5, 6, 100, 120),
            motion_source="dynamic_motion_blob",
        )

        hidden = bbox_viewer.record_details(source, "unreviewed", (640, 480))
        shown = bbox_viewer.record_details(
            source, "unreviewed", (640, 480), show_source_output=True
        )

        self.assertIn("Camera: Feeder left", hidden)
        self.assertIn("Camera/model output hidden", hidden)
        self.assertNotIn("Nuthatch", hidden)
        self.assertIn("Species guess: Nuthatch", shown)
        self.assertIn("Model boxes: 1", shown)
        self.assertIn("Motion crop: dynamic_motion_blob", shown)


class EditingGeometryTests(unittest.TestCase):
    def test_context_defaults_to_zero_percent(self) -> None:
        centered = bbox_viewer.bbox_with_context(BBox(20, 30, 50, 40), 100, 100)
        self.assertEqual(centered, BBox(20, 30, 50, 40))
        edge = bbox_viewer.bbox_with_context(BBox(0, 0, 50, 40), 100, 100)
        self.assertEqual(edge, BBox(0, 0, 50, 40))

    def test_context_supports_custom_and_zero_percentages(self) -> None:
        box = BBox(20, 30, 50, 40)
        self.assertEqual(bbox_viewer.bbox_with_context(box, 100, 100, 0), box)
        self.assertEqual(
            bbox_viewer.bbox_with_context(box, 100, 100, 50),
            BBox(7.5, 20, 75, 60),
        )
        self.assertEqual(bbox_viewer.parse_context_percent("12.5"), 12.5)
        with self.assertRaises(ValueError):
            bbox_viewer.parse_context_percent("-1")

    def test_hit_test_prefers_handles_then_box_interior(self) -> None:
        box = BBox(10, 20, 30, 40)
        self.assertEqual(bbox_viewer.hit_test_bbox(box, 11, 21, 3), "nw")
        self.assertEqual(bbox_viewer.hit_test_bbox(box, 25, 35, 3), "move")
        self.assertIsNone(bbox_viewer.hit_test_bbox(box, 0, 0, 3))

    def test_move_preserves_size_and_clamps_to_image(self) -> None:
        box = bbox_viewer.move_bbox(BBox(10, 10, 30, 20), 500, -500, 100, 80)
        self.assertEqual(box, BBox(70, 0, 30, 20))

    def test_resize_supports_crossing_the_opposite_corner(self) -> None:
        box = bbox_viewer.resize_bbox(BBox(10, 10, 20, 20), "nw", 50, 60, 100, 80)
        self.assertEqual(box, BBox(30, 30, 20, 30))
        clipped = bbox_viewer.resize_bbox(BBox(10, 10, 20, 20), "se", 500, -5, 100, 80)
        self.assertEqual(clipped, BBox(10, 0, 90, 10))

    def test_multi_box_hit_test_prefers_active_handles_then_topmost_box(self) -> None:
        boxes = [BBox(10, 10, 40, 40), BBox(20, 20, 40, 40)]
        self.assertEqual(bbox_viewer.hit_test_bboxes(boxes, 25, 25, 3), (1, "move"))
        self.assertEqual(
            bbox_viewer.hit_test_bboxes(boxes, 10, 10, 3, active_index=0),
            (0, "nw"),
        )


class ReviewSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = tuple(record(index, box=index != 2) for index in (10, 2, 1, 3))
        self.dataset = BBoxDataset(self.records, Path("manifest.csv"), "test")
        self.reviews = {
            "inat:2": GroundTruthRecord(
                "inat:2", "inat", "2", "birds/image_2.jpg", 100, 80, "no_bird"
            ),
            "orphan:99": GroundTruthRecord(
                "orphan:99", "old", "99", "old.jpg", 10, 10, "skipped"
            ),
        }

    def test_natural_sort_filtering_and_current_identity(self) -> None:
        session = bbox_viewer.ReviewSession(self.dataset, self.reviews)
        self.assertEqual([item.source_id for item in session.filtered], [
            "inat:1", "inat:2", "inat:3", "inat:10"
        ])
        session.position = 1
        session.apply_filters(query="sparrow", preserve_source_id="inat:2")
        self.assertEqual(session.current.source_id, "inat:2")
        session.apply_filters(source_box_filter="without box")
        self.assertEqual([item.source_id for item in session.filtered], ["inat:2"])
        session.apply_filters(review_filter="no_bird")
        self.assertEqual(session.current.source_id, "inat:2")

    def test_review_counts_ignore_orphans_and_navigation_wraps(self) -> None:
        session = bbox_viewer.ReviewSession(self.dataset, self.reviews)
        self.assertEqual(session.counts(), {
            "unreviewed": 3, "boxed": 0, "no_bird": 1, "unusable": 0, "skipped": 0
        })
        session.position = len(session.filtered) - 1
        self.assertTrue(session.next_unreviewed())
        self.assertEqual(session.current.source_id, "inat:1")

    def test_review_filter_reacts_to_new_decision(self) -> None:
        session = bbox_viewer.ReviewSession(self.dataset, self.reviews)
        session.apply_filters(review_filter="unreviewed")
        current = session.current
        self.assertIsNotNone(current)
        assert current is not None
        session.reviews[current.source_id] = GroundTruthRecord.from_image_record(
            current, "skipped"
        )
        session.apply_filters()
        self.assertNotIn(current.source_id, {item.source_id for item in session.filtered})

    def test_saved_identity_mismatch_is_rejected(self) -> None:
        bad = dict(self.reviews)
        bad["inat:2"] = GroundTruthRecord(
            "inat:2", "inat", "different", "birds/image_2.jpg", 100, 80, "skipped"
        )
        with self.assertRaisesRegex(ValueError, "image_id"):
            bbox_viewer.validate_review_identities(self.dataset, bad)

    def test_blind_search_does_not_use_prediction_species_or_status(self) -> None:
        predicted = ImageRecord(
            "capture:prediction",
            "esp32_capture",
            "prediction",
            "capture.jpg",
            width=100,
            height=80,
            camera_name="Feeder",
            inference_status="complete",
            predicted_species="Secret Robin Guess",
            prediction_confidence=0.99,
        )
        session = bbox_viewer.ReviewSession(
            BBoxDataset((predicted,), Path("images"), "image_directory"), {}
        )

        session.apply_filters(query="secret robin")

        self.assertEqual(session.filtered, ())

    def test_decoded_dimensions_may_correct_stale_manifest_dimensions(self) -> None:
        corrected = dict(self.reviews)
        corrected["inat:2"] = GroundTruthRecord(
            "inat:2", "inat", "2", "birds/image_2.jpg", 200, 160, "skipped"
        )
        bbox_viewer.validate_review_identities(self.dataset, corrected)


class CliTests(unittest.TestCase):
    def test_cli_paths_and_default_output(self) -> None:
        arguments = bbox_viewer.parse_args([
            "manifest.csv", "--image-root", "images", "--ground-truth", "review.csv",
            "--read-only", "--start", "inat:42",
        ])
        self.assertEqual(arguments.source, Path("manifest.csv"))
        self.assertEqual(arguments.image_root, Path("images"))
        self.assertEqual(arguments.ground_truth, Path("review.csv"))
        self.assertTrue(arguments.read_only)
        self.assertEqual(arguments.start, "inat:42")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "birds.csv"
            manifest.touch()
            self.assertEqual(
                bbox_viewer.default_ground_truth_path(manifest),
                root / "birds_ground_truth.csv",
            )
            self.assertEqual(
                bbox_viewer.default_ground_truth_path(root), root / "bbox_ground_truth.csv"
            )


class TkWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        if hasattr(self, "root"):
            self.root.destroy()
        if hasattr(self, "temporary"):
            self.temporary.cleanup()

    def test_review_save_failure_rolls_back_memory_and_success_round_trips(self) -> None:
        from unittest import mock
        from PIL import Image
        from bbox_data import load_ground_truth

        directory = Path(self.temporary.name)
        image = directory / "bird.jpg"
        Image.new("RGB", (100, 80), "navy").save(image)
        dataset = BBoxDataset(
            (ImageRecord("inat:1", "inat", "1", "bird.jpg", width=100, height=80),),
            directory / "manifest.csv",
            "test",
            directory,
        )
        output = directory / "reviews.csv"
        app = bbox_viewer.BBoxViewerApp(self.root, ground_truth_path=output)
        app.ground_truth_path = output
        app.session = bbox_viewer.ReviewSession(dataset, {})
        app.load_current()
        boxes = [BBox(10, 15, 30, 25), BBox(60, 20, 20, 30)]
        app.draft_bboxes = boxes
        app.active_box_index = 1

        with mock.patch("bbox_viewer.save_ground_truth", side_effect=OSError("disk full")), \
             mock.patch("bbox_viewer.messagebox.showerror"):
            self.assertFalse(app._save_review("boxed"))
        self.assertEqual(app.session.reviews, {})

        self.assertTrue(app._save_review("boxed"))
        loaded = load_ground_truth(output)
        self.assertEqual(loaded["inat:1"].bboxes, boxes)
        with output.open("r", encoding="utf-8") as handle:
            self.assertEqual(len(handle.read().splitlines()), 3)

    def test_shortcuts_do_not_fire_while_typing_in_search(self) -> None:
        from unittest import mock
        app = bbox_viewer.BBoxViewerApp(self.root)
        event = type("Event", (), {"widget": app.search_entry})()
        action = mock.Mock()
        self.assertIsNone(app._shortcut(event, action))
        action.assert_not_called()

    def test_context_control_defaults_to_zero(self) -> None:
        app = bbox_viewer.BBoxViewerApp(self.root)
        self.assertEqual(app.context_percent_var.get(), "0")

    def test_species_fields_share_editable_recent_species_dropdown(self) -> None:
        app = bbox_viewer.BBoxViewerApp(self.root)
        app._remember_species(["Robin", "Blue Tit"])

        self.assertEqual(app.species_entry.winfo_class(), "TCombobox")
        self.assertEqual(app.active_species_entry.winfo_class(), "TCombobox")
        self.assertEqual(tuple(app.species_entry.cget("values")), ("Robin", "Blue Tit"))
        self.assertEqual(
            tuple(app.active_species_entry.cget("values")), ("Robin", "Blue Tit")
        )
        self.assertEqual(str(app.species_entry.cget("state")), "normal")

    def test_successful_save_adds_species_to_the_top_of_both_dropdowns(self) -> None:
        from PIL import Image

        directory = Path(self.temporary.name)
        image = directory / "bird.jpg"
        Image.new("RGB", (100, 80)).save(image)
        source = ImageRecord(
            "capture:1", "capture", "1", "bird.jpg", width=100, height=80
        )
        output = directory / "reviews.csv"
        app = bbox_viewer.BBoxViewerApp(self.root, ground_truth_path=output)
        app.ground_truth_path = output
        app.session = bbox_viewer.ReviewSession(
            BBoxDataset((source,), output, "detector_annotations_csv", directory), {}
        )
        app.detector_manifest_mode = True
        app.load_current()
        app._remember_species(["Robin"])
        app.draft_bboxes = [BBox(10, 15, 30, 25)]
        app.species_var.set("Blue Tit")

        self.assertTrue(app._save_review("boxed"))
        self.assertEqual(tuple(app.species_entry.cget("values")), ("Blue Tit", "Robin"))
        self.assertEqual(
            tuple(app.active_species_entry.cget("values")), ("Blue Tit", "Robin")
        )

    def test_keep_species_repeats_only_into_images_without_an_existing_species(self) -> None:
        from PIL import Image

        directory = Path(self.temporary.name)
        for name in ("first.jpg", "second.jpg", "third.jpg"):
            Image.new("RGB", (100, 80)).save(directory / name)
        dataset = BBoxDataset(
            (
                ImageRecord("capture:1", "capture", "1", "first.jpg", width=100, height=80),
                ImageRecord("capture:2", "capture", "2", "second.jpg", width=100, height=80),
                ImageRecord(
                    "capture:3", "capture", "3", "third.jpg",
                    width=100, height=80, class_name="Robin",
                ),
            ),
            directory / "manifest.csv",
            "detector_annotations_csv",
            directory,
        )
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(dataset, {})
        app.load_current()
        app.keep_species_var.set(True)
        app.species_var.set("Blue Tit")
        app.draft_dirty = False

        app.next()
        self.assertEqual(app.species_var.get(), "Blue Tit")

        app.draft_dirty = False
        app.next()
        self.assertEqual(app.species_var.get(), "Robin")

    def test_keep_species_repeats_selected_mixed_name_without_repeating_boxes(self) -> None:
        from PIL import Image

        directory = Path(self.temporary.name)
        for name in ("first.jpg", "second.jpg"):
            Image.new("RGB", (100, 80)).save(directory / name)
        records = (
            ImageRecord("capture:1", "capture", "1", "first.jpg", width=100, height=80),
            ImageRecord("capture:2", "capture", "2", "second.jpg", width=100, height=80),
        )
        dataset = BBoxDataset(
            records, directory / "manifest.csv", "detector_annotations_csv", directory
        )
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(dataset, {})
        app.load_current()
        app.keep_species_var.set(True)
        app.mixed_species_var.set(True)
        app.draft_bboxes = [BBox(10, 10, 20, 20)]
        app.draft_box_species = ["Blue Tit"]
        app.active_box_index = 0
        app.active_species_var.set("Blue Tit")
        app.draft_dirty = False

        app.next()

        self.assertEqual(app.draft_bboxes, [])
        self.assertTrue(app.mixed_species_var.get())
        self.assertEqual(app.active_species_var.get(), "Blue Tit")

        app.transform = bbox_viewer.ImageCanvasTransform.fit(100, 80, 100, 80)
        event = type("Event", (), {"x": 10, "y": 10, "state": 0})()
        app._drag_started(event)
        self.assertEqual(app.draft_box_species, ["Blue Tit"])

    def test_keep_boxes_repeats_clipped_geometry_over_reviewed_images(self) -> None:
        from PIL import Image

        directory = Path(self.temporary.name)
        Image.new("RGB", (100, 80)).save(directory / "first.jpg")
        Image.new("RGB", (80, 60)).save(directory / "second.jpg")
        Image.new("RGB", (100, 80)).save(directory / "third.jpg")
        records = (
            ImageRecord("capture:1", "capture", "1", "first.jpg", width=100, height=80),
            ImageRecord("capture:2", "capture", "2", "second.jpg", width=80, height=60),
            ImageRecord("capture:3", "capture", "3", "third.jpg", width=100, height=80),
        )
        saved_third = GroundTruthRecord.from_image_record(
            records[2], "boxed", bbox=BBox(5, 6, 20, 25), species="Robin", is_bird=True
        )
        dataset = BBoxDataset(
            records, directory / "manifest.csv", "detector_annotations_csv", directory
        )
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(dataset, {records[2].source_id: saved_third})
        app.load_current()
        app.keep_boxes_var.set(True)
        app.draft_bboxes = [BBox(70, 50, 30, 30)]
        app.draft_dirty = False

        app.next()
        self.assertEqual(app.draft_bboxes, [BBox(70, 50, 10, 10)])
        self.assertTrue(app.draft_dirty)

        app.draft_dirty = False
        app.next()
        self.assertEqual(app.draft_bboxes, [BBox(70, 50, 10, 10)])
        self.assertEqual(app.saved_bboxes, [BBox(5, 6, 20, 25)])
        self.assertEqual(app.species_var.get(), "Robin")
        self.assertTrue(app.draft_dirty)
        saved_items = app.image_canvas.find_withtag("saved-box")
        self.assertEqual(len(saved_items), 1)
        self.assertEqual(app.image_canvas.itemcget(saved_items[0], "outline"), "#ff4d6d")

    def test_keep_species_and_boxes_repeat_mixed_labels_without_overwriting_saved_review(self) -> None:
        from PIL import Image

        directory = Path(self.temporary.name)
        Image.new("RGB", (100, 80)).save(directory / "first.jpg")
        Image.new("RGB", (80, 60)).save(directory / "second.jpg")
        Image.new("RGB", (100, 80)).save(directory / "third.jpg")
        records = (
            ImageRecord("capture:1", "capture", "1", "first.jpg", width=100, height=80),
            ImageRecord(
                "capture:2", "capture", "2", "second.jpg",
                width=80, height=60, class_name="Source-level guess",
            ),
            ImageRecord("capture:3", "capture", "3", "third.jpg", width=100, height=80),
        )
        saved_third = GroundTruthRecord.from_image_record(
            records[2],
            "boxed",
            bboxes=[BBox(5, 6, 20, 25), BBox(40, 10, 15, 20)],
            mixed_species=True,
            box_species=["Robin", "Sparrow"],
            is_bird=True,
        )
        dataset = BBoxDataset(
            records, directory / "manifest.csv", "detector_annotations_csv", directory
        )
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(dataset, {records[2].source_id: saved_third})
        app.load_current()
        app.keep_species_var.set(True)
        app.keep_boxes_var.set(True)
        app.mixed_species_var.set(True)
        app.draft_bboxes = [
            BBox(90, 70, 5, 5),
            BBox(10, 10, 20, 20),
            BBox(75, 50, 20, 20),
        ]
        app.draft_box_species = ["Blackbird", "Blue Tit", "Goldfinch"]
        app.draft_dirty = False

        app.next()
        self.assertEqual(
            app.draft_bboxes,
            [BBox(10, 10, 20, 20), BBox(75, 50, 5, 10)],
        )
        self.assertEqual(app.draft_box_species, ["Blue Tit", "Goldfinch"])
        self.assertTrue(app.mixed_species_var.get())
        self.assertEqual(app.active_species_var.get(), "Goldfinch")
        self.assertTrue(app.draft_dirty)

        app.draft_dirty = False
        app.next()
        self.assertEqual(
            app.draft_bboxes,
            [BBox(10, 10, 20, 20), BBox(75, 50, 5, 10)],
        )
        self.assertEqual(app.draft_box_species, ["Blue Tit", "Goldfinch"])
        self.assertEqual(app.saved_bboxes, saved_third.bboxes)
        self.assertTrue(app.mixed_species_var.get())

    def test_cancelled_filter_change_preserves_draft_and_restores_controls(self) -> None:
        from unittest import mock
        dataset = BBoxDataset((record(1), record(2)), Path("manifest.csv"), "test")
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(dataset, {})
        app.draft_bboxes = [BBox(1, 2, 3, 4), BBox(10, 20, 30, 40)]
        app.active_box_index = 1
        app.draft_dirty = True
        app.search_var.set("new search")
        if app.filter_job is not None:
            self.root.after_cancel(app.filter_job)
            app.filter_job = None
        with mock.patch("bbox_viewer.messagebox.askyesno", return_value=False):
            app._apply_filters()
        self.assertEqual(app.draft_bboxes, [BBox(1, 2, 3, 4), BBox(10, 20, 30, 40)])
        self.assertEqual(app.search_var.get(), "")

    def test_missing_image_can_still_be_marked_no_bird_from_manifest_dimensions(self) -> None:
        directory = Path(self.temporary.name)
        dataset = BBoxDataset(
            (ImageRecord("inat:1", "inat", "1", "missing.jpg", width=100, height=80),),
            directory / "manifest.csv", "test", directory,
        )
        output = directory / "reviews.csv"
        app = bbox_viewer.BBoxViewerApp(self.root, ground_truth_path=output)
        app.ground_truth_path = output
        app.ground_truth_signature = None
        app.session = bbox_viewer.ReviewSession(dataset, {})
        app.load_current()
        self.assertIsNone(app.image_size)
        self.assertTrue(app._save_review("no_bird"))

    def test_machine_source_box_is_not_loaded_or_drawn(self) -> None:
        from PIL import Image

        directory = Path(self.temporary.name)
        (directory / "birds").mkdir()
        Image.new("RGB", (100, 80)).save(directory / "birds" / "image_1.jpg")
        dataset = BBoxDataset((record(1),), directory / "manifest.csv", "test", directory)
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(dataset, {})

        app.load_current()
        app._render()

        self.assertEqual(app.draft_bboxes, [])
        self.assertEqual(app.image_canvas.find_withtag("source-box"), ())

    def test_camera_source_overlays_are_opt_in_and_never_become_draft_boxes(self) -> None:
        from PIL import Image

        directory = Path(self.temporary.name)
        Image.new("RGB", (100, 80)).save(directory / "capture.jpg")
        source = ImageRecord(
            "capture:1",
            "esp32_capture",
            "capture:1",
            "capture.jpg",
            width=100,
            height=80,
            camera_id="esp32cam-test",
            inference_status="complete",
            source_boxes=(SourceBox(BBox(10, 12, 30, 20), 0.8, "Robin"),),
            motion_bbox=BBox(5, 6, 60, 50),
            motion_source="dynamic_motion_blob",
        )
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(
            BBoxDataset((source,), directory, "image_directory", directory), {}
        )

        app.load_current()
        app._render()
        self.assertEqual(app.draft_bboxes, [])
        self.assertEqual(app.image_canvas.find_withtag("source-box"), ())
        self.assertEqual(app.image_canvas.find_withtag("motion-box"), ())

        app.show_source_output_var.set(True)
        app._source_output_changed()
        app._render()

        self.assertEqual(app.draft_bboxes, [])
        self.assertGreater(len(app.image_canvas.find_withtag("source-box")), 0)
        self.assertGreater(len(app.image_canvas.find_withtag("motion-box")), 0)

    def test_source_output_reveal_resets_when_navigating_to_another_capture(self) -> None:
        from PIL import Image

        directory = Path(self.temporary.name)
        records = []
        for index in (1, 2):
            Image.new("RGB", (100, 80)).save(directory / f"capture{index}.jpg")
            records.append(ImageRecord(
                f"capture:{index}", "esp32_capture", f"capture:{index}",
                f"capture{index}.jpg", width=100, height=80,
                camera_id="esp32cam-test", inference_status="complete",
                source_boxes=(SourceBox(BBox(10, 12, 30, 20), 0.8, "Robin"),),
            ))
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(
            BBoxDataset(tuple(records), directory, "image_directory", directory), {}
        )

        app.load_current()
        app.show_source_output_var.set(True)
        self.assertTrue(app.session.navigate(1))
        app.load_current()

        self.assertFalse(app.show_source_output_var.get())
        self.assertIn("hidden for blind review", app.details_var.get())

    def test_unusable_review_round_trips_without_boxes(self) -> None:
        from bbox_data import load_ground_truth

        directory = Path(self.temporary.name)
        dataset = BBoxDataset((record(1),), directory / "manifest.csv", "test")
        output = directory / "reviews.csv"
        app = bbox_viewer.BBoxViewerApp(self.root, ground_truth_path=output)
        app.ground_truth_path = output
        app.session = bbox_viewer.ReviewSession(dataset, {})

        self.assertTrue(app._save_review("unusable"))
        self.assertEqual(load_ground_truth(output)["inat:1"].review_state, "unusable")
        self.assertEqual(load_ground_truth(output)["inat:1"].bboxes, [])

    def test_external_review_change_blocks_overwrite(self) -> None:
        from unittest import mock
        from bbox_data import save_ground_truth
        directory = Path(self.temporary.name)
        image = directory / "bird.jpg"
        from PIL import Image
        Image.new("RGB", (100, 80)).save(image)
        dataset = BBoxDataset((record(1),), directory / "manifest.csv", "test", directory)
        output = directory / "reviews.csv"
        app = bbox_viewer.BBoxViewerApp(self.root, ground_truth_path=output)
        app.ground_truth_path = output
        app.session = bbox_viewer.ReviewSession(dataset, {})
        app.load_current()
        app.draft_bboxes = [BBox(1, 2, 3, 4)]
        app.active_box_index = 0
        save_ground_truth(output, [GroundTruthRecord.from_image_record(record(1), "skipped")])
        with mock.patch("bbox_viewer.messagebox.showerror"):
            self.assertFalse(app._save_review("boxed"))
        self.assertEqual(app.session.reviews, {})

    def test_saved_multi_box_review_loads_and_delete_removes_only_selected_box(self) -> None:
        from PIL import Image

        directory = Path(self.temporary.name)
        Image.new("RGB", (100, 80)).save(directory / "bird.jpg")
        source = ImageRecord("inat:1", "inat", "1", "bird.jpg", width=100, height=80)
        dataset = BBoxDataset((source,), directory / "manifest.csv", "test", directory)
        boxes = [BBox(1, 2, 20, 30), BBox(50, 10, 25, 35)]
        review = GroundTruthRecord.from_image_record(source, "boxed", bboxes=boxes)
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(dataset, {source.source_id: review})

        app.load_current()
        self.assertEqual(app.draft_bboxes, boxes)
        self.assertEqual(app.active_box_index, 1)

        app.delete_draft()
        self.assertEqual(app.draft_bboxes, boxes[:1])
        self.assertEqual(app.active_box_index, 0)

    def test_delete_picture_confirms_bundle_removal_and_advances(self) -> None:
        from unittest import mock
        from PIL import Image

        directory = Path(self.temporary.name)
        first_image = directory / "first.jpg"
        second_image = directory / "second.jpg"
        first_json = directory / "first.json"
        first_inference = directory / "first.inference.json"
        first_motion = directory / "first.motion.json"
        first_review = directory / "first.review.json"
        unrelated = directory / "first.notes.json"
        Image.new("RGB", (100, 80)).save(first_image)
        Image.new("RGB", (100, 80)).save(second_image)
        first_json.write_text('{"filename": "first.jpg"}', encoding="utf-8")
        first_inference.write_text("{}", encoding="utf-8")
        first_motion.write_text("{}", encoding="utf-8")
        first_review.write_text("{}", encoding="utf-8")
        unrelated.write_text("{}", encoding="utf-8")
        first = ImageRecord("capture:1", "capture", "1", "first.jpg", width=100, height=80)
        second = ImageRecord("capture:2", "capture", "2", "second.jpg", width=100, height=80)
        dataset = BBoxDataset(
            (first, second), directory / "manifest.csv", "test", directory
        )
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(dataset, {})
        app.load_current()
        app.draft_bboxes = [BBox(1, 2, 3, 4)]
        app.draft_dirty = True

        with mock.patch("bbox_viewer.messagebox.askyesno", return_value=True) as confirmation:
            self.assertTrue(app.delete_picture())

        confirmation.assert_called_once()
        prompt = confirmation.call_args.args[1]
        for name in ("first.jpg", "first.json", "first.inference.json", "first.motion.json"):
            self.assertIn(name, prompt)
        self.assertFalse(first_image.exists())
        self.assertFalse(first_json.exists())
        self.assertFalse(first_inference.exists())
        self.assertFalse(first_motion.exists())
        self.assertFalse(first_review.exists())
        self.assertTrue(unrelated.exists())
        self.assertTrue(second_image.exists())
        self.assertEqual([item.source_id for item in app.session.dataset], ["capture:2"])
        self.assertEqual(app.session.current.source_id, "capture:2")
        self.assertFalse(app.draft_dirty)

    def test_delete_picture_removes_saved_review_from_review_file(self) -> None:
        from unittest import mock
        from PIL import Image
        from bbox_data import load_ground_truth, save_ground_truth

        directory = Path(self.temporary.name)
        image = directory / "bird.jpg"
        image.with_suffix(".json").write_text("{}", encoding="utf-8")
        Image.new("RGB", (100, 80)).save(image)
        source = ImageRecord(
            "capture:1", "capture", "1", "bird.jpg", width=100, height=80
        )
        review = GroundTruthRecord.from_image_record(source, "no_bird")
        output = directory / "reviews.csv"
        save_ground_truth(output, [review])
        dataset = BBoxDataset(
            (source,), directory / "manifest.csv", "test", directory
        )
        app = bbox_viewer.BBoxViewerApp(self.root, ground_truth_path=output)
        app.ground_truth_path = output
        app.ground_truth_signature = bbox_viewer.file_signature(output)
        app.session = bbox_viewer.ReviewSession(dataset, {source.source_id: review})
        app.load_current()

        with mock.patch("bbox_viewer.messagebox.askyesno", return_value=True):
            self.assertTrue(app.delete_picture())

        self.assertEqual(load_ground_truth(output), {})
        self.assertEqual(app.session.reviews, {})
        self.assertIsNone(app.session.current)

    def test_delete_unreviewed_picture_removes_detector_manifest_row(self) -> None:
        import csv
        from unittest import mock
        from PIL import Image
        from bbox_data import save_detector_manifest

        directory = Path(self.temporary.name)
        first_image = directory / "first.jpg"
        second_image = directory / "second.jpg"
        Image.new("RGB", (100, 80)).save(first_image)
        Image.new("RGB", (100, 80)).save(second_image)
        first = ImageRecord("capture:1", "capture", "1", "first.jpg", width=100, height=80)
        second = ImageRecord("capture:2", "capture", "2", "second.jpg", width=100, height=80)
        dataset = BBoxDataset((first, second), directory, "image_directory", directory)
        output = directory / "bird_annotations.csv"
        save_detector_manifest(output, dataset, {})
        app = bbox_viewer.BBoxViewerApp(self.root, ground_truth_path=output)
        app.ground_truth_path = output
        app.ground_truth_signature = bbox_viewer.file_signature(output)
        app.detector_manifest_mode = True
        app.session = bbox_viewer.ReviewSession(dataset, {})
        app.load_current()

        with mock.patch("bbox_viewer.messagebox.askyesno", return_value=True):
            self.assertTrue(app.delete_picture())

        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["sample_id"] for row in rows], ["capture:2"])

    def test_delete_picture_cancellation_leaves_bundle_and_session_unchanged(self) -> None:
        from unittest import mock
        from PIL import Image

        directory = Path(self.temporary.name)
        image = directory / "bird.jpg"
        Image.new("RGB", (100, 80)).save(image)
        image.with_suffix(".json").write_text("{}", encoding="utf-8")
        source = ImageRecord("capture:1", "capture", "1", "bird.jpg", width=100, height=80)
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(
            BBoxDataset((source,), directory, "image_directory", directory), {}
        )
        app.load_current()

        with mock.patch("bbox_viewer.messagebox.askyesno", return_value=False):
            self.assertFalse(app.delete_picture())

        self.assertTrue(image.exists())
        self.assertTrue(image.with_suffix(".json").exists())
        self.assertEqual(app.session.current.source_id, source.source_id)

    def test_delete_picture_rolls_back_files_and_session_on_manifest_conflict(self) -> None:
        from unittest import mock
        from PIL import Image
        from bbox_data import save_ground_truth

        directory = Path(self.temporary.name)
        image = directory / "bird.jpg"
        Image.new("RGB", (100, 80)).save(image)
        image.with_suffix(".json").write_text("{}", encoding="utf-8")
        source = ImageRecord("capture:1", "capture", "1", "bird.jpg", width=100, height=80)
        review = GroundTruthRecord.from_image_record(source, "no_bird")
        output = directory / "reviews.csv"
        save_ground_truth(output, [review])
        app = bbox_viewer.BBoxViewerApp(self.root, ground_truth_path=output)
        app.ground_truth_path = output
        app.ground_truth_signature = bbox_viewer.file_signature(output)
        app.session = bbox_viewer.ReviewSession(
            BBoxDataset((source,), directory, "image_directory", directory),
            {source.source_id: review},
        )
        app.load_current()
        output.write_text("changed externally", encoding="utf-8")

        with mock.patch("bbox_viewer.messagebox.askyesno", return_value=True), \
                mock.patch("bbox_viewer.messagebox.showerror") as error:
            self.assertFalse(app.delete_picture())

        error.assert_called_once()
        self.assertTrue(image.exists())
        self.assertTrue(image.with_suffix(".json").exists())
        self.assertEqual(app.session.current.source_id, source.source_id)
        self.assertEqual(app.session.reviews, {source.source_id: review})

    def test_review_file_lock_rejects_a_concurrent_writer(self) -> None:
        if bbox_viewer.os.name != "nt":
            self.skipTest("Windows lock behavior is exercised on the project platform")
        path = Path(self.temporary.name) / "reviews.csv"
        with bbox_viewer.review_file_lock(path):
            with self.assertRaisesRegex(OSError, "another reviewer"):
                with bbox_viewer.review_file_lock(path):
                    pass


if __name__ == "__main__":
    unittest.main()