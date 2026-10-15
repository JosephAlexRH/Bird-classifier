"""Regression tests for the bounding-box viewer's non-GUI behavior."""

from __future__ import annotations

import tempfile
import tkinter as tk
import unittest
from pathlib import Path

import bbox_viewer
from bbox_data import BBox, BBoxDataset, GroundTruthRecord, ImageRecord


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


class EditingGeometryTests(unittest.TestCase):
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
            "unreviewed": 3, "boxed": 0, "no_bird": 1, "skipped": 0
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
        app.draft_bbox = BBox(10, 15, 30, 25)

        with mock.patch("bbox_viewer.save_ground_truth", side_effect=OSError("disk full")), \
             mock.patch("bbox_viewer.messagebox.showerror"):
            self.assertFalse(app._save_review("boxed"))
        self.assertEqual(app.session.reviews, {})

        self.assertTrue(app._save_review("boxed"))
        loaded = load_ground_truth(output)
        self.assertEqual(loaded["inat:1"].bbox, BBox(10, 15, 30, 25))

    def test_shortcuts_do_not_fire_while_typing_in_search(self) -> None:
        from unittest import mock
        app = bbox_viewer.BBoxViewerApp(self.root)
        event = type("Event", (), {"widget": app.search_entry})()
        action = mock.Mock()
        self.assertIsNone(app._shortcut(event, action))
        action.assert_not_called()

    def test_cancelled_filter_change_preserves_draft_and_restores_controls(self) -> None:
        from unittest import mock
        dataset = BBoxDataset((record(1), record(2)), Path("manifest.csv"), "test")
        app = bbox_viewer.BBoxViewerApp(self.root)
        app.session = bbox_viewer.ReviewSession(dataset, {})
        app.draft_bbox = BBox(1, 2, 3, 4)
        app.draft_dirty = True
        app.search_var.set("new search")
        if app.filter_job is not None:
            self.root.after_cancel(app.filter_job)
            app.filter_job = None
        with mock.patch("bbox_viewer.messagebox.askyesno", return_value=False):
            app._apply_filters()
        self.assertEqual(app.draft_bbox, BBox(1, 2, 3, 4))
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
        app.draft_bbox = BBox(1, 2, 3, 4)
        save_ground_truth(output, [GroundTruthRecord.from_image_record(record(1), "skipped")])
        with mock.patch("bbox_viewer.messagebox.showerror"):
            self.assertFalse(app._save_review("boxed"))
        self.assertEqual(app.session.reviews, {})

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