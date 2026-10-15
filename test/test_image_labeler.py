"""Scale and metadata regression tests for the desktop image labeler."""

from __future__ import annotations

import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

import image_labeler


class ThumbnailRangeTests(unittest.TestCase):
    def test_large_collection_materializes_only_viewport_rows(self) -> None:
        visible = image_labeler.visible_thumbnail_range(
            3500,
            visible_top=1700 * image_labeler.THUMBNAIL_ROW_HEIGHT,
            viewport_height=780,
        )

        self.assertLessEqual(len(visible), 11)
        self.assertLess(visible.start, 1700)
        self.assertGreater(visible.stop, 1700)

    def test_visible_range_clamps_to_collection_edges(self) -> None:
        at_start = image_labeler.visible_thumbnail_range(3500, -100, 500)
        at_end = image_labeler.visible_thumbnail_range(
            3500,
            3500 * image_labeler.THUMBNAIL_ROW_HEIGHT,
            500,
        )

        self.assertEqual(at_start.start, 0)
        self.assertEqual(at_end.stop, 3500)
        self.assertGreaterEqual(at_end.start, 0)
        self.assertLessEqual(at_end.start, at_end.stop)

    def test_empty_collection_has_no_visible_rows(self) -> None:
        self.assertEqual(list(image_labeler.visible_thumbnail_range(0, 0, 780)), [])


class ImageLabelerMetadataTests(unittest.TestCase):
    def test_discovery_and_label_state_still_pair_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "a.jpg").write_bytes(b"not decoded during discovery")
            (directory / "a.json").write_text(
                json.dumps(
                    {
                        "filename": "a.jpg",
                        "manual_label": {"bird_present": True, "species": "Robin"},
                    }
                ),
                encoding="utf-8",
            )
            (directory / "b.jpg").write_bytes(b"not decoded during discovery")

            records = image_labeler.discover_records(directory)
            labeled, species = image_labeler.collect_label_state(records)

            self.assertEqual([record.image_path.name for record in records], ["a.jpg", "b.jpg"])
            self.assertEqual(records[0].pairing, "matching stem")
            self.assertEqual(records[1].pairing, "new sidecar")
            self.assertEqual({path.name for path in labeled}, {"a.jpg"})
            self.assertEqual(species, ["Robin", image_labeler.UNKNOWN_SPECIES])


class VirtualGalleryTkTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.geometry("1360x780+0+0")
        try:
            self.root.attributes("-alpha", 0.0)
        except tk.TclError:
            pass

    def tearDown(self) -> None:
        if hasattr(self, "root"):
            self.root.destroy()

    def test_3500_record_gallery_keeps_tk_resources_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "preview.jpg"
            Image.new("RGB", (640, 480), (40, 90, 140)).save(image_path)
            records = [
                image_labeler.ImageRecord(
                    image_path,
                    Path(temporary) / f"image_{index:04d}.json",
                    "new sidecar",
                )
                for index in range(3500)
            ]
            app = image_labeler.BirdLabelerApp(self.root)
            self.root.geometry("1360x780+0+0")
            self.root.update()
            app.records = records
            app.index = 0
            app.selected_indices = {0}
            app._build_thumbnails()
            self.root.update_idletasks()
            command_count = len(app.thumbnail_canvas._tclCommands or [])
            first_item = app.thumbnail_items[0]
            app.thumbnail_canvas.addtag_withtag("current", first_item.preview_id)
            with mock.patch.object(app, "_thumbnail_clicked", return_value="break") as clicked:
                event = SimpleNamespace(state=0)
                self.assertEqual(app._thumbnail_canvas_clicked(event), "break")
                clicked.assert_called_once_with(event, 0)
            maximum_rows = 0
            maximum_images = 0
            maximum_canvas_items = 0

            for _pass in range(3):
                for position in (0.0, 0.25, 0.5, 0.75, 1.0):
                    app.thumbnail_canvas.yview_moveto(position)
                    app._refresh_visible_thumbnails()
                    self.root.update_idletasks()
                    maximum_rows = max(maximum_rows, len(app.thumbnail_items))
                    maximum_images = max(maximum_images, len(app.thumbnail_images))
                    maximum_canvas_items = max(
                        maximum_canvas_items, len(app.thumbnail_canvas.find_all())
                    )

            app.index = len(records) - 1
            app._show_current_thumbnail()
            self.root.update_idletasks()
            visible_top = app.thumbnail_canvas.canvasy(0)
            visible_bottom = visible_top + app.thumbnail_canvas.winfo_height()
            last_row_top = app.index * image_labeler.THUMBNAIL_ROW_HEIGHT
            last_row_bottom = last_row_top + image_labeler.THUMBNAIL_ROW_HEIGHT

            self.assertLessEqual(maximum_rows, 11)
            self.assertEqual(maximum_images, maximum_rows)
            self.assertLessEqual(maximum_canvas_items, maximum_rows * 3)
            self.assertEqual(len(app.thumbnail_canvas._tclCommands or []), command_count)
            self.assertGreaterEqual(last_row_top, visible_top - 1)
            self.assertLessEqual(last_row_bottom, visible_bottom + 1)


if __name__ == "__main__":
    unittest.main()