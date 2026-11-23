"""The hub's Review data tab is read-only model output: boxes, species, birds guessed and visits."""

from __future__ import annotations

import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from server.collection_dashboard import collection_page
from server.raspberry_pi_server import create_app

VISIT = "esp32cam-a/capture_20261120T151442_963898Z_36842f78"


def write_capture(root: Path, capture_id: str, inference: dict) -> None:
    (root / f"{capture_id}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (root / f"{capture_id}.json").write_text(json.dumps({
        "capture_id": capture_id, "filename": f"{capture_id}.jpg", "received_at": "2026-11-20T15:14:42Z",
        "camera_id": "esp32cam-a", "camera_registry_id": "reg-a", "camera_name": "Feeder"}))
    (root / f"{capture_id}.inference.json").write_text(json.dumps(inference))


class Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.ids.extend(value for key, value in attrs if key == "id")


class ReviewTabTests(unittest.TestCase):
    def test_capture_listing_carries_the_closed_visit_result(self):
        capture_id = "capture_20261120T151442_963898Z_36842f78"
        inference = {"status": "complete", "is_bird": True, "bird_probability": 0.99,
                     "visit": {"visit_id": VISIT, "frame_index": 0, "provisional": True},
                     "localizer": {"role": "bird_frame", "detections": [
                         {"score": 0.68, "box": {"left": 1, "top": 1, "right": 9, "bottom": 9}, "static": True}]},
                     "species": {"localised": False, "box_source": "selected_crop_fallback", "boxes": []}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_capture(root, capture_id, inference)
            client = create_app(root).test_client()
            first = client.get("/api/captures?limit=5").get_json()["captures"][0]
            self.assertIsNone(first["visit_result"])  # still open: no visits.jsonl line yet
            line = {"visit_id": VISIT, "captures": 1, "species_frames": 1, "capture_ids": [capture_id],
                    "result": {"decision": "review", "reason": "too_few_frames"}}
            (root / "visits.jsonl").write_text(json.dumps(line) + "\n")
            summary = client.get("/api/captures?limit=5").get_json()["captures"][0]
            self.assertEqual(summary["visit_result"]["result"]["reason"], "too_few_frames")
            self.assertNotIn("capture_ids", summary["visit_result"])
            with (root / "visits.jsonl").open("a") as handle:  # a later rebuild line wins
                handle.write(json.dumps({**line, "captures": 2, "result": {"decision": "species", "label": "House Sparrow"}}) + "\n")
            summary = client.get("/api/captures?limit=5").get_json()["captures"][0]
            self.assertEqual((summary["visit_result"]["captures"], summary["visit_result"]["result"]["decision"]),
                             (2, "species"))
            self.assertTrue(summary["inference"]["localizer"]["detections"][0]["static"])

    def test_review_pane_is_read_only(self):
        html = collection_page(remote=False)
        for element in ('id="prediction"', 'id="species-list"', 'id="visit"', 'id="show-weak"', 'id="overlay"',
                        'Birds guessed'):
            self.assertIn(element, html)
        for removed in ("data-review", "clear-boxes", "restore-boxes", "/api/collection/export.csv",
                        "/review',{method:'POST'", 'id="species"', 'id="note"'):
            self.assertNotIn(removed, html)
        parser = Ids()
        parser.feed(collection_page(remote=True))
        self.assertEqual(len(parser.ids), len(set(parser.ids)))


if __name__ == "__main__":
    unittest.main()
