from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "github pics" / "render_bird_captures.py"
SPEC = importlib.util.spec_from_file_location("render_bird_captures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class RenderBirdCapturesTests(unittest.TestCase):
    def test_extracts_each_species_box_prediction_and_confidence_in_sidecar_order(self) -> None:
        inference = {
            "status": "complete",
            "is_bird": True,
            "classification": {"label": "Wrong global label", "confidence": 0.99},
            "species": {
                "boxes": [
                    {
                        "box": {"left": 10, "top": 20, "right": 50, "bottom": 70},
                        "localizer_score": 0.4,
                        "top_k": [{"label": "House Sparrow", "probability": 0.61}],
                    },
                    {
                        "box": {"left": 60, "top": 25, "right": 100, "bottom": 75},
                        "localizer_score": 0.3,
                        "top_k": [{"label": "Blue Jay", "probability": 0.82}],
                    },
                ]
            },
        }
        boxes = renderer.inference_boxes(inference, 120, 90)
        self.assertEqual([box.prediction for box in boxes], ["House Sparrow", "Blue Jay"])
        self.assertEqual([box.confidence for box in boxes], [0.61, 0.82])
        self.assertEqual([box.detector_confidence for box in boxes], [0.4, 0.3])

    def test_intentional_below_threshold_species_box_is_not_filtered(self) -> None:
        path = ROOT / "sample" / "capture_20261119T201628_979321Z_f83f88d3.inference.json"
        inference = json.loads(path.read_text(encoding="utf-8"))
        boxes = renderer.inference_boxes(inference, 640, 480)
        self.assertEqual(len(boxes), 1)
        self.assertLess(boxes[0].detector_confidence, inference["species"]["box_threshold"])
        self.assertEqual(boxes[0].prediction, "House Sparrow")
        self.assertAlmostEqual(boxes[0].confidence, 0.60546875)

    def test_non_bird_sidecar_draws_no_boxes(self) -> None:
        inference = {"status": "complete", "is_bird": False, "species": {"boxes": []}}
        self.assertEqual(renderer.inference_boxes(inference, 640, 480), [])

    def test_render_keeps_native_dimensions_and_changes_annotated_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "capture.jpg"
            output_path = root / "out.png"
            Image.new("RGB", (160, 100), "#405060").save(image_path)
            image_path.with_suffix(".inference.json").write_text(json.dumps({
                "status": "complete",
                "is_bird": True,
                "species": {"boxes": [{
                    "box": {"left": 30, "top": 25, "right": 90, "bottom": 80},
                    "localizer_score": 0.7,
                    "top_k": [{"label": "Blue Jay", "probability": 0.875}],
                }]},
            }), encoding="utf-8")
            args = renderer.parse_args([str(image_path), "--format", "png"])
            style = renderer.style_from_args(args)
            boxes, size = renderer.render_capture(image_path, output_path, style, 95)
            with Image.open(image_path) as original, Image.open(output_path) as rendered:
                self.assertEqual(size, (160, 100))
                self.assertEqual(rendered.size, original.size)
                self.assertNotEqual(rendered.convert("RGB").tobytes(), original.convert("RGB").tobytes())
            self.assertEqual(len(boxes), 1)

    def test_color_and_label_template_options(self) -> None:
        args = renderer.parse_args([
            "--box-color", "#12abcdef",
            "--label-background-color", "transparent",
            "--box-style", "corners",
            "--label-style", "text",
            "--label-template", "{prediction} ({confidence})",
            "--confidence-decimals", "0",
        ])
        style = renderer.style_from_args(args)
        self.assertEqual(style.box_color, (18, 171, 205, 239))
        self.assertEqual(style.label_background_color, (0, 0, 0, 0))
        self.assertEqual(style.box_style, "corners")
        self.assertEqual(
            renderer.label_text(renderer.Box(0, 0, 1, 1, "Blue Jay", 0.875), style),
            "Blue Jay (88%)",
        )


if __name__ == "__main__":
    unittest.main()