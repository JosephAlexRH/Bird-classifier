"""Bird-centred species crops: the crop rule, the species contract, the ROI localiser and the engine."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from server.bird_inference import (
    BirdInferenceEngine,
    RoiLocalizer,
    SpeciesContract,
    species_crop_bounds,
    square_around,
)

LABELS = ["alpha", "beta", "gamma"]
GRID, ANCHORS = 8, [[0.09, 0.09], [0.2, 0.2], [0.31, 0.31]]


def jpeg(size=(640, 480), color=(90, 120, 60)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, "JPEG", quality=95)
    return output.getvalue()


def species_contract(**overrides) -> dict:
    data = {
        "model_file": "model_uint8.tflite", "classes": LABELS,
        "input_shape": [1, 224, 224, 3], "input_dtype": "float32", "output_dtype": "float32",
        "geometry": "detector_box_square_context_crop_then_letterbox", "geometry_location": "outside the model",
        "crop_rule": {"context": 1.5, "min_size_px": 48},
        "color_order": "RGB", "jpeg_decode_channels": 3, "resize_method": "TensorFlow bilinear",
        "resize_antialias": True, "letterbox_padding_value": 0,
        "numeric_scaling": "[0,255] to [-1,1] inside the model",
        "automation": {"species_min_confidence": 0.8, "group_min_confidence": 0.7, "auto_species": ["beta"],
                       "groups": {"ab": ["alpha", "beta"]}},
    }
    data.update(overrides)
    return data


def make_fake(*, localizer_hit: bool = True, bird: float = 0.95):
    """One fake interpreter class serving the is-bird, species and v3 localiser models by file name."""

    class Fake:
        instances: dict[str, "Fake"] = {}
        species_inputs: list[np.ndarray] = []

        def __init__(self, model_path: str, **_kwargs) -> None:
            self.name = Path(model_path).name
            self.tensor = None
            Fake.instances[self.name] = self

        def allocate_tensors(self) -> None:
            pass

        def get_input_details(self):
            size, dtype, quant = (128, np.uint8, (1.0, 0)) if self.name == "localizer.tflite" else (224, np.float32, (0.0, 0))
            return [{"index": 0, "shape": np.array([1, size, size, 3]), "dtype": dtype, "quantization": quant}]

        def get_output_details(self):
            if self.name == "localizer.tflite":
                return [
                    {"index": 1, "shape": np.array([1, GRID, GRID, 3]), "dtype": np.float32, "quantization": (0.0, 0)},
                    {"index": 2, "shape": np.array([1, GRID, GRID, 12]), "dtype": np.float32, "quantization": (0.0, 0)},
                ]
            width = 3 if self.name == "species.tflite" else 1
            return [{"index": 1, "shape": np.array([1, width]), "dtype": np.float32, "quantization": (0.0, 0)}]

        def set_tensor(self, _index: int, tensor: np.ndarray) -> None:
            self.tensor = tensor
            if self.name == "species.tflite":
                Fake.species_inputs.append(tensor.copy())

        def invoke(self) -> None:
            assert self.tensor is not None

        def get_tensor(self, index: int) -> np.ndarray:
            if self.name == "binary.tflite":
                return np.array([[bird]], np.float32)
            if self.name == "species.tflite":
                return np.array([[0.1, 0.85, 0.05]], np.float32)
            if index == 1:  # scores (logits): one confident cell, centre of the grid, middle anchor
                scores = np.full((1, GRID, GRID, 3), -30.0, np.float32)
                if localizer_hit:
                    scores[0, 4, 4, 1] = 6.0
                return scores
            return np.zeros((1, GRID, GRID, 12), np.float32)  # tx = ty = tw = th = 0

    return Fake


class Bundle:
    def __init__(self, root: Path, contract: dict | None = None, labels=LABELS):
        self.root = root
        self.binary, self.species, self.localizer = root / "binary.tflite", root / "species.tflite", root / "localizer.tflite"
        for path in (self.binary, self.species, self.localizer):
            path.write_bytes(path.name.encode())
        (root / "threshold.json").write_text(json.dumps({"bird_probability_threshold": 0.5}))
        (root / "labels.txt").write_text("\n".join(labels) + "\n")
        (root / "species_contract.json").write_text(json.dumps(contract or species_contract()))
        (root / "esp32_detector_contract.json").write_text(json.dumps({
            "schema_version": 2, "model_file": "localizer.tflite",
            "model_sha256": hashlib.sha256(self.localizer.read_bytes()).hexdigest(),
            "anchors_normalized_wh": ANCHORS,
            "decode": {"candidate_score_threshold": 0.05, "nms_iou": 0.45, "max_detections": 5},
            "pi_crop": {"pad_value": 114},
        }))
        self.image = root / "capture.jpg"
        self.image.write_bytes(jpeg())


class CropRuleTests(unittest.TestCase):
    def test_square_crop_is_centred_with_context(self):
        self.assertEqual(species_crop_bounds((300, 200, 340, 260), 640, 480), (275, 185, 365, 275))

    def test_minimum_size_applies_to_tiny_boxes(self):
        left, top, right, bottom = species_crop_bounds((100, 100, 110, 108), 640, 480)
        self.assertEqual((right - left, bottom - top), (48, 48))

    def test_crop_is_shifted_inside_the_frame_not_shrunk(self):
        self.assertEqual(species_crop_bounds((0, 400, 60, 480), 640, 480), (0, 360, 120, 480))

    def test_crop_never_exceeds_the_frame(self):
        left, top, right, bottom = species_crop_bounds((0, 0, 600, 470), 640, 480)
        self.assertTrue(0 <= left < right <= 640 and 0 <= top < bottom <= 480)
        self.assertEqual(bottom - top, 480)

    def test_empty_box_is_rejected(self):
        with self.assertRaises(ValueError):
            species_crop_bounds((10, 10, 10, 20), 640, 480)

    def test_square_around_keeps_long_side_and_stays_inside(self):
        self.assertEqual(square_around((500, 100, 640, 400), 640, 480), (340, 100, 640, 400))


class SpeciesContractTests(unittest.TestCase):
    def test_valid_contract_and_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "species_contract.json"
            path.write_text(json.dumps(species_contract()))
            contract = SpeciesContract(path)
            self.assertEqual((contract.image_size, contract.context, contract.min_size), (224, 1.5, 48))
            for bad in ({"letterbox_padding_value": 114}, {"geometry": "letterbox"},
                        {"crop_rule": {"context": 0, "min_size_px": 48}}, {"classes": []}):
                with self.subTest(bad=bad):
                    path.write_text(json.dumps(species_contract(**bad)))
                    with self.assertRaises(ValueError):
                        SpeciesContract(path)


class EngineSpeciesCropTests(unittest.TestCase):
    def test_is_bird_gate_with_localiser_runs_species_on_the_bird_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary))
            fake = make_fake()
            engine = BirdInferenceEngine(bundle.binary, bundle.species, localizer_model=bundle.localizer,
                                         interpreter_factory=fake)
            result = engine.infer(bundle.image)
        species = result["species"]
        self.assertTrue(species["localised"])
        self.assertEqual(species["box_source"], "localizer")
        box = species["boxes"][0]["box"]
        # middle anchor (0.2) at the grid centre of a 480 px square around the whole 640x480 frame
        self.assertAlmostEqual(box["right"] - box["left"], 0.2 * 480, places=3)
        expected = species_crop_bounds((box["left"], box["top"], box["right"], box["bottom"]), 640, 480)
        self.assertEqual(tuple(species["boxes"][0]["crop"].values()), expected)
        self.assertEqual(result["guess"], "beta")
        self.assertEqual(len(species["boxes"][0]["probabilities"]), 3)
        self.assertEqual(result["preprocessing"]["species"]["method"], "detector_box_square_context_crop_then_letterbox")
        self.assertEqual(fake.species_inputs[-1].shape, (1, 224, 224, 3))
        self.assertEqual(len(result["localizer"]["detections"]), 1)

    def test_no_localisation_falls_back_to_the_selected_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary))
            engine = BirdInferenceEngine(bundle.binary, bundle.species, localizer_model=bundle.localizer,
                                         interpreter_factory=make_fake(localizer_hit=False))
            result = engine.infer(bundle.image)
        self.assertFalse(result["species"]["localised"])
        self.assertEqual(result["species"]["box_source"], "selected_crop_fallback")
        self.assertEqual(result["species"]["boxes"][0]["crop"], {"left": 0, "top": 0, "right": 640, "bottom": 480})

    def test_species_is_skipped_when_the_gate_says_no_bird(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary))
            fake = make_fake(bird=0.1)
            engine = BirdInferenceEngine(bundle.binary, bundle.species, localizer_model=bundle.localizer,
                                         interpreter_factory=fake)
            result = engine.infer(bundle.image)
        self.assertFalse(result["is_bird"])
        self.assertIsNone(result["guess"])
        self.assertNotIn("species", result)
        self.assertEqual(fake.species_inputs, [])

    def test_species_can_be_switched_off_and_on_with_the_is_bird_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary))
            engine = BirdInferenceEngine(bundle.binary, bundle.species, localizer_model=bundle.localizer,
                                         interpreter_factory=make_fake())
            engine.set_species_enabled(False)
            off = engine.infer(bundle.image)
            engine.set_species_enabled(True)
            on = engine.infer(bundle.image)
        self.assertIsNone(off["guess"])
        self.assertNotIn("species", off)
        self.assertEqual(on["guess"], "beta")

    def test_label_mismatch_and_double_localisation_fail_at_startup(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary), labels=["beta", "alpha", "gamma"])
            with self.assertRaises(ValueError):
                BirdInferenceEngine(bundle.binary, bundle.species, interpreter_factory=make_fake())
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary))
            with self.assertRaises(ValueError):
                BirdInferenceEngine(None, bundle.species, detector_model=bundle.binary,
                                    localizer_model=bundle.localizer, interpreter_factory=make_fake())

    def test_localizer_contract_must_match_the_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary))
            bundle.localizer.write_bytes(b"a different model")
            with self.assertRaises(ValueError):
                RoiLocalizer(bundle.localizer, interpreter_factory=make_fake())


class StaticBoxEngineTests(unittest.TestCase):
    """The engine skips localiser boxes a box_filter marks static, and records empty-scene references."""

    def test_static_boxes_are_recorded_but_never_classified(self):
        def all_static(detections):
            return [{**d, "static": True} for d in detections], {"applied": True, "suppressed": len(detections)}

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary))
            engine = BirdInferenceEngine(bundle.binary, bundle.species, localizer_model=bundle.localizer,
                                         interpreter_factory=make_fake())
            result = engine.infer(bundle.image, box_filter=all_static)
        species = result["species"]
        self.assertFalse(species["localised"])
        self.assertEqual(species["box_source"], "selected_crop_fallback")
        self.assertEqual(species["static_boxes_skipped"], 1)
        self.assertEqual(species["boxes"][0]["crop"], {"left": 0, "top": 0, "right": 640, "bottom": 480})
        self.assertEqual(result["localizer"]["role"], "bird_frame")
        self.assertTrue(result["localizer"]["detections"][0]["static"])
        self.assertEqual(result["localizer"]["static_filter"], {"applied": True, "suppressed": 1})

    def test_non_static_boxes_are_classified_as_before(self):
        def none_static(detections):
            return [{**d, "static": False} for d in detections], {"applied": True, "suppressed": 0}

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary))
            engine = BirdInferenceEngine(bundle.binary, bundle.species, localizer_model=bundle.localizer,
                                         interpreter_factory=make_fake())
            result = engine.infer(bundle.image, box_filter=none_static)
        self.assertTrue(result["species"]["localised"])
        self.assertEqual(result["species"]["static_boxes_skipped"], 0)

    def test_confident_empty_frame_records_an_empty_scene_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary))
            fake = make_fake(bird=0.1)
            engine = BirdInferenceEngine(bundle.binary, bundle.species, localizer_model=bundle.localizer,
                                         interpreter_factory=fake)
            crop = {"left": 0, "top": 0, "right": 640, "bottom": 480}
            result = engine.infer(bundle.image, empty_reference_crop=crop)
        self.assertFalse(result["is_bird"])
        self.assertNotIn("species", result)
        self.assertEqual(fake.species_inputs, [])
        self.assertEqual(result["localizer"]["role"], "empty_reference")
        self.assertEqual(result["localizer"]["crop"], crop)
        self.assertEqual(len(result["localizer"]["detections"]), 1)
        self.assertGreater(result["timing_ms"]["localizer"], 0)

    def test_uncertain_non_bird_and_bird_frames_take_no_empty_reference(self):
        crop = {"left": 0, "top": 0, "right": 640, "bottom": 480}
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Bundle(Path(temporary))
            uncertain = BirdInferenceEngine(bundle.binary, bundle.species, localizer_model=bundle.localizer,
                                            interpreter_factory=make_fake(bird=0.4)).infer(
                bundle.image, empty_reference_crop=crop)
            bird = BirdInferenceEngine(bundle.binary, bundle.species, localizer_model=bundle.localizer,
                                       interpreter_factory=make_fake()).infer(bundle.image, empty_reference_crop=crop)
        self.assertNotIn("localizer", uncertain)  # 0.4 is not confidently empty (limit 0.25)
        self.assertEqual(bird["localizer"]["role"], "bird_frame")


if __name__ == "__main__":
    unittest.main()
