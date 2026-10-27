"""Tests for transport-independent Pi inference and laptop polling."""

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
    DetectorContract,
    TFLiteModel,
    decode_detector_outputs,
    letterbox_array,
    letterbox_rgb,
    roi_crop_geometry,
    validate_selected_crop,
    tf_resize_with_pad_geometry,
    validate_normalized_roi,
    read_inference_record,
    write_inference_record,
)
from server.pi_monitor import PiPoller, normalize_pi_url


class FakeInterpreter:
    outputs: dict[str, np.ndarray] = {}
    input_dtype = np.float32
    output_dtype = np.float32
    input_quantization = (0.0, 0)
    output_quantization = (0.0, 0)

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.tensor = None

    def allocate_tensors(self) -> None:
        pass

    def get_input_details(self):
        return [{
            "index": 0,
            "shape": np.array([1, 224, 224, 3]),
            "dtype": self.input_dtype,
            "quantization": self.input_quantization,
        }]

    def get_output_details(self):
        output = self.outputs[Path(self.model_path).name]
        return [{
            "index": 1,
            "shape": np.array(output.shape),
            "dtype": self.output_dtype,
            "quantization": self.output_quantization,
        }]

    def set_tensor(self, _index: int, tensor: np.ndarray) -> None:
        self.tensor = tensor

    def invoke(self) -> None:
        if self.tensor is None:
            raise AssertionError("input was not set")

    def get_tensor(self, _index: int) -> np.ndarray:
        return self.outputs[Path(self.model_path).name].astype(self.output_dtype)


def jpeg_bytes(color=(20, 80, 160), size=(80, 40)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, "JPEG")
    return output.getvalue()


class BirdInferenceTests(unittest.TestCase):
    def test_letterbox_centers_landscape_image_on_black(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wide.jpg"
            path.write_bytes(jpeg_bytes(color=(250, 20, 10), size=(100, 50)))
            pixels = letterbox_rgb(path)
        self.assertEqual(pixels.shape, (224, 224, 3))
        self.assertTrue(np.all(pixels[:50] == 0))
        self.assertGreater(int(pixels[112, 112, 0]), 200)
        self.assertTrue(np.all(pixels[-50:] == 0))


    def test_model_quantizes_uint8_input_and_dequantizes_output(self) -> None:
        class QuantizedFake(FakeInterpreter):
            outputs = {"binary.tflite": np.array([[128]], dtype=np.uint8)}
            input_dtype = np.uint8
            output_dtype = np.uint8
            input_quantization = (1.0, 0)
            output_quantization = (1 / 256, 0)

        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "binary.tflite"
            model_path.write_bytes(b"model")
            model = TFLiteModel(model_path, output_size=1, interpreter_factory=QuantizedFake)
            output = model.predict_pixels(np.full((224, 224, 3), 300, dtype=np.float32))
        self.assertAlmostEqual(float(output[0]), 0.5)
        self.assertEqual(model.interpreter.tensor.dtype, np.uint8)
        self.assertEqual(int(model.interpreter.tensor.max()), 255)

    def test_engine_returns_binary_probability_and_stable_species_top_k(self) -> None:
        class EngineFake(FakeInterpreter):
            outputs = {
                "binary.tflite": np.array([[0.91]], dtype=np.float32),
                "species.tflite": np.array([[0.2, 0.7, 0.1]], dtype=np.float32),
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "binary.tflite"
            species = root / "species.tflite"
            binary.write_bytes(b"binary")
            species.write_bytes(b"species")
            (root / "labels.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            (root / "threshold.json").write_text(
                json.dumps({"bird_probability_threshold": 0.8}), encoding="utf-8"
            )
            image = root / "capture.jpg"
            image.write_bytes(jpeg_bytes())
            engine = BirdInferenceEngine(
                binary, species, top_k=2, interpreter_factory=EngineFake
            )
            result = engine.infer(image, roi={"malformed": True}, roi_source="irrelevant")
            write_inference_record(image, result)
            restored = read_inference_record(image)
        self.assertTrue(result["is_bird"])
        self.assertAlmostEqual(result["bird_probability"], 0.91, places=5)
        self.assertEqual(result["guess"], "beta")
        self.assertEqual([item["label"] for item in result["species_top_k"]], ["beta", "alpha"])
        self.assertEqual(restored["classification"]["label"], "beta")


UINT8_CONTRACT = {
    "model_file": "binary.tflite",
    "labels": {"0": "not_bird", "1": "bird"},
    "bird_probability_threshold": 0.9453125,
    "uint8_output_threshold": 242,
    "input_shape": [1, 224, 224, 3],
    "input_dtype": "uint8",
    "input_quantization": [1.0, 0],
    "color_order": "RGB",
    "jpeg_decode_channels": 3,
    "geometry": "letterbox",
    "geometry_location": "outside the model",
    "resize_method": "TensorFlow bilinear",
    "resize_antialias": True,
    "letterbox_padding_value": 0,
    "numeric_scaling": "[0,255] to [-1,1] inside the model",
    "output_dtype": "uint8",
    "output_quantization": [0.00390625, 0],
}

ROI_CONTRACT = dict(
    UINT8_CONTRACT,
    geometry="roi_crop_then_letterbox",
    roi_source="capture_sidecar.camera.requested.motion.roi",
    roi_coordinate_space="normalized_original_frame",
    roi_padding_fraction=0.2,
    roi_padding_semantics="total_dimension_increase",
    roi_pixel_rounding="floor_min_ceil_max",
)

SELECTED_CROP_CONTRACT = dict(
    UINT8_CONTRACT,
    geometry="selected_crop_then_letterbox",
    crop_source="capture_motion_record.selection.final_bounds_px",
    crop_coordinate_space="original_decoded_frame_pixels",
    crop_bounds_semantics="half_open_left_top_right_bottom",
    crop_is_final=True,
    additional_runtime_margin=0,
)


def quantized_fake(raw_output: int):
    class QuantizedFake(FakeInterpreter):
        outputs = {
            "binary.tflite": np.array([[raw_output]], dtype=np.uint8),
            "species.tflite": np.array([[0.2, 0.7, 0.1]], dtype=np.float32),
        }
        input_dtype = np.uint8
        output_dtype = np.uint8
        input_quantization = (1.0, 0)
        output_quantization = (0.00390625, 0)

        def get_output_details(self):
            details = super().get_output_details()
            if Path(self.model_path).name == "species.tflite":
                details[0]["dtype"] = np.float32
                details[0]["quantization"] = (0.0, 0)
            return details

        def get_tensor(self, _index: int) -> np.ndarray:
            return self.outputs[Path(self.model_path).name]

    return QuantizedFake


class ContractTests(unittest.TestCase):
    def build(self, root: Path, raw_output: int, contract=UINT8_CONTRACT, **kwargs):
        binary = root / "binary.tflite"
        species = root / "species.tflite"
        binary.write_bytes(b"binary")
        species.write_bytes(b"species")
        (root / "labels.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        (root / "inference_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        return BirdInferenceEngine(
            binary, species, interpreter_factory=quantized_fake(raw_output), **kwargs
        )

    def infer(self, engine: BirdInferenceEngine, root: Path) -> dict:
        image = root / "capture.jpg"
        image.write_bytes(jpeg_bytes(size=(64, 48)))
        return engine.infer(image)

    def test_contract_uses_exact_uint8_rule(self) -> None:
        for raw, expected in ((242, True), (241, False), (255, True)):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                engine = self.build(root, raw)
                result = self.infer(engine, root)
                self.assertEqual(engine.decision_rule, "raw_output >= 242")
                self.assertIs(result["is_bird"], expected)
                self.assertEqual(result["bird_decision"]["raw_output"], raw)
                self.assertAlmostEqual(result["bird_probability"], raw / 256)
                self.assertEqual(result["classification"]["confidence"], result["bird_probability"])


class DetectorInferenceTests(unittest.TestCase):
    MODEL_NAME = "detector_float.tflite"

    def write_bundle(self, root: Path) -> tuple[Path, Path, Path]:
        detector = root / self.MODEL_NAME
        species = root / "species.tflite"
        anchors_path = root / "anchors.json"
        detector.write_bytes(b"float detector")
        species.write_bytes(b"species")
        (root / "labels.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        anchors = [
            [0.30197447538375854, 0.30662021040916443],
            [0.5027866959571838, 0.5528455972671509],
            [0.831632673740387, 0.6747719645500183],
        ]
        anchors_path.write_text(
            json.dumps({"input_size": 224, "anchors": anchors}), encoding="utf-8"
        )
        tensor_details_path = root / "tflite_tensor_details.json"
        tensor_details_path.write_text(json.dumps({
            "float": {
                "path": str(detector),
                "sha256": hashlib.sha256(detector.read_bytes()).hexdigest(),
                "inputs": [{
                    "name": "serving_default_image:0",
                    "shape": [1, 224, 224, 3],
                    "dtype": "float32",
                    "quantization": [0.0, 0],
                }],
                "outputs": [
                    {
                        "name": "StatefulPartitionedCall_1:0",
                        "shape": [1, 14, 14, 3, 4],
                        "dtype": "float32",
                        "quantization": [0.0, 0],
                    },
                    {
                        "name": "StatefulPartitionedCall_1:1",
                        "shape": [1, 14, 14, 3, 1],
                        "dtype": "float32",
                        "quantization": [0.0, 0],
                    },
                ],
            }
        }), encoding="utf-8")
        contract = {
            "model_file": detector.name,
            "model_sha256": hashlib.sha256(detector.read_bytes()).hexdigest(),
            "anchors_file": anchors_path.name,
            "anchors_sha256": hashlib.sha256(anchors_path.read_bytes()).hexdigest(),
            "tensor_details_file": tensor_details_path.name,
            "tensor_details_sha256": hashlib.sha256(tensor_details_path.read_bytes()).hexdigest(),
            "input": {
                "name": "serving_default_image:0",
                "shape": [1, 224, 224, 3],
                "dtype": "float32",
                "quantization": [0.0, 0],
                "color_order": "RGB",
                "geometry": "selected_crop_then_letterbox",
                "geometry_location": "outside the model",
                "crop_source": "capture_motion_record.selection.final_bounds_px",
                "crop_coordinate_space": "original_decoded_frame_pixels",
                "crop_bounds_semantics": "half_open_left_top_right_bottom",
                "crop_is_final": True,
                "additional_runtime_margin": 0,
                "letterbox_padding_value": 114,
                "resize_method": "bilinear",
                "resize_antialias": True,
                "numeric_scaling": "[0,255] to [-1,1] inside the model",
            },
            "outputs": {
                "box": {
                    "name": "StatefulPartitionedCall_1:0",
                    "shape": [1, 14, 14, 3, 4],
                    "dtype": "float32",
                    "quantization": [0.0, 0],
                    "output_position": 0,
                },
                "score": {
                    "name": "StatefulPartitionedCall_1:1",
                    "shape": [1, 14, 14, 3, 1],
                    "dtype": "float32",
                    "quantization": [0.0, 0],
                    "output_position": 1,
                },
            },
            "anchors_normalized_wh": anchors,
            "decode": {
                "center_x": "(sigmoid(tx) * 3 - 1 + grid_x) / grid_width",
                "center_y": "(sigmoid(ty) * 3 - 1 + grid_y) / grid_height",
                "width": "anchor_width * exp(clamp(tw, -4, 4))",
                "height": "anchor_height * exp(clamp(th, -4, 4))",
                "score": "sigmoid(bird_score_logit)",
            },
            "postprocess": {
                "candidate_score_threshold": 0.01,
                "deployment_score_threshold": 0.5,
                "nms_iou": 0.45,
                "max_detections": 5,
            },
        }
        contract_path = root / "detector_contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        return detector, species, contract_path

    @staticmethod
    def fake_interpreter(*, wrong_input=False, wrong_score_shape=False):
        class SplitHeadFake:
            instances = {}
            box_output = np.zeros((1, 14, 14, 3, 4), dtype=np.float32)
            score_output = np.full((1, 14, 14, 3, 1), -30.0, dtype=np.float32)

            def __init__(self, model_path: str, **_kwargs) -> None:
                self.model_path = model_path
                self.tensor = None
                self.instances[Path(model_path).name] = self

            def allocate_tensors(self) -> None:
                pass

            def get_input_details(self):
                size = 288 if wrong_input and Path(self.model_path).name != "species.tflite" else 224
                return [{
                    "index": 0,
                    "name": "serving_default_image:0",
                    "shape": np.array([1, size, size, 3]),
                    "dtype": np.float32,
                    "quantization": (0.0, 0),
                }]

            def get_output_details(self):
                if Path(self.model_path).name == "species.tflite":
                    return [{
                        "index": 3,
                        "name": "species",
                        "shape": np.array([1, 3]),
                        "dtype": np.float32,
                        "quantization": (0.0, 0),
                    }]
                score_shape = [1, 7, 7, 3, 1] if wrong_score_shape else [1, 14, 14, 3, 1]
                return [
                    {
                        "index": 1,
                        "name": "StatefulPartitionedCall_1:0",
                        "shape": np.array([1, 14, 14, 3, 4]),
                        "dtype": np.float32,
                        "quantization": (0.0, 0),
                    },
                    {
                        "index": 2,
                        "name": "StatefulPartitionedCall_1:1",
                        "shape": np.array(score_shape),
                        "dtype": np.float32,
                        "quantization": (0.0, 0),
                    },
                ]

            def set_tensor(self, _index: int, tensor: np.ndarray) -> None:
                self.tensor = tensor

            def invoke(self) -> None:
                if self.tensor is None:
                    raise AssertionError("input was not set")

            def get_tensor(self, index: int) -> np.ndarray:
                if Path(self.model_path).name == "species.tflite":
                    return np.array([[0.2, 0.7, 0.1]], dtype=np.float32)
                return self.box_output.copy() if index == 1 else self.score_output.copy()

        return SplitHeadFake

    def test_detector_contract_fails_startup_on_hash_input_output_and_anchor_mismatch(self) -> None:
        cases = ("hash", "input", "output", "anchors", "metadata")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                detector, species, contract_path = self.write_bundle(root)
                fake = self.fake_interpreter(
                    wrong_input=case == "input", wrong_score_shape=case == "output"
                )
                if case == "hash":
                    detector.write_bytes(b"different model")
                if case == "anchors":
                    (root / "anchors.json").write_text("{}", encoding="utf-8")
                if case == "metadata":
                    (root / "tflite_tensor_details.json").write_text("{}", encoding="utf-8")
                with self.assertRaises(ValueError):
                    BirdInferenceEngine(
                        None,
                        species,
                        detector_model=detector,
                        detector_contract_file=contract_path,
                        interpreter_factory=fake,
                    )

    def test_decode_uses_span_three_formula_full_frame_offsets_and_five_box_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, contract_path = self.write_bundle(root)
            contract = DetectorContract(contract_path)
            boxes = np.zeros((1, 14, 14, 3, 4), dtype=np.float32)
            scores = np.full((1, 14, 14, 3, 1), -30.0, dtype=np.float32)
            boxes[0, 5, 4, 0, 0] = np.log(3.0)
            boxes[0, 5, 4, 0, 1] = -np.log(3.0)
            scores[0, 5, 4, 0, 0] = 6.0
            decoded = decode_detector_outputs(boxes, scores, contract, (20, 10, 160, 150))
            expected_center_x = 20 + ((0.75 * 3 - 1 + 4) / 14) * 140
            expected_center_y = 10 + ((0.25 * 3 - 1 + 5) / 14) * 140
            self.assertAlmostEqual(
                (decoded[0]["box"]["left"] + decoded[0]["box"]["right"]) / 2,
                expected_center_x,
                places=4,
            )
            self.assertAlmostEqual(
                (decoded[0]["box"]["top"] + decoded[0]["box"]["bottom"]) / 2,
                expected_center_y,
                places=4,
            )

            boxes.fill(0)
            scores.fill(-30)
            for rank, (y, x) in enumerate(((1, 1), (1, 5), (1, 9), (6, 1), (6, 5), (6, 9))):
                scores[0, y, x, 0, 0] = 8.0 - rank * 0.1
            capped = decode_detector_outputs(boxes, scores, contract, (0, 0, 224, 224))
            self.assertEqual(len(capped), 5)

    def test_engine_uses_final_crop_pad_114_threshold_point_five_and_whole_frame_species(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector, species, contract_path = self.write_bundle(root)
            fake = self.fake_interpreter()
            fake.score_output[0, 7, 7, 0, 0] = 0.0
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            image[:, :] = (0, 0, 250)
            image[20:80, 40:160] = (250, 0, 0)
            Image.fromarray(image).save(root / "capture.png")
            engine = BirdInferenceEngine(
                None,
                species,
                detector_model=detector,
                detector_contract_file=contract_path,
                interpreter_factory=fake,
            )
            result = engine.infer(
                root / "capture.png",
                selected_crop={"left": 40, "top": 20, "right": 160, "bottom": 80},
                crop_source="capture_motion_record.selection.final_bounds_px",
            )
        detector_tensor = fake.instances[self.MODEL_NAME].tensor[0]
        species_tensor = fake.instances["species.tflite"].tensor[0]
        self.assertTrue(np.all(detector_tensor[:56] == 114))
        self.assertGreater(float(detector_tensor[112, :, 0].mean()), 200)
        self.assertGreater(float(species_tensor[112, 20, 2]), 200)
        self.assertTrue(result["is_bird"])
        self.assertAlmostEqual(result["detector"]["top_score"], 0.5)
        self.assertEqual(result["detector"]["threshold"], 0.5)
        self.assertTrue(result["detector"]["decision"])
        self.assertEqual(result["detector"]["model_sha256"], hashlib.sha256(b"float detector").hexdigest())
        self.assertGreaterEqual(result["detector"]["box"]["left"], 40)
        self.assertGreaterEqual(result["detector"]["box"]["top"], 20)
        self.assertEqual(result["preprocessing"]["crop"]["additional_runtime_margin"], 0)


class ContractCompatibilityTests(unittest.TestCase):
    def build(self, root: Path, raw_output: int, contract=UINT8_CONTRACT, **kwargs):
        binary = root / "binary.tflite"
        species = root / "species.tflite"
        binary.write_bytes(b"binary")
        species.write_bytes(b"species")
        (root / "labels.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        (root / "inference_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        return BirdInferenceEngine(
            binary, species, interpreter_factory=quantized_fake(raw_output), **kwargs
        )

    def infer(self, engine: BirdInferenceEngine, root: Path) -> dict:
        image = root / "capture.jpg"
        image.write_bytes(jpeg_bytes(size=(64, 48)))
        return engine.infer(image)

    def test_override_uses_probability_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = self.build(root, 200, bird_threshold=0.5)
            result = self.infer(engine, root)
        self.assertEqual(engine.decision_rule, "bird_probability >= 0.5")
        self.assertTrue(result["is_bird"])

    def test_contract_rejects_mismatched_model_tensors(self) -> None:
        contract = dict(UINT8_CONTRACT, input_dtype="float32", input_quantization=[0.0, 0])
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "input dtype"):
            self.build(Path(temporary), 242, contract=contract)

    def test_contract_rejects_unsupported_geometry(self) -> None:
        contract = dict(UINT8_CONTRACT, geometry="center_crop")
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "geometry"):
            self.build(Path(temporary), 242, contract=contract)

    def test_crop_contract_requires_every_geometry_field(self) -> None:
        required = (
            "geometry_location", "roi_source", "roi_coordinate_space",
            "roi_padding_fraction", "roi_padding_semantics", "roi_pixel_rounding",
        )
        for key in required:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                contract = dict(ROI_CONTRACT)
                del contract[key]
                with self.assertRaisesRegex(ValueError, key):
                    self.build(Path(temporary), 242, contract=contract)

    def test_crop_contract_rejects_wrong_geometry_values(self) -> None:
        bad_values = {
            "roi_source": "somewhere.else",
            "roi_coordinate_space": "pixels",
            "roi_padding_fraction": 0.1,
            "roi_padding_semantics": "each_side",
            "roi_pixel_rounding": "round",
        }
        for key, value in bad_values.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                contract = dict(ROI_CONTRACT, **{key: value})
                with self.assertRaisesRegex(ValueError, key):
                    self.build(Path(temporary), 242, contract=contract)

    def test_contract_and_threshold_file_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "threshold.json").write_text(
                json.dumps({"bird_probability_threshold": 0.5}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "disagrees"):
                self.build(root, 242)


class GeometryTests(unittest.TestCase):
    def test_strict_normalized_roi_validation(self) -> None:
        valid = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
        self.assertEqual(validate_normalized_roi(valid), valid)
        invalid = (
            None,
            [0.1, 0.2, 0.3, 0.4],
            {"x": 0.1, "y": 0.2, "width": 0.3},
            dict(valid, extra=1),
            dict(valid, x=True),
            dict(valid, x="0.1"),
            dict(valid, x=float("nan")),
            dict(valid, x=float("inf")),
            dict(valid, x=-0.01),
            dict(valid, width=0),
            dict(valid, width=-0.1),
            dict(valid, x=0.8, width=0.3),
            dict(valid, y=0.8, height=0.3),
        )
        for roi in invalid:
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                validate_normalized_roi(roi)

    def test_roi_padding_clips_and_rounds_outward(self) -> None:
        requested, padded, bounds = roi_crop_geometry(
            {"x": 0.03, "y": 0.2, "width": 0.5, "height": 0.3}, 101, 53
        )
        self.assertEqual(requested, {"x": 0.03, "y": 0.2, "width": 0.5, "height": 0.3})
        # x padding clips at zero; y gains 0.03 (10% of height) on each side.
        self.assertAlmostEqual(padded["x"], 0.0)
        self.assertAlmostEqual(padded["width"], 0.58)
        self.assertAlmostEqual(padded["y"], 0.17)
        self.assertAlmostEqual(padded["height"], 0.36)
        self.assertEqual(bounds, (0, 9, 59, 29))

    def test_selected_crop_validation_is_strict_half_open_pixels(self) -> None:
        self.assertEqual(
            validate_selected_crop({"left": 10, "top": 20, "right": 30, "bottom": 40}, 100, 80),
            (10, 20, 30, 40),
        )
        for crop in (
            {"left": 10.0, "top": 20, "right": 30, "bottom": 40},
            {"left": 10, "top": 20, "right": 10, "bottom": 40},
            {"left": -1, "top": 20, "right": 30, "bottom": 40},
            {"left": 10, "top": 20, "right": 101, "bottom": 40},
        ):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                validate_selected_crop(crop, 100, 80)

    def test_resize_with_pad_geometry_matches_tensorflow(self) -> None:
        # Values from tf.image.resize_with_pad, including its float32 floor quirks.
        self.assertEqual(tf_resize_with_pad_geometry(480, 640, 224), (168, 224, 28, 0))
        self.assertEqual(tf_resize_with_pad_geometry(100, 50, 224), (224, 112, 0, 56))
        self.assertEqual(tf_resize_with_pad_geometry(333, 111, 224), (223, 74, 0, 74))
        self.assertEqual(tf_resize_with_pad_geometry(1536, 2048, 224), (168, 223, 28, 0))

    def test_same_size_resize_is_identity(self) -> None:
        rgb = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        np.testing.assert_array_equal(letterbox_array(rgb), rgb.astype(np.float32))

    def test_letterbox_preserves_constant_color_and_float_output(self) -> None:
        rgb = np.full((480, 640, 3), (10, 120, 250), dtype=np.uint8)
        pixels = letterbox_array(rgb)
        self.assertEqual(pixels.dtype, np.float32)
        self.assertTrue(np.all(pixels[:28] == 0) and np.all(pixels[196:] == 0))
        np.testing.assert_allclose(pixels[28:196], np.broadcast_to([10, 120, 250], (168, 224, 3)), atol=1e-3)


class RoiInferenceTests(unittest.TestCase):
    def build(self, root: Path):
        class RecordingFake(FakeInterpreter):
            outputs = {
                "binary.tflite": np.array([[0.9]], dtype=np.float32),
                "species.tflite": np.array([[0.2, 0.7, 0.1]], dtype=np.float32),
            }
            instances = {}

            def __init__(self, model_path: str) -> None:
                super().__init__(model_path)
                self.instances[Path(model_path).name] = self

        binary = root / "binary.tflite"
        species = root / "species.tflite"
        binary.write_bytes(b"binary")
        species.write_bytes(b"species")
        (root / "labels.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        contract = dict(
            ROI_CONTRACT,
            input_dtype="float32",
            input_quantization=[0.0, 0],
            output_dtype="float32",
            output_quantization=[0.0, 0],
            bird_probability_threshold=0.5,
            uint8_output_threshold=None,
        )
        (root / "inference_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        return BirdInferenceEngine(binary, species, interpreter_factory=RecordingFake), RecordingFake

    def test_roi_is_required_with_matching_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, _ = self.build(root)
            image = root / "capture.jpg"
            image.write_bytes(jpeg_bytes(size=(100, 50)))
            with self.assertRaisesRegex(ValueError, "roi is required"):
                engine.infer(image)
            with self.assertRaisesRegex(ValueError, "roi_source is required"):
                engine.infer(image, roi={"x": 0.2, "y": 0.2, "width": 0.4, "height": 0.4})
            with self.assertRaisesRegex(ValueError, "does not match"):
                engine.infer(
                    image,
                    roi={"x": 0.2, "y": 0.2, "width": 0.4, "height": 0.4},
                    roi_source="caller.guess",
                )

    def test_crop_precedes_letterbox_and_species_uses_whole_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, fake = self.build(root)
            # ROI region is red, while the rest of the wide whole frame is blue.
            rgb = np.zeros((50, 100, 3), dtype=np.uint8)
            rgb[:] = (0, 0, 255)
            rgb[15:35, 30:70] = (255, 0, 0)
            image = root / "capture.png"
            Image.fromarray(rgb).save(image)
            roi = {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4}
            result = engine.infer(
                image,
                roi=roi,
                roi_source="capture_sidecar.camera.requested.motion.roi",
            )
            bird_tensor = fake.instances["binary.tflite"].tensor[0]
            species_tensor = fake.instances["species.tflite"].tensor[0]

        # Crop is 48x24 after 10% per-side padding, then letterboxed. Its centre is red.
        self.assertGreater(float(bird_tensor[112, 112, 0]), 200)
        self.assertLess(float(bird_tensor[112, 112, 2]), 30)
        # Species gets the uncropped wide frame (black bars top/bottom and blue left area).
        self.assertTrue(np.all(species_tensor[:50] == 0))
        self.assertGreater(float(species_tensor[112, 20, 2]), 200)
        self.assertEqual(result["preprocessing"]["is_bird"]["method"], "roi_crop_then_letterbox")
        self.assertEqual(result["preprocessing"]["species"]["method"], "letterbox")
        crop = result["preprocessing"]["crop"]
        self.assertEqual(crop["requested_roi"], roi)
        self.assertEqual(crop["pixel_bounds"], {
            "left": 26, "top": 13, "right": 75, "bottom": 38, "width": 49, "height": 25,
        })
        self.assertEqual(crop["roi_source"], "capture_sidecar.camera.requested.motion.roi")

    def test_letterbox_contract_remains_usable_without_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Reuse the normal float fake and override crop geometry back to legacy.
            engine, _ = self.build(root)
            engine.contract.geometry = "letterbox"
            image = root / "capture.jpg"
            image.write_bytes(jpeg_bytes())
            result = engine.infer(image)
        self.assertEqual(result["preprocessing"]["method"], "letterbox")
        self.assertIsNone(result["preprocessing"]["crop"])
        self.assertEqual(result["preprocessing"]["species"]["source"], "whole_original_frame")

    def test_selected_crop_is_applied_once_and_species_remains_whole_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            class RecordingFake(FakeInterpreter):
                outputs = {
                    "binary.tflite": np.array([[0.9]], dtype=np.float32),
                    "species.tflite": np.array([[0.2, 0.7, 0.1]], dtype=np.float32),
                }
                instances = {}

                def __init__(self, model_path: str) -> None:
                    super().__init__(model_path)
                    self.instances[Path(model_path).name] = self

            binary, species = root / "binary.tflite", root / "species.tflite"
            binary.write_bytes(b"binary")
            species.write_bytes(b"species")
            (root / "labels.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            contract = dict(
                SELECTED_CROP_CONTRACT,
                input_dtype="float32", input_quantization=[0.0, 0],
                output_dtype="float32", output_quantization=[0.0, 0],
                bird_probability_threshold=0.5, uint8_output_threshold=None,
            )
            (root / "inference_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            image = np.zeros((50, 100, 3), dtype=np.uint8)
            image[:, :] = (0, 0, 250)
            image[10:30, 40:60] = (250, 0, 0)
            Image.fromarray(image).save(root / "capture.png")
            engine = BirdInferenceEngine(binary, species, interpreter_factory=RecordingFake)
            result = engine.infer(
                root / "capture.png",
                selected_crop={"left": 40, "top": 10, "right": 60, "bottom": 30},
                crop_source="capture_motion_record.selection.final_bounds_px",
            )
        bird_tensor = RecordingFake.instances["binary.tflite"].tensor[0]
        species_tensor = RecordingFake.instances["species.tflite"].tensor[0]
        self.assertGreater(float(bird_tensor[..., 0].mean()), 200)
        self.assertGreater(float(species_tensor[112, 20, 2]), 200)
        crop = result["preprocessing"]["crop"]
        self.assertEqual(crop["pixel_bounds"]["width"], 20)
        self.assertEqual(crop["additional_runtime_margin"], 0)

    def test_selected_crop_contract_requires_final_crop_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, species = root / "binary.tflite", root / "species.tflite"
            binary.write_bytes(b"binary")
            species.write_bytes(b"species")
            (root / "labels.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            bad = dict(SELECTED_CROP_CONTRACT)
            del bad["crop_is_final"]
            (root / "inference_contract.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crop_is_final"):
                BirdInferenceEngine(binary, species, interpreter_factory=quantized_fake(200))


class PiPollerTests(unittest.TestCase):
    def test_pi_url_validation(self) -> None:
        self.assertEqual(normalize_pi_url("HTTP://PI.Local/"), "http://pi.local")
        for invalid in ("pi.local", "http://pi.local/path", "http://user@pi.local"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_pi_url(invalid)

    def test_poll_mirrors_once_and_advances_cursor(self) -> None:
        image = jpeg_bytes()
        digest = hashlib.sha256(image).hexdigest()

        class FakePoller(PiPoller):
            def _json(self, endpoint: str):
                if endpoint == "/health":
                    return {"status": "ok"}
                if endpoint.startswith("/api/captures?"):
                    if self.cursor:
                        return {"captures": []}
                    return {"captures": [{
                        "capture_id": "capture_001",
                        "sha256": digest,
                        "image_url": "/api/captures/capture_001/image",
                        "inference_status": "complete",
                        "inference": {"guess": "beta", "bird_probability": 0.9},
                    }]}
                raise AssertionError(endpoint)

            def _request(self, endpoint: str, max_bytes: int = 0):
                self.downloads += 1
                return image, 200

        with tempfile.TemporaryDirectory() as temporary:
            poller = FakePoller("http://pi.local", Path(temporary))
            poller.downloads = 0
            poller.poll_once()
            poller.poll_once()
            root = Path(temporary)
            self.assertEqual((root / "capture_001.jpg").read_bytes(), image)
            inference = json.loads((root / "capture_001.inference.json").read_text())
            self.assertEqual(inference["guess"], "beta")
            self.assertEqual((root / ".pi-monitor-cursor").read_text().strip(), "capture_001")
            self.assertEqual(poller.downloads, 1)

    def test_restart_recovers_mirrored_capture_with_pending_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "capture_001.jpg").write_bytes(jpeg_bytes())
            (root / ".pi-monitor-cursor").write_text("capture_001\n", encoding="utf-8")
            poller = PiPoller("http://pi.local", root)
            self.assertEqual(poller.cursor, "capture_001")
            self.assertEqual(poller.pending, {"capture_001"})


if __name__ == "__main__":
    unittest.main()