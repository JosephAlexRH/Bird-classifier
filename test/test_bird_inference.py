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
    TFLiteModel,
    letterbox_array,
    letterbox_rgb,
    tf_resize_with_pad_geometry,
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
            result = engine.infer(image)
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

    def test_contract_and_threshold_file_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "threshold.json").write_text(
                json.dumps({"bird_probability_threshold": 0.5}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "disagrees"):
                self.build(root, 242)


class GeometryTests(unittest.TestCase):
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