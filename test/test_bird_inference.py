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
    letterbox_rgb,
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