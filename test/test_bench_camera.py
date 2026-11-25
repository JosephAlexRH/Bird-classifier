import json
import sys
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_camera  # noqa: E402


def fake_jpeg(width: int, height: int) -> bytes:
    sof = b"\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03" + b"\x00" * 9
    return b"\xff\xd8" + b"\xff\xe0\x00\x04\x00\x00" + sof + b"\xff\xda\x00\x02" + b"\x11" * 100 + b"\xff\xd9"


SIZES = {"VGA": (640, 480), "UXGA": (1600, 1200)}


class FakeCamera:
    def __init__(self) -> None:
        self.profile = {"frame_size": "VGA", "jpeg_quality": 12, "xclk_hz": 20000000}
        self.posts: list[dict] = []
        camera = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/status":
                    return self._json({"camera_sensor_model": "OV5640"})
                if self.path == "/config":
                    return self._json({"active_sensor_key": "0x5640", "sensors": {"0x5640": dict(camera.profile)}})
                if self.path == "/capture":
                    data = fake_jpeg(*SIZES[camera.profile["frame_size"]])
                    meta = {"device": {"boot_id": "b1", "resources": {"chip_temperature_c": 41.0,
                            "heap": {"free_bytes": 150000, "minimum_free_bytes_since_boot": 120000},
                            "psram": {"free_bytes": 7000000}}}}
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("X-ESP32-Camera-Metadata", json.dumps(meta))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                camera.posts.append(body)
                camera.profile.update(body["sensors"]["0x5640"])
                self._json({"ok": True})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


class BenchCameraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FakeCamera()
        sleep = mock.patch.object(bench_camera.time, "sleep", lambda *_: None)  # skip settle delays
        sleep.start()
        self.addCleanup(sleep.stop)
        self.out = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        self.camera.server.shutdown()

    def test_jpeg_size_reads_sof_and_rejects_truncation(self) -> None:
        self.assertEqual(bench_camera.jpeg_size(fake_jpeg(800, 600)), (800, 600))
        self.assertIsNone(bench_camera.jpeg_size(fake_jpeg(800, 600)[:-2]))

    def test_size_sweep_measures_each_size_and_restores_profile(self) -> None:
        code = bench_camera.main(["--camera", self.camera.url, "--sizes", "vga,uxga", "--frames", "3",
                                  "--warmup", "0", "--out", str(self.out)])
        self.assertEqual(code, 0)
        report = json.loads(next(self.out.glob("bench_*.json")).read_text())
        results = report["cameras"][self.camera.url]["results"]
        self.assertEqual([r["resolution"] for r in results], ["640x480", "1600x1200"])
        self.assertEqual(results[0]["ok"], 3)
        self.assertEqual(self.camera.profile["frame_size"], "VGA")  # restored
        self.assertEqual(self.camera.posts[-1]["sensors"]["0x5640"]["xclk_hz"], 20000000)

    def test_soak_logs_rows_and_counts_no_problems(self) -> None:
        clock = iter(range(0, 10_000, 5))
        with mock.patch.object(bench_camera.time, "time", lambda: next(clock)):
            problems = bench_camera.soak([self.camera.url], minutes=0.5, interval=10, log_path=self.out / "s.csv")
        self.assertEqual(problems, 0)
        self.assertGreater(len((self.out / "s.csv").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
