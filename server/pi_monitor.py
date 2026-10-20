"""Laptop dashboard that polls a Raspberry Pi bird classifier and optionally mirrors results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, render_template_string, send_file


DEFAULT_MIRROR_DIR = Path.home() / "Pictures" / "BirdPiMirror"
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024


def normalize_pi_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Pi URL must be an HTTP(S) base URL with a host.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("Pi URL must not contain credentials, a path, query, or fragment.")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), netloc, "", "", ""))


def pi_url_argument(value: str) -> str:
    try:
        return normalize_pi_url(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class PiPoller:
    """Maintain a restart-safe local view of the Pi's cursor-based capture API."""

    def __init__(self, pi_url: str, mirror_dir: Path | None, interval: float = 2.0) -> None:
        self.pi_url = normalize_pi_url(pi_url)
        self.mirror_dir = Path(mirror_dir).expanduser().resolve() if mirror_dir else None
        self.interval = max(0.25, interval)
        self.lock = threading.Lock()
        self.latest: dict[str, Any] | None = None
        self.health: dict[str, Any] | None = None
        self.error: str | None = None
        self.cursor = ""
        self.pending: set[str] = set()
        self.running = False
        self._thread: threading.Thread | None = None
        if self.mirror_dir:
            self.mirror_dir.mkdir(parents=True, exist_ok=True)
            cursor_path = self.mirror_dir / ".pi-monitor-cursor"
            if cursor_path.is_file():
                self.cursor = cursor_path.read_text(encoding="utf-8").strip()
            # A previous process may have advanced its cursor before deferred
            # inference completed. Resume polling any mirrored JPEG without a result.
            self.pending.update(
                image.stem
                for image in self.mirror_dir.glob("capture_*.jpg")
                if not (self.mirror_dir / f"{image.stem}.inference.json").is_file()
            )

    def _request(self, endpoint: str, max_bytes: int = MAX_DOWNLOAD_BYTES) -> tuple[bytes, int]:
        target = urljoin(f"{self.pi_url}/", endpoint.lstrip("/"))
        try:
            with urlopen(Request(target, headers={"Accept": "application/json"}), timeout=8) as response:
                content = response.read(max_bytes + 1)
                if len(content) > max_bytes:
                    raise ValueError(f"Pi response exceeded {max_bytes} bytes.")
                return content, response.status
        except HTTPError as error:
            body = error.read(4096).decode("utf-8", "replace")
            raise RuntimeError(f"Pi returned HTTP {error.code} for {target}: {body}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Could not reach Pi at {target}: {error}") from error

    def _json(self, endpoint: str) -> dict[str, Any]:
        body, _ = self._request(endpoint, 2 * 1024 * 1024)
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError(f"Pi response from {endpoint} was not a JSON object.")
        return value

    def _mirror(self, capture: dict[str, Any]) -> None:
        if self.mirror_dir is None:
            return
        capture_id = capture["capture_id"]
        image_path = self.mirror_dir / f"{capture_id}.jpg"
        if not image_path.exists():
            image, _ = self._request(capture["image_url"])
            expected = capture.get("sha256")
            actual = hashlib.sha256(image).hexdigest()
            if expected and actual != expected:
                raise ValueError(f"SHA-256 mismatch while mirroring {capture_id}.")
            _atomic_write(image_path, image)
        inference = capture.get("inference")
        if isinstance(inference, dict):
            _atomic_write(
                self.mirror_dir / f"{capture_id}.inference.json",
                (json.dumps(inference, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        motion = capture.get("motion")
        if isinstance(motion, dict):
            _atomic_write(
                self.mirror_dir / f"{capture_id}.motion.json",
                (json.dumps(motion, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )

    def poll_once(self) -> None:
        with self.lock:
            cursor = self.cursor
            pending = set(self.pending)
            latest = self.latest
        health = self._json("/health")
        listing = self._json(f"/api/captures?after={cursor}&limit=100")
        captures = listing.get("captures", [])
        if not isinstance(captures, list):
            raise ValueError("Pi capture listing did not contain an array.")
        for capture in captures:
            if not isinstance(capture, dict) or not isinstance(capture.get("capture_id"), str):
                continue
            capture_id = capture["capture_id"]
            if capture.get("inference_status") == "pending":
                pending.add(capture_id)
            else:
                pending.discard(capture_id)
            self._mirror(capture)
            latest = capture
            if capture_id > cursor:
                cursor = capture_id
        for capture_id in list(pending):
            detail = self._json(f"/api/captures/{capture_id}")
            if detail.get("inference_status") != "pending":
                pending.discard(capture_id)
                self._mirror(detail)
                if latest is None or capture_id >= latest.get("capture_id", ""):
                    latest = detail
        if self.mirror_dir and cursor:
            _atomic_write(self.mirror_dir / ".pi-monitor-cursor", (cursor + "\n").encode())
        with self.lock:
            self.cursor = cursor
            self.pending = pending
            self.latest = latest
            self.health = health
            self.error = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "pi_url": self.pi_url,
                "mirror_dir": str(self.mirror_dir) if self.mirror_dir else None,
                "cursor": self.cursor or None,
                "pending": sorted(self.pending),
                "latest": self.latest,
                "pi_health": self.health,
                "error": self.error,
            }

    def _run(self) -> None:
        while self.running:
            try:
                self.poll_once()
            except Exception as error:
                with self.lock:
                    self.error = str(error)
            time.sleep(self.interval)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, name="pi-poller", daemon=True)
        self._thread.start()


MONITOR_HTML = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bird Pi monitor</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#101418;color:#edf2f7}
main,header{width:min(1000px,calc(100% - 2rem));margin:auto}header{padding:1.5rem 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}
section{background:#192128;border:1px solid #2d3943;border-radius:12px;padding:1rem}img{width:100%;min-height:260px;object-fit:contain;background:#080a0c;border-radius:8px}pre{white-space:pre-wrap;max-height:36rem;overflow:auto}.error{color:#ff8b8b}.muted{color:#aab5c0}
</style></head><body><header><h1>Bird Pi laptop monitor</h1><p id="connection" class="muted">Connecting…</p></header>
<main class="grid"><section><h2>Latest capture</h2><img id="image" alt="No Pi capture received yet"></section>
<section><h2>Inference</h2><pre id="inference">Waiting…</pre></section><section><h2>Pipeline health</h2><pre id="health">Waiting…</pre></section></main>
<script>async function refresh(){const r=await fetch('/api/state',{cache:'no-store'});const s=await r.json();
connection.textContent=`Pi: ${s.pi_url} | mirror: ${s.mirror_dir||'disabled'}`;inference.textContent=JSON.stringify((s.latest||{}).inference||s.error||'Waiting for inference',null,2);health.textContent=JSON.stringify({error:s.error,pending:s.pending,pi_health:s.pi_health},null,2);
if(s.latest)image.src=`/latest.jpg?t=${Date.now()}`;}refresh().catch(e=>inference.textContent=String(e));setInterval(()=>refresh().catch(e=>inference.textContent=String(e)),2000);</script></body></html>
"""


def create_app(poller: PiPoller, *, start_poller: bool = True) -> Flask:
    app = Flask(__name__)
    if start_poller:
        poller.start()

    @app.get("/")
    def dashboard() -> str:
        return render_template_string(MONITOR_HTML)

    @app.get("/api/state")
    def state() -> Response:
        return jsonify(poller.snapshot())

    @app.get("/latest.jpg")
    def latest_image():
        latest = poller.snapshot()["latest"]
        if latest is None:
            return jsonify(error="No Pi capture has been observed yet."), 404
        if poller.mirror_dir:
            local = poller.mirror_dir / f"{latest['capture_id']}.jpg"
            if local.is_file():
                return send_file(local, mimetype="image/jpeg", max_age=0)
        try:
            body, _ = poller._request(latest["image_url"])
        except Exception as error:
            return jsonify(error=str(error)), 502
        return Response(body, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-url", required=True, type=pi_url_argument, help="Pi server base URL")
    parser.add_argument("--host", default="127.0.0.1", help="Monitor bind address")
    parser.add_argument("--port", type=int, default=5050, help="Monitor port")
    parser.add_argument("--mirror-dir", type=Path, default=DEFAULT_MIRROR_DIR, help="Local mirror directory")
    parser.add_argument("--no-mirror", action="store_true", help="Display without downloading a local copy")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    monitor = PiPoller(
        arguments.pi_url,
        None if arguments.no_mirror else arguments.mirror_dir,
        arguments.poll_seconds,
    )
    application = create_app(monitor)
    print(f"Laptop monitor: http://127.0.0.1:{arguments.port}", flush=True)
    application.run(host=arguments.host, port=arguments.port, debug=False, threaded=True)