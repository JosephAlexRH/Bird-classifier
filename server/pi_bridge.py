"""Laptop-side Pi monitor routes. Never poll the cameras from a second host."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Response, jsonify, request

try:
    from .pi_monitor import normalize_pi_url, _atomic_write
    from .collection_dashboard import collection_page
except ImportError:
    from pi_monitor import normalize_pi_url, _atomic_write
    from collection_dashboard import collection_page


def register_pi_bridge(app, config_path: Path, pi_url: str | None = None) -> None:
    lock = threading.Lock()
    try:
        saved = json.loads(config_path.read_text(encoding="utf-8")).get("pi_url")
    except (OSError, ValueError):
        saved = None
    app.config["PI_URL"] = normalize_pi_url(pi_url or saved) if pi_url or saved else None

    @app.get("/pi")
    def pi_dashboard():
        return collection_page(remote=True)

    @app.route("/api/pi/config", methods=["GET", "POST"])
    def pi_config():
        if request.method == "POST":
            payload = request.get_json(silent=True)
            try:
                if not isinstance(payload, dict) or not isinstance(payload.get("pi_url"), str):
                    raise ValueError("Supply a Pi base URL.")
                target = normalize_pi_url(payload["pi_url"])
            except ValueError as error:
                return jsonify(error=str(error)), 400
            with lock:
                _atomic_write(config_path, (json.dumps({"pi_url": target}) + "\n").encode())
                app.config["PI_URL"] = target
                if not app.config.get("DIRECT_CAMERA_CONFIGURED") and app.config.get("CAMERA_REGISTRY"):
                    app.config["PI_MONITOR_ONLY"] = True
                    app.config["CAMERA_REGISTRY"].stop()
                    advertiser = app.config.get("HUB_ADVERTISER")
                    if advertiser is not None:
                        advertiser.stop()
                        app.config["HUB_ADVERTISER"] = None
        return jsonify(pi_url=app.config["PI_URL"])

    def proxy(endpoint: str, *, method="GET"):
        target = app.config["PI_URL"]
        if not target:
            return jsonify(error="Connect a Pi first using its base URL."), 503
        # Only routes below select endpoints; a response cannot redirect the
        # browser's image links to an arbitrary upstream supplied URL.
        query = {key: request.args[key] for key in ("order", "limit", "after", "before", "camera_id", "camera_registry_id")
                 if key in request.args}
        if endpoint == "/api/hub/fleet":
            query = {key: request.args[key] for key in ("since",) if key in request.args}
        elif endpoint != "/api/captures":
            query = {}
        data = None
        headers = {}
        if method in ("POST", "PATCH"):
            data = json.dumps(request.get_json(silent=True)).encode()
            headers["Content-Type"] = "application/json"
        outgoing = Request(target + endpoint + ("?" + urlencode(query) if query else ""),
                           data=data, headers=headers, method=method)
        try:
            timeout = 90 if endpoint == "/api/hub/settings" or endpoint.endswith("/pipeline") else 15
            with urlopen(outgoing, timeout=timeout) as upstream:
                body = upstream.read(12 * 1024 * 1024 + 1)
                if len(body) > 12 * 1024 * 1024:
                    return jsonify(error="Pi response exceeded monitor limit."), 502
                return Response(body, status=upstream.status, content_type=upstream.headers.get("Content-Type"),
                                headers={"Cache-Control": "no-store"})
        except HTTPError as error:
            return Response(error.read(4096), status=error.code, content_type="application/json")
        except (OSError, ValueError) as error:
            return jsonify(error=f"Pi unavailable: {error}"), 502

    @app.get("/api/pi/health")
    def pi_health():
        return proxy("/health")

    @app.route("/api/pi/hub/settings", methods=["GET", "POST"])
    def pi_hub_settings():
        return proxy("/api/hub/settings", method=request.method)

    @app.route("/api/pi/cameras/<camera_id>/<operation>", methods=["GET", "POST"])
    def pi_camera_tool(camera_id, operation):
        allowed = {"config": {"GET", "POST"}, "pipeline": {"POST"}, "capture": {"GET"},
                   "captures": {"POST"}, "profiles": {"POST"}, "status": {"GET"},
                   "metadata": {"GET"}, "latest": {"GET"}, "latest.jpg": {"GET"}}
        allowed['background'] = {'GET', 'POST'}
        if not valid_id(camera_id) or operation not in allowed or request.method not in allowed[operation]:
            return jsonify(error="Unsupported camera operation."), 404
        return proxy(f"/api/cameras/{camera_id}/{operation}", method=request.method)

    @app.delete("/api/pi/cameras/<camera_id>")
    def pi_camera_remove(camera_id):
        if not valid_id(camera_id):
            return jsonify(error="Unsupported camera operation."), 404
        return proxy(f"/api/cameras/{camera_id}", method="DELETE")

    @app.patch("/api/pi/cameras/<camera_id>")
    def pi_camera_update(camera_id):
        payload = request.get_json(silent=True)
        if not valid_id(camera_id) or not isinstance(payload, dict) or set(payload) - {"name", "color"}:
            return jsonify(error="Only name and color can be changed here."), 400
        return proxy(f"/api/cameras/{camera_id}", method="PATCH")

    @app.delete("/api/pi/fleet/<camera_id>")
    def pi_fleet_forget(camera_id):
        if not valid_id(camera_id):
            return jsonify(error="Unsupported camera operation."), 404
        return proxy(f"/api/fleet/{camera_id}", method="DELETE")

    @app.get("/api/pi/cameras/<camera_id>/stream")
    def pi_camera_stream(camera_id):
        target = app.config["PI_URL"]
        if not target or not valid_id(camera_id):
            return jsonify(error="Connect a Pi and select a camera first."), 400
        try:
            upstream = urlopen(target + f"/api/cameras/{camera_id}/stream", timeout=15)
        except HTTPError as error:
            return Response(error.read(4096), status=error.code, content_type="application/json")
        except OSError as error:
            return jsonify(error=str(error)), 502
        content_type = upstream.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/x-mixed-replace"):
            upstream.close()
            return jsonify(error="Pi did not return an MJPEG stream."), 502
        def generate():
            try:
                while True:
                    chunk = upstream.read(16 * 1024)
                    if not chunk:
                        break
                    yield chunk
            except OSError:
                pass
            finally:
                upstream.close()
        response = Response(generate(), content_type=content_type, headers={"Cache-Control": "no-store"})
        response.call_on_close(upstream.close)
        return response

    @app.route("/api/pi/time", methods=["GET", "POST"])
    def pi_time():
        return proxy("/api/time", method=request.method)

    @app.get("/api/pi/captures")
    def pi_captures():
        return proxy("/api/captures")

    @app.get("/api/pi/hub/specs")
    def pi_specs():
        return proxy("/api/hub/specs")

    @app.get("/api/pi/hub/fleet")
    def pi_fleet():
        return proxy("/api/hub/fleet")

    @app.get("/api/pi/collection/export.csv")
    def pi_export():
        result = proxy("/api/collection/export.csv")
        if isinstance(result, Response) and result.status_code == 200:
            result.headers["Content-Disposition"] = 'attachment; filename="deployment_collection.csv"'
        return result

    @app.get("/api/pi/captures/<capture_id>/image")
    def pi_image(capture_id):
        if not valid_id(capture_id):
            return jsonify(error="Invalid capture ID."), 400
        return proxy(f"/api/captures/{capture_id}/image")

    @app.get("/api/pi/captures/<capture_id>/crops/<int:index>/image")
    def pi_crop(capture_id, index):
        if not valid_id(capture_id):
            return jsonify(error="Invalid capture ID."), 400
        return proxy(f"/api/captures/{capture_id}/crops/{index}/image")

    @app.post("/api/pi/captures/<capture_id>/review")
    def pi_review(capture_id):
        if not valid_id(capture_id):
            return jsonify(error="Invalid capture ID."), 400
        return proxy(f"/api/captures/{capture_id}/review", method="POST")

    @app.get("/api/pi/captures/<capture_id>/review-crops/<int:index>/image")
    def pi_reviewed_crop(capture_id, index):
        if not valid_id(capture_id):
            return jsonify(error="Invalid capture ID."), 400
        return proxy(f"/api/captures/{capture_id}/review-crops/{index}/image")


def valid_id(value: str) -> bool:
    return bool(value) and all(c.isascii() and (c.isalnum() or c in "_-") for c in value)
