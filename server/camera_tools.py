"""Browser camera tools on the hub, coordinated with its capture workers."""
from __future__ import annotations

import json
import threading
from urllib.request import Request, urlopen
from flask import Response, jsonify, request

try:
    from .capture_storage import METADATA_HEADER, is_jpeg, parse_camera_metadata, parse_calibration_type
except ImportError:
    from capture_storage import METADATA_HEADER, is_jpeg, parse_camera_metadata, parse_calibration_type


def register_camera_tools(app, registry, store, camera_proxy, on_capture):
    def camera_or_error(camera_id):
        camera = registry.resolve(camera_id)
        if camera is None:
            return None, (jsonify(error="Camera not found or no default selected."), 404)
        if not camera["enabled"]:
            return None, (jsonify(error="Camera is disabled."), 503)
        return camera, None

    def capture(camera_id=None):
        camera, error = camera_or_error(camera_id)
        if error is not None:
            return error
        options = request.get_json(silent=True)
        if options is None:
            options = {}
        if not isinstance(options, dict):
            return jsonify(error="Expected a JSON object."), 400
        try:
            calibration = parse_calibration_type(options.get("calibration_type", "normal"))
        except ValueError as error:
            return jsonify(error=str(error)), 422
        response = app.make_response(camera_proxy("capture", camera_id=camera["id"]))
        if response.status_code != 200:
            return response
        image = response.get_data()
        if response.mimetype != "image/jpeg" or not is_jpeg(image):
            return jsonify(error="Camera did not return a complete JPEG."), 502
        metadata, metadata_error = parse_camera_metadata(response.headers.get(METADATA_HEADER))
        record = store.save_capture(image, source="manual", camera_metadata=metadata,
            metadata_error=metadata_error, calibration_type=calibration,
            label=str(options.get("label", ""))[:120], note=str(options.get("note", ""))[:500],
            extra={"camera_registry_id": camera["id"], "camera_name": camera["name"]})
        on_capture(record)
        return jsonify(status="ok", **record)

    def profile(camera_id=None):
        camera, error = camera_or_error(camera_id)
        if error is not None:
            return error
        snapshot = {"camera_registry_id": camera["id"], "camera_name": camera["name"]}
        for endpoint in ("metadata", "config", "status"):
            response = app.make_response(camera_proxy(endpoint, camera_id=camera["id"]))
            if response.status_code != 200:
                return response
            try:
                snapshot[endpoint] = json.loads(response.get_data())
            except ValueError:
                return jsonify(error=f"Invalid camera {endpoint} JSON."), 502
        return jsonify(result="ok", **store.save_profile(snapshot))

    def stream(camera_id=None):
        camera, error = camera_or_error(camera_id)
        if error is not None:
            return error
        operation = registry.operation_lock(camera["id"])
        # The Pi hub queues dashboard requests behind its poller (see
        # acquire_camera); other servers keep the immediate "busy" answer.
        acquire = app.config.get("ACQUIRE_CAMERA") or (lambda _camera, lock: lock.acquire(blocking=False))
        if operation is None or not acquire(camera, operation):
            return jsonify(error="Camera is busy. Retry after the current capture."), 409
        upstream = None
        try:
            upstream = urlopen(Request(camera["url"] + "/stream"), timeout=10)
            content_type = upstream.headers.get("Content-Type", "")
            if upstream.status != 200 or not content_type.lower().startswith("multipart/x-mixed-replace"):
                raise ValueError("Camera did not return an MJPEG stream.")
        except (OSError, ValueError) as error:
            if upstream is not None:
                upstream.close()
            operation.release()
            return jsonify(error=str(error)), 502
        registry.set_streaming(camera["id"], True)
        closed = threading.Event()
        close_lock = threading.Lock()

        def close():
            with close_lock:
                if closed.is_set():
                    return
                closed.set()
                upstream.close()
                registry.set_streaming(camera["id"], False)
                operation.release()

        def generate():
            try:
                while not closed.is_set():
                    chunk = upstream.read(16 * 1024)
                    if not chunk:
                        break
                    yield chunk
            except OSError:
                pass
            finally:
                close()

        response = Response(generate(), content_type=content_type,
                            headers={"Cache-Control": "no-store"})
        response.call_on_close(close)
        return response

    app.add_url_rule("/api/cameras/<camera_id>/captures", "hub_manual_capture", capture, methods=["POST"])
    app.add_url_rule("/api/cameras/<camera_id>/profiles", "hub_camera_profile", profile, methods=["POST"])
    app.add_url_rule("/api/cameras/<camera_id>/stream", "hub_camera_stream", stream)
    app.add_url_rule("/api/captures", "hub_default_capture", capture, methods=["POST"])
    app.add_url_rule("/api/profiles", "hub_default_profile", profile, methods=["POST"])
    app.add_url_rule("/camera/stream", "hub_default_stream", stream)
