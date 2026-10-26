"""Capture reviewable environment frames directly from an ESP32-CAM.

Run without ``--interval-minutes`` for one capture, which is suitable for
Windows Task Scheduler. Supplying an interval keeps the process running and
takes the first capture immediately.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    from .capture_storage import METADATA_HEADER, CaptureStore, is_jpeg, parse_camera_metadata
except ImportError:  # Direct execution: python server/camera_capture_job.py
    from capture_storage import METADATA_HEADER, CaptureStore, is_jpeg, parse_camera_metadata


DEFAULT_SAVE_DIR = Path.home() / "Pictures" / "ESP32Environment"
MAX_JPEG_BYTES = 10 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 8.0
COLLECTION_PURPOSE = "environment_negative_candidate"


class CameraCaptureError(RuntimeError):
    """Raised when the camera does not provide a usable capture."""


def normalize_camera_url(value: str) -> str:
    """Validate and normalize an ESP32 HTTP base URL."""
    candidate = value.strip()
    if not candidate:
        raise ValueError("Camera URL must not be empty.")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"Invalid camera URL: {error}.") from error
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Camera URL must use http:// or https://.")
    if not parsed.hostname:
        raise ValueError("Camera URL must include a host.")
    if any(character.isspace() for character in parsed.hostname):
        raise ValueError("Camera URL host must not contain whitespace.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Camera URL must not include credentials.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Camera URL must be a base URL without a path, query, or fragment.")

    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, "", "", ""))


def camera_url_argument(value: str) -> str:
    try:
        return normalize_camera_url(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def positive_number(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Value must be a number greater than zero.") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Value must be a number greater than zero.")
    return number


def _content_type(headers: Any) -> str:
    get_content_type = getattr(headers, "get_content_type", None)
    if callable(get_content_type):
        return str(get_content_type()).lower()
    return str(headers.get("Content-Type", "")).partition(";")[0].strip().lower()


def fetch_camera_capture(
    camera_url: str,
    *,
    endpoint: str = "capture",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_jpeg_bytes: int = MAX_JPEG_BYTES,
    opener: Callable[..., Any] = urlopen,
) -> tuple[bytes, str | None]:
    """Fetch one bounded JPEG and its optional firmware metadata header."""
    camera_url = normalize_camera_url(camera_url)
    if not endpoint or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in endpoint):
        raise ValueError("Camera endpoint must be one lowercase path segment.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero.")
    if max_jpeg_bytes < 4:
        raise ValueError("max_jpeg_bytes must be at least four.")

    target = f"{camera_url}/{endpoint}"
    outgoing = Request(target, headers={"Accept": "image/jpeg"}, method="GET")
    try:
        with opener(outgoing, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise CameraCaptureError(f"Camera capture returned HTTP {status} from {target}.")
            if _content_type(response.headers) != "image/jpeg":
                received = response.headers.get("Content-Type", "not supplied")
                raise CameraCaptureError(
                    f"Camera capture returned Content-Type {received!r}; expected image/jpeg."
                )
            declared_length = response.headers.get("Content-Length")
            if declared_length:
                try:
                    if int(declared_length) > max_jpeg_bytes:
                        raise CameraCaptureError(
                            f"Camera JPEG exceeds the {max_jpeg_bytes}-byte limit."
                        )
                except ValueError:
                    pass
            image = response.read(max_jpeg_bytes + 1)
            metadata_header = response.headers.get(METADATA_HEADER)
    except HTTPError as error:
        raise CameraCaptureError(f"Camera capture returned HTTP {error.code} from {target}.") from error
    except (URLError, TimeoutError, OSError) as error:
        reason = error.reason if isinstance(error, URLError) else error
        raise CameraCaptureError(f"Camera request to {target} failed: {reason}") from error

    if len(image) > max_jpeg_bytes:
        raise CameraCaptureError(f"Camera JPEG exceeds the {max_jpeg_bytes}-byte limit.")
    if not is_jpeg(image):
        raise CameraCaptureError("Camera did not return a complete JPEG image.")
    return image, metadata_header


def capture_environment_frame(
    camera_url: str,
    store: CaptureStore,
    *,
    collection_mode: str = "one_shot",
    interval_seconds: float | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_jpeg_bytes: int = MAX_JPEG_BYTES,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Capture and durably store one unlabelled environment-negative candidate."""
    if collection_mode not in {"one_shot", "interval"}:
        raise ValueError("collection_mode must be one_shot or interval.")
    if collection_mode == "interval" and (
        interval_seconds is None or not math.isfinite(interval_seconds) or interval_seconds <= 0
    ):
        raise ValueError("Interval captures require a positive interval_seconds value.")

    image, metadata_header = fetch_camera_capture(
        camera_url,
        timeout_seconds=timeout_seconds,
        max_jpeg_bytes=max_jpeg_bytes,
        opener=opener,
    )
    metadata, metadata_error = parse_camera_metadata(metadata_header)
    collection = {
        "purpose": COLLECTION_PURPOSE,
        "review_status": "unreviewed",
        "mode": collection_mode,
        "interval_seconds": interval_seconds if collection_mode == "interval" else None,
    }
    return store.save_capture(
        image,
        source="environment",
        camera_metadata=metadata,
        metadata_error=metadata_error,
        extra={"collection": collection},
    )


def run_interval(
    capture: Callable[[], dict[str, Any]],
    interval_seconds: float,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_success: Callable[[dict[str, Any]], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
    max_attempts: int | None = None,
) -> None:
    """Run immediate, fixed-rate attempts while skipping slots missed by overruns."""
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero.")
    if max_attempts is not None and max_attempts < 1:
        raise ValueError("max_attempts must be positive when supplied.")

    attempts = 0
    next_due = monotonic()
    while max_attempts is None or attempts < max_attempts:
        delay = next_due - monotonic()
        if delay > 0:
            sleep(delay)
        try:
            record = capture()
            if on_success is not None:
                on_success(record)
        except Exception as error:
            if on_error is not None:
                on_error(error)
        attempts += 1
        if max_attempts is not None and attempts >= max_attempts:
            break

        next_due += interval_seconds
        now = monotonic()
        if next_due < now:
            missed_slots = math.floor((now - next_due) / interval_seconds) + 1
            next_due += missed_slots * interval_seconds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-url", required=True, type=camera_url_argument)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"Capture directory (default: {DEFAULT_SAVE_DIR})",
    )
    parser.add_argument(
        "--interval-minutes",
        type=positive_number,
        help="Keep running and capture at this interval; omit for one capture.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_number,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-capture camera timeout (default: {DEFAULT_TIMEOUT_SECONDS:g}).",
    )
    return parser.parse_args(argv)


def _print_saved(store: CaptureStore, record: dict[str, Any]) -> None:
    print(
        f"Saved environment candidate: {store.root / record['filename']} "
        f"({record['bytes']} bytes, camera={record.get('camera_model') or 'unknown'})",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        store = CaptureStore(arguments.save_dir)
    except OSError as error:
        print(f"Capture setup failed: {error}", file=sys.stderr, flush=True)
        return 1

    if arguments.interval_minutes is None:
        try:
            record = capture_environment_frame(
                arguments.camera_url,
                store,
                timeout_seconds=arguments.timeout_seconds,
            )
        except (CameraCaptureError, OSError, ValueError) as error:
            print(f"Capture failed: {error}", file=sys.stderr, flush=True)
            return 1
        _print_saved(store, record)
        return 0

    interval_seconds = arguments.interval_minutes * 60.0
    print(
        f"Capturing immediately and every {arguments.interval_minutes:g} minute(s) into {store.root}. "
        "Press Ctrl+C to stop.",
        flush=True,
    )

    def capture() -> dict[str, Any]:
        return capture_environment_frame(
            arguments.camera_url,
            store,
            collection_mode="interval",
            interval_seconds=interval_seconds,
            timeout_seconds=arguments.timeout_seconds,
        )

    try:
        run_interval(
            capture,
            interval_seconds,
            on_success=lambda record: _print_saved(store, record),
            on_error=lambda error: print(f"Capture failed: {error}", file=sys.stderr, flush=True),
        )
    except KeyboardInterrupt:
        print("Capture job stopped.", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())