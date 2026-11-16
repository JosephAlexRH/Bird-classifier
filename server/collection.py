"""Lossless candidate bird crops with original-frame coordinates and provenance."""
from __future__ import annotations

import hashlib
import math
import os
import threading
import json
import csv
import io
import time
from pathlib import Path
from typing import Any

from PIL import Image

# Keep the deployment detector-manifest columns in the same order as bbox_data.
MANIFEST_FIELDS = tuple("annotation_id sample_id source_dataset source_image_id image_path relative_path width_px height_px review_state annotation_trust annotation_origin eligible_for_training detector_class_id detector_class_name species_id species_name scientific_name box_id xmin ymin xmax ymax confidence source_split model_split split_group projected_area_px2 bird_size is_Bird mixed_species bird_count context_percent context_xmin context_ymin context_xmax context_ymax".split())
PROVENANCE_FIELDS = ("camera_id", "camera_registry_id", "deployment_id", "collection_run_id", "received_at", "image_sha256", "detector_sha256")


def write_review_record(path: Path, record: dict[str, Any]) -> None:
    """Reviews can change; acquisition and model-derived records stay untouched."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(record, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        for attempt in range(4):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                # Windows indexers/viewers can briefly hold the destination.
                # Retain the old valid JSON if the bounded retry still fails.
                if attempt == 3:
                    raise
                time.sleep(.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def manifest_csv(root: Path):
    """Stream original-frame annotations; only human reviews qualify for training."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS + PROVENANCE_FIELDS)
    writer.writeheader()
    yield output.getvalue()
    for image_path in sorted(root.glob("capture_*.jpg")):
        def read(suffix):
            try:
                return json.loads(image_path.with_suffix(suffix).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
        sidecar, inference, review = read(".json"), read(".inference.json"), read(".review.json")
        # A review belongs to exact original bytes, never a replacement image.
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if review and review.get("image_sha256") != digest:
            review = {}
        state = review.get("decision", "unreviewed")
        verified = state in {"boxed", "no_bird", "unusable"}
        # The existing bbox editor loads any CSV box as an existing review.
        # Keep model proposals only in inference/crop records, never in the
        # canonical annotation columns of an unreviewed training manifest.
        boxes = review.get("boxes", []) if state == "boxed" else []
        with Image.open(image_path) as original:
            width, height = original.size
        for index, box in enumerate(boxes or [None]):
            row = {key: "" for key in MANIFEST_FIELDS + PROVENANCE_FIELDS}
            row.update(annotation_id=f"{image_path.stem}:{index}", sample_id=image_path.stem,
                       source_dataset="image_directory", source_image_id=image_path.stem,
                       image_path=image_path.name, relative_path=image_path.name,
                       width_px=width, height_px=height, review_state=state,
                       annotation_trust="human_verified" if verified else "unreviewed",
                       annotation_origin="pi_collection_review" if verified else "pi_collection_detector",
                       eligible_for_training=state in {"boxed", "no_bird"},
                       species_name=review.get("species_name") or "",
                       is_Bird="TRUE" if state == "boxed" else "FALSE" if state == "no_bird" else "",
                       bird_count=len(boxes) if state == "boxed" else 0 if state == "no_bird" else "",
                       mixed_species="", context_percent=0,
                       camera_id=sidecar.get("camera_id") or (sidecar.get("device") or {}).get("camera_id", ""),
                       camera_registry_id=sidecar.get("camera_registry_id", ""),
                       deployment_id=sidecar.get("deployment_id", ""), collection_run_id=sidecar.get("collection_run_id", ""),
                       received_at=sidecar.get("received_at", ""), image_sha256=digest,
                       detector_sha256=(inference.get("detector") or {}).get("model_sha256", ""))
            if row["deployment_id"] and row["camera_id"]:
                row["split_group"] = f"{row['camera_id']}:{row['deployment_id']}"
            if box:
                left, top, right, bottom = (box[k] for k in ("left", "top", "right", "bottom"))
                row.update(box_id=index, detector_class_id=0, detector_class_name="bird",
                           xmin=left, ymin=top, xmax=right, ymax=bottom,
                           context_xmin=left, context_ymin=top, context_xmax=right, context_ymax=bottom,
                           projected_area_px2=(right-left)*(bottom-top))
            output.seek(0)
            output.truncate(0)
            writer.writerow(row)
            yield output.getvalue()

try:
    from .bird_inference import decode_rgb
except ImportError:
    from bird_inference import decode_rgb


def export_bird_crops(image_path: Path, inference: dict[str, Any], *, reviewed: bool = False) -> dict[str, Any]:
    """Keep native pixels; never stretch a box or treat a prediction as a label."""
    if inference.get("status") != "complete":
        return {"status": "inference_error", "crops": [], "review_status": "unreviewed"}
    detector = inference.get("detector") or {}
    threshold = detector.get("threshold", inference.get("bird_threshold", 0.5))
    rgb = None
    crops = []
    for index, detection in enumerate(detector.get("detections") or []):
        score, box = detection.get("score"), detection.get("box")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or score < threshold:
            continue
        if not isinstance(box, dict):
            continue
        values = [box.get(key) for key in ("left", "top", "right", "bottom")]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            continue
        if rgb is None:
            rgb = decode_rgb(image_path)
        height, width = rgb.shape[:2]
        left, top = max(0, min(width, math.floor(values[0]))), max(0, min(height, math.floor(values[1])))
        right, bottom = max(0, min(width, math.ceil(values[2]))), max(0, min(height, math.ceil(values[3])))
        if right <= left or bottom <= top:
            continue
        folder = "review_crops" if reviewed else "crops"
        route = "review-crops" if reviewed else "crops"
        directory = image_path.parent / folder
        directory.mkdir(exist_ok=True)
        destination = directory / f"{image_path.stem}_bird_{index:02d}.png"
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            Image.fromarray(rgb[top:bottom, left:right]).save(temporary, format="PNG")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        crops.append({
            "detection_index": index, "score": float(score),
            "predicted_box": box,
            "pixel_bounds": {"left": left, "top": top, "right": right, "bottom": bottom},
            "bounds_semantics": "half_open_left_top_right_bottom",
            "width": right - left, "height": bottom - top,
            "filename": f"{folder}/{destination.name}",
            "image_url": f"/api/captures/{image_path.stem}/{route}/{index}/image",
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "species_label": None, "review_status": "human_verified" if reviewed else "unreviewed",
        })
    return {
        "schema_version": 1, "status": "complete", "review_status": "unreviewed",
        "box_source": "human_review" if reviewed else "model_prediction", "original_image": image_path.name,
        "original_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "detector_sha256": detector.get("model_sha256"), "threshold": threshold,
        "pixel_policy": "native_resolution_tight_box_no_padding_no_resize",
        "crops": crops,
    }
