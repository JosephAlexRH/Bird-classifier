"""Train and export a motion-crop bird detector for TensorFlow Lite.

This is the crop-oriented successor to the full-scene Keras continuation in
``train_bird_detector_keras_tflite_continuation.ipynb``.  The Raspberry Pi has
already selected ``capture_motion_record.selection.final_bounds_px`` before
this model runs, so training data is converted to the same crop domain.

Key differences from the old full-scene detector:

* 288x288 input by default (configurable to 256 or 320).
* MobileNetV2 ``block_6_expand_relu`` finder grid (stride 8).
* Every bird is positive in its center cell and valid 8-connected neighbours.
* Positive/object-present loss is divided by the number of unique birds, not
  by the number of detector slots or expanded positive cells.
* Native 160x120 captures are excluded from training, calibration, and metrics.
* Evaluation is crop-only because deployment never sends a whole frame.

TensorFlow is imported lazily so the data audit and unit-testable geometry
helpers can run in environments that do not have the training stack installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image


SEED = 1234
DEFAULT_INPUT_SIZE = 288
ALLOWED_INPUT_SIZES = (256, 288, 320)
BACKBONE_ALPHA = 0.35
FEATURE_LAYER = "block_6_expand_relu"
FEATURE_STRIDE = 8
NUM_ANCHORS = 3
NEIGHBOUR_RADIUS = 1
CENTER_DECODE_SPAN_CELLS = 3.0
MOTION_CROP_MARGIN_FRACTION = 0.20
MOTION_CROP_MINIMUM_SIZE = 160
NATIVE_CROP_SIZE = (160, 120)
NEGATIVE_CROPS_PER_TRAINING_SCENE = 2

BATCH_SIZE = 8
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 32
FINE_TUNE_BACKBONE_LAYERS = 24
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 8

REPRESENTATIVE_IMAGES = 500
PRE_NMS_SCORE = 0.01
NMS_IOU = 0.45
MAX_DETECTIONS = 5
EVALUATION_IOU = 0.50
CONFIDENCE_GRID = (
    tuple(round(value / 100, 2) for value in range(1, 90, 2))
    + tuple(round(value / 1000, 3) for value in range(900, 1000, 5))
)
MAX_FALSE_POSITIVES_PER_NEGATIVE_CROP = 0.10

POSITIVE_OBJECTNESS_WEIGHT = 0.25
NEGATIVE_OBJECTNESS_WEIGHT = 0.75
BOX_LOSS_WEIGHT = 8.0
OBJECTNESS_NORMALIZER = "unique_birds_in_batch"

TARGET_BOX_SLICE = slice(0, 4)
TARGET_OBJECTNESS = 4
TARGET_BIRD_COUNT = 5
TARGET_CHANNELS = 6


@dataclass(frozen=True)
class RunPaths:
    source_run: Path
    artifact_dir: Path
    checkpoint_dir: Path
    export_dir: Path
    head_weights: Path
    final_weights: Path
    anchors: Path
    float_tflite: Path
    uint8_tflite: Path


def portable_path(raw_path: str | os.PathLike[str]) -> Path:
    """Map Windows paths to WSL and /mnt/<drive> paths to Windows."""
    raw = str(raw_path).strip().strip('"').strip("'")
    windows_path = PureWindowsPath(raw)
    if windows_path.drive:
        if os.name == "nt":
            return Path(raw)
        drive = windows_path.drive.rstrip(":").lower()
        return Path("/mnt") / drive / Path(*windows_path.parts[1:])
    posix_path = PurePosixPath(raw)
    if os.name == "nt" and len(posix_path.parts) >= 4 and posix_path.parts[:2] == ("/", "mnt"):
        drive = posix_path.parts[2]
        if len(drive) == 1 and drive.isalpha():
            return Path(f"{drive.upper()}:\\", *posix_path.parts[3:])
    return Path(raw)


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "Models" / "Multi_Object_Detector" / "dataset_dirs.txt").is_file():
            return candidate
    raise FileNotFoundError("Could not find Models/Multi_Object_Detector/dataset_dirs.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def configuration_fingerprint_payload(manifest_sha256: str, input_size: int) -> dict[str, Any]:
    """Return every training/export setting that makes checkpoints incompatible."""
    return {
        "manifest": manifest_sha256,
        "input_size": input_size,
        "backbone_alpha": BACKBONE_ALPHA,
        "feature_layer": FEATURE_LAYER,
        "feature_stride": FEATURE_STRIDE,
        "num_anchors": NUM_ANCHORS,
        "neighbour_radius": NEIGHBOUR_RADIUS,
        "decode_span": CENTER_DECODE_SPAN_CELLS,
        "motion_crop_margin_fraction": MOTION_CROP_MARGIN_FRACTION,
        "motion_crop_minimum_size": MOTION_CROP_MINIMUM_SIZE,
        "training_positive_crop_mode": "per_bird",
        "evaluation_positive_crop_mode": "scene_union",
        "native_160x120_excluded": True,
        "positive_objectness_weight": POSITIVE_OBJECTNESS_WEIGHT,
        "negative_objectness_weight": NEGATIVE_OBJECTNESS_WEIGHT,
        "box_loss_weight": BOX_LOSS_WEIGHT,
        "objectness_normalizer": OBJECTNESS_NORMALIZER,
        "max_detections": MAX_DETECTIONS,
        "confidence_grid": CONFIDENCE_GRID,
    }


def is_complete_source_run(path: Path) -> bool:
    return (
        (path / "joined_bird_detector_annotations.csv").is_file()
        and (path / "data_audit_summary.json").is_file()
    )


def resolve_source_run(project_root: Path, override: str | None) -> Path:
    runs_dir = project_root / "Models" / "Multi_Object_Detector" / "runs"
    if override:
        candidate = portable_path(override)
        if not candidate.is_absolute():
            candidate = runs_dir / candidate
        candidate = candidate.resolve()
        if not is_complete_source_run(candidate):
            raise FileNotFoundError(f"Incomplete detector source run: {candidate}")
        return candidate
    candidates = [path for path in runs_dir.iterdir() if path.is_dir() and is_complete_source_run(path)]
    if not candidates:
        raise FileNotFoundError(f"No completed detector runs found below {runs_dir}")
    return max(
        candidates,
        key=lambda path: (
            (path / "joined_bird_detector_annotations.csv").stat().st_mtime_ns,
            path.name,
        ),
    ).resolve()


def make_run_paths(source_run: Path) -> RunPaths:
    artifact_dir = source_run / "keras_motion_crop_detector"
    checkpoint_dir = artifact_dir / "checkpoints"
    export_dir = source_run / "exports" / "keras_motion_crop_detector"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        source_run=source_run,
        artifact_dir=artifact_dir,
        checkpoint_dir=checkpoint_dir,
        export_dir=export_dir,
        head_weights=checkpoint_dir / "head_complete.weights.h5",
        final_weights=checkpoint_dir / "fine_tune_complete.weights.h5",
        anchors=artifact_dir / "anchors.json",
        float_tflite=export_dir / "bird_detector_motion_crop_float32.tflite",
        uint8_tflite=export_dir / "bird_detector_motion_crop_uint8.tflite",
    )


def is_native_crop(width: int, height: int) -> bool:
    return (int(width), int(height)) == NATIVE_CROP_SIZE


def final_crop_bounds(
    bounds: Sequence[float],
    frame_width: int,
    frame_height: int,
    margin_fraction: float = MOTION_CROP_MARGIN_FRACTION,
    minimum_size: int = MOTION_CROP_MINIMUM_SIZE,
) -> tuple[int, int, int, int]:
    """Match the Pi motion crop's total-margin and edge-shifting semantics."""
    left, top, right, bottom = (float(value) for value in bounds)
    if not (0 <= left < right <= frame_width and 0 <= top < bottom <= frame_height):
        raise ValueError(f"Crop seed is outside {frame_width}x{frame_height}: {bounds}")
    width, height = right - left, bottom - top
    target_width = min(frame_width, max(minimum_size, int(math.ceil(width * (1 + margin_fraction)))))
    target_height = min(frame_height, max(minimum_size, int(math.ceil(height * (1 + margin_fraction)))))
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    crop_left = min(max(0, int(math.floor(center_x - target_width / 2))), frame_width - target_width)
    crop_top = min(max(0, int(math.floor(center_y - target_height / 2))), frame_height - target_height)
    return crop_left, crop_top, crop_left + target_width, crop_top + target_height


def deterministic_negative_crop_bounds(
    sample_id: str,
    width: int,
    height: int,
    index: int,
) -> tuple[int, int, int, int]:
    """Create a stable crop-like negative without pretending it is a full scene."""
    maximum = min(320, width, height)
    minimum = min(MOTION_CROP_MINIMUM_SIZE, maximum)
    digest = hashlib.sha256(f"{SEED}:{sample_id}:{index}".encode("utf-8")).digest()
    span = maximum - minimum
    size = minimum + (int.from_bytes(digest[:2], "little") % (span + 1) if span else 0)
    max_left, max_top = width - size, height - size
    left = int.from_bytes(digest[2:6], "little") % (max_left + 1) if max_left else 0
    top = int.from_bytes(digest[6:10], "little") % (max_top + 1) if max_top else 0
    return left, top, left + size, top + size


def load_scene_records(manifest_path: Path) -> list[dict[str, Any]]:
    required = {
        "annotation_id", "sample_id", "source_dataset", "image_path", "width_px",
        "height_px", "model_split", "eligible_for_training", "annotation_trust",
        "detector_class_id", "xmin", "ymin", "xmax", "ymax",
    }
    frame = pd.read_csv(manifest_path, usecols=sorted(required), low_memory=False)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Joined annotation manifest is missing columns: {sorted(missing)}")
    # The joined manifest contains tens of thousands of provisional iNaturalist
    # rows whose model_split is "excluded".  They are neither trainable nor part
    # of the ESP32 crop/full-scene evaluation, so discard them before grouping.
    frame = frame.loc[
        frame["eligible_for_training"].eq(True)
        | frame["source_dataset"].eq("esp32_capture")
    ].copy()
    frame = frame.sort_values("annotation_id", kind="stable")
    image_rows = frame.drop_duplicates("sample_id", keep="first")
    positive = frame.loc[
        frame["detector_class_id"].notna()
        & frame["xmax"].gt(frame["xmin"])
        & frame["ymax"].gt(frame["ymin"]),
        ["sample_id", "xmin", "ymin", "xmax", "ymax"],
    ]
    boxes_by_sample = {
        str(sample_id): rows[["xmin", "ymin", "xmax", "ymax"]].to_numpy(np.float32)
        for sample_id, rows in positive.groupby("sample_id", sort=False)
    }
    records = []
    for row in image_rows.itertuples(index=False):
        sample_id = str(row.sample_id)
        width, height = int(row.width_px), int(row.height_px)
        records.append({
            "sample_id": sample_id,
            "source_dataset": str(row.source_dataset),
            "image_path": portable_path(str(row.image_path)),
            "width": width,
            "height": height,
            "boxes": boxes_by_sample.get(sample_id, np.empty((0, 4), np.float32)),
            "model_split": str(row.model_split),
            "eligible_for_training": bool(row.eligible_for_training),
            "annotation_trust": str(row.annotation_trust),
            "scene_geometry": "native_crop" if is_native_crop(width, height) else "full_scene",
        })
    return records


def boxes_in_crop(boxes: np.ndarray, bounds: Sequence[int]) -> np.ndarray:
    if not len(boxes):
        return np.empty((0, 4), dtype=np.float32)
    left, top, right, bottom = bounds
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    selected = boxes[
        (centers_x >= left) & (centers_x < right) & (centers_y >= top) & (centers_y < bottom)
    ].copy()
    if not len(selected):
        return np.empty((0, 4), dtype=np.float32)
    selected[:, [0, 2]] = np.clip(selected[:, [0, 2]], left, right) - left
    selected[:, [1, 3]] = np.clip(selected[:, [1, 3]], top, bottom) - top
    keep = (selected[:, 2] > selected[:, 0]) & (selected[:, 3] > selected[:, 1])
    return selected[keep].astype(np.float32)


def make_crop_records(
    scenes: Sequence[Mapping[str, Any]],
    *,
    negative_crops_per_scene: int,
    positive_crop_mode: str = "per_bird",
) -> list[dict[str, Any]]:
    if positive_crop_mode not in {"per_bird", "scene_union"}:
        raise ValueError(f"Unsupported positive crop mode: {positive_crop_mode}")
    records: list[dict[str, Any]] = []
    for scene in scenes:
        width, height = int(scene["width"]), int(scene["height"])
        boxes = np.asarray(scene["boxes"], dtype=np.float32).reshape((-1, 4))
        if scene["scene_geometry"] == "native_crop":
            bounds = (0, 0, width, height)
            candidates = [("native", bounds, "native_crop")]
        elif len(boxes):
            seen: set[tuple[int, int, int, int]] = set()
            candidates = []
            crop_seeds = boxes
            if positive_crop_mode == "scene_union":
                crop_seeds = np.asarray([[
                    boxes[:, 0].min(), boxes[:, 1].min(),
                    boxes[:, 2].max(), boxes[:, 3].max(),
                ]], dtype=np.float32)
            for index, box in enumerate(crop_seeds):
                bounds = final_crop_bounds(box, width, height)
                if bounds not in seen:
                    candidates.append((f"motion_{index:03d}", bounds, "simulated_motion_crop"))
                    seen.add(bounds)
        else:
            candidates = [
                (
                    f"negative_{index:03d}",
                    deterministic_negative_crop_bounds(str(scene["sample_id"]), width, height, index),
                    "simulated_motion_crop",
                )
                for index in range(negative_crops_per_scene)
            ]
        for suffix, bounds, geometry in candidates:
            left, top, right, bottom = bounds
            records.append({
                **scene,
                "sample_id": f"{scene['sample_id']}::{suffix}",
                "source_sample_id": scene["sample_id"],
                "crop_bounds": np.asarray(bounds, dtype=np.int32),
                "width": right - left,
                "height": bottom - top,
                "boxes": boxes_in_crop(boxes, bounds),
                "evaluation_geometry": geometry,
            })
    return records


def select_representative_records(
    records: Sequence[Mapping[str, Any]],
    count: int = REPRESENTATIVE_IMAGES,
) -> list[dict[str, Any]]:
    """Choose deterministic, class-balanced Pi-style crops for INT8 calibration.

    Use only crops simulated from full-resolution ESP32 scenes. Native 160x120
    legacy captures are excluded from the detector pipeline entirely.
    """
    candidates = [
        record for record in records
        if record["source_dataset"] == "esp32_capture"
        and record["evaluation_geometry"] == "simulated_motion_crop"
    ]

    def stable_order(record: Mapping[str, Any]) -> str:
        return hashlib.sha256(f"{SEED}:{record['sample_id']}".encode("utf-8")).hexdigest()

    positives = sorted((record for record in candidates if len(record["boxes"])), key=stable_order)
    negatives = sorted((record for record in candidates if not len(record["boxes"])), key=stable_order)
    positive_limit = min(len(positives), (count + 1) // 2)
    negative_limit = min(len(negatives), count // 2)
    selected = positives[:positive_limit] + negatives[:negative_limit]
    selected_ids = {record["sample_id"] for record in selected}
    remainder = sorted(
        (record for record in candidates if record["sample_id"] not in selected_ids),
        key=stable_order,
    )
    selected.extend(remainder[: max(0, count - len(selected))])
    return sorted(selected[:count], key=stable_order)


def select_datasets(scenes: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    train_scenes = [
        scene for scene in scenes
        if scene["model_split"] == "train"
        and scene["eligible_for_training"]
        and scene["scene_geometry"] != "native_crop"
    ]
    esp32_validation = [
        scene for scene in scenes
        if scene["source_dataset"] == "esp32_capture" and scene["model_split"] == "validation"
        and scene["scene_geometry"] != "native_crop"
    ]
    esp32_test = [
        scene for scene in scenes
        if scene["source_dataset"] == "esp32_capture" and scene["model_split"] == "test"
        and scene["scene_geometry"] != "native_crop"
    ]
    train = make_crop_records(
        train_scenes,
        negative_crops_per_scene=NEGATIVE_CROPS_PER_TRAINING_SCENE,
    )
    validation_crop = make_crop_records(
        esp32_validation,
        negative_crops_per_scene=1,
        positive_crop_mode="scene_union",
    )
    test_crop = make_crop_records(
        esp32_test,
        negative_crops_per_scene=1,
        positive_crop_mode="scene_union",
    )
    calibration = select_representative_records(train)
    return {
        "train_crop": train,
        "validation_crop": validation_crop,
        "test_crop": test_crop,
        "calibration_crop": calibration,
    }


def audit_datasets(
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    verify_images: bool = False,
) -> pd.DataFrame:
    rows = []
    for name, records in datasets.items():
        unique_paths = {Path(record["image_path"]) for record in records}
        rows.append({
            "dataset": name,
            "images": len(records),
            "positive_images": sum(bool(len(record["boxes"])) for record in records),
            "negative_images": sum(not len(record["boxes"]) for record in records),
            "birds": sum(len(record["boxes"]) for record in records),
            "missing_images": (
                sum(not path.is_file() for path in unique_paths)
                if verify_images else None
            ),
        })
    return pd.DataFrame(rows)


def letterbox_geometry(width: float, height: float, size: int) -> tuple[float, float, float]:
    scale = min(size / width, size / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    return scale, (size - resized_width) / 2.0, (size - resized_height) / 2.0


def normalized_letterbox_wh(records: Sequence[Mapping[str, Any]], input_size: int) -> np.ndarray:
    values = []
    for record in records:
        scale, _, _ = letterbox_geometry(record["width"], record["height"], input_size)
        for xmin, ymin, xmax, ymax in record["boxes"]:
            values.append([(xmax - xmin) * scale / input_size, (ymax - ymin) * scale / input_size])
    result = np.asarray(values, dtype=np.float32).reshape((-1, 2))
    return result[(result > 0).all(axis=1)]


def wh_iou(box_wh: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    intersection = np.minimum(box_wh[:, None, :], anchors[None, :, :]).prod(axis=-1)
    union = box_wh.prod(axis=-1)[:, None] + anchors.prod(axis=-1)[None, :] - intersection
    return intersection / np.maximum(union, 1e-9)


def fit_anchors(box_wh: np.ndarray, count: int, seed: int = SEED, iterations: int = 100) -> np.ndarray:
    if len(box_wh) < count:
        raise ValueError(f"Need at least {count} boxes to fit anchors")
    rng = np.random.default_rng(seed)
    centers = box_wh[rng.choice(len(box_wh), count, replace=False)].copy()
    for _ in range(iterations):
        assignments = wh_iou(box_wh, centers).argmax(axis=1)
        updated = centers.copy()
        for index in range(count):
            members = box_wh[assignments == index]
            if len(members):
                updated[index] = np.median(members, axis=0)
        if np.allclose(updated, centers, rtol=0, atol=1e-6):
            break
        centers = updated
    return centers[np.argsort(centers.prod(axis=1))].astype(np.float32)


def encode_targets_numpy(
    boxes: np.ndarray,
    anchors: np.ndarray,
    grid_size: int,
    neighbour_radius: int = NEIGHBOUR_RADIUS,
) -> np.ndarray:
    """Encode birds into center and neighbouring cells with unique-bird count."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape((-1, 4))
    target = np.zeros((grid_size, grid_size, len(anchors), TARGET_CHANNELS), dtype=np.float32)
    target[..., TARGET_BIRD_COUNT] = float(len(boxes))
    if not len(boxes):
        return target
    widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)
    centers_x = np.clip((boxes[:, 0] + boxes[:, 2]) / 2, 0, 1 - 1e-6)
    centers_y = np.clip((boxes[:, 1] + boxes[:, 3]) / 2, 0, 1 - 1e-6)
    normalized_boxes = np.stack([centers_x, centers_y, widths, heights], axis=-1)
    anchor_ious = wh_iou(normalized_boxes[:, 2:4], np.asarray(anchors, dtype=np.float32))
    bird_order = np.argsort(anchor_ious.max(axis=1))[::-1]
    for bird_index in bird_order:
        center_cell_x = int(math.floor(float(centers_x[bird_index]) * grid_size))
        center_cell_y = int(math.floor(float(centers_y[bird_index]) * grid_size))
        ranked_anchors = np.argsort(anchor_ious[bird_index])[::-1]
        for dy in range(-neighbour_radius, neighbour_radius + 1):
            for dx in range(-neighbour_radius, neighbour_radius + 1):
                cell_x, cell_y = center_cell_x + dx, center_cell_y + dy
                if not (0 <= cell_x < grid_size and 0 <= cell_y < grid_size):
                    continue
                free = [anchor for anchor in ranked_anchors if target[cell_y, cell_x, anchor, TARGET_OBJECTNESS] < 0.5]
                if not free:
                    continue
                anchor = int(free[0])
                target[cell_y, cell_x, anchor, TARGET_BOX_SLICE] = normalized_boxes[bird_index]
                target[cell_y, cell_x, anchor, TARGET_OBJECTNESS] = 1.0
    return target


def require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for training/export. Use the project's Python 3.12 WSL "
            "environment, or run this script with --audit-only on this interpreter."
        ) from error
    return tf


def build_model(tf: Any, input_size: int) -> tuple[Any, Any, int]:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        alpha=BACKBONE_ALPHA,
        include_top=False,
        weights="imagenet",
    )
    feature_extractor = tf.keras.Model(
        base.input,
        base.get_layer(FEATURE_LAYER).output,
        name="mobilenetv2_block6_finder",
    )
    inputs = tf.keras.Input((input_size, input_size, 3), name="image", dtype=tf.float32)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="normalize_uint8_rgb")(inputs)
    x = feature_extractor(x)
    x = tf.keras.layers.SeparableConv2D(96, 3, padding="same", use_bias=False, name="finder_sepconv")(x)
    x = tf.keras.layers.BatchNormalization(name="finder_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="finder_relu6")(x)
    x = tf.keras.layers.Conv2D(NUM_ANCHORS * 5, 1, padding="same", name="detector_logits")(x)
    grid_height, grid_width = int(x.shape[1]), int(x.shape[2])
    if grid_height != grid_width:
        raise ValueError(f"Expected square finder grid, got {x.shape}")
    expected_grid = input_size // FEATURE_STRIDE
    if grid_height != expected_grid:
        raise ValueError(
            f"{FEATURE_LAYER} produced {grid_height}, expected stride-{FEATURE_STRIDE} grid {expected_grid}"
        )
    output = tf.keras.layers.Reshape((grid_height, grid_width, NUM_ANCHORS, 5), name="raw_detections")(x)
    return tf.keras.Model(inputs, output, name="bird_detector_motion_crop"), feature_extractor, grid_height


def make_dataset(
    tf: Any,
    records: Sequence[Mapping[str, Any]],
    anchors: np.ndarray,
    grid_size: int,
    input_size: int,
    batch_size: int,
    training: bool,
) -> Any:
    def generator() -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
        for record in records:
            bounds = record.get("crop_bounds", (0, 0, record["width"], record["height"]))
            yield str(record["image_path"]), np.asarray(bounds, np.int32), np.asarray(record["boxes"], np.float32)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec([], tf.string),
            tf.TensorSpec([4], tf.int32),
            tf.TensorSpec([None, 4], tf.float32),
        ),
    )
    if training:
        dataset = dataset.shuffle(min(len(records), 4096), seed=SEED, reshuffle_each_iteration=True)

    def load(path: Any, bounds: Any, boxes: Any) -> tuple[Any, Any]:
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        left, top, right, bottom = tf.unstack(bounds)
        image = tf.cast(image[top:bottom, left:right], tf.float32)
        height, width = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
        scale = tf.minimum(input_size / width, input_size / height)
        new_width, new_height = tf.cast(tf.round(width * scale), tf.int32), tf.cast(tf.round(height * scale), tf.int32)
        resized = tf.image.resize(image, [new_height, new_width], antialias=True)
        pad_left, pad_top = (input_size - new_width) // 2, (input_size - new_height) // 2
        image = tf.pad(
            resized,
            [[pad_top, input_size - new_height - pad_top], [pad_left, input_size - new_width - pad_left], [0, 0]],
            constant_values=114.0,
        )
        image.set_shape([input_size, input_size, 3])
        vector = tf.cast(tf.stack([pad_left, pad_top, pad_left, pad_top]), tf.float32)
        boxes = tf.clip_by_value((boxes * scale + vector) / float(input_size), 0.0, 1.0)
        return image, boxes

    def augment(image: Any, boxes: Any) -> tuple[Any, Any]:
        flip = tf.random.uniform(()) < 0.5
        image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
        boxes = tf.cond(
            flip,
            lambda: tf.stack([1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1),
            lambda: boxes,
        )
        image = tf.image.random_brightness(image, 12.0)
        image = tf.image.random_contrast(image, 0.85, 1.15)
        return tf.clip_by_value(image, 0.0, 255.0), boxes

    def encode(image: Any, boxes: Any) -> tuple[Any, Any]:
        target = tf.numpy_function(
            lambda value: encode_targets_numpy(value, anchors, grid_size),
            [boxes],
            tf.float32,
        )
        target.set_shape([grid_size, grid_size, NUM_ANCHORS, TARGET_CHANNELS])
        return image, target

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.map(encode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compile_model(tf: Any, model: Any, anchors: np.ndarray, grid_size: int, learning_rate: float) -> None:
    anchor_tensor = tf.constant(anchors, tf.float32)[None, None, None, :, :]
    grid_y, grid_x = tf.meshgrid(
        tf.range(grid_size, dtype=tf.float32),
        tf.range(grid_size, dtype=tf.float32),
        indexing="ij",
    )
    grid = tf.stack([grid_x, grid_y], axis=-1)[None, :, :, None, :]

    def decode(raw: Any) -> tuple[Any, Any]:
        center = (
            tf.sigmoid(raw[..., :2]) * CENTER_DECODE_SPAN_CELLS
            - (CENTER_DECODE_SPAN_CELLS - 1) / 2
            + grid
        ) / float(grid_size)
        size = anchor_tensor * tf.exp(tf.clip_by_value(raw[..., 2:4], -4.0, 4.0))
        return tf.concat([center, size], axis=-1), raw[..., 4]

    def detector_loss(y_true: Any, y_pred: Any) -> Any:
        true_box, object_mask = y_true[..., :4], y_true[..., TARGET_OBJECTNESS]
        bird_counts = tf.reduce_max(y_true[..., TARGET_BIRD_COUNT], axis=[1, 2, 3])
        bird_count = tf.maximum(1.0, tf.reduce_sum(bird_counts))
        predicted_box, logits = decode(y_pred)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=logits)
        probability = tf.sigmoid(logits)
        probability_true = object_mask * probability + (1 - object_mask) * (1 - probability)
        focal = tf.square(1 - probability_true) * bce
        positive_loss = POSITIVE_OBJECTNESS_WEIGHT * tf.reduce_sum(focal * object_mask) / bird_count
        negative_mask = 1 - object_mask
        negative_loss = NEGATIVE_OBJECTNESS_WEIGHT * tf.reduce_sum(focal * negative_mask) / bird_count
        coordinate_error = tf.keras.losses.Huber(reduction="none")(true_box, predicted_box)
        box_loss = tf.reduce_sum(coordinate_error * object_mask) / bird_count
        return positive_loss + negative_loss + BOX_LOSS_WEIGHT * box_loss

    def mean_positive_objectness(y_true: Any, y_pred: Any) -> Any:
        mask = y_true[..., TARGET_OBJECTNESS]
        return tf.reduce_sum(tf.sigmoid(y_pred[..., 4]) * mask) / tf.maximum(1.0, tf.reduce_sum(mask))

    def mean_negative_objectness(y_true: Any, y_pred: Any) -> Any:
        mask = 1 - y_true[..., TARGET_OBJECTNESS]
        return tf.reduce_sum(tf.sigmoid(y_pred[..., 4]) * mask) / tf.maximum(1.0, tf.reduce_sum(mask))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=detector_loss,
        metrics=[mean_positive_objectness, mean_negative_objectness],
    )


def train_model(
    tf: Any,
    model: Any,
    feature_extractor: Any,
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    anchors: np.ndarray,
    grid_size: int,
    input_size: int,
    paths: RunPaths,
    batch_size: int,
) -> None:
    train = make_dataset(tf, datasets["train_crop"], anchors, grid_size, input_size, batch_size, True)
    validation = make_dataset(tf, datasets["validation_crop"], anchors, grid_size, input_size, batch_size, False)

    def callbacks(stage: str, append: bool) -> list[Any]:
        return [
            tf.keras.callbacks.BackupAndRestore(backup_dir=paths.checkpoint_dir / f"{stage}_backup"),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
            tf.keras.callbacks.CSVLogger(paths.artifact_dir / "training_log.csv", append=append),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

    if paths.final_weights.is_file():
        model.load_weights(paths.final_weights)
        return
    if paths.head_weights.is_file():
        model.load_weights(paths.head_weights)
    else:
        feature_extractor.trainable = False
        compile_model(tf, model, anchors, grid_size, HEAD_LEARNING_RATE)
        model.fit(train, validation_data=validation, epochs=HEAD_EPOCHS, callbacks=callbacks("head", False))
        model.save_weights(paths.head_weights)
    feature_extractor.trainable = True
    for layer in feature_extractor.layers[:-FINE_TUNE_BACKBONE_LAYERS]:
        layer.trainable = False
    for layer in feature_extractor.layers[-FINE_TUNE_BACKBONE_LAYERS:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(tf, model, anchors, grid_size, FINE_TUNE_LEARNING_RATE)
    model.fit(train, validation_data=validation, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks("fine_tune", True))
    model.save_weights(paths.final_weights)


def interpreter_details(tf: Any, path: Path) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    operations = []
    if hasattr(interpreter, "_get_ops_details"):
        operations = sorted({item["op_name"] for item in interpreter._get_ops_details() if item["op_name"] != "DELEGATE"})
    def details(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{
            "name": item["name"],
            "shape": item["shape"].tolist(),
            "dtype": np.dtype(item["dtype"]).name,
            "quantization": list(item["quantization"]),
        } for item in items]
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "inputs": details(interpreter.get_input_details()),
        "outputs": details(interpreter.get_output_details()),
        "operations": operations,
    }


def export_models(
    tf: Any,
    model: Any,
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    anchors: np.ndarray,
    grid_size: int,
    input_size: int,
    paths: RunPaths,
) -> dict[str, Any]:
    calibration = make_dataset(
        tf, datasets["calibration_crop"], anchors, grid_size, input_size, 1, False
    )
    def representative_dataset() -> Iterator[list[np.ndarray]]:
        for image, _ in calibration.unbatch().batch(1):
            yield [image.numpy().astype(np.float32)]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    paths.float_tflite.write_bytes(converter.convert())
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    paths.uint8_tflite.write_bytes(converter.convert())
    details = {
        "float": interpreter_details(tf, paths.float_tflite),
        "uint8": interpreter_details(tf, paths.uint8_tflite),
    }
    expected_input = [1, input_size, input_size, 3]
    expected_output = [1, grid_size, grid_size, NUM_ANCHORS, 5]
    if details["uint8"]["inputs"][0]["shape"] != expected_input:
        raise RuntimeError(f"Unexpected UINT8 input shape: {details['uint8']['inputs'][0]['shape']}")
    if details["uint8"]["outputs"][0]["shape"] != expected_output:
        raise RuntimeError(f"Unexpected UINT8 output shape: {details['uint8']['outputs'][0]['shape']}")
    if details["uint8"]["inputs"][0]["dtype"] != "uint8":
        raise RuntimeError("Full-integer export did not produce a UINT8 input tensor")
    if details["uint8"]["outputs"][0]["dtype"] != "uint8":
        raise RuntimeError("Full-integer export did not produce a UINT8 output tensor")
    if details["uint8"]["inputs"][0]["quantization"][0] <= 0:
        raise RuntimeError("UINT8 input tensor is missing quantization parameters")
    if details["uint8"]["outputs"][0]["quantization"][0] <= 0:
        raise RuntimeError("UINT8 output tensor is missing quantization parameters")
    (paths.export_dir / "tflite_tensor_details.json").write_text(
        json.dumps(json_safe(details), indent=2), encoding="utf-8"
    )
    return details


def sigmoid_np(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def quantize_array(array: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype in (np.uint8, np.int8):
        scale, zero_point = detail["quantization"]
        limits = np.iinfo(dtype)
        return np.clip(np.round(array / scale + zero_point), limits.min, limits.max).astype(dtype)
    return array.astype(dtype)


def dequantize_array(array: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    if detail["dtype"] in (np.uint8, np.int8):
        scale, zero_point = detail["quantization"]
        return (array.astype(np.float32) - zero_point) * scale
    return array.astype(np.float32)


def prepare_record_image(record: Mapping[str, Any], input_size: int) -> tuple[np.ndarray, float, float, float]:
    with Image.open(record["image_path"]) as source:
        image = source.convert("RGB")
        if "crop_bounds" in record:
            image = image.crop(tuple(int(value) for value in record["crop_bounds"]))
        width, height = image.size
        scale, _, _ = letterbox_geometry(width, height, input_size)
        resized_width, resized_height = round(width * scale), round(height * scale)
        pad_left, pad_top = (input_size - resized_width) // 2, (input_size - resized_height) // 2
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        resized = np.asarray(image.resize((resized_width, resized_height), Image.Resampling.BILINEAR), dtype=np.uint8)
        canvas[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = resized
    return canvas, scale, float(pad_left), float(pad_top)


def box_iou_array(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area_a = np.maximum(0, box[2:] - box[:2]).prod()
    area_b = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def decode_output(
    raw: np.ndarray,
    anchors: np.ndarray,
    input_size: int,
    original_width: int,
    original_height: int,
    scale: float,
    pad_x: float,
    pad_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid_height, grid_width = raw.shape[:2]
    grid_y, grid_x = np.meshgrid(np.arange(grid_height), np.arange(grid_width), indexing="ij")
    center_x = (sigmoid_np(raw[..., 0]) * CENTER_DECODE_SPAN_CELLS - 1 + grid_x[..., None]) / grid_width
    center_y = (sigmoid_np(raw[..., 1]) * CENTER_DECODE_SPAN_CELLS - 1 + grid_y[..., None]) / grid_height
    width = anchors[None, None, :, 0] * np.exp(np.clip(raw[..., 2], -4, 4))
    height = anchors[None, None, :, 1] * np.exp(np.clip(raw[..., 3], -4, 4))
    scores = sigmoid_np(raw[..., 4])
    selected = scores >= PRE_NMS_SCORE
    if not selected.any():
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
    boxes = np.stack([center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2], axis=-1)[selected]
    scores = scores[selected]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] * input_size - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] * input_size - pad_y) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    order, keep = np.argsort(scores)[::-1], []
    while len(order) and len(keep) < MAX_DETECTIONS:
        current = int(order[0]); keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou_array(boxes[current], boxes[remaining]) < NMS_IOU]
    return boxes[np.asarray(keep, np.int32)], scores[np.asarray(keep, np.int32)]


def predict_tflite(
    tf: Any,
    model_path: Path,
    records: Sequence[Mapping[str, Any]],
    anchors: np.ndarray,
    input_size: int,
) -> pd.DataFrame:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    predictions = []
    for record in records:
        tensor, scale, pad_x, pad_y = prepare_record_image(record, input_size)
        interpreter.set_tensor(input_detail["index"], quantize_array(tensor[None].astype(np.float32), input_detail))
        interpreter.invoke()
        raw = dequantize_array(interpreter.get_tensor(output_detail["index"]), output_detail)[0]
        boxes, scores = decode_output(raw, anchors, input_size, record["width"], record["height"], scale, pad_x, pad_y)
        predictions.extend({
            "sample_id": record["sample_id"], "xmin": float(box[0]), "ymin": float(box[1]),
            "xmax": float(box[2]), "ymax": float(box[3]), "score": float(score),
        } for box, score in zip(boxes, scores))
    return pd.DataFrame(predictions, columns=["sample_id", "xmin", "ymin", "xmax", "ymax", "score"])


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def evaluate_predictions(records: Sequence[Mapping[str, Any]], predictions: pd.DataFrame, threshold: float) -> dict[str, Any]:
    true_positives = false_positives = false_negatives = negative_false_positives = 0
    negative_images = sum(not len(record["boxes"]) for record in records)
    matched_ious = []
    for record in records:
        truth = record["boxes"]
        selected = predictions.loc[
            predictions.sample_id.eq(record["sample_id"]) & predictions.score.ge(threshold)
        ].sort_values("score", ascending=False)
        matched: set[int] = set()
        for prediction in selected.itertuples(index=False):
            candidates = [
                (box_iou((prediction.xmin, prediction.ymin, prediction.xmax, prediction.ymax), box), index)
                for index, box in enumerate(truth) if index not in matched
            ]
            best_iou, best_index = max(candidates, default=(0.0, None))
            if best_index is not None and best_iou >= EVALUATION_IOU:
                matched.add(best_index); true_positives += 1; matched_ious.append(best_iou)
            else:
                false_positives += 1
                if not len(truth):
                    negative_false_positives += 1
        false_negatives += len(truth) - len(matched)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    return {
        "confidence_threshold": threshold,
        "true_positives": true_positives, "false_positives": false_positives,
        "false_negatives": false_negatives, "precision": precision, "recall": recall,
        "f2": 5 * precision * recall / max(1e-12, 4 * precision + recall),
        "negative_images": negative_images,
        "false_positives_per_negative_image": negative_false_positives / max(1, negative_images),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def select_threshold(records: Sequence[Mapping[str, Any]], predictions: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    sweep = pd.DataFrame([evaluate_predictions(records, predictions, threshold) for threshold in CONFIDENCE_GRID])
    feasible = sweep[sweep.false_positives_per_negative_image <= MAX_FALSE_POSITIVES_PER_NEGATIVE_CROP]
    candidates = feasible if not feasible.empty else sweep
    best = candidates.sort_values(["recall", "f2", "precision", "confidence_threshold"], ascending=[False, False, False, True]).iloc[0]
    return float(best.confidence_threshold), sweep


def write_contract(
    paths: RunPaths,
    input_size: int,
    grid_size: int,
    anchors: np.ndarray,
    audit: pd.DataFrame,
    export_details: Mapping[str, Any] | None,
    thresholds: Mapping[str, float],
) -> None:
    contract = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(paths.source_run),
        "task": "single-class bird detection on Pi-selected motion crops",
        "input_geometry": {
            "geometry": "selected_crop_then_letterbox",
            "geometry_location": "outside the model",
            "crop_source": "capture_motion_record.selection.final_bounds_px",
            "crop_coordinate_space": "original_decoded_frame_pixels",
            "crop_bounds_semantics": "half_open_left_top_right_bottom",
            "crop_is_final": True,
            "additional_runtime_margin": 0,
            "letterbox_padding_value": 114,
        },
        "architecture": {
            "backbone": "MobileNetV2 ImageNet transfer learning",
            "backbone_alpha": BACKBONE_ALPHA,
            "feature_layer": FEATURE_LAYER,
            "feature_stride": FEATURE_STRIDE,
            "input_size": input_size,
            "grid_size": grid_size,
            "anchors_normalized_wh": anchors.tolist(),
            "neighbour_radius": NEIGHBOUR_RADIUS,
            "center_decode_span_cells": CENTER_DECODE_SPAN_CELLS,
            "raw_output_shape": [1, grid_size, grid_size, NUM_ANCHORS, 5],
            "output_order": ["tx", "ty", "tw", "th", "objectness_logit"],
            "decode": {
                "center_x": "(sigmoid(tx) * 3 - 1 + grid_x) / grid_width",
                "center_y": "(sigmoid(ty) * 3 - 1 + grid_y) / grid_height",
                "width": "anchor_width * exp(clamp(tw, -4, 4))",
                "height": "anchor_height * exp(clamp(th, -4, 4))",
                "score": "sigmoid(objectness_logit)",
            },
        },
        "training_loss": {
            "positive_objectness_normalizer": "unique birds in batch",
            "negative_objectness_normalizer": "unique birds in batch",
            "box_normalizer": "unique birds in batch",
            "positive_objectness_weight": POSITIVE_OBJECTNESS_WEIGHT,
            "negative_objectness_weight": NEGATIVE_OBJECTNESS_WEIGHT,
            "box_loss_weight": BOX_LOSS_WEIGHT,
            "logged_metrics": ["mean_positive_objectness", "mean_negative_objectness"],
            "target_only_channels": ["bird_count"],
        },
        "evaluation_policy": {
            "native_160x120_excluded": True,
            "full_scene_metrics": "not run; deployment input is always a Pi-selected motion crop",
            "threshold_selection_dataset": "ESP32 validation motion crops",
            "confidence_grid": list(CONFIDENCE_GRID),
        },
        "quantization": {
            "representative_dataset": "deterministic class-balanced ESP32 simulated motion crops",
            "representative_images": int(
                audit.loc[audit["dataset"].eq("calibration_crop"), "images"].iloc[0]
            ),
            "native_160x120_allowed_to_dominate": False,
        },
        "limitations": [
            "Historical full-scene captures do not contain Pi motion-selection bounds; their training crops are annotation-centred simulations using the same final margin and minimum-size rules.",
            "A desktop TFLite conversion does not prove the model fits the target runtime's memory budget.",
        ],
        "data_audit": audit.to_dict(orient="records"),
        "selected_thresholds": dict(thresholds),
        "exports": export_details,
    }
    (paths.artifact_dir / "run_contract.json").write_text(
        json.dumps(json_safe(contract), indent=2, allow_nan=False), encoding="utf-8"
    )


def write_decoder_metadata(
    paths: RunPaths,
    export_details: Mapping[str, Any],
    anchors: np.ndarray,
    deployment_threshold: float | None,
) -> Path:
    uint8 = export_details["uint8"]
    metadata = {
        "schema_version": 2,
        "model": paths.uint8_tflite.name,
        "model_sha256": sha256_file(paths.uint8_tflite),
        "class_map": {"0": "bird"},
        "input": {
            **uint8["inputs"][0],
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
        "output": {
            **uint8["outputs"][0],
            "order": ["tx", "ty", "tw", "th", "objectness_logit"],
            "anchors_normalized_wh": anchors.tolist(),
            "decode": {
                "dequantize": "real_value = (quantized_value - zero_point) * scale",
                "center_x": "(sigmoid(tx) * 3 - 1 + grid_x) / grid_width",
                "center_y": "(sigmoid(ty) * 3 - 1 + grid_y) / grid_height",
                "width": "anchor_width * exp(clamp(tw, -4, 4))",
                "height": "anchor_height * exp(clamp(th, -4, 4))",
                "score": "sigmoid(objectness_logit)",
            },
        },
        "postprocess": {
            "candidate_score_threshold": PRE_NMS_SCORE,
            "deployment_score_threshold": deployment_threshold,
            "deployment_threshold_source": (
                "ESP32 validation motion-crop recall-first sweep"
                if deployment_threshold is not None else "not evaluated"
            ),
            "nms_iou": NMS_IOU,
            "max_detections": MAX_DETECTIONS,
        },
        "operations": uint8.get("operations", []),
    }
    path = paths.export_dir / "bird_detector_motion_crop_metadata.json"
    path.write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False), encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=os.getenv("BIRD_DETECTOR_SOURCE_RUN", ""))
    parser.add_argument("--input-size", type=int, choices=ALLOWED_INPUT_SIZES, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(SEED); np.random.seed(SEED)
    project_root = find_project_root(Path(__file__).resolve().parent)
    source_run = resolve_source_run(project_root, args.source_run or None)
    paths = make_run_paths(source_run)
    scenes = load_scene_records(source_run / "joined_bird_detector_annotations.csv")
    datasets = select_datasets(scenes)
    audit = audit_datasets(datasets, verify_images=args.verify_images)
    audit.to_csv(paths.artifact_dir / "crop_dataset_audit.csv", index=False)
    print(audit.to_string(index=False))
    if args.verify_images and audit.loc[
        audit.dataset.isin(["train_crop", "validation_crop"]), "missing_images"
    ].sum():
        raise FileNotFoundError("Training or validation images are missing; see crop_dataset_audit.csv")
    if args.audit_only:
        (paths.artifact_dir / "crop_data_contract.json").write_text(
            json.dumps({
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_run": str(source_run),
                "input_size": args.input_size,
                "feature_layer": FEATURE_LAYER,
                "feature_stride": FEATURE_STRIDE,
                "grid_size": args.input_size // FEATURE_STRIDE,
                "native_160x120_excluded": True,
                "full_scene_metrics": "not generated",
                "data_audit": audit.to_dict(orient="records"),
            }, indent=2),
            encoding="utf-8",
        )
        return 0

    tf = require_tensorflow()
    tf.random.set_seed(SEED)
    model, feature_extractor, grid_size = build_model(tf, args.input_size)
    fingerprint_payload = configuration_fingerprint_payload(
        sha256_file(source_run / "joined_bird_detector_annotations.csv"),
        args.input_size,
    )
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    configuration_path = paths.artifact_dir / "configuration.json"
    if configuration_path.is_file():
        existing_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing_configuration.get("config_fingerprint") != fingerprint:
            raise RuntimeError(
                "Existing motion-crop checkpoints/exports belong to a different configuration. "
                f"Archive or remove {paths.artifact_dir} and {paths.export_dir} before retraining."
            )
    else:
        configuration_path.write_text(
            json.dumps({
                "config_fingerprint": fingerprint,
                "configuration": fingerprint_payload,
            }, indent=2),
            encoding="utf-8",
        )
    if paths.anchors.is_file():
        anchor_contract = json.loads(paths.anchors.read_text(encoding="utf-8"))
        if anchor_contract.get("config_fingerprint") != fingerprint:
            raise RuntimeError("Existing anchors belong to a different crop/model configuration")
        anchors = np.asarray(anchor_contract["anchors"], np.float32)
    else:
        anchors = fit_anchors(normalized_letterbox_wh(datasets["train_crop"], args.input_size), NUM_ANCHORS)
        paths.anchors.write_text(json.dumps({
            "config_fingerprint": fingerprint, "input_size": args.input_size,
            "feature_layer": FEATURE_LAYER, "anchors": anchors.tolist(),
        }, indent=2), encoding="utf-8")
    if not args.skip_training:
        train_model(tf, model, feature_extractor, datasets, anchors, grid_size, args.input_size, paths, args.batch_size)
    elif not paths.final_weights.is_file():
        raise FileNotFoundError(f"Training skipped but weights do not exist: {paths.final_weights}")
    model.load_weights(paths.final_weights)
    export_details = None
    if not args.skip_export:
        export_details = export_models(tf, model, datasets, anchors, grid_size, args.input_size, paths)
    else:
        missing_exports = [path for path in (paths.float_tflite, paths.uint8_tflite) if not path.is_file()]
        if missing_exports:
            raise FileNotFoundError(f"Export skipped but models do not exist: {missing_exports}")
        details_path = paths.export_dir / "tflite_tensor_details.json"
        if details_path.is_file():
            export_details = json.loads(details_path.read_text(encoding="utf-8"))

    thresholds: dict[str, float] = {}
    if not args.skip_evaluation:
        rows = []
        for name, model_path in (("float", paths.float_tflite), ("uint8", paths.uint8_tflite)):
            validation_predictions = predict_tflite(tf, model_path, datasets["validation_crop"], anchors, args.input_size)
            threshold, sweep = select_threshold(datasets["validation_crop"], validation_predictions)
            thresholds[name] = threshold
            sweep.to_csv(paths.artifact_dir / f"{name}_validation_crop_threshold_sweep.csv", index=False)
            for dataset_name in ("validation_crop", "test_crop"):
                records = datasets[dataset_name]
                predictions = validation_predictions if dataset_name == "validation_crop" else predict_tflite(tf, model_path, records, anchors, args.input_size)
                rows.append({"model": name, "dataset": dataset_name, **evaluate_predictions(records, predictions, threshold)})
        pd.DataFrame(rows).to_csv(paths.artifact_dir / "crop_metrics.csv", index=False)
    if export_details is None:
        export_details = {
            "float": interpreter_details(tf, paths.float_tflite),
            "uint8": interpreter_details(tf, paths.uint8_tflite),
        }
    metadata_path = write_decoder_metadata(
        paths,
        export_details,
        anchors,
        thresholds.get("uint8"),
    )
    write_contract(paths, args.input_size, grid_size, anchors, audit, export_details, thresholds)
    print("Artifacts:", paths.artifact_dir)
    print("UINT8 TFLite:", paths.uint8_tflite)
    print("Decoder metadata:", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())