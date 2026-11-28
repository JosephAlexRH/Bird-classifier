"""Small, dependency-free data layer for bird bounding-box tools.

The loaders in this module deliberately retain only the columns needed by the
viewer/annotator.  In particular, loading the wide generated CUB manifest does
not retain its hundreds of attribute columns.  Image files are never opened by
the loaders; paths are resolved only when :meth:`ImageRecord.resolve_image_path`
or :meth:`BBoxDataset.resolve_image_path` is called.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Iterator, Mapping, Sequence


REVIEW_STATES = frozenset({"boxed", "no_bird", "unusable", "skipped"})
# Folders the Pi hub writes beside captures; they hold derived images, never captures.
HUB_DERIVED_DIRECTORIES = frozenset({"crops", "review_crops", "backgrounds"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"})
LEGACY_GROUND_TRUTH_FIELDS = (
    "source_id",
    "source_dataset",
    "image_id",
    "relative_path",
    "source_width",
    "source_height",
    "review_state",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
)
GROUND_TRUTH_FIELDS = (*LEGACY_GROUND_TRUTH_FIELDS, "context_percent")
DETECTOR_MANIFEST_FIELDS = (
    "annotation_id",
    "sample_id",
    "source_dataset",
    "source_image_id",
    "image_path",
    "relative_path",
    "width_px",
    "height_px",
    "review_state",
    "annotation_trust",
    "annotation_origin",
    "eligible_for_training",
    "detector_class_id",
    "detector_class_name",
    "species_id",
    "species_name",
    "scientific_name",
    "box_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "confidence",
    "source_split",
    "model_split",
    "split_group",
    "projected_area_px2",
    "bird_size",
    "is_Bird",
    "mixed_species",
    "bird_count",
    "context_percent",
    "context_xmin",
    "context_ymin",
    "context_xmax",
    "context_ymax",
)

_TRUE = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE = frozenset({"0", "false", "f", "no", "n", "off"})


DATE_DIR_GLOB = "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]"  # hub date folders: YYYY/MM/DD

def parse_optional_float(value: object) -> float | None:
    """Return a finite float, or ``None`` for blank/malformed/non-finite data."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def parse_optional_int(value: object) -> int | None:
    """Parse an integer, accepting integral CSV floats such as ``"53.0"``."""

    number = parse_optional_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def parse_optional_bool(value: object) -> bool | None:
    """Parse common CSV boolean spellings, returning ``None`` when unknown."""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().casefold()
    if not text:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


@dataclass(frozen=True, slots=True)
class BBox:
    """A zero-based ``x, y, width, height`` rectangle in pixel coordinates."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bounding-box coordinates must be finite")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def is_valid(self) -> bool:
        return self.width > 0.0 and self.height > 0.0

    def normalized(self) -> BBox:
        """Return an equivalent rectangle with non-negative dimensions."""

        left, right = sorted((self.x, self.right))
        top, bottom = sorted((self.y, self.bottom))
        return BBox(left, top, right - left, bottom - top)

    def clipped(self, width: float, height: float) -> BBox | None:
        """Intersect this rectangle with an image; return ``None`` if empty."""

        if not (math.isfinite(width) and math.isfinite(height)) or width <= 0 or height <= 0:
            raise ValueError("clip dimensions must be positive and finite")
        box = self.normalized()
        left = min(max(box.x, 0.0), width)
        top = min(max(box.y, 0.0), height)
        right = min(max(box.right, 0.0), width)
        bottom = min(max(box.bottom, 0.0), height)
        if right <= left or bottom <= top:
            return None
        return BBox(left, top, right - left, bottom - top)


@dataclass(frozen=True, slots=True)
class SourceBox:
    """A read-only camera/model box kept separate from human annotations."""

    bbox: BBox
    score: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.bbox.is_valid:
            raise ValueError("source bounding box must be valid")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("source bounding-box score must be finite")


def bbox_with_context(
    box: BBox,
    image_width: float,
    image_height: float,
    context_percent: float = 0.0,
) -> BBox:
    """Expand a tight box by a total percentage, shifting at image edges.

    The default of 0 leaves the tight box unchanged. A value of 20 makes the
    target width and height 120% of the tight box, normally adding 10% on each
    side. Near an edge the expanded rectangle is shifted inward so its requested
    size is retained whenever the frame allows.
    """

    if not math.isfinite(context_percent) or context_percent < 0:
        raise ValueError("context_percent must be a finite non-negative number")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    tight = box.normalized().clipped(image_width, image_height)
    if tight is None:
        raise ValueError("bounding box lies outside the image")
    scale = 1.0 + context_percent / 100.0
    target_width = min(image_width, tight.width * scale)
    target_height = min(image_height, tight.height * scale)
    center_x = tight.x + tight.width / 2.0
    center_y = tight.y + tight.height / 2.0
    left = min(max(center_x - target_width / 2.0, 0.0), image_width - target_width)
    top = min(max(center_y - target_height / 2.0, 0.0), image_height - target_height)
    return BBox(left, top, target_width, target_height)


def _path_parts(value: str) -> tuple[str, ...]:
    """Split either Windows or POSIX relative path syntax."""

    return tuple(part for part in re.split(r"[\\/]", value.strip()) if part not in ("", "."))


def _under_root(root: str | os.PathLike[str], relative_path: str) -> Path:
    parts = _path_parts(relative_path)
    if any(part == ".." for part in parts):
        raise ValueError(f"relative image path escapes its root: {relative_path!r}")
    # Discard a drive/root if a malformed "relative" path contains one.
    if parts and re.fullmatch(r"[A-Za-z]:", parts[0]):
        parts = parts[1:]
    return Path(root).joinpath(*parts)


def _portable_stored_path(value: str) -> Path:
    """Interpret a stored absolute path from Windows, POSIX, or WSL."""

    value = value.strip()
    if not value:
        return Path()
    windows = PureWindowsPath(value)
    if windows.drive and windows.root:
        if os.name == "nt":
            return Path(windows)
        drive = windows.drive.rstrip(":").lower()
        return Path("/mnt", drive, *windows.parts[1:])
    posix = PurePosixPath(value)
    if os.name == "nt" and len(posix.parts) >= 4 and posix.parts[:2] == ("/", "mnt"):
        drive = posix.parts[2]
        if len(drive) == 1 and drive.isalpha():
            return Path(f"{drive.upper()}:\\", *posix.parts[3:])
    if "\\" in value:
        return Path(*_path_parts(value))
    return Path(value)


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Compact immutable metadata for one source image."""

    source_id: str
    source_dataset: str
    image_id: str
    relative_path: str
    stored_path: str = ""
    width: int | None = None
    height: int | None = None
    bbox: BBox | None = None
    bbox_status: str = ""
    bbox_source: str = ""
    confidence: float | None = None
    detection_count: int | None = None
    class_id: str = ""
    class_name: str = ""
    split: str = ""
    is_training: bool | None = None
    is_bird: bool | None = None
    mixed_species: bool = False
    bird_count: int | None = None
    captured_at: str = ""
    camera_id: str = ""
    camera_name: str = ""
    camera_model: str = ""
    event_id: str = ""
    deployment_id: str = ""
    inference_status: str = ""
    predicted_is_bird: bool | None = None
    predicted_species: str = ""
    prediction_confidence: float | None = None
    predicted_bird_count: int | None = None
    source_boxes: tuple[SourceBox, ...] = ()
    motion_bbox: BBox | None = None
    motion_source: str = ""

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.relative_path and not self.stored_path:
            raise ValueError("an image record needs a relative or stored path")
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive when present")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive when present")
        if self.bird_count is not None and self.bird_count < 0:
            raise ValueError("bird_count must not be negative")
        if self.predicted_bird_count is not None and self.predicted_bird_count < 0:
            raise ValueError("predicted_bird_count must not be negative")
        if self.prediction_confidence is not None and not math.isfinite(self.prediction_confidence):
            raise ValueError("prediction_confidence must be finite when present")

    @property
    def has_bbox(self) -> bool:
        return self.bbox is not None and self.bbox.is_valid

    @property
    def image_width(self) -> int | None:
        return self.width

    @property
    def image_height(self) -> int | None:
        return self.height

    @property
    def detection_status(self) -> str:
        return self.bbox_status

    def resolve_image_path(self, image_root: str | os.PathLike[str] | None = None) -> Path:
        """Resolve lazily, preferring an explicit portable image-root override."""

        if image_root is not None:
            return _under_root(image_root, self.relative_path)
        if self.stored_path:
            return _portable_stored_path(self.stored_path)
        return _under_root(Path(), self.relative_path)


@dataclass(frozen=True, slots=True)
class BBoxDataset(Sequence[ImageRecord]):
    """An immutable collection and its path-resolution context."""

    records: tuple[ImageRecord, ...]
    source_path: Path
    format: str
    image_root: Path | None = None

    def __iter__(self) -> Iterator[ImageRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | slice) -> ImageRecord | tuple[ImageRecord, ...]:
        return self.records[index]

    def resolve_image_path(
        self,
        record: ImageRecord,
        image_root: str | os.PathLike[str] | None = None,
    ) -> Path:
        root = image_root if image_root is not None else self.image_root
        return record.resolve_image_path(root)


def _text(row: list[str], columns: Mapping[str, int], name: str) -> str:
    index = columns.get(name)
    return row[index].strip() if index is not None and index < len(row) else ""


def _first(row: list[str], columns: Mapping[str, int], *names: str) -> str:
    for name in names:
        value = _text(row, columns, name)
        if value:
            return value
    return ""


def _parsed_bbox(
    row: list[str],
    columns: Mapping[str, int],
    source_path: Path,
    line_number: int,
) -> BBox | None:
    names = ("bbox_x", "bbox_y", "bbox_width", "bbox_height")
    texts = tuple(_text(row, columns, key) for key in names)
    if not any(texts):
        return None
    if not all(texts):
        raise ValueError(f"{source_path}:{line_number}: incomplete bounding box")
    values = tuple(parse_optional_float(text) for text in texts)
    if any(value is None for value in values):
        raise ValueError(f"{source_path}:{line_number}: invalid bounding box")
    box = BBox(*values)  # type: ignore[arg-type]
    if not box.is_valid:
        raise ValueError(f"{source_path}:{line_number}: invalid bounding box dimensions")
    return box


def _csv_kind(columns: Mapping[str, int]) -> str:
    detector_required = {
        "sample_id", "image_path", "width_px", "height_px",
        "xmin", "ymin", "xmax", "ymax",
    }
    if detector_required <= columns.keys():
        return "detector_annotations_csv"
    if "bbox_detection_status" in columns or "bridge_image_id" in columns:
        return "inaturalist_bbox_csv"
    required = {"image_id", "relative_path", "bbox_x", "bbox_y", "bbox_width", "bbox_height"}
    if required <= columns.keys():
        return "cub_manifest_csv"
    raise ValueError("unsupported bbox CSV header")


def load_manifest_csv(
    path: str | os.PathLike[str],
    image_root: str | os.PathLike[str] | None = None,
) -> BBoxDataset:
    """Load a generated iNaturalist bbox or generated CUB image manifest CSV."""

    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty CSV: {source_path}") from error
    columns = {name.strip(): index for index, name in enumerate(header)}
    if _csv_kind(columns) == "detector_annotations_csv":
        dataset, _reviews = load_detector_manifest(source_path, image_root=image_root)
        return dataset

    records: list[ImageRecord] = []
    seen_source_ids: set[str] = set()
    declared_root = ""
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty CSV: {source_path}") from error
        columns = {name.strip(): index for index, name in enumerate(header)}
        kind = _csv_kind(columns)
        for line_number, row in enumerate(reader, 2):
            if not row or not any(cell.strip() for cell in row):
                continue
            image_id = _first(row, columns, "source_image_id", "image_id")
            source_dataset = _first(row, columns, "source_dataset") or (
                "inaturalist" if kind == "inaturalist_bbox_csv" else "cub_200_2011"
            )
            source_id = _first(row, columns, "bridge_image_id")
            if not source_id:
                source_id = f"{source_dataset}:{image_id}" if image_id else ""
            if not source_id:
                raise ValueError(f"{source_path}:{line_number}: missing stable image identity")
            if source_id in seen_source_ids:
                raise ValueError(f"{source_path}:{line_number}: duplicate source_id {source_id!r}")
            seen_source_ids.add(source_id)
            relative_path = _first(
                row, columns, "photo_relative_path", "relative_path", "source_relative_path"
            )
            stored_path = _first(row, columns, "absolute_path", "image_path")
            if not relative_path and stored_path:
                relative_path = _portable_stored_path(stored_path).name
            if not relative_path and not stored_path:
                raise ValueError(f"{source_path}:{line_number}: missing image path")
            if not declared_root:
                declared_root = _first(row, columns, "photos_directory")
            width = parse_optional_int(_first(row, columns, "bbox_image_width_px", "width_px"))
            height = parse_optional_int(_first(row, columns, "bbox_image_height_px", "height_px"))
            records.append(
                ImageRecord(
                    source_id=source_id,
                    source_dataset=source_dataset,
                    image_id=image_id,
                    relative_path=relative_path,
                    stored_path=stored_path,
                    width=width,
                    height=height,
                    bbox=_parsed_bbox(row, columns, source_path, line_number),
                    bbox_status=_first(row, columns, "bbox_detection_status")
                    or ("ground_truth" if kind == "cub_manifest_csv" else ""),
                    bbox_source=_first(row, columns, "bbox_source")
                    or ("cub_annotation" if kind == "cub_manifest_csv" else ""),
                    confidence=parse_optional_float(_text(row, columns, "bbox_confidence")),
                    detection_count=parse_optional_int(_text(row, columns, "bird_detection_count")),
                    class_id=_first(row, columns, "cub_class_id", "class_id"),
                    class_name=_first(
                        row, columns, "common_name", "class_name_friendly", "class_name"
                    ),
                    split=_text(row, columns, "split"),
                    is_training=parse_optional_bool(_text(row, columns, "is_training_image")),
                )
            )
    root_value: str | os.PathLike[str] | None = image_root
    if root_value is None and declared_root:
        root_value = _portable_stored_path(declared_root)
    return BBoxDataset(tuple(records), source_path, kind, Path(root_value) if root_value else None)


def _read_id_table(path: Path, expected_values: int) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split(maxsplit=expected_values)
            if not parts:
                continue
            if len(parts) != expected_values + 1:
                raise ValueError(f"{path}:{line_number}: malformed CUB metadata row")
            result[parts[0]] = tuple(parts[1:])
    return result


def load_cub_directory(
    path: str | os.PathLike[str],
    image_root: str | os.PathLike[str] | None = None,
) -> BBoxDataset:
    """Load an unmodified standard ``CUB_200_2011`` directory."""

    root = Path(path)
    required = (
        "images.txt", "bounding_boxes.txt", "image_class_labels.txt",
        "train_test_split.txt", "classes.txt",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"not a standard CUB_200_2011 directory; missing {', '.join(missing)}")
    images = _read_id_table(root / "images.txt", 1)
    boxes = _read_id_table(root / "bounding_boxes.txt", 4)
    labels = _read_id_table(root / "image_class_labels.txt", 1)
    splits = _read_id_table(root / "train_test_split.txt", 1)
    classes = _read_id_table(root / "classes.txt", 1)
    records: list[ImageRecord] = []
    for image_id, path_values in images.items():
        if image_id not in boxes or image_id not in labels or image_id not in splits:
            raise ValueError(f"incomplete CUB metadata for image id {image_id}")
        box_values = tuple(parse_optional_float(value) for value in boxes[image_id])
        if any(value is None for value in box_values):
            raise ValueError(f"invalid CUB bounding box for image id {image_id}")
        box = BBox(*box_values)  # type: ignore[arg-type]
        class_id = labels[image_id][0]
        training = parse_optional_bool(splits[image_id][0])
        relative_path = path_values[0]
        records.append(
            ImageRecord(
                source_id=f"cub_200_2011:{image_id}",
                source_dataset="cub_200_2011",
                image_id=image_id,
                relative_path=relative_path,
                stored_path=str(root / "images" / Path(*_path_parts(relative_path))),
                bbox=box if box.is_valid else None,
                bbox_status="ground_truth",
                bbox_source="cub_annotation",
                class_id=class_id,
                class_name=classes.get(class_id, ("",))[0].replace("_", " "),
                split="train" if training else "test" if training is False else "",
                is_training=training,
            )
        )
    chosen_root = Path(image_root) if image_root is not None else root / "images"
    return BBoxDataset(tuple(records), root, "cub_raw", chosen_root)


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _sidecar_manual_label(
    path: Path, data: Mapping[str, object] | None = None
) -> tuple[bool | None, str]:
    if data is None:
        data = _load_json_object(path.with_suffix(".json"))
    if data is None:
        return None, ""
    label = data.get("manual_label")
    if not isinstance(label, dict) or type(label.get("bird_present")) is not bool:
        return None, ""
    species = label.get("species")
    return label["bird_present"], species.strip() if isinstance(species, str) else ""


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _acquisition_matches_image(image_path: Path, acquisition: Mapping[str, object]) -> bool:
    """Bind acquisition metadata and any manual label to the selected image."""

    capture_id = _nonempty_text(acquisition.get("capture_id"))
    filename = _nonempty_text(acquisition.get("filename"))
    if (capture_id and capture_id != image_path.stem) or (filename and filename != image_path.name):
        return False
    expected_hash = _nonempty_text(acquisition.get("sha256")).casefold()
    if not expected_hash:
        return True
    actual_hash = _sha256_file(image_path)
    return actual_hash is not None and actual_hash.casefold() == expected_hash


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, dict) else None


def _nonempty_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sidecar_hash_matches(
    acquisition: Mapping[str, object], sidecar_hash: object
) -> bool:
    """Accept absent hashes, but reject a sidecar that names different image bytes."""

    expected = _nonempty_text(acquisition.get("sha256")).casefold()
    observed = _nonempty_text(sidecar_hash).casefold()
    return not expected or not observed or expected == observed


def _box_from_edges(value: object, width: int | None, height: int | None) -> BBox | None:
    data = _mapping(value)
    if data is None:
        return None
    left = parse_optional_float(data.get("left"))
    top = parse_optional_float(data.get("top"))
    right = parse_optional_float(data.get("right"))
    bottom = parse_optional_float(data.get("bottom"))
    if None in (left, top, right, bottom):
        return None
    assert left is not None and top is not None and right is not None and bottom is not None
    box = BBox(left, top, right - left, bottom - top)
    if not box.is_valid:
        return None
    if width is not None and height is not None:
        return box.clipped(width, height)
    return box


@dataclass(frozen=True, slots=True)
class _CameraCaptureData:
    capture_id: str
    width: int | None
    height: int | None
    captured_at: str = ""
    camera_id: str = ""
    camera_name: str = ""
    camera_model: str = ""
    event_id: str = ""
    deployment_id: str = ""
    inference_status: str = ""
    predicted_is_bird: bool | None = None
    predicted_species: str = ""
    prediction_confidence: float | None = None
    predicted_bird_count: int | None = None
    source_boxes: tuple[SourceBox, ...] = ()
    motion_bbox: BBox | None = None
    motion_source: str = ""


def _camera_capture_data(
    image_path: Path,
    acquisition: Mapping[str, object] | None,
    decoded_width: int | None,
    decoded_height: int | None,
) -> _CameraCaptureData | None:
    """Read a Pi camera capture bundle without treating model output as a label."""

    if acquisition is None:
        return None
    capture_id = _nonempty_text(acquisition.get("capture_id"))
    filename = _nonempty_text(acquisition.get("filename"))
    if not capture_id or capture_id != image_path.stem or (filename and filename != image_path.name):
        return None

    camera = _mapping(acquisition.get("camera")) or {}
    observed = _mapping(camera.get("observed")) or {}
    observed_capture = _mapping(observed.get("capture")) or {}
    width = decoded_width or parse_optional_int(observed_capture.get("width"))
    height = decoded_height or parse_optional_int(observed_capture.get("height"))

    inference = _load_json_object(image_path.with_suffix(".inference.json"))
    if inference is not None:
        inference_capture_id = _nonempty_text(inference.get("capture_id"))
        image = _mapping(inference.get("image")) or {}
        inference_filename = _nonempty_text(image.get("filename"))
        if ((inference_capture_id and inference_capture_id != capture_id)
                or (inference_filename and inference_filename != image_path.name)
                or not _sidecar_hash_matches(acquisition, image.get("sha256"))):
            inference = None

    predicted_is_bird: bool | None = None
    prediction_confidence: float | None = None
    predicted_species = ""
    inference_status = ""
    source_boxes: list[SourceBox] = []
    if inference is not None:
        inference_status = _nonempty_text(inference.get("status"))
        if type(inference.get("is_bird")) is bool:
            predicted_is_bird = bool(inference["is_bird"])
        classification = _mapping(inference.get("classification")) or {}
        if predicted_is_bird is None and type(classification.get("is_bird")) is bool:
            predicted_is_bird = bool(classification["is_bird"])
        prediction_confidence = parse_optional_float(inference.get("bird_probability"))
        if prediction_confidence is None:
            prediction_confidence = parse_optional_float(classification.get("confidence"))
        predicted_species = _nonempty_text(classification.get("label")) or _nonempty_text(
            inference.get("guess")
        )
        top_k = inference.get("species_top_k")
        if not predicted_species and isinstance(top_k, list) and top_k:
            predicted_species = _nonempty_text((_mapping(top_k[0]) or {}).get("label"))

        # Species entries are the accepted localizer boxes and preserve each box's guess.
        species = _mapping(inference.get("species")) or {}
        species_boxes = species.get("boxes")
        if predicted_is_bird is True and isinstance(species_boxes, list):
            for entry in species_boxes:
                item = _mapping(entry) or {}
                box = _box_from_edges(item.get("box"), width, height)
                if box is None:
                    continue
                entry_top_k = item.get("top_k")
                label = ""
                if isinstance(entry_top_k, list) and entry_top_k:
                    label = _nonempty_text((_mapping(entry_top_k[0]) or {}).get("label"))
                source_boxes.append(SourceBox(
                    box,
                    parse_optional_float(item.get("localizer_score")),
                    label,
                ))

        # Fall back to thresholded localizer output when species inference is absent.
        if predicted_is_bird is True and not source_boxes:
            localizer = _mapping(inference.get("localizer")) or {}
            localizer_model = _mapping(localizer.get("model")) or {}
            threshold = parse_optional_float(localizer_model.get("deployment_threshold"))
            if threshold is None:
                threshold = 0.0
            detections = localizer.get("detections")
            if isinstance(detections, list):
                for entry in detections:
                    item = _mapping(entry) or {}
                    score = parse_optional_float(item.get("score"))
                    if score is None or score < threshold:
                        continue
                    box = _box_from_edges(item.get("box"), width, height)
                    if box is not None:
                        source_boxes.append(SourceBox(box, score, predicted_species))

        # Compatibility with the older detector block used by previous Pi deployments.
        if predicted_is_bird is True and not source_boxes:
            detector = _mapping(inference.get("detector")) or {}
            threshold = parse_optional_float(detector.get("threshold")) or 0.0
            detections = detector.get("detections")
            if isinstance(detections, list):
                for entry in detections:
                    item = _mapping(entry) or {}
                    score = parse_optional_float(item.get("score"))
                    if score is None or score < threshold:
                        continue
                    box = _box_from_edges(item.get("box"), width, height)
                    if box is not None:
                        source_boxes.append(SourceBox(box, score, predicted_species))

    motion = _load_json_object(image_path.with_suffix(".motion.json"))
    if motion is not None:
        capture = _mapping(motion.get("capture")) or {}
        motion_capture_id = _nonempty_text(capture.get("capture_id"))
        motion_filename = _nonempty_text(capture.get("filename"))
        if ((motion_capture_id and motion_capture_id != capture_id)
                or (motion_filename and motion_filename != image_path.name)
                or not _sidecar_hash_matches(acquisition, capture.get("image_sha256"))):
            motion = None
    selection = _mapping(motion.get("selection")) if motion is not None else None
    motion_bbox = _box_from_edges(selection.get("final_bounds_px"), width, height) if selection else None
    motion_source = _nonempty_text(selection.get("source")) if selection else ""

    predicted_bird_count = (
        len(source_boxes) if predicted_is_bird is True
        else 0 if predicted_is_bird is False
        else None
    )
    return _CameraCaptureData(
        capture_id=capture_id,
        width=width,
        height=height,
        captured_at=_nonempty_text(acquisition.get("received_at")),
        camera_id=_nonempty_text(acquisition.get("camera_id")),
        camera_name=_nonempty_text(acquisition.get("camera_name")),
        camera_model=_nonempty_text(acquisition.get("camera_model")),
        event_id=_nonempty_text(acquisition.get("event_id")),
        deployment_id=_nonempty_text(acquisition.get("deployment_id")),
        inference_status=inference_status,
        predicted_is_bird=predicted_is_bird,
        predicted_species=predicted_species,
        prediction_confidence=prediction_confidence,
        predicted_bird_count=predicted_bird_count,
        source_boxes=tuple(source_boxes),
        motion_bbox=motion_bbox,
        motion_source=motion_source,
    )


def load_image_directory(
    path: str | os.PathLike[str], *, recursive: bool = True
) -> tuple[BBoxDataset, dict[str, GroundTruthRecord]]:
    """Discover images and build an unsaved detector-style dataset.

    ``recursive=False`` limits discovery to files directly inside ``path`` (plus the
    hub's ``YYYY/MM/DD`` capture folders).  This
    is useful for capture directories that also contain generated crops or other
    derived images in subdirectories. The hub's own derived folders (``crops``,
    ``review_crops``, ``backgrounds``) are skipped even when recursive.
    """

    root = Path(path).expanduser().resolve(strict=True)
    candidates = root.rglob("*") if recursive else [*root.iterdir(), *root.glob(f"{DATE_DIR_GLOB}/*")]
    images = sorted(
        (item for item in candidates if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS
         and not HUB_DERIVED_DIRECTORIES.intersection(
             part.casefold() for part in item.relative_to(root).parts[:-1])),
        key=lambda item: item.relative_to(root).as_posix().casefold(),
    )
    records: list[ImageRecord] = []
    reviews: dict[str, GroundTruthRecord] = {}
    source_ids: set[str] = set()
    for index, image_path in enumerate(images, 1):
        relative_path = image_path.relative_to(root).as_posix()
        image_id = image_path.stem
        width: int | None = None
        height: int | None = None
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, ValueError):
            pass
        acquisition = _load_json_object(image_path.with_suffix(".json"))
        label_acquisition = acquisition
        if acquisition is not None and not _acquisition_matches_image(image_path, acquisition):
            acquisition = None
            label_acquisition = {}
        capture = _camera_capture_data(image_path, acquisition, width, height)
        source_id = capture.capture_id if capture else image_id
        if source_id in source_ids:
            if capture:
                raise ValueError(f"duplicate camera capture_id {source_id!r} in {root}")
            source_id = f"image:{index}:{relative_path}"
        source_ids.add(source_id)
        if capture:
            image_id = capture.capture_id
            width, height = capture.width, capture.height
        is_bird, species = _sidecar_manual_label(image_path, label_acquisition)
        record = ImageRecord(
            source_id=source_id,
            source_dataset="esp32_capture" if capture else "image_directory",
            image_id=image_id,
            relative_path=relative_path,
            stored_path=str(image_path),
            width=width,
            height=height,
            class_name=species,
            is_bird=is_bird,
            bird_count=0 if is_bird is False else None,
            captured_at=capture.captured_at if capture else "",
            camera_id=capture.camera_id if capture else "",
            camera_name=capture.camera_name if capture else "",
            camera_model=capture.camera_model if capture else "",
            event_id=capture.event_id if capture else "",
            deployment_id=capture.deployment_id if capture else "",
            inference_status=capture.inference_status if capture else "",
            predicted_is_bird=capture.predicted_is_bird if capture else None,
            predicted_species=capture.predicted_species if capture else "",
            prediction_confidence=capture.prediction_confidence if capture else None,
            predicted_bird_count=capture.predicted_bird_count if capture else None,
            source_boxes=capture.source_boxes if capture else (),
            motion_bbox=capture.motion_bbox if capture else None,
            motion_source=capture.motion_source if capture else "",
        )
        records.append(record)
        if is_bird is not None and width and height:
            state = "no_bird" if not is_bird else "skipped"
            reviews[source_id] = GroundTruthRecord.from_image_record(
                record, state, species=species, is_bird=is_bird, bird_count=0 if not is_bird else 1
            )
    return BBoxDataset(tuple(records), root, "image_directory", root), reviews


def _detector_bbox(row: Mapping[str, str], source: Path, line_number: int) -> BBox | None:
    values = tuple(parse_optional_float(row.get(name)) for name in ("xmin", "ymin", "xmax", "ymax"))
    if not any(value is not None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{source}:{line_number}: incomplete detector bounding box")
    xmin, ymin, xmax, ymax = values
    assert xmin is not None and ymin is not None and xmax is not None and ymax is not None
    box = BBox(xmin, ymin, xmax - xmin, ymax - ymin)
    if not box.is_valid:
        raise ValueError(f"{source}:{line_number}: invalid detector bounding box")
    return box


def load_detector_manifest(
    path: str | os.PathLike[str], image_root: str | os.PathLike[str] | None = None
) -> tuple[BBoxDataset, dict[str, GroundTruthRecord]]:
    """Load the detector training CSV, grouping its one-row-per-box representation."""

    source = Path(path)
    grouped: dict[str, list[tuple[int, dict[str, str], BBox | None]]] = {}
    order: list[str] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {
            "sample_id", "image_path", "width_px", "height_px", "xmin", "ymin", "xmax", "ymax"
        } <= set(reader.fieldnames):
            raise ValueError(f"unsupported detector CSV header: {source}")
        for line_number, row in enumerate(reader, 2):
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"{source}:{line_number}: missing sample_id")
            if sample_id not in grouped:
                grouped[sample_id] = []
                order.append(sample_id)
            grouped[sample_id].append((line_number, dict(row), _detector_bbox(row, source, line_number)))

    records: list[ImageRecord] = []
    reviews: dict[str, GroundTruthRecord] = {}
    for sample_id in order:
        rows = grouped[sample_id]
        first = rows[0][1]
        width = parse_optional_int(first.get("width_px"))
        height = parse_optional_int(first.get("height_px"))
        if width is None or height is None:
            raise ValueError(f"{source}:{rows[0][0]}: invalid source dimensions")
        stored_path = (first.get("image_path") or "").strip()
        relative_path = (first.get("relative_path") or "").strip()
        if not relative_path and stored_path:
            relative_path = _portable_stored_path(stored_path).name
        source_dataset = (first.get("source_dataset") or "detector_manifest").strip()
        boxes = [box for _line, _row, box in rows if box is not None]
        row_species = [(row.get("species_name") or "").strip() for _line, row, box in rows if box is not None]
        mixed = parse_optional_bool(first.get("mixed_species"))
        if mixed is None:
            mixed = len({name.casefold() for name in row_species if name}) > 1
        species = "" if mixed else next((name for name in row_species if name), "")
        is_bird = parse_optional_bool(first.get("is_Bird"))
        state = (first.get("review_state") or "").strip()
        if is_bird is None:
            is_bird = (
                state in {"boxed", "unusable"} or bool(boxes)
                if state in REVIEW_STATES or boxes else None
            )
        bird_count = parse_optional_int(first.get("bird_count"))
        if bird_count is None:
            bird_count = len(boxes) if is_bird else 0 if is_bird is False else None
        context_percent = parse_optional_float(first.get("context_percent"))
        if context_percent is None or context_percent < 0:
            context_percent = 0.0
        record = ImageRecord(
            source_id=sample_id,
            source_dataset=source_dataset,
            image_id=(first.get("source_image_id") or sample_id).strip(),
            relative_path=relative_path,
            stored_path=stored_path,
            width=width,
            height=height,
            bbox=boxes[0] if boxes else None,
            bbox_status=state,
            bbox_source=(first.get("annotation_origin") or "").strip(),
            confidence=parse_optional_float(first.get("confidence")),
            detection_count=bird_count,
            class_id=(first.get("species_id") or "").strip(),
            class_name=species,
            split=(first.get("model_split") or "").strip(),
            is_training=parse_optional_bool(first.get("eligible_for_training")),
            is_bird=is_bird,
            mixed_species=bool(mixed),
            bird_count=bird_count,
        )
        records.append(record)
        if state in REVIEW_STATES or is_bird is not None or boxes:
            review_state = state if state in REVIEW_STATES else "boxed" if boxes else "no_bird"
            reviews[sample_id] = GroundTruthRecord.from_image_record(
                record,
                review_state,
                bboxes=boxes if review_state == "boxed" else [],
                species=species,
                mixed_species=bool(mixed),
                box_species=row_species,
                is_bird=is_bird,
                bird_count=bird_count,
                context_percent=context_percent,
            )
    root = Path(image_root) if image_root is not None else None
    return BBoxDataset(tuple(records), source, "detector_annotations_csv", root), reviews


def load_dataset(
    path: str | os.PathLike[str],
    image_root: str | os.PathLike[str] | None = None,
) -> BBoxDataset:
    """Autodetect and load a supported manifest CSV or raw CUB directory."""

    source = Path(path)
    if source.is_dir():
        required = (
            "images.txt", "bounding_boxes.txt", "image_class_labels.txt",
            "train_test_split.txt", "classes.txt",
        )
        if all((source / name).is_file() for name in required):
            return load_cub_directory(source, image_root=image_root)
        dataset, _reviews = load_image_directory(source)
        return dataset
    if not source.is_file():
        raise FileNotFoundError(source)
    return load_manifest_csv(source, image_root=image_root)


load_bbox_dataset = load_dataset


def filter_records(
    records: Iterable[ImageRecord],
    *,
    statuses: Iterable[str] | None = None,
    has_bbox: bool | None = None,
    source_dataset: str | None = None,
    query: str | None = None,
) -> tuple[ImageRecord, ...]:
    """Filter records without mutating their dataset/order."""

    accepted_statuses = {status.casefold() for status in statuses} if statuses is not None else None
    needle = query.strip().casefold() if query else ""
    result = []
    for record in records:
        if accepted_statuses is not None and record.bbox_status.casefold() not in accepted_statuses:
            continue
        if has_bbox is not None and record.has_bbox is not has_bbox:
            continue
        if source_dataset is not None and record.source_dataset != source_dataset:
            continue
        if needle and not any(
            needle in value.casefold()
            for value in (
                record.source_id, record.image_id, record.relative_path,
                record.class_id, record.class_name, record.bbox_status,
                record.camera_id, record.camera_name, record.camera_model,
                record.event_id, record.deployment_id, record.captured_at,
                record.inference_status, record.predicted_species,
            )
        ):
            continue
        result.append(record)
    return tuple(result)


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def sort_records(
    records: Iterable[ImageRecord],
    key: str = "source_id",
    reverse: bool = False,
) -> tuple[ImageRecord, ...]:
    """Return records sorted naturally by a supported display field."""

    aliases = {"path": "relative_path", "status": "bbox_status", "dataset": "source_dataset"}
    field = aliases.get(key, key)
    fields = {
        "source_id", "source_dataset", "image_id", "relative_path", "bbox_status",
        "class_id", "class_name", "split", "confidence", "detection_count",
        "captured_at", "camera_id", "camera_name", "camera_model", "event_id",
        "deployment_id", "inference_status", "predicted_species", "prediction_confidence",
        "predicted_bird_count",
    }
    if field not in fields:
        raise ValueError(f"unsupported record sort key: {key}")

    def value(record: ImageRecord) -> object:
        item = getattr(record, field)
        if isinstance(item, str):
            return _natural_key(item)
        return (item is None, item if item is not None else 0)

    return tuple(sorted(records, key=value, reverse=reverse))


@dataclass(frozen=True, slots=True)
class ImageCanvasTransform:
    """Aspect-preserving mapping between image and canvas coordinates."""

    image_width: float
    image_height: float
    canvas_width: float
    canvas_height: float
    scale: float
    offset_x: float
    offset_y: float
    display_width: float
    display_height: float

    @classmethod
    def fit(
        cls,
        image_width: float,
        image_height: float,
        canvas_width: float,
        canvas_height: float,
    ) -> ImageCanvasTransform:
        values = (image_width, image_height, canvas_width, canvas_height)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("image and canvas dimensions must be positive and finite")
        scale = min(canvas_width / image_width, canvas_height / image_height)
        display_width = image_width * scale
        display_height = image_height * scale
        return cls(
            image_width, image_height, canvas_width, canvas_height, scale,
            (canvas_width - display_width) / 2.0,
            (canvas_height - display_height) / 2.0,
            display_width, display_height,
        )

    def image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.offset_x + x * self.scale, self.offset_y + y * self.scale

    def canvas_to_image(self, x: float, y: float, clamp: bool = True) -> tuple[float, float]:
        image_x = (x - self.offset_x) / self.scale
        image_y = (y - self.offset_y) / self.scale
        if clamp:
            image_x = min(max(image_x, 0.0), self.image_width)
            image_y = min(max(image_y, 0.0), self.image_height)
        return image_x, image_y

    def image_bbox_to_canvas(self, bbox: BBox, clip: bool = True) -> BBox | None:
        box = bbox.clipped(self.image_width, self.image_height) if clip else bbox.normalized()
        if box is None:
            return None
        left, top = self.image_to_canvas(box.x, box.y)
        return BBox(left, top, box.width * self.scale, box.height * self.scale)

    def canvas_bbox_to_image(
        self,
        bbox_or_x: BBox | float,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
    ) -> BBox | None:
        """Map and clip a canvas BBox (or x, y, width, height) to the image."""

        if isinstance(bbox_or_x, BBox):
            if any(value is not None for value in (y, width, height)):
                raise TypeError("pass either a BBox or x, y, width, height")
            box = bbox_or_x
        else:
            if y is None or width is None or height is None:
                raise TypeError("x, y, width and height are required")
            box = BBox(float(bbox_or_x), y, width, height)
        box = box.normalized()
        left, top = self.canvas_to_image(box.x, box.y, clamp=False)
        right, bottom = self.canvas_to_image(box.right, box.bottom, clamp=False)
        return BBox(left, top, right - left, bottom - top).clipped(
            self.image_width, self.image_height
        )


@dataclass(frozen=True, slots=True, init=False)
class GroundTruthRecord:
    """One image-level review decision containing zero or more bird boxes.

    ``bboxes`` is the canonical representation.  The legacy ``bbox`` argument
    and property remain available for callers that work with one box.
    """

    source_id: str
    source_dataset: str
    image_id: str
    relative_path: str
    source_width: int
    source_height: int
    review_state: str
    bboxes: list[BBox]
    is_bird: bool
    species: str
    mixed_species: bool
    box_species: list[str]
    bird_count: int
    context_percent: float

    def __init__(
        self,
        source_id: str,
        source_dataset: str,
        image_id: str,
        relative_path: str,
        source_width: int,
        source_height: int,
        review_state: str,
        bbox: BBox | None = None,
        *,
        bboxes: Iterable[BBox] | None = None,
        is_bird: bool | None = None,
        species: str = "",
        mixed_species: bool = False,
        box_species: Iterable[str] | None = None,
        bird_count: int | None = None,
        context_percent: float = 0.0,
    ) -> None:
        if bbox is not None and bboxes is not None:
            raise ValueError("specify bbox or bboxes, not both")
        boxes = list(bboxes) if bboxes is not None else ([] if bbox is None else [bbox])
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_dataset", source_dataset)
        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "source_width", source_width)
        object.__setattr__(self, "source_height", source_height)
        object.__setattr__(self, "review_state", review_state)
        object.__setattr__(self, "bboxes", boxes)
        inferred_bird = review_state in {"boxed", "unusable"} if is_bird is None else is_bird
        labels = list(box_species or [])
        if len(labels) < len(boxes):
            labels.extend([""] * (len(boxes) - len(labels)))
        object.__setattr__(self, "is_bird", bool(inferred_bird))
        object.__setattr__(self, "species", species.strip())
        object.__setattr__(self, "mixed_species", bool(mixed_species))
        object.__setattr__(self, "box_species", [label.strip() for label in labels[:len(boxes)]])
        object.__setattr__(
            self,
            "bird_count",
            len(boxes) if review_state == "boxed" else int(bird_count or 0),
        )
        object.__setattr__(self, "context_percent", float(context_percent))
        self.__post_init__()

    @property
    def bbox(self) -> BBox | None:
        """Return the first box for compatibility with single-box callers."""

        return self.bboxes[0] if self.bboxes else None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("source dimensions must be positive")
        if self.review_state not in REVIEW_STATES:
            raise ValueError(f"invalid review state: {self.review_state!r}")
        if self.review_state == "boxed":
            if not self.bboxes:
                raise ValueError("boxed review state requires a valid bounding box (at least one)")
            tolerance = 1e-12 * max(self.source_width, self.source_height, 1)
            for box in self.bboxes:
                if not isinstance(box, BBox) or not box.is_valid:
                    raise ValueError("boxed review state requires valid bounding boxes")
                normalized = box.normalized()
                if (
                    normalized.x < -tolerance
                    or normalized.y < -tolerance
                    or normalized.right > self.source_width + tolerance
                    or normalized.bottom > self.source_height + tolerance
                ):
                    raise ValueError("ground-truth bounding boxes must be within source dimensions")
        elif self.bboxes:
            raise ValueError(f"{self.review_state} review state must not contain bounding boxes")
        if self.bird_count < 0:
            raise ValueError("bird_count must not be negative")
        if not math.isfinite(self.context_percent) or self.context_percent < 0:
            raise ValueError("context_percent must be a finite non-negative number")
        if not self.is_bird and self.bird_count != 0:
            raise ValueError("non-bird records must have bird_count 0")
        if self.review_state == "boxed" and self.bird_count != len(self.bboxes):
            raise ValueError("boxed bird_count must match the number of bounding boxes")
        if self.mixed_species:
            if len(self.box_species) != len(self.bboxes):
                raise ValueError("mixed-species records need one species value per box")
            if any(not label for label in self.box_species):
                raise ValueError("enter a species for every bird box")

    @classmethod
    def from_image_record(
        cls,
        record: ImageRecord,
        review_state: str,
        bbox: BBox | None = None,
        *,
        bboxes: Iterable[BBox] | None = None,
        source_width: int | None = None,
        source_height: int | None = None,
        is_bird: bool | None = None,
        species: str = "",
        mixed_species: bool = False,
        box_species: Iterable[str] | None = None,
        bird_count: int | None = None,
        context_percent: float = 0.0,
    ) -> GroundTruthRecord:
        width = source_width if source_width is not None else record.width
        height = source_height if source_height is not None else record.height
        if width is None or height is None:
            raise ValueError("source dimensions are required when absent from the manifest")
        return cls(
            record.source_id, record.source_dataset, record.image_id, record.relative_path,
            width, height, review_state, bbox, bboxes=bboxes,
            is_bird=is_bird,
            species=species,
            mixed_species=mixed_species,
            box_species=box_species,
            bird_count=bird_count,
            context_percent=context_percent,
        )


# A short alias useful in GUI type annotations.
ReviewRecord = GroundTruthRecord


def load_ground_truth(path: str | os.PathLike[str]) -> dict[str, GroundTruthRecord]:
    """Load image decisions, combining one boxed CSV row per bird box."""

    source = Path(path)
    if not source.exists():
        return {}
    result: dict[str, GroundTruthRecord] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(LEGACY_GROUND_TRUTH_FIELDS) <= set(reader.fieldnames):
            raise ValueError(f"unsupported ground-truth CSV header: {source}")
        for line_number, row in enumerate(reader, 2):
            width = parse_optional_int(row.get("source_width"))
            height = parse_optional_int(row.get("source_height"))
            state = (row.get("review_state") or "").strip()
            source_id = (row.get("source_id") or "").strip()
            if width is None or height is None:
                raise ValueError(f"{source}:{line_number}: invalid source dimensions")
            values = tuple(parse_optional_float(row.get(key)) for key in (
                "bbox_x", "bbox_y", "bbox_width", "bbox_height"
            ))
            bbox: BBox | None = None
            if any(value is not None for value in values):
                if any(value is None for value in values):
                    raise ValueError(f"{source}:{line_number}: incomplete bounding box")
                bbox = BBox(*values)  # type: ignore[arg-type]
            try:
                context_percent = parse_optional_float(row.get("context_percent"))
                record = GroundTruthRecord(
                    source_id=source_id,
                    source_dataset=(row.get("source_dataset") or "").strip(),
                    image_id=(row.get("image_id") or "").strip(),
                    relative_path=(row.get("relative_path") or "").strip(),
                    source_width=width,
                    source_height=height,
                    review_state=state,
                    bbox=bbox,
                    is_bird=True if state == "boxed" else False if state == "no_bird" else None,
                    context_percent=0.0 if context_percent is None else context_percent,
                )
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            previous = result.get(source_id)
            if previous is None:
                result[source_id] = record
                continue
            identity = (
                "source_dataset", "image_id", "relative_path",
                "source_width", "source_height", "review_state", "context_percent",
            )
            mismatches = [name for name in identity if getattr(previous, name) != getattr(record, name)]
            if mismatches:
                raise ValueError(
                    f"{source}:{line_number}: inconsistent rows for source_id {source_id!r}: "
                    + ", ".join(mismatches)
                )
            if record.review_state != "boxed":
                raise ValueError(f"{source}:{line_number}: duplicate source_id {source_id!r}")
            result[source_id] = GroundTruthRecord(
                previous.source_id,
                previous.source_dataset,
                previous.image_id,
                previous.relative_path,
                previous.source_width,
                previous.source_height,
                previous.review_state,
                bboxes=[*previous.bboxes, *record.bboxes],
                is_bird=previous.is_bird,
                species=previous.species,
                mixed_species=previous.mixed_species,
                box_species=[*previous.box_species, *record.box_species],
                bird_count=len(previous.bboxes) + len(record.bboxes),
                context_percent=previous.context_percent,
            )
    return result


def _csv_number(value: float) -> str:
    # repr() is the shortest spelling guaranteed to reconstruct the exact same
    # binary float. Fewer digits can make an edge-aligned box exceed its image
    # after a save/load round trip.
    return repr(value)


def save_ground_truth(
    path: str | os.PathLike[str],
    records: Iterable[GroundTruthRecord] | Mapping[str, GroundTruthRecord],
) -> None:
    """Atomically save decisions, writing one row for each bird box."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = records.values() if isinstance(records, Mapping) else records
    ordered = sorted(values, key=lambda record: _natural_key(record.source_id))
    seen: set[str] = set()
    validated: list[GroundTruthRecord] = []
    for record in ordered:
        if record.source_id in seen:
            raise ValueError(f"duplicate source_id {record.source_id!r}")
        seen.add(record.source_id)
        # Reconstruct so a caller cannot persist an invalid, subsequently
        # mutated bboxes list from this otherwise frozen record.
        validated.append(GroundTruthRecord(
            record.source_id,
            record.source_dataset,
            record.image_id,
            record.relative_path,
            record.source_width,
            record.source_height,
            record.review_state,
            bboxes=record.bboxes,
            is_bird=record.is_bird,
            species=record.species,
            mixed_species=record.mixed_species,
            box_species=record.box_species,
            bird_count=record.bird_count,
            context_percent=record.context_percent,
        ))
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=GROUND_TRUTH_FIELDS, lineterminator="\n")
            writer.writeheader()
            for record in validated:
                for bbox in record.bboxes or [None]:
                    writer.writerow({
                        "source_id": record.source_id,
                        "source_dataset": record.source_dataset,
                        "image_id": record.image_id,
                        "relative_path": record.relative_path,
                        "source_width": record.source_width,
                        "source_height": record.source_height,
                        "review_state": record.review_state,
                        "bbox_x": _csv_number(bbox.x) if bbox else "",
                        "bbox_y": _csv_number(bbox.y) if bbox else "",
                        "bbox_width": _csv_number(bbox.width) if bbox else "",
                        "bbox_height": _csv_number(bbox.height) if bbox else "",
                        "context_percent": _csv_number(record.context_percent),
                    })
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def save_detector_manifest(
    path: str | os.PathLike[str],
    dataset: BBoxDataset,
    reviews: Mapping[str, GroundTruthRecord],
) -> None:
    """Atomically create or update a detector-compatible one-row-per-box CSV."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_fields: list[str] = []
    existing_by_sample: dict[str, list[dict[str, str]]] = {}
    if target.is_file():
        with target.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            for row in reader:
                sample_id = (row.get("sample_id") or "").strip()
                if sample_id:
                    existing_by_sample.setdefault(sample_id, []).append(dict(row))
    fields = list(existing_fields)
    for name in DETECTOR_MANIFEST_FIELDS:
        if name not in fields:
            fields.append(name)

    output_rows: list[dict[str, object]] = []
    for record in dataset:
        review = reviews.get(record.source_id)
        if review is None:
            state = ""
            is_bird = None
            species = ""
            mixed_species = False
            bird_count = ""
            boxes: list[BBox] = []
            box_species: list[str] = []
            context_percent = 0.0
        else:
            state = review.review_state
            is_bird = review.is_bird
            species = review.species
            mixed_species = review.mixed_species
            bird_count = review.bird_count
            boxes = list(review.bboxes)
            box_species = list(review.box_species)
            context_percent = review.context_percent

        existing_rows = existing_by_sample.get(record.source_id) or [{}]
        image_path = record.stored_path
        if not image_path:
            image_path = str(dataset.resolve_image_path(record))
        row_boxes: list[BBox | None] = boxes or [None]
        for box_index, box in enumerate(row_boxes):
            context_box = (
                bbox_with_context(
                    box,
                    review.source_width if review else record.width or 1,
                    review.source_height if review else record.height or 1,
                    context_percent,
                )
                if box else None
            )
            base = dict(existing_rows[min(box_index, len(existing_rows) - 1)])
            row: dict[str, object] = {name: base.get(name, "") for name in fields}
            row.update({
                "annotation_id": f"{record.source_id}:{box_index}",
                "sample_id": record.source_id,
                "source_dataset": record.source_dataset,
                "source_image_id": record.image_id,
                "image_path": image_path,
                "relative_path": record.relative_path,
                "width_px": review.source_width if review else record.width or "",
                "height_px": review.source_height if review else record.height or "",
                "review_state": state,
                "annotation_trust": "human_verified" if state in REVIEW_STATES else base.get("annotation_trust", ""),
                "annotation_origin": "combined_label_bbox_editor" if state in REVIEW_STATES else base.get("annotation_origin", ""),
                "detector_class_id": 0 if box else "",
                "detector_class_name": "bird" if box else "",
                "species_name": (
                    box_species[box_index]
                    if box and mixed_species and box_index < len(box_species)
                    else species if box else ""
                ),
                "box_id": box_index if box else "",
                "xmin": _csv_number(box.x) if box else "",
                "ymin": _csv_number(box.y) if box else "",
                "xmax": _csv_number(box.right) if box else "",
                "ymax": _csv_number(box.bottom) if box else "",
                "projected_area_px2": _csv_number(box.area) if box else "",
                "is_Bird": "" if is_bird is None else str(is_bird),
                "mixed_species": str(bool(mixed_species)),
                "bird_count": bird_count,
                "context_percent": _csv_number(context_percent),
                "context_xmin": _csv_number(context_box.x) if context_box else "",
                "context_ymin": _csv_number(context_box.y) if context_box else "",
                "context_xmax": _csv_number(context_box.right) if context_box else "",
                "context_ymax": _csv_number(context_box.bottom) if context_box else "",
            })
            output_rows.append(row)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", prefix=f".{target.name}.", suffix=".tmp",
            dir=target.parent, delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(output_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


__all__ = [
    "BBox",
    "BBoxDataset",
    "DETECTOR_MANIFEST_FIELDS",
    "GROUND_TRUTH_FIELDS",
    "GroundTruthRecord",
    "ImageCanvasTransform",
    "ImageRecord",
    "REVIEW_STATES",
    "ReviewRecord",
    "SourceBox",
    "bbox_with_context",
    "filter_records",
    "load_bbox_dataset",
    "load_cub_directory",
    "load_dataset",
    "load_detector_manifest",
    "load_ground_truth",
    "load_image_directory",
    "load_manifest_csv",
    "parse_optional_bool",
    "parse_optional_float",
    "parse_optional_int",
    "save_ground_truth",
    "save_detector_manifest",
    "sort_records",
]