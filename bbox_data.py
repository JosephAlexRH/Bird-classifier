"""Small, dependency-free data layer for bird bounding-box tools.

The loaders in this module deliberately retain only the columns needed by the
viewer/annotator.  In particular, loading the wide generated CUB manifest does
not retain its hundreds of attribute columns.  Image files are never opened by
the loaders; paths are resolved only when :meth:`ImageRecord.resolve_image_path`
or :meth:`BBoxDataset.resolve_image_path` is called.
"""

from __future__ import annotations

import csv
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Iterator, Mapping, Sequence


REVIEW_STATES = frozenset({"boxed", "no_bird", "unusable", "skipped"})
GROUND_TRUTH_FIELDS = (
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

_TRUE = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE = frozenset({"0", "false", "f", "no", "n", "off"})


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

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.relative_path and not self.stored_path:
            raise ValueError("an image record needs a relative or stored path")
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive when present")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive when present")

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


def load_dataset(
    path: str | os.PathLike[str],
    image_root: str | os.PathLike[str] | None = None,
) -> BBoxDataset:
    """Autodetect and load a supported manifest CSV or raw CUB directory."""

    source = Path(path)
    if source.is_dir():
        return load_cub_directory(source, image_root=image_root)
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
    ) -> GroundTruthRecord:
        width = source_width if source_width is not None else record.width
        height = source_height if source_height is not None else record.height
        if width is None or height is None:
            raise ValueError("source dimensions are required when absent from the manifest")
        return cls(
            record.source_id, record.source_dataset, record.image_id, record.relative_path,
            width, height, review_state, bbox, bboxes=bboxes,
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
        if reader.fieldnames is None or not set(GROUND_TRUTH_FIELDS) <= set(reader.fieldnames):
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
                record = GroundTruthRecord(
                    source_id=source_id,
                    source_dataset=(row.get("source_dataset") or "").strip(),
                    image_id=(row.get("image_id") or "").strip(),
                    relative_path=(row.get("relative_path") or "").strip(),
                    source_width=width,
                    source_height=height,
                    review_state=state,
                    bbox=bbox,
                )
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            previous = result.get(source_id)
            if previous is None:
                result[source_id] = record
                continue
            identity = (
                "source_dataset", "image_id", "relative_path",
                "source_width", "source_height", "review_state",
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


__all__ = [
    "BBox",
    "BBoxDataset",
    "GROUND_TRUTH_FIELDS",
    "GroundTruthRecord",
    "ImageCanvasTransform",
    "ImageRecord",
    "REVIEW_STATES",
    "ReviewRecord",
    "filter_records",
    "load_bbox_dataset",
    "load_cub_directory",
    "load_dataset",
    "load_ground_truth",
    "load_manifest_csv",
    "parse_optional_bool",
    "parse_optional_float",
    "parse_optional_int",
    "save_ground_truth",
    "sort_records",
]