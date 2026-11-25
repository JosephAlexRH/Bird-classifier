#!/usr/bin/env python3
"""Render polished JPEG presentation cards from bird captures and JSON sidecars.

The renderer looks for sidecars that share the image stem:

    capture.jpg
    capture.json
    capture.inference.json
    capture.motion.json

When inference metadata is available, the highest-confidence detector box is
drawn. Otherwise, a motion selection or configured normalized motion ROI is
drawn and clearly labelled as such.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (7, 18, 26)
PANEL = (13, 30, 40)
PANEL_BORDER = (31, 57, 68)
TEXT = (239, 246, 245)
MUTED = (153, 174, 179)
GOLD = (255, 190, 55)
CYAN = (64, 214, 205)
RED = (244, 104, 94)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float
    label: str
    source: str
    confidence: float | None = None
    detail: str | None = None
    dashed: bool = False

    def clamped(self, width: int, height: int) -> "Box":
        left = max(0.0, min(float(width), self.left))
        top = max(0.0, min(float(height), self.top))
        right = max(left + 1.0, min(float(width), self.right))
        bottom = max(top + 1.0, min(float(height), self.bottom))
        return Box(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            label=self.label,
            source=self.source,
            confidence=self.confidence,
            detail=self.detail,
            dashed=self.dashed,
        )


@dataclass(frozen=True)
class CaptureFiles:
    image: Path
    capture: Path
    inference: Path
    motion: Path


def nested(data: Any, *keys: str, default: Any = None) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read {path.name}: {exc}", file=sys.stderr)
        return {}
    return value if isinstance(value, dict) else {}


def paired_files(image_path: Path) -> CaptureFiles:
    stem = image_path.stem
    parent = image_path.parent
    return CaptureFiles(
        image=image_path,
        capture=parent / f"{stem}.json",
        inference=parent / f"{stem}.inference.json",
        motion=parent / f"{stem}.motion.json",
    )


def discover_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported input image: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"input path does not exist: {input_path}")
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def box_from_dict(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, dict):
        return None
    left = number(value.get("left"))
    top = number(value.get("top"))
    right = number(value.get("right"))
    bottom = number(value.get("bottom"))
    if None in (left, top, right, bottom):
        return None
    assert left is not None and top is not None and right is not None and bottom is not None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def inference_boxes(inference: dict[str, Any]) -> list[Box]:
    boxes: list[Box] = []
    species_boxes = nested(inference, "species", "boxes", default=[])
    if isinstance(species_boxes, list):
        for item in species_boxes:
            if not isinstance(item, dict):
                continue
            coords = box_from_dict(item.get("box"))
            if coords is None:
                continue
            top_k = item.get("top_k")
            best = top_k[0] if isinstance(top_k, list) and top_k and isinstance(top_k[0], dict) else {}
            label = str(best.get("label") or nested(inference, "classification", "label", default="Bird"))
            species_confidence = number(best.get("probability"))
            detector_confidence = number(item.get("localizer_score"))
            detail = (
                f"detector {detector_confidence:.1%}"
                if detector_confidence is not None
                else None
            )
            boxes.append(
                Box(
                    *coords,
                    label=label,
                    source="model detection",
                    confidence=species_confidence,
                    detail=detail,
                )
            )

    if boxes:
        return sorted(boxes, key=lambda box: box.confidence or 0.0, reverse=True)

    detections = nested(inference, "localizer", "detections", default=[])
    if isinstance(detections, list):
        for item in detections:
            if not isinstance(item, dict):
                continue
            coords = box_from_dict(item.get("box"))
            if coords is None:
                continue
            boxes.append(
                Box(
                    *coords,
                    label=str(nested(inference, "classification", "label", default="Bird")),
                    source="model detection",
                    confidence=number(item.get("score")),
                )
            )
    return sorted(boxes, key=lambda box: box.confidence or 0.0, reverse=True)


def motion_box(motion: dict[str, Any]) -> Box | None:
    coords = box_from_dict(nested(motion, "selection", "final_bounds_px"))
    if coords is None:
        return None
    source = str(nested(motion, "selection", "source", default="motion selection"))
    readable_source = source.replace("_", " ")
    return Box(
        *coords,
        label="Motion crop",
        source=readable_source,
        detail="fallback region" if nested(motion, "detection", "status") == "fallback" else None,
        dashed=True,
    )


def roi_box(capture: dict[str, Any], width: int, height: int) -> Box | None:
    roi = nested(capture, "camera", "requested", "motion", "roi")
    if not isinstance(roi, dict):
        return None
    x = number(roi.get("x"))
    y = number(roi.get("y"))
    roi_width = number(roi.get("width"))
    roi_height = number(roi.get("height"))
    if None in (x, y, roi_width, roi_height):
        return None
    assert x is not None and y is not None and roi_width is not None and roi_height is not None
    if roi_width <= 0 or roi_height <= 0:
        return None
    return Box(
        x * width,
        y * height,
        (x + roi_width) * width,
        (y + roi_height) * height,
        label="Motion ROI",
        source="configured motion ROI",
        detail="not a detector box",
        dashed=True,
    )


def select_boxes(
    mode: str,
    capture: dict[str, Any],
    inference: dict[str, Any],
    motion: dict[str, Any],
    width: int,
    height: int,
    show_all: bool,
) -> list[Box]:
    candidates: list[Box] = []
    if mode in {"auto", "detection"}:
        candidates = inference_boxes(inference)
        if candidates:
            return [box.clamped(width, height) for box in (candidates if show_all else candidates[:1])]
        if mode == "detection":
            return []
    if mode in {"auto", "roi"}:
        selected_motion_box = motion_box(motion)
        if selected_motion_box is not None:
            return [selected_motion_box.clamped(width, height)]
        selected_roi_box = roi_box(capture, width, height)
        if selected_roi_box is not None:
            return [selected_roi_box.clamped(width, height)]
    return []


def font_candidates(weight: str) -> Iterable[Path]:
    windows_fonts = Path("C:/Windows/Fonts")
    names = (
        ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf")
        if weight == "bold"
        else ("segoeui.ttf", "arial.ttf", "calibri.ttf")
    )
    for name in names:
        yield windows_fonts / name
    yield Path("DejaVuSans-Bold.ttf" if weight == "bold" else "DejaVuSans.ttf")


def load_font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in font_candidates(weight):
        try:
            return ImageFont.truetype(str(candidate), size=max(8, size))
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def clean_label(value: Any, fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def format_timestamp(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Time unavailable"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S UTC")
    except ValueError:
        return raw


def parse_hex_color(value: str) -> tuple[int, int, int]:
    cleaned = value.strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(character * 2 for character in cleaned)
    if len(cleaned) != 6:
        raise ValueError(f"invalid color: {value}")
    try:
        return tuple(int(cleaned[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"invalid color: {value}") from exc


@dataclass(frozen=True)
class RenderStyle:
    box_color: str = "#ffbe37"
    text_color: str = "#ffffff"
    panel_color: str = "#071219"
    line_width: int = 3
    line_style: str = "auto"
    font_size: int = 16
    panel_opacity: int = 190
    metadata_position: str = "top-left"
    box_mode: str = "auto"
    show_metadata: bool = True
    show_box_label: bool = True
    show_all: bool = False
    output_width: int = 0
    quality: int = 95


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int, int],
    width: int,
    dash: int,
    gap: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance <= 0:
        return
    ux, uy = (x2 - x1) / distance, (y2 - y1) / distance
    position = 0.0
    while position < distance:
        finish = min(distance, position + dash)
        draw.line(
            (x1 + ux * position, y1 + uy * position, x1 + ux * finish, y1 + uy * finish),
            fill=fill,
            width=width,
        )
        position += dash + gap


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[float, float, float, float],
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    left, top, right, bottom = bounds
    dash, gap = max(7, width * 3), max(5, width * 2)
    draw_dashed_line(draw, (left, top), (right, top), fill, width, dash, gap)
    draw_dashed_line(draw, (right, top), (right, bottom), fill, width, dash, gap)
    draw_dashed_line(draw, (right, bottom), (left, bottom), fill, width, dash, gap)
    draw_dashed_line(draw, (left, bottom), (left, top), fill, width, dash, gap)


def draw_corner_rectangle(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[float, float, float, float],
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    left, top, right, bottom = bounds
    length = max(12, int(min(right - left, bottom - top) * 0.25))
    for points in (
        (left, top + length, left, top, left + length, top),
        (right - length, top, right, top, right, top + length),
        (right, bottom - length, right, bottom, right - length, bottom),
        (left + length, bottom, left, bottom, left, bottom - length),
    ):
        draw.line(points, fill=fill, width=width, joint="curve")


def metadata_lines(
    capture: dict[str, Any],
    inference: dict[str, Any],
    boxes: list[Box],
    image_size: tuple[int, int],
) -> tuple[str, list[str]]:
    title = clean_label(
        nested(inference, "classification", "label"),
        clean_label(inference.get("guess"), clean_label(capture.get("label"), "Bird capture")),
    )
    bird_probability = number(inference.get("bird_probability"))
    species_probability = number(inference.get("species_probability"))
    confidence_parts: list[str] = []
    if bird_probability is not None:
        confidence_parts.append(f"Bird {bird_probability:.1%}")
    if species_probability is not None:
        confidence_parts.append(f"Species {species_probability:.1%}")

    width, height = image_size
    camera = clean_label(
        nested(capture, "camera", "sensor", "model"),
        clean_label(capture.get("camera_model"), "Unknown camera"),
    )
    jpeg_quality = nested(capture, "camera", "observed", "capture", "jpeg_quality")
    camera_line = f"{camera}  |  {width} x {height}"
    if isinstance(jpeg_quality, (int, float)):
        camera_line += f"  |  JPEG Q{jpeg_quality:g}"

    exposure = nested(capture, "camera", "observed", "ov5640_registers", "aec", "aec_lines")
    gain = number(nested(capture, "camera", "observed", "ov5640_registers", "gain", "nominal_multiplier"))
    exposure_parts: list[str] = []
    if isinstance(exposure, (int, float)):
        exposure_parts.append(f"Exposure {exposure:g} lines")
    if gain is not None:
        exposure_parts.append(f"Gain {gain:.2f}x")

    lines = [format_timestamp(capture.get("received_at") or inference.get("completed_at"))]
    if confidence_parts:
        lines.append("  |  ".join(confidence_parts))
    lines.append(camera_line)
    if exposure_parts:
        lines.append("  |  ".join(exposure_parts))
    lines.append(f"Box: {boxes[0].source.title()}" if boxes else "Box: none")
    deployment = clean_label(capture.get("deployment_id"), "")
    if deployment:
        lines.append(deployment)
    return title, lines


def draw_metadata_panel(canvas: Image.Image, title: str, lines: list[str], style: RenderStyle) -> None:
    width, height = canvas.size
    scale = max(1.0, width / 640.0)
    font_size = max(9, round(style.font_size * scale))
    title_font = load_font(round(font_size * 1.2), "bold")
    body_font = load_font(font_size)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    padding = max(8, round(9 * scale))
    line_gap = max(2, round(3 * scale))
    title_width, title_height = text_size(draw, title, title_font)
    body_sizes = [text_size(draw, line, body_font) for line in lines]
    content_width = max([title_width, *(item[0] for item in body_sizes)])
    content_height = title_height + line_gap * 2 + sum(item[1] + line_gap for item in body_sizes)
    panel_width = min(width - padding * 2, content_width + padding * 2)
    panel_height = content_height + padding * 2
    edge = max(8, round(10 * scale))
    x = width - edge - panel_width if style.metadata_position.endswith("right") else edge
    y = height - edge - panel_height if style.metadata_position.startswith("bottom") else edge
    draw.rounded_rectangle(
        (x, y, x + panel_width, y + panel_height),
        radius=max(5, round(7 * scale)),
        fill=parse_hex_color(style.panel_color) + (style.panel_opacity,),
    )
    text_x = x + padding
    text_y = y + padding
    text_color = parse_hex_color(style.text_color)
    draw.text((text_x, text_y), title, font=title_font, fill=text_color + (255,))
    text_y += title_height + line_gap * 2
    for line, (_, line_height) in zip(lines, body_sizes):
        draw.text((text_x, text_y), line, font=body_font, fill=text_color + (245,))
        text_y += line_height + line_gap
    canvas.alpha_composite(overlay)


def render_image(files: CaptureFiles, style: RenderStyle) -> tuple[Image.Image, list[Box]]:
    capture = load_json(files.capture)
    inference = load_json(files.inference)
    motion = load_json(files.motion)
    with Image.open(files.image) as opened:
        source = opened.convert("RGB")
    original_width, original_height = source.size
    boxes = select_boxes(
        style.box_mode,
        capture,
        inference,
        motion,
        original_width,
        original_height,
        style.show_all,
    )

    output_width = style.output_width if style.output_width > 0 else original_width
    output_height = round(original_height * output_width / original_width)
    if output_width != original_width:
        source = source.resize((output_width, output_height), Image.Resampling.LANCZOS)
    canvas = source.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    scale_x, scale_y = output_width / original_width, output_height / original_height
    color = parse_hex_color(style.box_color) + (255,)
    line_width = max(1, round(style.line_width * max(scale_x, scale_y)))
    label_font = load_font(max(9, round(style.font_size * max(scale_x, scale_y))), "bold")

    for box in boxes:
        bounds = (box.left * scale_x, box.top * scale_y, box.right * scale_x, box.bottom * scale_y)
        line_style = style.line_style
        if line_style == "auto":
            line_style = "dashed" if box.dashed else "solid"
        if line_style == "dashed":
            draw_dashed_rectangle(draw, bounds, color, line_width)
        elif line_style == "corners":
            draw_corner_rectangle(draw, bounds, color, line_width)
        else:
            draw.rectangle(bounds, outline=(0, 0, 0, 190), width=line_width + 2)
            draw.rectangle(bounds, outline=color, width=line_width)

        if style.show_box_label:
            label = box.label
            if box.confidence is not None:
                label += f"  {box.confidence:.1%}"
            label_width, label_height = text_size(draw, label, label_font)
            pad_x, pad_y = max(4, line_width * 2), max(3, line_width)
            left, top, _, _ = bounds
            label_x = max(0, min(left, output_width - label_width - pad_x * 2))
            label_y = top - label_height - pad_y * 2 - 2
            if label_y < 0:
                label_y = top + 2
            draw.rectangle(
                (label_x, label_y, label_x + label_width + pad_x * 2, label_y + label_height + pad_y * 2),
                fill=parse_hex_color(style.panel_color) + (style.panel_opacity,),
            )
            draw.text(
                (label_x + pad_x, label_y + pad_y - 1),
                label,
                font=label_font,
                fill=parse_hex_color(style.text_color) + (255,),
            )

    if style.show_metadata:
        title, lines = metadata_lines(capture, inference, boxes, (original_width, original_height))
        draw_metadata_panel(canvas, title, lines, style)
    return canvas.convert("RGB"), boxes


def export_image(files: CaptureFiles, output_path: Path, style: RenderStyle) -> tuple[list[Box], tuple[int, int]]:
    rendered, boxes = render_image(files, style)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(
        output_path,
        format="JPEG",
        quality=style.quality,
        subsampling=0,
        optimize=True,
        progressive=True,
    )
    return boxes, rendered.size


def export_batch(input_path: Path, output_dir: Path, style: RenderStyle, overwrite: bool) -> tuple[int, int]:
    images = discover_images(input_path)
    exported = 0
    skipped = 0
    for image_path in images:
        output_path = output_dir / f"{image_path.stem}_annotated.jpg"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        boxes, size = export_image(paired_files(image_path), output_path, style)
        source = boxes[0].source if boxes else "no box"
        print(f"exported: {output_path.name} [{size[0]}x{size[1]}, {source}]")
        exported += 1
    return exported, skipped


class BirdCaptureUI:
    def __init__(self, root: Any) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("Bird Capture Annotator")
        self.root.minsize(980, 650)
        self.root.geometry("1180x760")
        self.base_dir = Path(__file__).resolve().parent
        self.input_var = tk.StringVar(value=str(self.base_dir))
        self.output_var = tk.StringVar(value=str(self.base_dir / "exported_ui"))
        self.image_var = tk.StringVar()
        self.box_mode_var = tk.StringVar(value="auto")
        self.line_style_var = tk.StringVar(value="auto")
        self.position_var = tk.StringVar(value="top-left")
        self.box_color_var = tk.StringVar(value="#ffbe37")
        self.text_color_var = tk.StringVar(value="#ffffff")
        self.panel_color_var = tk.StringVar(value="#071219")
        self.line_width_var = tk.IntVar(value=3)
        self.font_size_var = tk.IntVar(value=16)
        self.opacity_var = tk.IntVar(value=190)
        self.output_width_var = tk.IntVar(value=0)
        self.quality_var = tk.IntVar(value=95)
        self.show_metadata_var = tk.BooleanVar(value=True)
        self.show_label_var = tk.BooleanVar(value=True)
        self.show_all_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Choose an image to preview, or export the whole folder.")
        self.preview_photo: Any = None
        self.image_paths: list[Path] = []
        self._build()
        self.refresh_images()

    def _build(self) -> None:
        from tkinter import ttk

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="nsw")
        preview_frame = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        preview_frame.grid(row=0, column=1, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_frame, text="Preview", anchor="center", relief="sunken")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        ttk.Label(preview_frame, textvariable=self.status_var, anchor="w").grid(row=1, column=0, sticky="ew", pady=(8, 0))

        row = 0
        ttk.Label(controls, text="Bird Capture Annotator", font=("Segoe UI", 16, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 12))
        row += 1
        row = self._path_row(controls, row, "Input", self.input_var, self.choose_input)
        row = self._path_row(controls, row, "Output", self.output_var, self.choose_output)
        ttk.Label(controls, text="Image").grid(row=row, column=0, sticky="w", pady=4)
        self.image_combo = ttk.Combobox(controls, textvariable=self.image_var, state="readonly", width=31)
        self.image_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.image_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_preview())
        row += 1
        ttk.Separator(controls).grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1
        row = self._combo_row(controls, row, "Box source", self.box_mode_var, ("auto", "detection", "roi", "none"))
        row = self._combo_row(controls, row, "Line style", self.line_style_var, ("auto", "solid", "dashed", "corners"))
        row = self._combo_row(controls, row, "Metadata", self.position_var, ("top-left", "top-right", "bottom-left", "bottom-right"))
        row = self._spin_row(controls, row, "Line width", self.line_width_var, 1, 20)
        row = self._spin_row(controls, row, "Font size", self.font_size_var, 9, 48)
        row = self._spin_row(controls, row, "Panel opacity", self.opacity_var, 0, 255)
        row = self._spin_row(controls, row, "Output width", self.output_width_var, 0, 4096, "0 = original")
        row = self._spin_row(controls, row, "JPEG quality", self.quality_var, 1, 100)
        row = self._color_row(controls, row, "Box color", self.box_color_var)
        row = self._color_row(controls, row, "Text color", self.text_color_var)
        row = self._color_row(controls, row, "Panel color", self.panel_color_var)
        for label, variable in (
            ("Show sidecar data", self.show_metadata_var),
            ("Show box label", self.show_label_var),
            ("Show all detections", self.show_all_var),
            ("Overwrite exports", self.overwrite_var),
        ):
            ttk.Checkbutton(controls, text=label, variable=variable, command=self.update_preview).grid(row=row, column=0, columnspan=3, sticky="w")
            row += 1
        ttk.Button(controls, text="Update preview", command=self.update_preview).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        row += 1
        ttk.Button(controls, text="Export selected image", command=self.export_selected).grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1
        ttk.Button(controls, text="Export all images", command=self.export_all).grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        for variable in (self.box_mode_var, self.line_style_var, self.position_var):
            variable.trace_add("write", lambda *_args: self.update_preview())

    def _path_row(self, parent: Any, row: int, label: str, variable: Any, command: Any) -> int:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        self.ttk.Entry(parent, textvariable=variable, width=27).grid(row=row, column=1, sticky="ew", pady=4)
        self.ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, padx=(5, 0), pady=4)
        return row + 1

    def _combo_row(self, parent: Any, row: int, label: str, variable: Any, values: tuple[str, ...]) -> int:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        self.ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=18).grid(row=row, column=1, columnspan=2, sticky="ew", pady=3)
        return row + 1

    def _spin_row(self, parent: Any, row: int, label: str, variable: Any, minimum: int, maximum: int, note: str = "") -> int:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        self.ttk.Spinbox(parent, from_=minimum, to=maximum, textvariable=variable, width=10, command=self.update_preview).grid(row=row, column=1, sticky="w", pady=3)
        self.ttk.Label(parent, text=note).grid(row=row, column=2, sticky="w")
        return row + 1

    def _color_row(self, parent: Any, row: int, label: str, variable: Any) -> int:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        self.ttk.Entry(parent, textvariable=variable, width=12).grid(row=row, column=1, sticky="ew", pady=3)
        self.ttk.Button(parent, text="Choose", command=lambda: self.choose_color(variable)).grid(row=row, column=2, padx=(5, 0), pady=3)
        return row + 1

    def choose_input(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(initialdir=self.input_var.get(), title="Choose capture folder")
        if selected:
            self.input_var.set(selected)
            self.refresh_images()

    def choose_output(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(initialdir=self.output_var.get(), title="Choose output folder")
        if selected:
            self.output_var.set(selected)

    def choose_color(self, variable: Any) -> None:
        from tkinter import colorchooser

        _rgb, selected = colorchooser.askcolor(color=variable.get(), parent=self.root)
        if selected:
            variable.set(selected)
            self.update_preview()

    def refresh_images(self) -> None:
        try:
            self.image_paths = discover_images(Path(self.input_var.get()).expanduser().resolve())
        except ValueError as exc:
            self.image_paths = []
            self.status_var.set(str(exc))
        names = [path.name for path in self.image_paths]
        self.image_combo["values"] = names
        if names:
            self.image_var.set(names[0])
            self.update_preview()
        else:
            self.image_var.set("")
            self.preview_label.configure(image="", text="No images found")

    def current_style(self) -> RenderStyle:
        return RenderStyle(
            box_color=self.box_color_var.get(),
            text_color=self.text_color_var.get(),
            panel_color=self.panel_color_var.get(),
            line_width=max(1, self.line_width_var.get()),
            line_style=self.line_style_var.get(),
            font_size=max(8, self.font_size_var.get()),
            panel_opacity=max(0, min(255, self.opacity_var.get())),
            metadata_position=self.position_var.get(),
            box_mode=self.box_mode_var.get(),
            show_metadata=self.show_metadata_var.get(),
            show_box_label=self.show_label_var.get(),
            show_all=self.show_all_var.get(),
            output_width=max(0, self.output_width_var.get()),
            quality=max(1, min(100, self.quality_var.get())),
        )

    def selected_path(self) -> Path | None:
        selected = self.image_var.get()
        return next((path for path in self.image_paths if path.name == selected), None)

    def update_preview(self) -> None:
        from PIL import ImageTk

        selected = self.selected_path()
        if selected is None:
            return
        try:
            rendered, boxes = render_image(paired_files(selected), self.current_style())
            rendered.thumbnail((800, 650), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(rendered)
            self.preview_label.configure(image=self.preview_photo, text="")
            self.status_var.set(f"Preview: {selected.name} | {len(boxes)} box(es)")
        except (OSError, ValueError, self.tk.TclError) as exc:
            self.status_var.set(f"Preview error: {exc}")

    def export_selected(self) -> None:
        from tkinter import messagebox

        selected = self.selected_path()
        if selected is None:
            messagebox.showwarning("No image", "Choose an image first.", parent=self.root)
            return
        output = Path(self.output_var.get()).expanduser().resolve() / f"{selected.stem}_annotated.jpg"
        if output.exists() and not self.overwrite_var.get():
            messagebox.showwarning("File exists", f"Enable overwrite or choose another output folder.\n\n{output}", parent=self.root)
            return
        try:
            export_image(paired_files(selected), output, self.current_style())
        except (OSError, ValueError) as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.root)
            return
        self.status_var.set(f"Exported {output}")
        messagebox.showinfo("Export complete", f"Saved:\n{output}", parent=self.root)

    def export_all(self) -> None:
        from tkinter import messagebox

        if not self.image_paths:
            messagebox.showwarning("No images", "No images were found in the input folder.", parent=self.root)
            return
        output_dir = Path(self.output_var.get()).expanduser().resolve()
        try:
            exported, skipped = export_batch(
                Path(self.input_var.get()).expanduser().resolve(),
                output_dir,
                self.current_style(),
                self.overwrite_var.get(),
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.root)
            return
        self.status_var.set(f"Exported {exported}; skipped {skipped}. Output: {output_dir}")
        messagebox.showinfo("Export complete", f"Exported: {exported}\nSkipped: {skipped}\n\n{output_dir}", parent=self.root)


def launch_ui() -> int:
    try:
        import tkinter as tk
    except ImportError as exc:
        print(f"error: Tkinter is not available: {exc}", file=sys.stderr)
        return 1
    root = tk.Tk()
    BirdCaptureUI(root)
    root.mainloop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate bird captures with bounding boxes and sidecar data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="?", type=Path, help="Image or directory to export in batch mode.")
    parser.add_argument("-o", "--output", type=Path, help="Batch output directory.")
    parser.add_argument("--gui", action="store_true", help="Open the graphical interface.")
    parser.add_argument("--box-mode", choices=("auto", "detection", "roi", "none"), default="auto")
    parser.add_argument("--line-style", choices=("auto", "solid", "dashed", "corners"), default="auto")
    parser.add_argument("--box-color", default="#ffbe37")
    parser.add_argument("--text-color", default="#ffffff")
    parser.add_argument("--panel-color", default="#071219")
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--panel-opacity", type=int, default=190)
    parser.add_argument("--metadata-position", choices=("top-left", "top-right", "bottom-left", "bottom-right"), default="top-left")
    parser.add_argument("--width", type=int, default=0, help="Output width; 0 preserves the original dimensions.")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--no-box-label", action="store_true")
    parser.add_argument("--all-detections", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gui or args.input is None:
        return launch_ui()
    if args.width < 0 or args.line_width < 1 or not 1 <= args.quality <= 100 or not 0 <= args.panel_opacity <= 255:
        print("error: invalid width, line width, quality, or opacity", file=sys.stderr)
        return 2
    try:
        parse_hex_color(args.box_color)
        parse_hex_color(args.text_color)
        parse_hex_color(args.panel_color)
        input_path = args.input.expanduser().resolve()
        output_dir = (
            args.output.expanduser().resolve()
            if args.output
            else (input_path / "exported_ui" if input_path.is_dir() else input_path.parent / "exported_ui")
        )
        style = RenderStyle(
            box_color=args.box_color,
            text_color=args.text_color,
            panel_color=args.panel_color,
            line_width=args.line_width,
            line_style=args.line_style,
            font_size=args.font_size,
            panel_opacity=args.panel_opacity,
            metadata_position=args.metadata_position,
            box_mode=args.box_mode,
            show_metadata=not args.no_metadata,
            show_box_label=not args.no_box_label,
            show_all=args.all_detections,
            output_width=args.width,
            quality=args.quality,
        )
        exported, skipped = export_batch(input_path, output_dir, style, args.overwrite)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"done: {exported} exported, {skipped} skipped -> {output_dir}")
    return 0 if exported or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())