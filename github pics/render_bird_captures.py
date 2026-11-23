#!/usr/bin/env python3
"""Draw inference-sidecar bird boxes and per-box predictions on captures.

For ``capture.jpg`` the renderer reads ``capture.inference.json``. Output is
the original image at its native size with no presentation card, capture
metadata, motion crop, ROI, title, footer, or other decoration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageColor, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TRANSPARENT_NAMES = {"none", "transparent"}


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float
    prediction: str
    confidence: float | None
    detector_confidence: float | None = None

    def clamped(self, width: int, height: int) -> "Box | None":
        left = max(0.0, min(float(width), self.left))
        top = max(0.0, min(float(height), self.top))
        right = max(0.0, min(float(width), self.right))
        bottom = max(0.0, min(float(height), self.bottom))
        if right <= left or bottom <= top:
            return None
        return Box(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            prediction=self.prediction,
            confidence=self.confidence,
            detector_confidence=self.detector_confidence,
        )


@dataclass(frozen=True)
class RenderStyle:
    box_color: tuple[int, int, int, int]
    box_width: int
    box_style: str
    dash_length: int
    dash_gap: int
    label_text_color: tuple[int, int, int, int]
    label_background_color: tuple[int, int, int, int]
    label_border_color: tuple[int, int, int, int]
    label_border_width: int
    label_style: str
    label_position: str
    label_padding_x: int
    label_padding_y: int
    label_gap: int
    label_radius: int
    text_stroke_width: int
    text_stroke_color: tuple[int, int, int, int]
    font: ImageFont.ImageFont
    label_template: str
    confidence_decimals: int


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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read {path.name}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(value, dict):
        print(f"warning: {path.name} does not contain a JSON object", file=sys.stderr)
        return {}
    return value


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


def prediction_from_item(item: dict[str, Any], species: dict[str, Any]) -> tuple[str, float | None]:
    top_k = item.get("top_k")
    if isinstance(top_k, list) and top_k and isinstance(top_k[0], dict):
        prediction = str(top_k[0].get("label") or "Bird").strip() or "Bird"
        return prediction, number(top_k[0].get("probability"))

    labels = species.get("labels")
    probabilities = item.get("probabilities")
    if (
        isinstance(labels, list)
        and isinstance(probabilities, list)
        and labels
        and len(labels) == len(probabilities)
    ):
        parsed = [number(value) for value in probabilities]
        valid = [(value, index) for index, value in enumerate(parsed) if value is not None]
        if valid:
            confidence, index = max(valid)
            prediction = str(labels[index] or "Bird").strip() or "Bird"
            return prediction, confidence
    return "Bird", None


def inference_boxes(inference: dict[str, Any], width: int, height: int) -> list[Box]:
    """Return display boxes in sidecar order, with prediction local to each box."""
    if inference.get("status") not in (None, "complete") or inference.get("is_bird") is False:
        return []

    species = inference.get("species")
    if isinstance(species, dict) and isinstance(species.get("boxes"), list):
        boxes: list[Box] = []
        for item in species["boxes"]:
            if not isinstance(item, dict):
                continue
            coords = box_from_dict(item.get("box"))
            if coords is None:
                continue
            prediction, confidence = prediction_from_item(item, species)
            candidate = Box(
                *coords,
                prediction=prediction,
                confidence=confidence,
                detector_confidence=number(item.get("localizer_score")),
            ).clamped(width, height)
            if candidate is not None:
                boxes.append(candidate)
        return boxes

    # Compatibility with older sidecars that only recorded detector boxes.
    detections = nested(inference, "localizer", "detections", default=[])
    if not isinstance(detections, list):
        return []
    prediction = str(nested(inference, "classification", "label", default="Bird") or "Bird").strip()
    boxes = []
    for item in detections:
        if not isinstance(item, dict) or item.get("static") is True:
            continue
        coords = box_from_dict(item.get("box"))
        if coords is None:
            continue
        score = number(item.get("score"))
        candidate = Box(
            *coords,
            prediction=prediction or "Bird",
            confidence=score,
            detector_confidence=score,
        ).clamped(width, height)
        if candidate is not None:
            boxes.append(candidate)
    return boxes


def parse_color(value: str) -> tuple[int, int, int, int]:
    if value.strip().casefold() in TRANSPARENT_NAMES:
        return 0, 0, 0, 0
    try:
        parsed = ImageColor.getrgb(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid color {value!r}") from exc
    if len(parsed) == 3:
        return parsed[0], parsed[1], parsed[2], 255
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def font_candidates(weight: str) -> Iterable[Path]:
    windows_fonts = Path("C:/Windows/Fonts")
    names = (
        ("seguisb.ttf", "arialbd.ttf", "calibrib.ttf")
        if weight == "bold"
        else ("segoeui.ttf", "arial.ttf", "calibri.ttf")
    )
    for name in names:
        yield windows_fonts / name
    for name in (("DejaVuSans-Bold.ttf",) if weight == "bold" else ("DejaVuSans.ttf",)):
        yield Path(name)


def load_font(size: int, weight: str, path: Path | None = None) -> ImageFont.ImageFont:
    candidates = [path] if path is not None else list(font_candidates(weight))
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    if path is not None:
        raise ValueError(f"could not load font: {path}")
    return ImageFont.load_default()


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
    ux = (x2 - x1) / distance
    uy = (y2 - y1) / distance
    position = 0.0
    while position < distance:
        segment_end = min(distance, position + dash)
        draw.line(
            (
                x1 + ux * position,
                y1 + uy * position,
                x1 + ux * segment_end,
                y1 + uy * segment_end,
            ),
            fill=fill,
            width=width,
        )
        position += dash + gap


def draw_box(draw: ImageDraw.ImageDraw, box: Box, style: RenderStyle) -> None:
    bounds = (box.left, box.top, box.right, box.bottom)
    if style.box_style == "solid":
        draw.rectangle(bounds, outline=style.box_color, width=style.box_width)
        return
    if style.box_style == "dashed":
        left, top, right, bottom = bounds
        for start, end in (
            ((left, top), (right, top)),
            ((right, top), (right, bottom)),
            ((right, bottom), (left, bottom)),
            ((left, bottom), (left, top)),
        ):
            draw_dashed_line(
                draw, start, end, style.box_color, style.box_width,
                style.dash_length, style.dash_gap,
            )
        return

    length = max(8, min(28, round(min(box.right - box.left, box.bottom - box.top) * 0.25)))
    left, top, right, bottom = bounds
    for points in (
        (left, top + length, left, top, left + length, top),
        (right - length, top, right, top, right, top + length),
        (right, bottom - length, right, bottom, right - length, bottom),
        (left + length, bottom, left, bottom, left, bottom - length),
    ):
        draw.line(points, fill=style.box_color, width=style.box_width, joint="curve")


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, stroke: int) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    return right - left, bottom - top


def label_text(box: Box, style: RenderStyle) -> str:
    decimals = style.confidence_decimals
    confidence = "?" if box.confidence is None else f"{box.confidence:.{decimals}%}"
    detector = "?" if box.detector_confidence is None else f"{box.detector_confidence:.{decimals}%}"
    try:
        return style.label_template.format(
            prediction=box.prediction,
            confidence=confidence,
            confidence_value="" if box.confidence is None else f"{box.confidence:.{decimals + 2}f}",
            detector_confidence=detector,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid --label-template: {exc}") from exc


def intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def label_bounds(
    box: Box,
    label_size: tuple[int, int],
    image_size: tuple[int, int],
    position: str,
    gap: int,
) -> tuple[int, int, int, int]:
    label_width, label_height = label_size
    image_width, image_height = image_size
    left = round(box.left)
    top = round(box.top)
    bottom = round(box.bottom)
    if position == "above":
        y = top - gap - label_height
    elif position == "below":
        y = bottom + gap
    elif position == "inside-bottom":
        y = bottom - gap - label_height
    else:
        y = top + gap
    x = max(0, min(left, image_width - label_width))
    y = max(0, min(y, image_height - label_height))
    return x, y, x + label_width, y + label_height


def choose_label_bounds(
    box: Box,
    size: tuple[int, int],
    image_size: tuple[int, int],
    style: RenderStyle,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    positions = (
        ("above", "below", "inside-top", "inside-bottom")
        if style.label_position == "auto"
        else (style.label_position,)
    )
    candidates = [label_bounds(box, size, image_size, position, style.label_gap) for position in positions]
    return min(candidates, key=lambda bounds: sum(intersects(bounds, other) for other in occupied))


def draw_label(
    draw: ImageDraw.ImageDraw,
    box: Box,
    image_size: tuple[int, int],
    style: RenderStyle,
    occupied: list[tuple[int, int, int, int]],
) -> None:
    text = label_text(box, style)
    width, height = text_size(draw, text, style.font, style.text_stroke_width)
    pad_x = 0 if style.label_style == "text" else style.label_padding_x
    pad_y = 0 if style.label_style == "text" else style.label_padding_y
    size = (width + pad_x * 2, height + pad_y * 2)
    bounds = choose_label_bounds(box, size, image_size, style, occupied)
    occupied.append(bounds)

    if style.label_style != "text":
        radius = 0 if style.label_style == "box" else min(style.label_radius, size[1] // 2)
        draw.rounded_rectangle(
            bounds,
            radius=radius,
            fill=style.label_background_color,
            outline=style.label_border_color if style.label_border_width else None,
            width=style.label_border_width,
        )
    draw.text(
        (bounds[0] + pad_x, bounds[1] + pad_y),
        text,
        font=style.font,
        fill=style.label_text_color,
        stroke_width=style.text_stroke_width,
        stroke_fill=style.text_stroke_color,
    )


def render_capture(
    image_path: Path,
    output_path: Path,
    style: RenderStyle,
    quality: int,
) -> tuple[list[Box], tuple[int, int]]:
    inference = load_json(image_path.with_suffix(".inference.json"))
    with Image.open(image_path) as opened:
        image = opened.convert("RGBA")
    boxes = inference_boxes(inference, image.width, image.height)
    draw = ImageDraw.Draw(image, "RGBA")
    occupied: list[tuple[int, int, int, int]] = []
    for box in boxes:
        draw_box(draw, box, style)
        draw_label(draw, box, image.size, style, occupied)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".png":
        image.save(output_path, format="PNG", optimize=True)
    else:
        image.convert("RGB").save(
            output_path,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=True,
            progressive=True,
        )
    return boxes, image.size


def build_parser() -> argparse.ArgumentParser:
    default_input = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Draw inference-sidecar boxes, predictions, and confidence on captures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="?", type=Path, default=default_input,
                        help="Capture image or directory of captures.")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output directory; defaults to 'exported' beside the input.")
    parser.add_argument("--suffix", default="_annotated",
                        help="Text appended to each output filename stem.")
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg",
                        help="Output image format.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality from 1 to 100.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")

    box = parser.add_argument_group("box style")
    box.add_argument("--box-color", type=parse_color, default=parse_color("#ffbe37"),
                     help="Pillow/CSS color name or #RGB[A]/#RRGGBB[AA].")
    box.add_argument("--box-width", type=positive_int, default=3, help="Box line width in pixels.")
    box.add_argument("--box-style", choices=("solid", "dashed", "corners"), default="solid")
    box.add_argument("--dash-length", type=positive_int, default=10)
    box.add_argument("--dash-gap", type=nonnegative_int, default=6)

    label = parser.add_argument_group("label style")
    label.add_argument("--label-template", default="{prediction}  {confidence}",
                       help="Fields: {prediction}, {confidence}, {confidence_value}, {detector_confidence}.")
    label.add_argument("--confidence-decimals", type=nonnegative_int, default=1)
    label.add_argument("--label-position",
                       choices=("auto", "above", "below", "inside-top", "inside-bottom"),
                       default="auto")
    label.add_argument("--label-style", choices=("box", "pill", "text"), default="box")
    label.add_argument("--label-text-color", type=parse_color, default=parse_color("white"))
    label.add_argument("--label-background-color", type=parse_color, default=parse_color("#081218e6"))
    label.add_argument("--label-border-color", type=parse_color, default=parse_color("#ffbe37"))
    label.add_argument("--label-border-width", type=nonnegative_int, default=1)
    label.add_argument("--label-padding-x", type=nonnegative_int, default=8)
    label.add_argument("--label-padding-y", type=nonnegative_int, default=5)
    label.add_argument("--label-gap", type=nonnegative_int, default=4)
    label.add_argument("--label-radius", type=nonnegative_int, default=8)
    label.add_argument("--text-stroke-width", type=nonnegative_int, default=0)
    label.add_argument("--text-stroke-color", type=parse_color, default=parse_color("black"))
    label.add_argument("--font", type=Path, help="Optional TrueType/OpenType font file.")
    label.add_argument("--font-size", type=positive_int, default=18)
    label.add_argument("--font-weight", choices=("regular", "bold"), default="bold")
    return parser


def style_from_args(args: argparse.Namespace) -> RenderStyle:
    font_path = args.font.expanduser().resolve() if args.font else None
    return RenderStyle(
        box_color=args.box_color,
        box_width=args.box_width,
        box_style=args.box_style,
        dash_length=args.dash_length,
        dash_gap=args.dash_gap,
        label_text_color=args.label_text_color,
        label_background_color=args.label_background_color,
        label_border_color=args.label_border_color,
        label_border_width=args.label_border_width,
        label_style=args.label_style,
        label_position=args.label_position,
        label_padding_x=args.label_padding_x,
        label_padding_y=args.label_padding_y,
        label_gap=args.label_gap,
        label_radius=args.label_radius,
        text_stroke_width=args.text_stroke_width,
        text_stroke_color=args.text_stroke_color,
        font=load_font(args.font_size, args.font_weight, font_path),
        label_template=args.label_template,
        confidence_decimals=args.confidence_decimals,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.quality <= 100:
        print("error: --quality must be between 1 and 100", file=sys.stderr)
        return 2
    try:
        style = style_from_args(args)
        # Validate the template even when the selected directory has no boxes.
        label_text(Box(0, 0, 1, 1, "Bird", 0.5, 0.4), style)
        input_path = args.input.expanduser().resolve()
        images = discover_images(input_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not images:
        print(f"error: no supported images found in {input_path}", file=sys.stderr)
        return 1

    if args.output:
        output_dir = args.output.expanduser().resolve()
    elif input_path.is_dir():
        output_dir = input_path / "exported"
    else:
        output_dir = input_path.parent / "exported"
    extension = ".png" if args.format == "png" else ".jpg"

    exported = skipped = failed = 0
    for image_path in images:
        output_path = output_dir / f"{image_path.stem}{args.suffix}{extension}"
        if output_path.exists() and not args.overwrite:
            print(f"skip: {output_path.name} already exists (use --overwrite)")
            skipped += 1
            continue
        try:
            boxes, size = render_capture(image_path, output_path, style, args.quality)
        except (OSError, ValueError) as exc:
            print(f"error: failed to render {image_path.name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"exported: {output_path.name} [{size[0]}x{size[1]}, {len(boxes)} boxes]")
        exported += 1

    print(f"done: {exported} exported, {skipped} skipped, {failed} failed -> {output_dir}")
    return 0 if (exported or skipped) and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())