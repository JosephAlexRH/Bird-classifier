"""
    python image_labeler.py path/to/captures

Labels are added without changing the existing capture metadata:

    "manual_label": {
      "bird_present": true,
      "species": "European robin"
    }

"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


from PIL import Image, ImageOps, ImageTk



IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
MANUAL_LABEL_KEY = "manual_label"
UNKNOWN_SPECIES = "Unknown"
THUMBNAIL_SIZE = (140, 96)
THUMBNAIL_ROW_HEIGHT = 126
THUMBNAIL_OVERSCAN_ROWS = 2


@dataclass(frozen=True)
class ImageRecord:
    """An image and the JSON file"""

    image_path: Path
    sidecar_path: Path
    pairing: str


@dataclass(frozen=True)
class ThumbnailCanvasItem:
    """Canvas objects for one currently materialized thumbnail row."""

    background_id: int
    preview_id: int
    name_id: int


def visible_thumbnail_range(
    record_count: int,
    visible_top: float,
    viewport_height: int,
    *,
    row_height: int = THUMBNAIL_ROW_HEIGHT,
    overscan_rows: int = THUMBNAIL_OVERSCAN_ROWS,
) -> range:
    """Return the bounded rows needed for a virtualized thumbnail viewport."""
    if record_count <= 0 or row_height <= 0:
        return range(0)
    overscan_rows = max(overscan_rows, 0)
    first = min(
        max(int(max(visible_top, 0) // row_height) - overscan_rows, 0),
        record_count,
    )
    last_visible = int((max(visible_top, 0) + max(viewport_height, 1)) // row_height)
    end = min(last_visible + overscan_rows + 1, record_count)
    return range(first, end)


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a sidecar and reject non-object JSON values."""
    with path.open("r", encoding="utf-8-sig") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError("the JSON root must be an object")
    return value


def discover_records(directory: Path) -> list[ImageRecord]:
    """Find images and pair them with sidecars in one directory.

    Pairing order is deliberately conservative: an identical stem wins, then a
    sidecar whose ``filename`` field names the image. A one-to-one fallback is
    used only when exactly one image and one otherwise-unmatched JSON file are
    present. Remaining images get a same-stem path where a new sidecar can be
    created on first save.
    """
    directory = directory.expanduser().resolve()
    images = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    json_paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".json"
            and path.name.casefold() != "manifest.json"
        ),
        key=lambda path: path.name.casefold(),
    )

    records_by_image: dict[Path, ImageRecord] = {}
    claimed_json: set[Path] = set()
    json_by_lower_name = {path.name.casefold(): path for path in json_paths}

    # First prefer the normal capture layout: image.jpg + image.json.
    for image_path in images:
        expected_name = f"{image_path.stem}.json".casefold()
        sidecar_path = json_by_lower_name.get(expected_name)
        if sidecar_path is not None and sidecar_path not in claimed_json:
            records_by_image[image_path] = ImageRecord(image_path, sidecar_path, "matching stem")
            claimed_json.add(sidecar_path)

    # Next honor the explicit filename stored in capture sidecars.
    filename_to_json: dict[str, list[Path]] = {}
    for sidecar_path in json_paths:
        if sidecar_path in claimed_json:
            continue
        try:
            filename = _read_json_object(sidecar_path).get("filename")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(filename, str) and Path(filename).name == filename:
            filename_to_json.setdefault(filename.casefold(), []).append(sidecar_path)

    for image_path in images:
        if image_path in records_by_image:
            continue
        candidates = filename_to_json.get(image_path.name.casefold(), [])
        candidates = [path for path in candidates if path not in claimed_json]
        if len(candidates) == 1:
            sidecar_path = candidates[0]
            records_by_image[image_path] = ImageRecord(
                image_path, sidecar_path, "JSON filename field"
            )
            claimed_json.add(sidecar_path)

    # This makes a folder containing one sample of each type usable even when a
    # sample was renamed or copied without its matching partner.
    unpaired_images = [path for path in images if path not in records_by_image]
    unclaimed_json = [path for path in json_paths if path not in claimed_json]
    if len(unpaired_images) == 1 and len(unclaimed_json) == 1:
        image_path = unpaired_images[0]
        sidecar_path = unclaimed_json[0]
        records_by_image[image_path] = ImageRecord(
            image_path, sidecar_path, "only image/JSON in folder (names differ)"
        )
        claimed_json.add(sidecar_path)

    # Images without a sidecar remain labelable; saving creates a minimal one.
    for image_path in images:
        if image_path not in records_by_image:
            sidecar_path = image_path.with_suffix(".json")
            if sidecar_path in claimed_json:
                # Different image extensions can share a stem. Never let, for
                # example, frame.jpg and frame.png both write to frame.json.
                sidecar_path = image_path.with_name(f"{image_path.name}.json")
            records_by_image[image_path] = ImageRecord(
                image_path, sidecar_path, "new sidecar"
            )

    return [records_by_image[path] for path in images]


def read_sidecar(record: ImageRecord) -> dict[str, Any]:
    """Read an existing sidecar, or build the base for a new one."""
    if record.sidecar_path.exists():
        return _read_json_object(record.sidecar_path)
    return {"filename": record.image_path.name}


def manual_label(data: dict[str, Any]) -> tuple[bool | None, str]:
    """Return a normalized label for display in the UI."""
    value = data.get(MANUAL_LABEL_KEY)
    if not isinstance(value, dict) or type(value.get("bird_present")) is not bool:
        return None, ""
    bird_present = value["bird_present"]
    species = value.get("species")
    if bird_present and isinstance(species, str):
        return True, species.strip()
    return bird_present, ""


def apply_manual_label(
    data: dict[str, Any], *, bird_present: bool, species: str | None
) -> dict[str, Any]:
    """Return a copy of a sidecar with a validated manual label applied."""
    normalized_species = (species or "").strip()
    if bird_present and not normalized_species:
        raise ValueError("Enter a species (use 'Unknown' if it cannot be identified).")

    updated = copy.deepcopy(data)
    updated[MANUAL_LABEL_KEY] = {
        "bird_present": bird_present,
        "species": normalized_species if bird_present else None,
    }
    return updated


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write formatted JSON and atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            json.dump(data, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            temporary_path = Path(output.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def collect_label_state(records: Iterable[ImageRecord]) -> tuple[set[Path], list[str]]:
    """Collect labeled image paths and species suggestions in one scan."""
    labeled_images: set[Path] = set()
    species = {UNKNOWN_SPECIES}
    for record in records:
        if not record.sidecar_path.exists():
            continue
        try:
            bird_present, name = manual_label(read_sidecar(record))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if bird_present is not None:
            labeled_images.add(record.image_path)
        if bird_present and name:
            species.add(name)
    return labeled_images, sorted(species, key=str.casefold)


class BirdLabelerApp:
    def __init__(self, root: tk.Tk, initial_directory: Path | None = None) -> None:
        self.root = root
        self.root.title("Bird Frame Labeler")
        self.root.geometry("1360x780")
        self.root.minsize(1050, 560)

        self.directory: Path | None = None
        self.records: list[ImageRecord] = []
        self.labeled_images: set[Path] = set()
        self.index = -1
        self.sidecar_data: dict[str, Any] = {}
        self.sidecar_load_error: str | None = None
        self.source_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.render_job: str | None = None
        self.loading_controls = False
        self.dirty = False
        self.selected_indices: set[int] = set()
        self.selection_anchor: int | None = None
        self.thumbnail_images: dict[int, ImageTk.PhotoImage] = {}
        self.thumbnail_items: dict[int, ThumbnailCanvasItem] = {}
        self.thumbnail_refresh_job: str | None = None

        self.presence_var = tk.StringVar(value="unlabeled")
        self.species_var = tk.StringVar()
        self.repeat_species_var = tk.BooleanVar(value=True)
        self.filename_var = tk.StringVar(value="Open a folder to begin")
        self.pairing_var = tk.StringVar()
        self.progress_var = tk.StringVar(value="0 / 0")
        self.selection_var = tk.StringVar(value="0 selected")
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self._bind_shortcuts()
        self.presence_var.trace_add("write", self._control_changed)
        self.species_var.trace_add("write", self._control_changed)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        if initial_directory is not None:
            self.open_directory(initial_directory)
        else:
            self.root.after(100, self.choose_directory)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Open Folder…", command=self.choose_directory).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(toolbar, textvariable=self.progress_var).pack(side=tk.RIGHT)

        separator = ttk.Separator(self.root)
        separator.pack(fill=tk.X)

        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        gallery = ttk.Frame(content, width=180)
        image_frame = ttk.Frame(content)
        controls = ttk.Frame(content, width=320, padding=(14, 6))
        content.add(gallery, weight=0)
        content.add(image_frame, weight=4)
        content.add(controls, weight=1)

        gallery_header = ttk.Frame(gallery)
        gallery_header.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(gallery_header, text="Images", font=("TkDefaultFont", 11, "bold")).pack(
            side=tk.LEFT
        )
        ttk.Label(gallery_header, textvariable=self.selection_var).pack(side=tk.RIGHT)
        gallery_actions = ttk.Frame(gallery)
        gallery_actions.pack(fill=tk.X, pady=(0, 5))
        self.select_all_button = ttk.Button(
            gallery_actions, text="Select All", command=self.select_all
        )
        self.select_all_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.only_current_button = ttk.Button(
            gallery_actions, text="Only Current", command=self.select_only_current
        )
        self.only_current_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        gallery_body = ttk.Frame(gallery)
        gallery_body.pack(fill=tk.BOTH, expand=True)
        gallery_scrollbar = ttk.Scrollbar(gallery_body, orient=tk.VERTICAL)
        gallery_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.thumbnail_canvas = tk.Canvas(
            gallery_body,
            width=170,
            background="#e6e6e6",
            highlightthickness=0,
            yscrollcommand=self._thumbnail_scroll_changed,
        )
        self.thumbnail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.thumbnail_scrollbar = gallery_scrollbar
        gallery_scrollbar.configure(command=self._scroll_thumbnail_view)
        self.thumbnail_canvas.bind("<Configure>", self._thumbnail_canvas_resized)
        self.thumbnail_canvas.tag_bind("thumbnail", "<Button-1>", self._thumbnail_canvas_clicked)
        self.thumbnail_canvas.bind("<MouseWheel>", self._scroll_thumbnails)
        self.thumbnail_canvas.bind("<Button-4>", self._scroll_thumbnails)
        self.thumbnail_canvas.bind("<Button-5>", self._scroll_thumbnails)
        ttk.Label(
            gallery,
            text="Click to view. Ctrl-click adds/removes; Shift-click selects a range.",
            wraplength=170,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(5, 0))

        self.canvas = tk.Canvas(
            image_frame,
            background="#202124",
            highlightthickness=0,
            width=760,
            height=600,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._schedule_render)

        ttk.Label(controls, text="Image", font=("TkDefaultFont", 12, "bold")).pack(
            anchor=tk.W
        )
        ttk.Label(
            controls,
            textvariable=self.filename_var,
            wraplength=290,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))
        self.pairing_label = ttk.Label(
            controls,
            textvariable=self.pairing_var,
            wraplength=290,
            justify=tk.LEFT,
            foreground="#8a5a00",
        )
        self.pairing_label.pack(anchor=tk.W, pady=(4, 16))

        ttk.Label(
            controls, text="1. Is a bird present?", font=("TkDefaultFont", 12, "bold")
        ).pack(anchor=tk.W)
        presence_frame = ttk.Frame(controls)
        presence_frame.pack(fill=tk.X, pady=(6, 18))
        ttk.Radiobutton(
            presence_frame,
            text="Yes",
            variable=self.presence_var,
            value="yes",
            command=self._presence_selected,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            presence_frame,
            text="No",
            variable=self.presence_var,
            value="no",
            command=self._presence_selected,
        ).pack(side=tk.LEFT, padx=(18, 0))

        ttk.Label(
            controls, text="2. What species?", font=("TkDefaultFont", 12, "bold")
        ).pack(anchor=tk.W)
        ttk.Label(
            controls,
            text="Type a name or choose a previous value. Use “Unknown” when needed.",
            wraplength=290,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 6))
        self.species_box = ttk.Combobox(controls, textvariable=self.species_var)
        self.species_box.pack(fill=tk.X, pady=(0, 6))
        self.repeat_species_check = ttk.Checkbutton(
            controls,
            text="Keep this bird species for the next unlabeled image",
            variable=self.repeat_species_var,
        )
        self.repeat_species_check.pack(anchor=tk.W, pady=(0, 18))

        self.save_next_button = ttk.Button(
            controls, text="Save && Next  (Enter)", command=self.save_and_next
        )
        self.save_next_button.pack(fill=tk.X)
        self.no_bird_button = ttk.Button(
            controls, text="No Bird && Next  (N)", command=self.mark_no_bird_and_next
        )
        self.no_bird_button.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(controls, text="Save  (Ctrl+S)", command=self.save).pack(
            fill=tk.X, pady=(6, 0)
        )
        self.save_selected_button = ttk.Button(
            controls,
            text="Save Label to Selected  (Ctrl+Shift+S)",
            command=self.save_selected,
        )
        self.save_selected_button.pack(fill=tk.X, pady=(6, 0))
        self.delete_selected_button = ttk.Button(
            controls,
            text="Delete Selected Photo(s)  (Delete)",
            command=self.delete_selected,
        )
        self.delete_selected_button.pack(fill=tk.X, pady=(6, 0))

        navigation = ttk.Frame(controls)
        navigation.pack(fill=tk.X, pady=(18, 0))
        ttk.Button(navigation, text="◀ Previous", command=self.previous).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(navigation, text="Next ▶", command=self.next).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0)
        )
        ttk.Button(controls, text="Next Unlabeled", command=self.next_unlabeled).pack(
            fill=tk.X, pady=(6, 0)
        )

        ttk.Separator(controls).pack(fill=tk.X, pady=18)
        ttk.Label(
            controls,
            text="Shortcuts: Y or B = bird, N = no bird + next, Enter = save + next, "
            "Ctrl+Shift+S = save to selected, Delete = delete selected, ←/→ = navigate. "
            "Shortcuts do not intercept species typing.",
            wraplength=290,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor=tk.W,
            relief=tk.SUNKEN,
            padding=(8, 4),
        )
        status.pack(fill=tk.X, side=tk.BOTTOM)

        self._set_controls_available(False)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-s>", lambda _event: self.save())
        self.root.bind("<Control-Shift-S>", lambda _event: self.save_selected())
        self.root.bind("<Delete>", self._delete_pressed)
        self.root.bind("<Return>", self._enter_pressed)
        self.root.bind("<Left>", self._left_pressed)
        self.root.bind("<Right>", self._right_pressed)
        self.root.bind("<KeyPress-b>", self._bird_shortcut)
        self.root.bind("<KeyPress-B>", self._bird_shortcut)
        self.root.bind("<KeyPress-y>", self._bird_shortcut)
        self.root.bind("<KeyPress-Y>", self._bird_shortcut)
        self.root.bind("<KeyPress-n>", self._no_bird_shortcut)
        self.root.bind("<KeyPress-N>", self._no_bird_shortcut)

    def _typing_species(self) -> bool:
        focused = self.root.focus_get()
        return focused is not None and str(focused).startswith(str(self.species_box))

    def _enter_pressed(self, _event: tk.Event[Any]) -> str:
        if self.records:
            self.save_and_next()
        return "break"

    def _left_pressed(self, _event: tk.Event[Any]) -> str | None:
        if self._typing_species():
            return None
        self.previous()
        return "break"

    def _right_pressed(self, _event: tk.Event[Any]) -> str | None:
        if self._typing_species():
            return None
        self.next()
        return "break"

    def _bird_shortcut(self, _event: tk.Event[Any]) -> str | None:
        if self._typing_species() or not self.records:
            return None
        self.presence_var.set("yes")
        self._presence_selected()
        return "break"

    def _no_bird_shortcut(self, _event: tk.Event[Any]) -> str | None:
        if self._typing_species() or not self.records:
            return None
        self.mark_no_bird_and_next()
        return "break"

    def _delete_pressed(self, _event: tk.Event[Any]) -> str | None:
        if self._typing_species():
            return None
        self.delete_selected()
        return "break"

    def _set_controls_available(self, available: bool) -> None:
        state = tk.NORMAL if available else tk.DISABLED
        self.save_next_button.configure(state=state)
        self.no_bird_button.configure(state=state)
        self.repeat_species_check.configure(state=state)
        self.select_all_button.configure(state=state)
        self.only_current_button.configure(state=state)
        self._update_selection_controls()
        self._update_species_state()

    def _update_selection_controls(self) -> None:
        count = len(self.selected_indices)
        self.selection_var.set(f"{count} selected")
        state = tk.NORMAL if self.records and count else tk.DISABLED
        self.save_selected_button.configure(state=state)
        self.delete_selected_button.configure(state=state)

    def _update_species_state(self) -> None:
        enabled = bool(self.records) and self.presence_var.get() == "yes"
        self.species_box.configure(state="normal" if enabled else "disabled")

    def _presence_selected(self) -> None:
        if self.presence_var.get() == "no":
            self.species_var.set("")
        self._update_species_state()
        if self.presence_var.get() == "yes":
            self.species_box.focus_set()

    def _control_changed(self, *_args: object) -> None:
        if not self.loading_controls and self.records:
            self.dirty = True
            self.status_var.set("Unsaved changes")
        self._update_species_state()

    def _thumbnail_scroll_changed(self, first: str, last: str) -> None:
        self.thumbnail_scrollbar.set(first, last)
        self._schedule_thumbnail_refresh()

    def _scroll_thumbnail_view(self, *args: str) -> None:
        self.thumbnail_canvas.yview(*args)
        self._schedule_thumbnail_refresh()

    def _thumbnail_canvas_resized(self, _event: tk.Event[Any]) -> None:
        width = max(self.thumbnail_canvas.winfo_width(), THUMBNAIL_SIZE[0] + 12)
        for index, item in self.thumbnail_items.items():
            row_top = index * THUMBNAIL_ROW_HEIGHT
            self.thumbnail_canvas.coords(
                item.background_id,
                3,
                row_top + 3,
                width - 4,
                row_top + THUMBNAIL_ROW_HEIGHT - 3,
            )
            self.thumbnail_canvas.coords(item.preview_id, width // 2, row_top + 52)
        self._schedule_thumbnail_refresh()

    def _schedule_thumbnail_refresh(self) -> None:
        if self.thumbnail_refresh_job is None:
            self.thumbnail_refresh_job = self.root.after_idle(self._refresh_visible_thumbnails)

    def _scroll_thumbnails(self, event: tk.Event[Any]) -> str:
        button = getattr(event, "num", None)
        if button in {4, 5}:
            direction = -1 if button == 4 else 1
        else:
            delta = getattr(event, "delta", 0)
            direction = -1 if delta > 0 else 1
        visible_top = max(self.thumbnail_canvas.canvasy(0) + direction * 32, 0)
        content_height = max(len(self.records) * THUMBNAIL_ROW_HEIGHT, 1)
        self.thumbnail_canvas.yview_moveto(visible_top / content_height)
        self._schedule_thumbnail_refresh()
        return "break"

    def _thumbnail_canvas_clicked(self, event: tk.Event[Any]) -> str | None:
        current = self.thumbnail_canvas.find_withtag("current")
        if not current:
            return None
        for tag in self.thumbnail_canvas.gettags(current[0]):
            if tag.startswith("thumbnail-index-"):
                return self._thumbnail_clicked(event, int(tag.removeprefix("thumbnail-index-")))
        return None

    @staticmethod
    def _selection_modifiers(event: tk.Event[Any]) -> tuple[bool, bool]:
        # Tk uses these state bits for Shift and Control on Windows/Linux.
        return bool(event.state & 0x0001), bool(event.state & 0x0004)

    def _thumbnail_clicked(self, event: tk.Event[Any], index: int) -> str:
        if index == self.index or self._save_dirty_before_navigation():
            shift, control = self._selection_modifiers(event)
            if shift and self.selection_anchor is not None:
                start, end = sorted((self.selection_anchor, index))
                ranged = set(range(start, end + 1))
                self.selected_indices = self.selected_indices | ranged if control else ranged
            elif control:
                if index in self.selected_indices:
                    self.selected_indices.remove(index)
                else:
                    self.selected_indices.add(index)
                self.selection_anchor = index
            else:
                self.selected_indices = {index}
                self.selection_anchor = index

            changed_image = index != self.index
            self.index = index
            self._refresh_thumbnail_styles()
            self._update_selection_controls()
            if changed_image:
                self.load_current(update_selection=False)
        return "break"

    def select_all(self) -> None:
        if not self.records:
            return
        self.selected_indices = set(range(len(self.records)))
        self.selection_anchor = self.index
        self._refresh_thumbnail_styles()
        self._update_selection_controls()

    def select_only_current(self) -> None:
        if not self.records:
            return
        self.selected_indices = {self.index}
        self.selection_anchor = self.index
        self._refresh_thumbnail_styles()
        self._update_selection_controls()

    def _build_thumbnails(self) -> None:
        if self.thumbnail_refresh_job is not None:
            self.root.after_cancel(self.thumbnail_refresh_job)
            self.thumbnail_refresh_job = None
        self.thumbnail_canvas.delete("all")
        self.thumbnail_images.clear()
        self.thumbnail_items.clear()
        content_height = max(len(self.records) * THUMBNAIL_ROW_HEIGHT, 1)
        self.thumbnail_canvas.configure(
            scrollregion=(0, 0, max(self.thumbnail_canvas.winfo_width(), 1), content_height),
        )
        self.thumbnail_canvas.yview_moveto(0)
        self._refresh_visible_thumbnails()

    def _load_thumbnail(self, index: int) -> ImageTk.PhotoImage:
        """Decode one small preview, asking supported formats to downsample early."""
        with Image.open(self.records[index].image_path) as opened:
            opened.draft("RGB", THUMBNAIL_SIZE)
            thumbnail = ImageOps.exif_transpose(opened).convert("RGB")
            thumbnail.thumbnail(
                THUMBNAIL_SIZE,
                resample=Image.Resampling.LANCZOS,
                reducing_gap=3.0,
            )
        return ImageTk.PhotoImage(thumbnail)

    def _create_thumbnail_item(self, index: int) -> None:
        width = max(self.thumbnail_canvas.winfo_width(), THUMBNAIL_SIZE[0] + 12)
        row_top = index * THUMBNAIL_ROW_HEIGHT
        tags = ("thumbnail", f"thumbnail-index-{index}")
        background_id = self.thumbnail_canvas.create_rectangle(
            3,
            row_top + 3,
            width - 4,
            row_top + THUMBNAIL_ROW_HEIGHT - 3,
            fill="#e6e6e6",
            outline="#e6e6e6",
            width=2,
            tags=tags,
        )
        try:
            photo = self._load_thumbnail(index)
            self.thumbnail_images[index] = photo
            preview_id = self.thumbnail_canvas.create_image(
                width // 2,
                row_top + 52,
                image=photo,
                anchor=tk.CENTER,
                tags=tags,
            )
        except (OSError, ValueError, MemoryError, Image.DecompressionBombError, tk.TclError):
            preview_id = self.thumbnail_canvas.create_text(
                width // 2,
                row_top + 52,
                text="Image unavailable",
                fill="#8b1a1a",
                font=("TkDefaultFont", 9, "bold"),
                tags=tags,
            )
        name_id = self.thumbnail_canvas.create_text(
            8,
            row_top + 112,
            text=self.records[index].image_path.name,
            anchor=tk.W,
            fill="#202020",
            font=("TkDefaultFont", 8),
            tags=tags,
        )
        self.thumbnail_items[index] = ThumbnailCanvasItem(background_id, preview_id, name_id)

    def _delete_thumbnail_item(self, index: int) -> None:
        item = self.thumbnail_items.pop(index)
        for canvas_id in (item.background_id, item.preview_id, item.name_id):
            self.thumbnail_canvas.delete(canvas_id)
        self.thumbnail_images.pop(index, None)

    def _refresh_visible_thumbnails(self) -> None:
        if self.thumbnail_refresh_job is not None:
            try:
                self.root.after_cancel(self.thumbnail_refresh_job)
            except tk.TclError:
                pass
        self.thumbnail_refresh_job = None
        if not self.records:
            return
        visible = visible_thumbnail_range(
            len(self.records),
            self.thumbnail_canvas.canvasy(0),
            self.thumbnail_canvas.winfo_height(),
        )
        wanted = set(visible)
        for index in set(self.thumbnail_items) - wanted:
            self._delete_thumbnail_item(index)
        for index in visible:
            if index not in self.thumbnail_items:
                self._create_thumbnail_item(index)
        self._refresh_thumbnail_styles()

    def _refresh_thumbnail_styles(self) -> None:
        for index, item in self.thumbnail_items.items():
            selected = index in self.selected_indices
            current = index == self.index
            background = "#cfe8ff" if selected else "#e6e6e6"
            border = "#0067c0" if current else "#58a6d8" if selected else "#e6e6e6"
            self.thumbnail_canvas.itemconfigure(
                item.background_id,
                fill=background,
                outline=border,
                width=2 if selected or current else 1,
            )
            self.thumbnail_canvas.itemconfigure(
                item.name_id,
                fill="#176b2c"
                if self.records[index].image_path in self.labeled_images
                else "#202020",
                font=("TkDefaultFont", 8, "bold" if current else "normal"),
            )

    def _show_current_thumbnail(self) -> None:
        if self.index < 0 or not self.records:
            return
        top = self.index * THUMBNAIL_ROW_HEIGHT
        bottom = top + THUMBNAIL_ROW_HEIGHT
        visible_top = self.thumbnail_canvas.canvasy(0)
        viewport_height = max(self.thumbnail_canvas.winfo_height(), THUMBNAIL_ROW_HEIGHT)
        visible_bottom = visible_top + viewport_height
        content_height = max(len(self.records) * THUMBNAIL_ROW_HEIGHT, 1)
        if top < visible_top:
            self.thumbnail_canvas.yview_moveto(top / content_height)
        elif bottom > visible_bottom:
            self.thumbnail_canvas.yview_moveto(
                max((bottom - viewport_height) / content_height, 0)
            )
        self._refresh_visible_thumbnails()

    def choose_directory(self) -> None:
        if not self._confirm_leave_unsaved():
            return
        initial = self.directory
        if initial is None:
            sample = Path(__file__).resolve().parent / "sample"
            initial = sample if sample.is_dir() else Path.cwd()
        selected = filedialog.askdirectory(
            title="Choose a folder containing images and JSON sidecars",
            initialdir=initial,
            mustexist=True,
        )
        if selected:
            self.open_directory(Path(selected))

    def open_directory(self, directory: Path) -> None:
        try:
            directory = directory.expanduser().resolve(strict=True)
            if not directory.is_dir():
                raise NotADirectoryError(f"Not a directory: {directory}")
            records = discover_records(directory)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot open folder", str(error), parent=self.root)
            return

        self.directory = directory
        self.records = records
        self.index = 0 if records else -1
        self.selected_indices = {0} if records else set()
        self.selection_anchor = self.index if records else None
        self.labeled_images, species = collect_label_state(records)
        self.species_box.configure(values=species)
        self.dirty = False
        self.root.title(f"Bird Frame Labeler — {directory.name}")

        if not records:
            self._clear_current(f"No supported images found in {directory}")
            return
        self._build_thumbnails()
        self._set_controls_available(True)
        self.load_current()

    def refresh(self) -> None:
        if self.directory is None:
            self.choose_directory()
            return
        if not self._confirm_leave_unsaved():
            return
        current_path = self.current_record.image_path if self.records else None
        records = discover_records(self.directory)
        self.records = records
        self.selected_indices.clear()
        self.selection_anchor = None
        self.labeled_images, species = collect_label_state(records)
        self.species_box.configure(values=species)
        if not records:
            self.index = -1
            self._clear_current(f"No supported images found in {self.directory}")
            return
        paths = [record.image_path for record in records]
        self.index = paths.index(current_path) if current_path in paths else 0
        self.selected_indices = {self.index}
        self.selection_anchor = self.index
        self._build_thumbnails()
        self._set_controls_available(True)
        self.load_current()

    @property
    def current_record(self) -> ImageRecord:
        return self.records[self.index]

    def _clear_current(self, status: str) -> None:
        self.records = []
        self.labeled_images.clear()
        self.selected_indices.clear()
        self.selection_anchor = None
        if self.thumbnail_refresh_job is not None:
            self.root.after_cancel(self.thumbnail_refresh_job)
            self.thumbnail_refresh_job = None
        self.thumbnail_canvas.delete("all")
        self.thumbnail_canvas.configure(scrollregion=(0, 0, 1, 1))
        self.thumbnail_images.clear()
        self.thumbnail_items.clear()
        self.index = -1
        self.sidecar_data = {}
        self.source_image = None
        self.tk_image = None
        self.canvas.delete("all")
        self.canvas.create_text(
            max(self.canvas.winfo_width() // 2, 1),
            max(self.canvas.winfo_height() // 2, 1),
            text="No images",
            fill="white",
            font=("TkDefaultFont", 16),
        )
        self.filename_var.set("No image selected")
        self.pairing_var.set("")
        self.progress_var.set("0 / 0")
        self._update_selection_controls()
        self.status_var.set(status)
        self._set_controls_available(False)

    def load_current(
        self, *, repeated_species: str | None = None, update_selection: bool = True
    ) -> None:
        record = self.current_record
        if update_selection:
            self.selected_indices = {self.index}
            self.selection_anchor = self.index
        self._refresh_thumbnail_styles()
        self._update_selection_controls()
        self._show_current_thumbnail()
        self.sidecar_load_error = None
        try:
            self.sidecar_data = read_sidecar(record)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.sidecar_data = {}
            self.sidecar_load_error = f"Could not read {record.sidecar_path.name}: {error}"

        try:
            with Image.open(record.image_path) as opened:
                target_size = (
                    max(self.root.winfo_screenwidth(), self.canvas.winfo_width(), 1),
                    max(self.root.winfo_screenheight(), self.canvas.winfo_height(), 1),
                )
                opened.draft("RGB", target_size)
                source_image = ImageOps.exif_transpose(opened).convert("RGB")
                source_image.thumbnail(
                    target_size,
                    resample=Image.Resampling.LANCZOS,
                    reducing_gap=3.0,
                )
                self.source_image = source_image
        except (OSError, ValueError, MemoryError, Image.DecompressionBombError) as error:
            self.source_image = None
            self.status_var.set(f"Could not open image: {error}")

        bird_present, species = manual_label(self.sidecar_data)
        if bird_present is None and repeated_species:
            bird_present = True
            species = repeated_species
        self.loading_controls = True
        try:
            self.presence_var.set(
                "yes" if bird_present is True else "no" if bird_present is False else "unlabeled"
            )
            self.species_var.set(species)
        finally:
            self.loading_controls = False
        self.dirty = False
        self._update_species_state()

        self.filename_var.set(record.image_path.name)
        if record.pairing == "matching stem":
            self.pairing_var.set(f"Sidecar: {record.sidecar_path.name}")
        elif record.pairing == "new sidecar":
            self.pairing_var.set(f"No sidecar yet; saving will create {record.sidecar_path.name}")
        else:
            self.pairing_var.set(
                f"Sidecar: {record.sidecar_path.name}\nPairing: {record.pairing}"
            )
        self._update_progress()
        if self.sidecar_load_error:
            self.status_var.set(self.sidecar_load_error)
        elif repeated_species and manual_label(self.sidecar_data)[0] is None:
            self.status_var.set(f"Ready to repeat bird species: {repeated_species}")
        elif bird_present is None:
            self.status_var.set("Unlabeled")
        else:
            self.status_var.set("Label loaded")
        self._render_image()

    def _schedule_render(self, _event: tk.Event[Any]) -> None:
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.render_job = self.root.after(80, self._render_image)

    def _render_image(self) -> None:
        self.render_job = None
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width() - 20, 1)
        height = max(self.canvas.winfo_height() - 20, 1)
        if self.source_image is None:
            self.canvas.create_text(
                width // 2,
                height // 2,
                text="Image unavailable",
                fill="white",
                font=("TkDefaultFont", 16),
            )
            return
        try:
            display = ImageOps.contain(
                self.source_image, (width, height), method=Image.Resampling.LANCZOS
            )
            self.tk_image = ImageTk.PhotoImage(display)
        except (MemoryError, tk.TclError) as error:
            self.tk_image = None
            self.canvas.create_text(
                width // 2,
                height // 2,
                text=f"Could not render image: {error}",
                fill="white",
                font=("TkDefaultFont", 12),
            )
            return
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.tk_image,
            anchor=tk.CENTER,
        )

    def _update_progress(self) -> None:
        self.progress_var.set(
            f"Image {self.index + 1} / {len(self.records)}  |  "
            f"Labeled {len(self.labeled_images)} / {len(self.records)}"
        )

    def save(self, *, quiet: bool = False) -> bool:
        if not self.records:
            return False
        if self.sidecar_load_error:
            messagebox.showerror(
                "Cannot save label",
                f"The existing sidecar could not be read, so it will not be overwritten.\n\n"
                f"{self.sidecar_load_error}",
                parent=self.root,
            )
            return False

        presence = self.presence_var.get()
        if presence not in {"yes", "no"}:
            messagebox.showwarning(
                "Label required", "Choose Yes or No for bird presence.", parent=self.root
            )
            return False
        try:
            updated = apply_manual_label(
                self.sidecar_data,
                bird_present=presence == "yes",
                species=self.species_var.get(),
            )
            write_json_atomic(self.current_record.sidecar_path, updated)
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not save label", str(error), parent=self.root)
            return False

        self.sidecar_data = updated
        self.labeled_images.add(self.current_record.image_path)
        self.dirty = False
        species_values = set(self.species_box.cget("values"))
        if presence == "yes":
            species_values.add(self.species_var.get().strip())
        self.species_box.configure(values=sorted(species_values, key=str.casefold))
        self._update_progress()
        self._refresh_thumbnail_styles()
        if not quiet:
            self.status_var.set(f"Saved {self.current_record.sidecar_path.name}")
        return True

    def save_selected(self) -> bool:
        """Apply the controls' label to every selected thumbnail."""
        indices = sorted(self.selected_indices)
        if not self.records or not indices:
            return False

        presence = self.presence_var.get()
        if presence not in {"yes", "no"}:
            messagebox.showwarning(
                "Label required", "Choose Yes or No for bird presence.", parent=self.root
            )
            return False

        bird_present = presence == "yes"
        species = self.species_var.get()
        try:
            # Validate the label before asking for confirmation or touching files.
            apply_manual_label({}, bird_present=bird_present, species=species)
        except ValueError as error:
            messagebox.showwarning("Label required", str(error), parent=self.root)
            return False

        if len(indices) > 1:
            existing_count = sum(
                1
                for index in indices
                if self.records[index].image_path in self.labeled_images
            )
            overwrite_note = (
                f"\n\nThis will replace {existing_count} existing manual label(s)."
                if existing_count
                else ""
            )
            if not messagebox.askyesno(
                "Apply label to selected images?",
                f"Save the same label to {len(indices)} selected images?{overwrite_note}",
                parent=self.root,
            ):
                return False

        updates: list[tuple[ImageRecord, dict[str, Any]]] = []
        for index in indices:
            record = self.records[index]
            try:
                data = read_sidecar(record)
                updated = apply_manual_label(
                    data, bird_present=bird_present, species=species
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                messagebox.showerror(
                    "Cannot save selected labels",
                    f"No files were changed because {record.sidecar_path.name} could not "
                    f"be read or validated:\n\n{error}",
                    parent=self.root,
                )
                return False
            updates.append((record, updated))

        saved_count = 0
        try:
            for record, updated in updates:
                write_json_atomic(record.sidecar_path, updated)
                saved_count += 1
        except OSError as error:
            messagebox.showerror(
                "Could not save selected labels",
                f"Saved {saved_count} of {len(updates)} labels before an error occurred:\n\n{error}",
                parent=self.root,
            )
            self.labeled_images.update(
                record.image_path for record, _updated in updates[:saved_count]
            )
            self._update_progress()
            self._refresh_thumbnail_styles()
            return False

        self.labeled_images.update(record.image_path for record, _updated in updates)
        if self.index in self.selected_indices:
            self.sidecar_data = updates[indices.index(self.index)][1]
            self.sidecar_load_error = None
            self.dirty = False
        species_values = set(self.species_box.cget("values"))
        if bird_present:
            species_values.add(species.strip())
        self.species_box.configure(values=sorted(species_values, key=str.casefold))
        self._update_progress()
        self._refresh_thumbnail_styles()
        label_description = species.strip() if bird_present else "No bird"
        self.status_var.set(
            f"Saved “{label_description}” to {len(updates)} selected image(s)"
        )
        return True

    def delete_selected(self) -> bool:
        """Permanently delete selected photos and their existing sidecars."""
        indices = sorted(
            index for index in self.selected_indices if 0 <= index < len(self.records)
        )
        if not indices:
            return False

        targets = [self.records[index] for index in indices]
        existing_sidecars = sum(record.sidecar_path.exists() for record in targets)
        names = "\n".join(f"  • {record.image_path.name}" for record in targets[:6])
        if len(targets) > 6:
            names += f"\n  • …and {len(targets) - 6} more"
        sidecar_note = (
            f"\n\n{existing_sidecars} corresponding JSON sidecar(s) will also be deleted."
            if existing_sidecars
            else "\n\nNo corresponding JSON sidecars currently exist."
        )
        unsaved_note = (
            "\n\nUnsaved label changes for the current photo will be discarded."
            if self.dirty and self.index in indices
            else ""
        )
        if not messagebox.askyesno(
            "Permanently delete selected photos?",
            f"Permanently delete {len(targets)} selected photo(s)?\n\n"
            f"{names}{sidecar_note}{unsaved_note}\n\nThis cannot be undone.",
            icon="warning",
            parent=self.root,
        ):
            return False

        old_records = self.records
        old_index = self.index
        current_path = self.current_record.image_path
        selected_paths = {
            old_records[index].image_path
            for index in self.selected_indices
            if 0 <= index < len(old_records)
        }
        deleted_paths: set[Path] = set()
        deleted_sidecars = 0
        errors: list[str] = []

        for record in targets:
            try:
                record.image_path.unlink()
            except FileNotFoundError:
                # Treat a concurrently removed photo as deleted and clean up its
                # paired sidecar so the UI can discard the stale record.
                pass
            except OSError as error:
                errors.append(f"{record.image_path.name}: {error}")
                continue

            deleted_paths.add(record.image_path)
            self.labeled_images.discard(record.image_path)
            try:
                if record.sidecar_path.exists():
                    record.sidecar_path.unlink()
                    deleted_sidecars += 1
            except OSError as error:
                errors.append(f"{record.sidecar_path.name}: {error}")

        if not deleted_paths:
            self.status_var.set("No photos were deleted")
            messagebox.showerror(
                "Could not delete selected photos",
                "No photos were deleted:\n\n" + "\n".join(errors),
                parent=self.root,
            )
            return False

        self.records = [
            record for record in old_records if record.image_path not in deleted_paths
        ]
        selected_paths.difference_update(deleted_paths)

        if not self.records:
            deleted_count = len(deleted_paths)
            self.dirty = False
            self._clear_current(
                f"Deleted {deleted_count} photo(s) and {deleted_sidecars} sidecar(s)"
            )
        else:
            remaining_paths = [record.image_path for record in self.records]
            current_deleted = current_path in deleted_paths
            if current_deleted:
                self.index = min(old_index, len(self.records) - 1)
            else:
                self.index = remaining_paths.index(current_path)

            self.selected_indices = {
                index
                for index, record in enumerate(self.records)
                if record.image_path in selected_paths
            }
            if not self.selected_indices:
                self.selected_indices = {self.index}
            self.selection_anchor = self.index
            self._build_thumbnails()
            self._set_controls_available(True)
            if current_deleted:
                self.load_current(update_selection=False)
            else:
                self._refresh_thumbnail_styles()
                self._update_selection_controls()
                self._show_current_thumbnail()
                self._update_progress()

            status = (
                f"Deleted {len(deleted_paths)} photo(s) and "
                f"{deleted_sidecars} sidecar(s)"
            )
            if self.dirty:
                status += "; current photo still has unsaved changes"
            self.status_var.set(status)

        if errors:
            messagebox.showerror(
                "Deletion incomplete",
                f"Deleted {len(deleted_paths)} of {len(targets)} selected photo(s), "
                "but some files could not be removed:\n\n" + "\n".join(errors),
                parent=self.root,
            )
            return False
        return True

    def save_and_next(self) -> None:
        repeated_species = None
        if self.repeat_species_var.get() and self.presence_var.get() == "yes":
            repeated_species = self.species_var.get().strip() or None
        if self.save(quiet=True):
            saved_name = self.current_record.sidecar_path.name
            if self.index < len(self.records) - 1:
                self.index += 1
                self.load_current(repeated_species=repeated_species)
                if repeated_species is None or manual_label(self.sidecar_data)[0] is not None:
                    self.status_var.set(f"Saved {saved_name}")
            else:
                self.status_var.set(f"Saved {saved_name} — reached the last image")

    def mark_no_bird_and_next(self) -> None:
        if not self.records:
            return
        self.presence_var.set("no")
        self.species_var.set("")
        self.save_and_next()

    def _navigate(self, offset: int) -> None:
        if not self.records or not self._save_dirty_before_navigation():
            return
        new_index = min(max(self.index + offset, 0), len(self.records) - 1)
        if new_index == self.index:
            self.status_var.set("Already at the first image" if offset < 0 else "Already at the last image")
            return
        self.index = new_index
        self.load_current()

    def previous(self) -> None:
        self._navigate(-1)

    def next(self) -> None:
        self._navigate(1)

    def next_unlabeled(self) -> None:
        if not self.records or not self._save_dirty_before_navigation():
            return
        order = list(range(self.index + 1, len(self.records))) + list(range(0, self.index))
        for candidate in order:
            if self.records[candidate].image_path not in self.labeled_images:
                self.index = candidate
                self.load_current()
                return
        self.status_var.set("All images are labeled")

    def _save_dirty_before_navigation(self) -> bool:
        return not self.dirty or self.save(quiet=True)

    def _confirm_leave_unsaved(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved changes",
            "Save the current label before continuing?",
            parent=self.root,
        )
        if answer is None:
            return False
        if answer:
            return self.save()
        self.dirty = False
        return True

    def close(self) -> None:
        if self._confirm_leave_unsaved():
            self.root.destroy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label bird presence and species in image JSON sidecars."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="capture directory to open (otherwise a folder chooser is shown)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    root = tk.Tk()
    app = BirdLabelerApp(root, arguments.directory)
    # Retain an explicit reference for Python/Tk implementations that otherwise
    # collect an application reachable only through bound callbacks.
    root._bird_labeler_app = app  # type: ignore[attr-defined]
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())