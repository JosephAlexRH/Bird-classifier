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


@dataclass(frozen=True)
class ImageRecord:
    """An image and the JSON file"""

    image_path: Path
    sidecar_path: Path
    pairing: str


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
        self.root.geometry("1180x780")
        self.root.minsize(800, 560)

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

        self.presence_var = tk.StringVar(value="unlabeled")
        self.species_var = tk.StringVar()
        self.repeat_species_var = tk.BooleanVar(value=True)
        self.filename_var = tk.StringVar(value="Open a folder to begin")
        self.pairing_var = tk.StringVar()
        self.progress_var = tk.StringVar(value="0 / 0")
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

        image_frame = ttk.Frame(content)
        controls = ttk.Frame(content, width=320, padding=(14, 6))
        content.add(image_frame, weight=4)
        content.add(controls, weight=1)

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
            "←/→ = navigate. Shortcuts do not intercept species typing.",
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

    def _set_controls_available(self, available: bool) -> None:
        state = tk.NORMAL if available else tk.DISABLED
        self.save_next_button.configure(state=state)
        self.no_bird_button.configure(state=state)
        self.repeat_species_check.configure(state=state)
        self._update_species_state()

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
        self.labeled_images, species = collect_label_state(records)
        self.species_box.configure(values=species)
        self.dirty = False
        self.root.title(f"Bird Frame Labeler — {directory.name}")

        if not records:
            self._clear_current(f"No supported images found in {directory}")
            return
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
        self.labeled_images, species = collect_label_state(records)
        self.species_box.configure(values=species)
        if not records:
            self.index = -1
            self._clear_current(f"No supported images found in {self.directory}")
            return
        paths = [record.image_path for record in records]
        self.index = paths.index(current_path) if current_path in paths else 0
        self._set_controls_available(True)
        self.load_current()

    @property
    def current_record(self) -> ImageRecord:
        return self.records[self.index]

    def _clear_current(self, status: str) -> None:
        self.records = []
        self.labeled_images.clear()
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
        self.status_var.set(status)
        self._set_controls_available(False)

    def load_current(self, *, repeated_species: str | None = None) -> None:
        record = self.current_record
        self.sidecar_load_error = None
        try:
            self.sidecar_data = read_sidecar(record)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.sidecar_data = {}
            self.sidecar_load_error = f"Could not read {record.sidecar_path.name}: {error}"

        try:
            with Image.open(record.image_path) as opened:
                self.source_image = ImageOps.exif_transpose(opened).convert("RGB")
        except (OSError, ValueError) as error:
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
        display = ImageOps.contain(
            self.source_image, (width, height), method=Image.Resampling.LANCZOS
        )
        self.tk_image = ImageTk.PhotoImage(display)
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
        if not quiet:
            self.status_var.set(f"Saved {self.current_record.sidecar_path.name}")
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