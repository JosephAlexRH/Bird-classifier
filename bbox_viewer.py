"""View and manually review single-bird bounding boxes.

The source manifest is never modified.  Human decisions are atomically saved to
a separate CSV through :mod:`bbox_data`, so a review session can be resumed.

PowerShell examples::

    python .\\bbox_viewer.py ".\\Notebooks\\Outputs\\iNaturalist Outputs\\Bounding Box Outputs\\inaturalist_bird_bbox_manifest.csv" --image-root "E:\\Datasets\\iNaturalist\\train_mini\\train_mini"
    python .\\bbox_viewer.py "E:\\Datasets\\CUB_200\\CUB_200_2011"

Drag on the image to create a box.  Drag inside it to move it and drag a corner
handle to resize it.  The orange dashed rectangle is the immutable source box;
the cyan rectangle is the human-review draft or saved box.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Iterable
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from bbox_data import (
    BBox,
    BBoxDataset,
    GroundTruthRecord,
    ImageCanvasTransform,
    ImageRecord,
    load_dataset,
    load_ground_truth,
    save_ground_truth,
    sort_records,
)


REVIEW_FILTERS = ("all", "unreviewed", "boxed", "no_bird", "skipped")
SOURCE_BOX_FILTERS = ("all", "with box", "without box")
SORT_KEYS = (
    "source_id",
    "image_id",
    "path",
    "status",
    "class_name",
    "confidence",
    "detection_count",
)
ROW_HEIGHT = 28
ROW_OVERSCAN = 3
HANDLE_RADIUS = 6
MIN_BOX_PIXELS = 1.0


def default_ground_truth_path(source: Path) -> Path:
    """Return a non-destructive default review path for a source dataset."""

    if source.is_dir():
        return source / "bbox_ground_truth.csv"
    return source.with_name(f"{source.stem}_ground_truth.csv")


def file_signature(path: Path) -> tuple[int, int, str] | None:
    """Return metadata plus content hash for external-modification detection."""

    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return stat.st_mtime_ns, stat.st_size, digest.hexdigest()


@contextmanager
def review_file_lock(path: Path) -> Iterable[None]:
    """Hold a non-blocking cross-process lock beside one review CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            raise OSError(
                f"another reviewer is currently saving {path}; try again"
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def validate_review_identities(
    dataset: BBoxDataset, reviews: dict[str, GroundTruthRecord]
) -> None:
    """Reject saved rows that reuse a current source ID for another image."""

    records = {record.source_id: record for record in dataset}
    for source_id, review in reviews.items():
        record = records.get(source_id)
        if record is None:  # Preserve decisions from a previous/superset manifest.
            continue
        mismatches = []
        if review.source_dataset != record.source_dataset:
            mismatches.append("source_dataset")
        if review.image_id != record.image_id:
            mismatches.append("image_id")
        if review.relative_path != record.relative_path:
            mismatches.append("relative_path")
        if mismatches:
            raise ValueError(
                f"saved review {source_id!r} does not match the source manifest: "
                + ", ".join(mismatches)
            )


def visible_row_range(
    count: int,
    visible_top: float,
    viewport_height: int,
    *,
    row_height: int = ROW_HEIGHT,
    overscan: int = ROW_OVERSCAN,
) -> range:
    """Return only rows around a virtual list's current viewport."""

    if count <= 0 or row_height <= 0:
        return range(0)
    first = max(int(max(visible_top, 0.0) // row_height) - max(overscan, 0), 0)
    last = int((max(visible_top, 0.0) + max(viewport_height, 1)) // row_height)
    return range(first, min(last + max(overscan, 0) + 1, count))


def bbox_handles(box: BBox) -> dict[str, tuple[float, float]]:
    """Return named corner locations for a normalized box."""

    box = box.normalized()
    return {
        "nw": (box.x, box.y),
        "ne": (box.right, box.y),
        "se": (box.right, box.bottom),
        "sw": (box.x, box.bottom),
    }


def hit_test_bbox(box: BBox, x: float, y: float, tolerance: float) -> str | None:
    """Return a corner name, ``move``, or ``None`` for an image-space point."""

    for name, (handle_x, handle_y) in bbox_handles(box).items():
        if math.hypot(x - handle_x, y - handle_y) <= tolerance:
            return name
    normalized = box.normalized()
    if normalized.x <= x <= normalized.right and normalized.y <= y <= normalized.bottom:
        return "move"
    return None


def move_bbox(box: BBox, dx: float, dy: float, width: float, height: float) -> BBox:
    """Move a box while keeping its size and all edges inside the image."""

    normalized = box.normalized()
    x = min(max(normalized.x + dx, 0.0), max(width - normalized.width, 0.0))
    y = min(max(normalized.y + dy, 0.0), max(height - normalized.height, 0.0))
    return BBox(x, y, min(normalized.width, width), min(normalized.height, height))


def resize_bbox(
    box: BBox,
    corner: str,
    x: float,
    y: float,
    image_width: float,
    image_height: float,
) -> BBox:
    """Move one corner and return a normalized box clipped to the image."""

    handles = bbox_handles(box)
    opposites = {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}
    if corner not in opposites:
        raise ValueError(f"unknown bounding-box corner: {corner}")
    fixed_x, fixed_y = handles[opposites[corner]]
    x = min(max(x, 0.0), image_width)
    y = min(max(y, 0.0), image_height)
    return BBox(fixed_x, fixed_y, x - fixed_x, y - fixed_y).normalized()


@dataclass(slots=True)
class ReviewSession:
    """Review-aware filtering and navigation independent of Tk."""

    dataset: BBoxDataset
    reviews: dict[str, GroundTruthRecord]
    filtered: tuple[ImageRecord, ...] = ()
    position: int = -1
    query: str = ""
    review_filter: str = "all"
    source_box_filter: str = "all"
    sort_key: str = "source_id"
    reverse: bool = False
    _counts_cache: dict[str, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.apply_filters()

    @property
    def current(self) -> ImageRecord | None:
        if 0 <= self.position < len(self.filtered):
            return self.filtered[self.position]
        return None

    @property
    def current_review(self) -> GroundTruthRecord | None:
        record = self.current
        return self.reviews.get(record.source_id) if record else None

    def apply_filters(
        self,
        *,
        query: str | None = None,
        review_filter: str | None = None,
        source_box_filter: str | None = None,
        sort_key: str | None = None,
        reverse: bool | None = None,
        preserve_source_id: str | None = None,
    ) -> None:
        if query is not None:
            self.query = query.strip()
        if review_filter is not None:
            if review_filter not in REVIEW_FILTERS:
                raise ValueError(f"unknown review filter: {review_filter}")
            self.review_filter = review_filter
        if source_box_filter is not None:
            if source_box_filter not in SOURCE_BOX_FILTERS:
                raise ValueError(f"unknown source-box filter: {source_box_filter}")
            self.source_box_filter = source_box_filter
        if sort_key is not None:
            self.sort_key = sort_key
        if reverse is not None:
            self.reverse = reverse

        old_id = preserve_source_id or (self.current.source_id if self.current else None)
        needle = self.query.casefold()
        records: Iterable[ImageRecord] = self.dataset
        accepted: list[ImageRecord] = []
        for record in records:
            review = self.reviews.get(record.source_id)
            state = review.review_state if review else "unreviewed"
            if self.review_filter != "all" and state != self.review_filter:
                continue
            if self.source_box_filter == "with box" and not record.has_bbox:
                continue
            if self.source_box_filter == "without box" and record.has_bbox:
                continue
            if needle and not any(
                needle in value.casefold()
                for value in (
                    record.source_id,
                    record.image_id,
                    record.relative_path,
                    record.class_name,
                    record.bbox_status,
                    state,
                )
            ):
                continue
            accepted.append(record)
        self.filtered = sort_records(accepted, self.sort_key, self.reverse)
        if not self.filtered:
            self.position = -1
            return
        if old_id:
            for index, record in enumerate(self.filtered):
                if record.source_id == old_id:
                    self.position = index
                    break
            else:
                self.position = min(max(self.position, 0), len(self.filtered) - 1)
        else:
            self.position = min(max(self.position, 0), len(self.filtered) - 1)

    def navigate(self, offset: int) -> bool:
        if not self.filtered:
            return False
        target = min(max(self.position + offset, 0), len(self.filtered) - 1)
        changed = target != self.position
        self.position = target
        return changed

    def next_unreviewed(self) -> bool:
        if not self.filtered:
            return False
        for index in chain(
            range(self.position + 1, len(self.filtered)), range(0, self.position)
        ):
            if self.filtered[index].source_id not in self.reviews:
                self.position = index
                return True
        return False

    def counts(self) -> dict[str, int]:
        if self._counts_cache is not None:
            return self._counts_cache.copy()
        counts = {state: 0 for state in ("unreviewed", "boxed", "no_bird", "skipped")}
        source_ids = {record.source_id for record in self.dataset}
        for source_id in source_ids:
            review = self.reviews.get(source_id)
            counts[review.review_state if review else "unreviewed"] += 1
        self._counts_cache = counts
        return counts.copy()

    def reviews_changed(self) -> None:
        self._counts_cache = None


@dataclass(slots=True)
class DragState:
    mode: str
    start_x: float
    start_y: float
    original: BBox | None


class BBoxViewerApp:
    """Tk viewer/editor for one bounding box and decision per source image."""

    def __init__(
        self,
        root: tk.Tk,
        source: Path | None = None,
        *,
        image_root: Path | None = None,
        ground_truth_path: Path | None = None,
        read_only: bool = False,
        start_source_id: str | None = None,
    ) -> None:
        self.root = root
        self.root.title("Bird Bounding-Box Reviewer")
        self.root.geometry("1450x850")
        self.root.minsize(950, 600)
        self.read_only = read_only
        self.requested_image_root = image_root
        self.requested_ground_truth = ground_truth_path
        self.start_source_id = start_source_id

        self.session: ReviewSession | None = None
        self.ground_truth_path: Path | None = None
        self.ground_truth_signature: tuple[int, int, str] | None = None
        self.preview_image: Image.Image | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self.rendered_size: tuple[int, int] | None = None
        self.image_size: tuple[int, int] | None = None
        self.transform: ImageCanvasTransform | None = None
        self.draft_bbox: BBox | None = None
        self.draft_dirty = False
        self.drag: DragState | None = None
        self.render_job: str | None = None
        self.filter_job: str | None = None
        self.list_job: str | None = None
        self.syncing_filters = False

        self.search_var = tk.StringVar()
        self.review_filter_var = tk.StringVar(value="all")
        self.source_filter_var = tk.StringVar(value="all")
        self.sort_var = tk.StringVar(value="source_id")
        self.reverse_var = tk.BooleanVar(value=False)
        self.progress_var = tk.StringVar(value="0 / 0")
        self.counts_var = tk.StringVar(value="No dataset loaded")
        self.title_var = tk.StringVar(value="Open a manifest or raw CUB directory")
        self.details_var = tk.StringVar()
        self.box_var = tk.StringVar(value="No box")
        self.status_var = tk.StringVar(value="Ready")
        self.filter_controls: list[tk.Widget] = []

        self._build_ui()
        self._bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        if source is not None:
            self.root.after_idle(lambda: self.open_source(source))

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(8, 7))
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Open Manifest…", command=self.choose_manifest).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Open Raw CUB…", command=self.choose_cub).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Image Root…", command=self.choose_image_root).pack(side=tk.LEFT)
        ttk.Label(toolbar, textvariable=self.progress_var).pack(side=tk.RIGHT)

        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        left = ttk.Frame(content, width=310)
        center = ttk.Frame(content)
        right = ttk.Frame(content, width=310, padding=(10, 4))
        content.add(left, weight=0)
        content.add(center, weight=5)
        content.add(right, weight=0)

        filters = ttk.Frame(left, padding=(0, 0, 5, 5))
        filters.pack(fill=tk.X)
        ttk.Label(filters, text="Search").grid(row=0, column=0, sticky=tk.W)
        search = ttk.Entry(filters, textvariable=self.search_var)
        search.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 5))
        self.search_entry = search
        ttk.Label(filters, text="Review").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(filters, text="Source box").grid(row=2, column=1, sticky=tk.W)
        review_filter = ttk.Combobox(
            filters, textvariable=self.review_filter_var, values=REVIEW_FILTERS,
            state="readonly", width=13,
        )
        review_filter.grid(row=3, column=0, sticky=tk.EW, padx=(0, 3))
        source_filter = ttk.Combobox(
            filters, textvariable=self.source_filter_var, values=SOURCE_BOX_FILTERS,
            state="readonly", width=13,
        )
        source_filter.grid(row=3, column=1, sticky=tk.EW)
        ttk.Label(filters, text="Sort").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        sort_box = ttk.Combobox(
            filters, textvariable=self.sort_var, values=SORT_KEYS, state="readonly", width=16,
        )
        sort_box.grid(row=5, column=0, sticky=tk.EW)
        ttk.Checkbutton(filters, text="Descending", variable=self.reverse_var).grid(
            row=5, column=1, sticky=tk.W
        )
        filters.columnconfigure(0, weight=1)
        filters.columnconfigure(1, weight=1)
        self.filter_controls = [search, review_filter, source_filter, sort_box]

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.record_canvas = tk.Canvas(
            list_frame, width=300, background="#f0f0f0", highlightthickness=0,
            yscrollcommand=self._list_scroll_changed,
        )
        self.record_canvas.pack(fill=tk.BOTH, expand=True)
        self.list_scroll = list_scroll
        list_scroll.configure(command=self._scroll_list)
        ttk.Label(left, textvariable=self.counts_var, wraplength=295).pack(fill=tk.X, pady=4)

        self.image_canvas = tk.Canvas(center, background="#171717", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right, textvariable=self.title_var, font=("TkDefaultFont", 12, "bold"),
                  wraplength=290).pack(fill=tk.X)
        ttk.Separator(right).pack(fill=tk.X, pady=8)
        ttk.Label(right, textvariable=self.details_var, justify=tk.LEFT, wraplength=290).pack(
            fill=tk.X
        )
        ttk.Label(right, textvariable=self.box_var, justify=tk.LEFT, wraplength=290).pack(
            fill=tk.X, pady=(8, 12)
        )

        self.accept_button = ttk.Button(
            right, text="Accept Box + Next  [1]", command=self.accept_box_and_next
        )
        self.accept_button.pack(fill=tk.X, pady=2)
        self.no_bird_button = ttk.Button(
            right, text="No Bird + Next  [2]", command=lambda: self.commit_and_next("no_bird")
        )
        self.no_bird_button.pack(fill=tk.X, pady=2)
        self.skip_button = ttk.Button(
            right, text="Skip + Next  [3]", command=lambda: self.commit_and_next("skipped")
        )
        self.skip_button.pack(fill=tk.X, pady=2)
        ttk.Separator(right).pack(fill=tk.X, pady=8)
        self.source_button = ttk.Button(right, text="Reset Draft to Source  [R]", command=self.use_source)
        self.source_button.pack(fill=tk.X, pady=2)
        self.delete_button = ttk.Button(right, text="Delete Draft  [Del]", command=self.delete_draft)
        self.delete_button.pack(fill=tk.X, pady=2)
        self.clear_button = ttk.Button(right, text="Clear Saved Review", command=self.clear_review)
        self.clear_button.pack(fill=tk.X, pady=2)
        ttk.Separator(right).pack(fill=tk.X, pady=8)
        nav = ttk.Frame(right)
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="← Previous", command=self.previous).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav, text="Next →", command=self.next).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0)
        )
        ttk.Button(right, text="Next Unreviewed  [N]", command=self.next_unreviewed).pack(
            fill=tk.X, pady=(5, 2)
        )

        status = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, padding=(8, 4))
        status.pack(fill=tk.X)

    def _bind_events(self) -> None:
        for variable in (
            self.search_var, self.review_filter_var, self.source_filter_var,
            self.sort_var, self.reverse_var,
        ):
            variable.trace_add("write", self._schedule_filter)
        self.record_canvas.bind("<Configure>", lambda _event: self._schedule_list_refresh())
        self.record_canvas.bind("<Button-1>", self._record_clicked)
        self.record_canvas.bind("<MouseWheel>", self._list_mousewheel)
        self.record_canvas.bind("<Button-4>", self._list_mousewheel)
        self.record_canvas.bind("<Button-5>", self._list_mousewheel)
        self.image_canvas.bind("<Configure>", self._schedule_render)
        self.image_canvas.bind("<ButtonPress-1>", self._drag_started)
        self.image_canvas.bind("<B1-Motion>", self._drag_moved)
        self.image_canvas.bind("<ButtonRelease-1>", self._drag_finished)
        self.image_canvas.bind("<Motion>", self._update_cursor)
        self.root.bind("<Left>", lambda event: self._shortcut(event, self.previous))
        self.root.bind("<Right>", lambda event: self._shortcut(event, self.next))
        self.root.bind("<Key-1>", lambda event: self._shortcut(event, self.accept_box_and_next))
        self.root.bind(
            "<Key-2>", lambda event: self._shortcut(event, lambda: self.commit_and_next("no_bird"))
        )
        self.root.bind(
            "<Key-3>", lambda event: self._shortcut(event, lambda: self.commit_and_next("skipped"))
        )
        self.root.bind("<Key-n>", lambda event: self._shortcut(event, self.next_unreviewed))
        self.root.bind("<Key-r>", lambda event: self._shortcut(event, self.use_source))
        self.root.bind("<Delete>", lambda event: self._shortcut(event, self.delete_draft))

    def _shortcut(self, event: tk.Event[Any], action: Any) -> str | None:
        """Run a global shortcut unless focus is in a text-entry control."""

        if event.widget.winfo_class() in {"Entry", "TEntry", "Text", "TCombobox", "Spinbox"}:
            return None
        action()
        return "break"

    def choose_manifest(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a generated bounding-box manifest",
            filetypes=(("CSV manifests", "*.csv"), ("All files", "*.*")),
        )
        if selected:
            self.open_source(Path(selected))

    def choose_cub(self) -> None:
        selected = filedialog.askdirectory(title="Choose the CUB_200_2011 directory", mustexist=True)
        if selected:
            self.open_source(Path(selected))

    def choose_image_root(self) -> None:
        selected = filedialog.askdirectory(title="Choose the directory containing class folders")
        if selected and self._confirm_discard_draft():
            self.requested_image_root = Path(selected)
            if self.session:
                self.session.dataset = BBoxDataset(
                    self.session.dataset.records,
                    self.session.dataset.source_path,
                    self.session.dataset.format,
                    Path(selected),
                )
                self.load_current()

    def open_source(self, source: Path) -> None:
        if not self._confirm_discard_draft():
            return
        self._cancel_pending_jobs()
        try:
            source = source.expanduser().resolve(strict=True)
            dataset = load_dataset(source, image_root=self.requested_image_root)
            output = self.requested_ground_truth or default_ground_truth_path(source)
            reviews = load_ground_truth(output)
            validate_review_identities(dataset, reviews)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot open dataset", str(error), parent=self.root)
            return
        self.ground_truth_path = output
        self.ground_truth_signature = file_signature(output)
        self.session = ReviewSession(dataset, reviews)
        self._sync_filter_controls()
        if self.start_source_id:
            for index, record in enumerate(self.session.filtered):
                if record.source_id == self.start_source_id:
                    self.session.position = index
                    break
            else:
                self.status_var.set(f"Start source ID not found: {self.start_source_id}")
        self.root.title(f"Bird Bounding-Box Reviewer — {source.name}")
        self._refresh_after_filter()
        self.load_current()

    def _schedule_filter(self, *_args: object) -> None:
        if self.syncing_filters:
            return
        if self.filter_job is not None:
            self.root.after_cancel(self.filter_job)
        self.filter_job = self.root.after(180, self._apply_filters)

    def _apply_filters(self) -> None:
        self.filter_job = None
        if not self.session:
            return
        if self.drag is not None:
            self.filter_job = self.root.after(180, self._apply_filters)
            return
        if not self._confirm_discard_draft():
            self._sync_filter_controls()
            return
        current_id = self.session.current.source_id if self.session.current else None
        try:
            self.session.apply_filters(
                query=self.search_var.get(),
                review_filter=self.review_filter_var.get(),
                source_box_filter=self.source_filter_var.get(),
                sort_key=self.sort_var.get(),
                reverse=self.reverse_var.get(),
                preserve_source_id=current_id,
            )
        except ValueError as error:
            self.status_var.set(str(error))
            return
        self._refresh_after_filter()
        self.load_current()

    def _sync_filter_controls(self) -> None:
        if not self.session:
            return
        self.syncing_filters = True
        try:
            self.search_var.set(self.session.query)
            self.review_filter_var.set(self.session.review_filter)
            self.source_filter_var.set(self.session.source_box_filter)
            self.sort_var.set(self.session.sort_key)
            self.reverse_var.set(self.session.reverse)
        finally:
            self.syncing_filters = False

    def _refresh_after_filter(self) -> None:
        if not self.session:
            return
        height = max(len(self.session.filtered) * ROW_HEIGHT, 1)
        self.record_canvas.configure(scrollregion=(0, 0, 1, height))
        self._update_counts()
        self._ensure_current_visible()
        self._schedule_list_refresh()

    def _update_counts(self) -> None:
        if not self.session:
            return
        counts = self.session.counts()
        self.counts_var.set(
            f"Showing {len(self.session.filtered):,} of {len(self.session.dataset):,}  •  "
            f"unreviewed {counts['unreviewed']:,}  boxed {counts['boxed']:,}  "
            f"no bird {counts['no_bird']:,}  skipped {counts['skipped']:,}"
        )
        position = self.session.position + 1 if self.session.current else 0
        self.progress_var.set(f"{position:,} / {len(self.session.filtered):,}")

    def _list_scroll_changed(self, first: str, last: str) -> None:
        self.list_scroll.set(first, last)
        self._schedule_list_refresh()

    def _scroll_list(self, *args: str) -> None:
        self.record_canvas.yview(*args)
        self._schedule_list_refresh()

    def _list_mousewheel(self, event: tk.Event[Any]) -> str:
        button = getattr(event, "num", None)
        direction = -1 if button == 4 or (button not in {4, 5} and event.delta > 0) else 1
        self.record_canvas.yview_scroll(direction * 3, "units")
        self._schedule_list_refresh()
        return "break"

    def _schedule_list_refresh(self) -> None:
        if self.list_job is None:
            self.list_job = self.root.after_idle(self._refresh_visible_rows)

    def _refresh_visible_rows(self) -> None:
        self.list_job = None
        self.record_canvas.delete("row")
        if not self.session:
            return
        width = max(self.record_canvas.winfo_width(), 1)
        rows = visible_row_range(
            len(self.session.filtered), self.record_canvas.canvasy(0),
            self.record_canvas.winfo_height(),
        )
        for index in rows:
            record = self.session.filtered[index]
            review = self.session.reviews.get(record.source_id)
            state = review.review_state if review else "unreviewed"
            selected = index == self.session.position
            fill = "#cce8ff" if selected else ("#e8f5e9" if review else "#f7f7f7")
            top = index * ROW_HEIGHT
            tags = ("row", f"index:{index}")
            self.record_canvas.create_rectangle(
                1, top + 1, width - 2, top + ROW_HEIGHT - 1,
                fill=fill, outline="#b7b7b7" if selected else "", tags=tags,
            )
            marker = {"boxed": "□", "no_bird": "×", "skipped": "–"}.get(state, "·")
            label = f"{marker}  {record.source_id}"
            self.record_canvas.create_text(
                7, top + ROW_HEIGHT / 2, text=label, anchor=tk.W,
                fill="#111111", tags=tags,
            )

    def _record_clicked(self, event: tk.Event[Any]) -> str:
        if not self.session or not self._confirm_discard_draft():
            return "break"
        index = int(self.record_canvas.canvasy(event.y) // ROW_HEIGHT)
        if 0 <= index < len(self.session.filtered):
            self.session.position = index
            self.load_current()
        return "break"

    def _ensure_current_visible(self) -> None:
        if not self.session or not self.session.current:
            return
        top = self.record_canvas.canvasy(0)
        viewport = max(self.record_canvas.winfo_height(), 1)
        row_top = self.session.position * ROW_HEIGHT
        row_bottom = row_top + ROW_HEIGHT
        total = max(len(self.session.filtered) * ROW_HEIGHT, 1)
        if row_top < top:
            self.record_canvas.yview_moveto(row_top / total)
        elif row_bottom > top + viewport:
            self.record_canvas.yview_moveto(max(row_bottom - viewport, 0) / total)

    def load_current(self) -> None:
        self.preview_image = None
        self.photo_image = None
        self.rendered_size = None
        self.image_size = None
        self.transform = None
        self.drag = None
        self.draft_bbox = None
        self.draft_dirty = False
        if not self.session or not self.session.current:
            self.title_var.set("No records match the filters")
            self.details_var.set("")
            self.status_var.set("No matching records")
            self._update_counts() if self.session else None
            self._render()
            self._schedule_list_refresh()
            self._update_button_states()
            return

        record = self.session.current
        path = self.session.dataset.resolve_image_path(record)
        try:
            with Image.open(path) as opened:
                original_size = opened.size
                target = (
                    max(self.root.winfo_screenwidth(), self.image_canvas.winfo_width(), 1),
                    max(self.root.winfo_screenheight(), self.image_canvas.winfo_height(), 1),
                )
                opened.draft("RGB", target)
                preview = opened.convert("RGB")
                preview.thumbnail(target, Image.Resampling.LANCZOS, reducing_gap=3.0)
            self.preview_image = preview
            self.image_size = original_size
        except (OSError, ValueError, MemoryError, Image.DecompressionBombError) as error:
            self.status_var.set(f"Could not open {path}: {error}")

        review = self.session.current_review
        if review and review.bbox:
            self.draft_bbox = review.bbox
        elif record.bbox and self.image_size:
            self.draft_bbox = record.bbox.clipped(*self.image_size)

        confidence = "—" if record.confidence is None else f"{record.confidence:.4f}"
        review_state = review.review_state if review else "unreviewed"
        self.title_var.set(record.source_id)
        self.details_var.set(
            f"Path: {record.relative_path}\n"
            f"Class: {record.class_name or '—'}\n"
            f"Source status: {record.bbox_status or '—'}\n"
            f"Source: {record.bbox_source or '—'}\n"
            f"Confidence: {confidence}\n"
            f"Detections: {record.detection_count if record.detection_count is not None else '—'}\n"
            f"Review: {review_state}\n"
            f"Image: {self.image_size[0]} × {self.image_size[1]}" if self.image_size else
            f"Path: {record.relative_path}\nReview: {review_state}\nImage unavailable"
        )
        self.status_var.set(
            f"Loaded saved {review_state} review" if review else
            "Source box loaded as a draft; explicitly accept it to save a human review"
        )
        self._update_box_text()
        self._update_counts()
        self._ensure_current_visible()
        self._schedule_list_refresh()
        self._update_button_states()
        self._render()

    def _schedule_render(self, _event: tk.Event[Any] | None = None) -> None:
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.render_job = self.root.after(40, self._render)

    def _render(self) -> None:
        self.render_job = None
        canvas = self.image_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        if self.preview_image is None or self.image_size is None:
            canvas.create_text(width / 2, height / 2, text="No image", fill="white",
                               font=("TkDefaultFont", 18))
            return
        self.transform = ImageCanvasTransform.fit(*self.image_size, width, height)
        display_size = (
            max(round(self.transform.display_width), 1),
            max(round(self.transform.display_height), 1),
        )
        if self.photo_image is None or self.rendered_size != display_size:
            displayed = self.preview_image.resize(display_size, Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(displayed)
            self.rendered_size = display_size
        canvas.create_image(width / 2, height / 2, image=self.photo_image)

        record = self.session.current if self.session else None
        if record and record.bbox:
            source_box = self.transform.image_bbox_to_canvas(record.bbox)
            if source_box:
                canvas.create_rectangle(
                    source_box.x, source_box.y, source_box.right, source_box.bottom,
                    outline="#ff9f1c", width=2, dash=(7, 4), tags="source-box",
                )
        if self.draft_bbox:
            review_box = self.transform.image_bbox_to_canvas(self.draft_bbox)
            if review_box:
                canvas.create_rectangle(
                    review_box.x, review_box.y, review_box.right, review_box.bottom,
                    outline="#00e5ff", width=3, tags="review-box",
                )
                for handle_x, handle_y in bbox_handles(review_box).values():
                    canvas.create_rectangle(
                        handle_x - HANDLE_RADIUS, handle_y - HANDLE_RADIUS,
                        handle_x + HANDLE_RADIUS, handle_y + HANDLE_RADIUS,
                        fill="#00e5ff", outline="#001f24", tags="review-box",
                    )

    def _canvas_image_point(
        self, event: tk.Event[Any], *, require_inside: bool = False
    ) -> tuple[float, float] | None:
        if not self.transform:
            return None
        if require_inside and not (
            self.transform.offset_x <= event.x <= self.transform.offset_x + self.transform.display_width
            and self.transform.offset_y <= event.y <= self.transform.offset_y + self.transform.display_height
        ):
            return None
        return self.transform.canvas_to_image(event.x, event.y, clamp=True)

    def _drag_started(self, event: tk.Event[Any]) -> str:
        point = self._canvas_image_point(event, require_inside=True)
        if point is None or self.read_only:
            return "break"
        x, y = point
        mode = None
        if self.draft_bbox and self.transform:
            mode = hit_test_bbox(self.draft_bbox, x, y, HANDLE_RADIUS / self.transform.scale)
        self.drag = DragState(mode or "draw", x, y, self.draft_bbox)
        if mode is None:
            self.draft_bbox = BBox(x, y, 0.0, 0.0)
        return "break"

    def _drag_moved(self, event: tk.Event[Any]) -> str:
        point = self._canvas_image_point(event)
        if not self.drag or point is None or self.image_size is None:
            return "break"
        x, y = point
        width, height = self.image_size
        if self.drag.mode == "draw":
            self.draft_bbox = BBox(
                self.drag.start_x, self.drag.start_y, x - self.drag.start_x, y - self.drag.start_y
            ).normalized()
        elif self.drag.mode == "move" and self.drag.original:
            self.draft_bbox = move_bbox(
                self.drag.original, x - self.drag.start_x, y - self.drag.start_y, width, height
            )
        elif self.drag.original:
            self.draft_bbox = resize_bbox(self.drag.original, self.drag.mode, x, y, width, height)
        self.draft_dirty = True
        self._update_box_text()
        self._update_button_states()
        self._render()
        return "break"

    def _drag_finished(self, event: tk.Event[Any]) -> str:
        if self.drag:
            self._drag_moved(event)
            if self.draft_bbox and (
                self.draft_bbox.width < MIN_BOX_PIXELS or self.draft_bbox.height < MIN_BOX_PIXELS
            ):
                self.draft_bbox = self.drag.original
                self.status_var.set("Ignored a box smaller than one source pixel")
            else:
                self.status_var.set("Draft changed — choose Accept Box to save it")
            self.drag = None
            self._update_box_text()
            self._render()
        return "break"

    def _update_cursor(self, event: tk.Event[Any]) -> None:
        point = self._canvas_image_point(event)
        cursor = "crosshair"
        if point and self.draft_bbox and self.transform:
            hit = hit_test_bbox(
                self.draft_bbox, point[0], point[1], HANDLE_RADIUS / self.transform.scale
            )
            cursor = {
                "move": "fleur", "nw": "top_left_corner", "se": "bottom_right_corner",
                "ne": "top_right_corner", "sw": "bottom_left_corner",
            }.get(hit, "crosshair")
        try:
            self.image_canvas.configure(cursor=cursor)
        except tk.TclError:
            self.image_canvas.configure(cursor="crosshair")

    def _update_box_text(self) -> None:
        if not self.draft_bbox:
            self.box_var.set("Draft: no box")
            return
        box = self.draft_bbox
        self.box_var.set(
            f"Draft box (source pixels)\n"
            f"x {box.x:.2f}   y {box.y:.2f}\n"
            f"width {box.width:.2f}   height {box.height:.2f}"
        )

    def _update_button_states(self) -> None:
        current = self.session.current if self.session else None
        dimensions_known = bool(
            current and (self.image_size or (current.width and current.height))
        )
        decision_state = tk.NORMAL if dimensions_known and not self.read_only else tk.DISABLED
        image_edit_state = tk.NORMAL if self.image_size and not self.read_only else tk.DISABLED
        self.accept_button.configure(
            state=tk.NORMAL if image_edit_state == tk.NORMAL and self.draft_bbox else tk.DISABLED
        )
        for button in (self.no_bird_button, self.skip_button):
            button.configure(state=decision_state)
        self.delete_button.configure(state=image_edit_state)
        self.clear_button.configure(
            state=tk.NORMAL
            if current and not self.read_only and current.source_id in self.session.reviews
            else tk.DISABLED
        )
        self.source_button.configure(
            state=tk.NORMAL
            if image_edit_state == tk.NORMAL and current and current.has_bbox else tk.DISABLED
        )

    def use_source(self) -> None:
        if not self.session or not self.session.current or not self.image_size or self.read_only:
            return
        source = self.session.current.bbox
        self.draft_bbox = source.clipped(*self.image_size) if source else None
        self.draft_dirty = True
        self.status_var.set("Source box copied to draft; choose Accept Box to save it")
        self._update_box_text()
        self._update_button_states()
        self._render()

    def delete_draft(self) -> None:
        if self.read_only:
            return
        self.draft_bbox = None
        self.draft_dirty = True
        self.status_var.set("Draft box removed; saved review is unchanged")
        self._update_box_text()
        self._update_button_states()
        self._render()

    def _save_review(self, state: str) -> bool:
        if self.read_only or not self.session or not self.session.current:
            return False
        dimensions = self.image_size or (
            (self.session.current.width, self.session.current.height)
            if self.session.current.width and self.session.current.height else None
        )
        if dimensions is None:
            messagebox.showerror(
                "Unknown image dimensions",
                "The image could not be opened and the manifest has no source dimensions.",
                parent=self.root,
            )
            return False
        if state == "boxed" and self.image_size is None:
            messagebox.showerror(
                "Image unavailable", "A box cannot be accepted without opening the source image.",
                parent=self.root,
            )
            return False
        bbox = self.draft_bbox if state == "boxed" else None
        if state == "boxed" and (bbox is None or not bbox.is_valid):
            messagebox.showerror("No valid box", "Draw or restore a valid box first.", parent=self.root)
            return False
        source_id = self.session.current.source_id
        previous = self.session.reviews.get(source_id)
        try:
            review = GroundTruthRecord.from_image_record(
                self.session.current, state, bbox,
                source_width=dimensions[0], source_height=dimensions[1],
            )
            self.session.reviews[review.source_id] = review
            if self.ground_truth_path is None:
                raise ValueError("no ground-truth output path is configured")
            with review_file_lock(self.ground_truth_path):
                if file_signature(self.ground_truth_path) != self.ground_truth_signature:
                    raise OSError(
                        "the ground-truth CSV changed outside this application; reload the dataset "
                        "before saving so another reviewer's changes are not overwritten"
                    )
                save_ground_truth(self.ground_truth_path, self.session.reviews)
        except (OSError, ValueError) as error:
            if previous is None:
                self.session.reviews.pop(source_id, None)
            else:
                self.session.reviews[source_id] = previous
            messagebox.showerror("Could not save review", str(error), parent=self.root)
            return False
        self.ground_truth_signature = file_signature(self.ground_truth_path)
        self.session.reviews_changed()
        self.draft_dirty = False
        self.status_var.set(f"Saved {state} to {self.ground_truth_path}")
        self._update_counts()
        self._schedule_list_refresh()
        return True

    def accept_box_and_next(self) -> None:
        if self._save_review("boxed"):
            self._advance_after_commit()

    def commit_and_next(self, state: str) -> None:
        if self._save_review(state):
            self._advance_after_commit()

    def _advance_after_commit(self) -> None:
        if not self.session:
            return
        current_id = self.session.current.source_id if self.session.current else None
        old_position = self.session.position
        self.session.apply_filters(preserve_source_id=current_id)
        # A review-aware filter can remove the row that was just committed.
        if current_id and (not self.session.current or self.session.current.source_id != current_id):
            self.session.position = min(old_position, len(self.session.filtered) - 1)
        elif self.session.position < len(self.session.filtered) - 1:
            self.session.position += 1
        self._refresh_after_filter()
        self.load_current()

    def clear_review(self) -> None:
        if self.read_only or not self.session or not self.session.current:
            return
        if not self._confirm_discard_draft():
            return
        source_id = self.session.current.source_id
        if source_id not in self.session.reviews:
            self.status_var.set("This image has no saved review")
            return
        previous = self.session.reviews.pop(source_id)
        try:
            if self.ground_truth_path is None:
                raise ValueError("no ground-truth output path is configured")
            with review_file_lock(self.ground_truth_path):
                if file_signature(self.ground_truth_path) != self.ground_truth_signature:
                    raise OSError(
                        "the ground-truth CSV changed outside this application; reload the dataset "
                        "before saving so another reviewer's changes are not overwritten"
                    )
                save_ground_truth(self.ground_truth_path, self.session.reviews)
        except (OSError, ValueError) as error:
            self.session.reviews[source_id] = previous
            messagebox.showerror("Could not save review", str(error), parent=self.root)
            return
        self.ground_truth_signature = file_signature(self.ground_truth_path)
        self.session.reviews_changed()
        self.status_var.set("Saved review cleared")
        self.session.apply_filters(preserve_source_id=source_id)
        self._refresh_after_filter()
        self.load_current()

    def _navigate(self, offset: int) -> None:
        if not self.session or not self._confirm_discard_draft():
            return
        if not self.session.navigate(offset):
            self.status_var.set("Already at the first image" if offset < 0 else "Already at the last image")
            return
        self.load_current()

    def previous(self) -> None:
        self._navigate(-1)

    def next(self) -> None:
        self._navigate(1)

    def next_unreviewed(self) -> None:
        if not self.session or not self._confirm_discard_draft():
            return
        if self.session.next_unreviewed():
            self.load_current()
        else:
            self.status_var.set("No other unreviewed images in the current filtered list")

    def _confirm_discard_draft(self) -> bool:
        if not self.draft_dirty:
            return True
        discard = messagebox.askyesno(
            "Unsaved draft",
            "The current box draft has not been accepted. Discard it?",
            parent=self.root,
        )
        if discard:
            self.draft_dirty = False
        return discard

    def close(self) -> None:
        if self._confirm_discard_draft():
            self._cancel_pending_jobs()
            self.root.destroy()

    def _cancel_pending_jobs(self) -> None:
        for name in ("filter_job", "render_job", "list_job"):
            job = getattr(self, name)
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except tk.TclError:
                    pass
                setattr(self, name, None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", type=Path,
        help="generated bbox manifest CSV or raw CUB_200_2011 directory",
    )
    parser.add_argument(
        "--image-root", type=Path,
        help="portable override for the directory directly containing class folders",
    )
    parser.add_argument(
        "--ground-truth", type=Path,
        help="review CSV to load/save (default: beside the source manifest)",
    )
    parser.add_argument("--read-only", action="store_true", help="disable review changes")
    parser.add_argument("--start", metavar="SOURCE_ID", help="stable source ID to select initially")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    root = tk.Tk()
    app = BBoxViewerApp(
        root,
        arguments.source,
        image_root=arguments.image_root,
        ground_truth_path=arguments.ground_truth,
        read_only=arguments.read_only,
        start_source_id=arguments.start,
    )
    root._bbox_viewer_app = app  # type: ignore[attr-defined]
    if arguments.source is None:
        root.after(100, app.choose_manifest)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())