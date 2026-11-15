#!/usr/bin/env python3
"""Copy or move confirmed captures and their JSON sidecars with robocopy.

Confirmation can come from either:

* ``bird_annotations.csv`` rows whose review state is ``boxed`` or ``no_bird``;
  use ``--annotations`` to select this mode. Multiple box rows for one capture
  are deduplicated.
* A valid ``manual_label.bird_present`` boolean in each JSON sidecar, retained
  as the backwards-compatible default when ``--annotations`` is not supplied.

The annotation CSV locates images, while each same-stem capture JSON supplies
the original metadata and optional SHA-256. Both files are copied or moved.

Usage (Windows):
    python copy_confirmed.py C:\\Users\\Joe\\Pictures\\ESP32Cam E:\\ESP32Cam_confirmed --annotations --dry-run
    python copy_confirmed.py C:\\Users\\Joe\\Pictures\\ESP32Cam E:\\ESP32Cam_confirmed --annotations --move --verify
    python copy_confirmed.py C:\\ESP32Cam E:\\ESP32Cam_confirmed --manual-labels

Options:
    --annotations [CSV]  use the detector annotation CSV; without a value, use
                         bird_annotations.csv beside this script
    --manual-labels      explicitly use labels stored inside JSON sidecars
    --dry-run            list what robocopy would do, copy or move nothing
    --verify             re-hash destination images after the copy/move
    --move               delete source files after a successful copy (robocopy /MOV)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
CONFIRMED_REVIEW_STATES = frozenset({"boxed", "no_bird"})
DEFAULT_ANNOTATIONS = Path(__file__).with_name("bird_annotations.csv")
MAX_ARG_CHARS = 20000


@dataclass(frozen=True)
class ConfirmedCapture:
    image_path: Path
    sidecar_path: Path
    expected_sha256: str


@dataclass
class DiscoveryResult:
    captures: list[ConfirmedCapture]
    sidecars_scanned: int = 0
    unlabeled_sidecars: int = 0
    invalid_sidecars: int = 0
    missing_images: list[Path] = field(default_factory=list)
    missing_sidecars: list[Path] = field(default_factory=list)
    invalid_sidecar_paths: list[Path] = field(default_factory=list)
    annotation_rows: int = 0
    duplicate_annotation_rows: int = 0
    ignored_annotation_rows: int = 0


@dataclass(frozen=True)
class SelectedCopyResult:
    """Summary returned after copying an explicit list of relative file paths."""

    files_queued: int
    robocopy_calls: int
    verified_files: int
    dry_run: bool


def chunk_names(names: Iterable[str]) -> Iterable[list[str]]:
    """Split file names into batches that fit on one command line."""
    batch: list[str] = []
    size = 0
    for name in names:
        cost = len(name) + 3  # quotes + space
        if batch and size + cost > MAX_ARG_CHARS:
            yield batch
            batch, size = [], 0
        batch.append(name)
        size += cost
    if batch:
        yield batch


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError("the JSON root must be an object")
    return value


def has_manual_label(data: dict[str, Any]) -> bool:
    """Match image_labeler.py's definition of a completed manual label."""
    label = data.get("manual_label")
    return isinstance(label, dict) and type(label.get("bird_present")) is bool


def find_image(sidecar_path: Path, data: dict[str, Any]) -> Path | None:
    """Find the image named by a sidecar, falling back to its matching stem."""
    filename = data.get("filename")
    if isinstance(filename, str) and Path(filename).name == filename:
        image_path = sidecar_path.with_name(filename)
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            return image_path

    candidates = [
        path
        for path in sidecar_path.parent.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem.casefold() == sidecar_path.stem.casefold()
    ]
    return candidates[0] if len(candidates) == 1 else None


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _safe_relative_path(value: str) -> Path | None:
    """Return a platform-independent safe relative path from a CSV field."""

    text = value.strip().replace("\\", "/")
    if not text:
        return None
    portable = PurePosixPath(text)
    if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
        return None
    if portable.parts and portable.parts[0].endswith(":"):
        return None
    return Path(*portable.parts)


def _resolve_annotation_image(row: Mapping[str, str], source_root: Path) -> Path | None:
    """Resolve one detector-manifest row to an image below ``source_root``."""

    relative = _safe_relative_path(row.get("relative_path", ""))
    if relative is not None:
        candidate = (source_root / relative).resolve()
        if (
            is_within(candidate, source_root)
            and candidate.is_file()
            and candidate.suffix.casefold() in IMAGE_EXTENSIONS
        ):
            return candidate

    stored_text = row.get("image_path", "").strip()
    if stored_text:
        stored = Path(stored_text).expanduser()
        if stored.is_absolute():
            candidate = stored.resolve()
            if (
                is_within(candidate, source_root)
                and candidate.is_file()
                and candidate.suffix.casefold() in IMAGE_EXTENSIONS
            ):
                return candidate

        filename = Path(stored_text.replace("\\", "/")).name
        if filename and filename not in {".", ".."}:
            candidate = (source_root / filename).resolve()
            if (
                is_within(candidate, source_root)
                and candidate.is_file()
                and candidate.suffix.casefold() in IMAGE_EXTENSIONS
            ):
                return candidate

    sample_id = row.get("sample_id", "").strip()
    if sample_id and Path(sample_id).name == sample_id:
        matches = sorted(
            (
                path
                for path in source_root.rglob(f"{sample_id}.*")
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=lambda path: str(path).casefold(),
        )
        if len(matches) == 1:
            return matches[0].resolve()
    return None


def discover_confirmed_from_annotations(
    annotations_path: Path,
    source_root: Path,
    excluded_root: Path | None = None,
) -> DiscoveryResult:
    """Discover unique confirmed captures selected by a detector annotation CSV."""

    annotations = annotations_path.expanduser().resolve()
    if not annotations.is_file():
        raise FileNotFoundError(f"Annotation CSV does not exist: {annotations}")

    result = DiscoveryResult(captures=[])
    seen: dict[str, tuple[str, str]] = {}
    with annotations.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Annotation CSV has no header: {annotations}")
        required = {"sample_id", "review_state", "image_path", "relative_path"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(
                f"Annotation CSV is missing required column(s): {', '.join(missing)}"
            )

        for line_number, row in enumerate(reader, 2):
            result.annotation_rows += 1
            review_state = (row.get("review_state") or "").strip().casefold()
            if review_state not in CONFIRMED_REVIEW_STATES:
                result.ignored_annotation_rows += 1
                continue

            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"{annotations}:{line_number}: missing sample_id")
            key = sample_id.casefold()
            identity = (
                (row.get("relative_path") or "").strip().casefold(),
                (row.get("image_path") or "").strip().casefold(),
            )
            previous = seen.get(key)
            if previous is not None:
                if previous != identity:
                    raise ValueError(
                        f"{annotations}:{line_number}: inconsistent paths for sample_id {sample_id!r}"
                    )
                result.duplicate_annotation_rows += 1
                continue
            seen[key] = identity

            image_path = _resolve_annotation_image(row, source_root)
            if image_path is None:
                requested = (row.get("relative_path") or row.get("image_path") or sample_id).strip()
                result.missing_images.append(source_root / requested)
                continue
            if excluded_root is not None and is_within(image_path, excluded_root):
                continue

            sidecar_path = image_path.with_suffix(".json")
            if not sidecar_path.is_file():
                result.missing_sidecars.append(sidecar_path)
                continue

            result.sidecars_scanned += 1
            try:
                data = read_json_object(sidecar_path)
            except (OSError, ValueError, json.JSONDecodeError):
                result.invalid_sidecars += 1
                result.invalid_sidecar_paths.append(sidecar_path)
                continue

            sidecar_image = find_image(sidecar_path, data)
            if sidecar_image is None or sidecar_image.resolve() != image_path:
                result.missing_images.append(sidecar_path)
                continue

            expected_sha256 = data.get("sha256")
            if not isinstance(expected_sha256, str):
                expected_sha256 = ""
            result.captures.append(
                ConfirmedCapture(
                    image_path=image_path,
                    sidecar_path=sidecar_path,
                    expected_sha256=expected_sha256.strip().casefold(),
                )
            )

    result.captures.sort(key=lambda capture: str(capture.image_path).casefold())
    return result


def copy_selected_files(
    source_root: str | Path,
    dest_root: str | Path,
    relative_paths: Iterable[str | Path],
    *,
    dry_run: bool = False,
    verify: bool = False,
) -> SelectedCopyResult:
    """Copy an explicit file selection while preserving directories below a root.

    This is the reusable counterpart to the confirmed-sidecar CLI.  It is useful
    when another trusted source, such as a dataset manifest, has already selected
    the files.  Relative paths must remain below ``source_root``; missing files and
    unsafe paths fail before robocopy is started.
    """
    source = Path(source_root).expanduser().resolve()
    destination = Path(dest_root).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Source folder does not exist or is not a directory: {source}")
    if source == destination:
        raise ValueError("Source and destination folders must be different.")

    selected: dict[str, tuple[Path, Path]] = {}
    for value in relative_paths:
        relative = Path(value)
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Path must be relative and remain below the source root: {value}")

        source_path = (source / relative).resolve()
        if not is_within(source_path, source):
            raise ValueError(f"Path escapes the source root: {value}")
        if not source_path.is_file():
            raise FileNotFoundError(f"Selected source file does not exist: {source_path}")

        destination_path = destination / relative
        key = str(relative).casefold()
        previous = selected.get(key)
        if previous is not None and previous[0] != source_path:
            raise ValueError(f"Case-insensitive destination path collision: {relative}")
        selected[key] = (source_path, destination_path)

    if not selected:
        raise ValueError("At least one relative file path must be selected.")

    groups: dict[tuple[Path, Path], list[str]] = defaultdict(list)
    for source_path, destination_path in selected.values():
        groups[(source_path.parent, destination_path.parent)].append(source_path.name)

    failed = False
    robocopy_calls = 0
    for (source_dir, destination_dir), names in groups.items():
        for batch in chunk_names(sorted(set(names), key=str.casefold)):
            command = [
                "robocopy",
                str(source_dir),
                str(destination_dir),
                *batch,
                "/COPY:DAT",
                "/R:2",
                "/W:2",
                "/NP",
                "/NJH",
                "/NDL",
            ]
            if dry_run:
                command.append("/L")
            result = subprocess.run(command, check=False)
            robocopy_calls += 1
            if result.returncode >= 8:
                failed = True

    if failed:
        raise RuntimeError("robocopy reported at least one file-copy failure")

    verified_files = 0
    if verify and not dry_run:
        failures: list[str] = []
        for source_path, destination_path in selected.values():
            if not destination_path.is_file():
                failures.append(f"missing destination: {destination_path}")
            elif sha256_of(source_path) != sha256_of(destination_path):
                failures.append(f"hash mismatch: {destination_path}")
            else:
                verified_files += 1
        if failures:
            details = "\n".join(f"  {failure}" for failure in failures[:20])
            raise RuntimeError(f"Copy verification failed for {len(failures)} file(s):\n{details}")

    return SelectedCopyResult(
        files_queued=len(selected),
        robocopy_calls=robocopy_calls,
        verified_files=verified_files,
        dry_run=dry_run,
    )


def discover_confirmed(source_root: Path, excluded_root: Path | None = None) -> DiscoveryResult:
    """Discover recursively all sidecars containing completed manual labels."""
    result = DiscoveryResult(captures=[])
    sidecars = sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".json"
        ),
        key=lambda path: str(path).casefold(),
    )
    for sidecar_path in sidecars:
        if excluded_root is not None and is_within(sidecar_path, excluded_root):
            continue
        if sidecar_path.name.casefold() == "manifest.json":
            continue

        result.sidecars_scanned += 1
        try:
            data = read_json_object(sidecar_path)
        except (OSError, ValueError, json.JSONDecodeError):
            result.invalid_sidecars += 1
            continue

        if not has_manual_label(data):
            result.unlabeled_sidecars += 1
            continue

        image_path = find_image(sidecar_path, data)
        if image_path is None:
            result.missing_images.append(sidecar_path)
            continue

        expected_sha256 = data.get("sha256")
        if not isinstance(expected_sha256, str):
            expected_sha256 = ""
        result.captures.append(
            ConfirmedCapture(
                image_path=image_path,
                sidecar_path=sidecar_path,
                expected_sha256=expected_sha256.strip().lower(),
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="folder containing images and JSON sidecars")
    parser.add_argument("dest", help="destination folder, e.g. E:\\ESP32Cam_confirmed")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--annotations",
        nargs="?",
        const=DEFAULT_ANNOTATIONS,
        type=Path,
        metavar="CSV",
        help=(
            "select boxed/no_bird captures from a detector annotation CSV; "
            f"default when omitted: {DEFAULT_ANNOTATIONS}"
        ),
    )
    selection.add_argument(
        "--manual-labels",
        action="store_true",
        help="select sidecars containing manual_label.bird_present (legacy mode)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--move", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    dest_root = Path(args.dest).expanduser().resolve()
    if not source_root.is_dir():
        sys.exit(f"Source folder does not exist or is not a directory: {source_root}")
    if source_root == dest_root:
        sys.exit("Source and destination folders must be different.")

    excluded_root = dest_root if is_within(dest_root, source_root) else None
    try:
        if args.annotations is not None:
            annotations_path = args.annotations.expanduser().resolve()
            discovery = discover_confirmed_from_annotations(
                annotations_path, source_root, excluded_root
            )
            print(f"Annotations:           {annotations_path}")
            print(
                f"CSV rows: {discovery.annotation_rows}   "
                f"duplicate box rows: {discovery.duplicate_annotation_rows}   "
                f"ignored states: {discovery.ignored_annotation_rows}"
            )
            print(
                f"Confirmed captures: {len(discovery.captures)}   "
                f"missing images: {len(discovery.missing_images)}   "
                f"missing sidecars: {len(discovery.missing_sidecars)}   "
                f"invalid sidecars: {discovery.invalid_sidecars}"
            )
        else:
            discovery = discover_confirmed(source_root, excluded_root)
            print(
                f"JSON sidecars scanned: {discovery.sidecars_scanned}   "
                f"confirmed labels: {len(discovery.captures)}"
            )
            print(
                f"Unlabeled: {discovery.unlabeled_sidecars}   "
                f"invalid JSON: {discovery.invalid_sidecars}   "
                f"labeled but missing image: {len(discovery.missing_images)}"
            )
    except (OSError, ValueError) as error:
        sys.exit(str(error))
    confirmed = discovery.captures

    if not confirmed:
        sys.exit("No confirmed image/sidecar pairs were found.")

    # Group files by (source dir, destination dir) so each robocopy call covers one folder.
    groups: dict[tuple[Path, Path], list[str]] = defaultdict(list)
    copied_images: list[tuple[Path, str]] = []
    for capture in confirmed:
        relative_dir = capture.sidecar_path.parent.relative_to(source_root)
        dest_dir = dest_root / relative_dir
        groups[(capture.image_path.parent, dest_dir)].append(capture.image_path.name)
        groups[(capture.sidecar_path.parent, dest_dir)].append(capture.sidecar_path.name)

        expected_sha256 = capture.expected_sha256
        if args.verify and not expected_sha256:
            expected_sha256 = sha256_of(capture.image_path)
        copied_images.append((dest_dir / capture.image_path.name, expected_sha256))

    failed = False
    for (src_dir, dest_dir), names in groups.items():
        for batch in chunk_names(sorted(set(names), key=str.casefold)):
            command = [
                "robocopy", str(src_dir), str(dest_dir), *batch,
                "/COPY:DAT", "/R:2", "/W:2", "/NP", "/NJH", "/NDL",
            ]
            if args.dry_run:
                command.append("/L")
            if args.move:
                command.append("/MOV")
            result = subprocess.run(command, check=False)
            # Robocopy exit codes are a bitmask: 0-7 = success, 8+ = at least one failure.
            if result.returncode >= 8:
                failed = True
                print(f"!! robocopy reported failures (exit {result.returncode}) for {src_dir}")

    print("\n--- Summary ---")
    print(f"Confirmed pairs:       {len(confirmed)}")
    print(f"Images queued:         {len(copied_images)}")
    print(f"Missing images:        {len(discovery.missing_images)}")
    print(f"Missing sidecars:      {len(discovery.missing_sidecars)}")
    print(f"Invalid sidecars:      {discovery.invalid_sidecars}")
    for path in discovery.missing_images[:20]:
        print(f"  missing or mismatched image: {path}")
    for path in discovery.missing_sidecars[:20]:
        print(f"  missing sidecar: {path}")
    for path in discovery.invalid_sidecar_paths[:20]:
        print(f"  invalid sidecar: {path}")

    if args.dry_run:
        print("\nDry run only - nothing was copied.")
        if failed:
            sys.exit(1)
        return

    if args.verify:
        bad = 0
        for path, expected in copied_images:
            if not path.is_file():
                bad += 1
                print(f"  not at destination: {path}")
            elif sha256_of(path) != expected:
                bad += 1
                print(f"  hash mismatch:      {path}")
        print(f"Verification:          {len(copied_images) - bad} ok, {bad} bad")
        failed = failed or bad > 0

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()