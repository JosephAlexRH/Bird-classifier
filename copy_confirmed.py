#!/usr/bin/env python3
"""
Copy manually labeled images and their JSON sidecars to another drive using robocopy.

The source folder is scanned recursively. A capture is confirmed when its JSON
sidecar contains a valid ``manual_label.bird_present`` boolean, as written by
image_labeler.py. Both bird and no-bird labels count as confirmed.

Usage (Windows):
    python copy_confirmed.py C:\\ESP32Cam E:\\ESP32Cam_confirmed --dry-run
    python copy_confirmed.py C:\\ESP32Cam E:\\ESP32Cam_confirmed
    python copy_confirmed.py C:\\ESP32Cam E:\\ESP32Cam_confirmed --verify

Options:
    --dry-run  list what robocopy would do, copy nothing
    --verify   re-hash copied images against the sidecar's sha256 value
    --move     delete source files after a successful copy (robocopy /MOV)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
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
    discovery = discover_confirmed(source_root, excluded_root)
    confirmed = discovery.captures

    print(
        f"JSON sidecars scanned: {discovery.sidecars_scanned}   "
        f"confirmed labels: {len(confirmed)}"
    )
    print(
        f"Unlabeled: {discovery.unlabeled_sidecars}   "
        f"invalid JSON: {discovery.invalid_sidecars}   "
        f"labeled but missing image: {len(discovery.missing_images)}"
    )
    if not confirmed:
        sys.exit("No manually labeled image/sidecar pairs were found.")

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
    for sidecar_path in discovery.missing_images[:20]:
        print(f"  no image for sidecar: {sidecar_path}")

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