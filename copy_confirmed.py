#!/usr/bin/env python3
"""
Copy labeled images and their sidecars from a manifest to another drive using robocopy.

Usage (Windows):
    python copy_confirmed.py manifest.tsv E:\\ESP32Cam_confirmed --dry-run
    python copy_confirmed.py manifest.tsv E:\\ESP32Cam_confirmed
    python copy_confirmed.py manifest.tsv E:\\ESP32Cam_confirmed --verify

Options:
    --column   manifest column to filter on        (default: manual_label_status)
    --status   value(s) that count as confirmed    (default: labeled)
    --dry-run  list what robocopy would do, copy nothing
    --verify   re-hash copied images against image_sha256 in the manifest
    --move     delete source files after a successful copy (robocopy /MOV)
"""

import argparse
import csv
import hashlib
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath


MAX_ARG_CHARS = 20000


def read_manifest(path):
    """Read a CSV/TSV manifest"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = f.readline()
        delimiter = "\t" if header.count("\t") >= header.count(",") else ","
        f.seek(0)
        return list(csv.DictReader(f, delimiter=delimiter))


def chunk_names(names):
    """Split file names into batches that fit on one command line."""
    batch, size = [], 0
    for name in names:
        cost = len(name) + 3  # quotes + space
        if batch and size + cost > MAX_ARG_CHARS:
            yield batch
            batch, size = [], 0
        batch.append(name)
        size += cost
    if batch:
        yield batch


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="path to the manifest (.tsv or .csv)")
    ap.add_argument("dest", help="destination folder on the SSD, e.g. E:\\ESP32Cam_confirmed")
    ap.add_argument("--column", default="manual_label_status")
    ap.add_argument("--status", nargs="+", default=["labeled"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    if not rows:
        sys.exit("Manifest is empty.")
    if args.column not in rows[0]:
        sys.exit(f"Column '{args.column}' not in manifest. Columns: {', '.join(rows[0].keys())}")

    wanted = {s.strip().lower() for s in args.status}
    confirmed = [r for r in rows if (r[args.column] or "").strip().lower() in wanted]

    print(f"Manifest rows: {len(rows)}   matching {args.column} in {sorted(wanted)}: {len(confirmed)}")
    if not confirmed:
        counts = Counter((r[args.column] or "").strip() or "<blank>" for r in rows)
        print(f"\nNothing matched. Values found in '{args.column}':")
        for value, n in counts.most_common():
            print(f"  {value}: {n}")
        sys.exit("\nRe-run with --status <value> (and --column <name> if needed).")

    dest_root = Path(args.dest)

    # Group files by (source dir, destination dir) so each robocopy call covers one folder.
    groups = defaultdict(list)
    missing_images, missing_sidecars = [], []
    copied_images = []  # (dest path, expected sha256)

    for r in confirmed:
        image = Path(r["image_path"])
        if not image.is_file():
            missing_images.append(str(image))
            continue

        # Keep any sub-folder structure from image_relative_path.
        rel = PureWindowsPath(r.get("image_relative_path") or image.name)
        dest_dir = dest_root.joinpath(*rel.parent.parts)
        groups[(image.parent, dest_dir)].append(image.name)
        copied_images.append((dest_dir / image.name, (r.get("image_sha256") or "").strip().lower()))

        sidecar_field = (r.get("sidecar_path") or "").strip()
        if sidecar_field:
            sidecar = Path(sidecar_field)
            if sidecar.is_file():
                srel = PureWindowsPath(r.get("sidecar_relative_path") or sidecar.name)
                sdest = dest_root.joinpath(*srel.parent.parts)
                groups[(sidecar.parent, sdest)].append(sidecar.name)
            else:
                missing_sidecars.append(str(sidecar))

    failed = False
    for (src_dir, dest_dir), names in groups.items():
        names = sorted(set(names))
        for batch in chunk_names(names):
            cmd = ["robocopy", str(src_dir), str(dest_dir), *batch,
                   "/COPY:DAT", "/R:2", "/W:2", "/NP", "/NJH", "/NDL"]
            if args.dry_run:
                cmd.append("/L")
            if args.move:
                cmd.append("/MOV")
            result = subprocess.run(cmd)
            # Robocopy exit codes are a bitmask: 0-7 = success, 8+ = at least one failure.
            if result.returncode >= 8:
                failed = True
                print(f"!! robocopy reported failures (exit {result.returncode}) for {src_dir}")

    print("\n--- Summary ---")
    print(f"Confirmed rows:        {len(confirmed)}")
    print(f"Images queued:         {len(copied_images)}")
    print(f"Missing images:        {len(missing_images)}")
    print(f"Missing sidecars:      {len(missing_sidecars)}")
    for p in missing_images[:20]:
        print(f"  missing image:   {p}")
    for p in missing_sidecars[:20]:
        print(f"  missing sidecar: {p}")

    if args.dry_run:
        print("\nDry run only - nothing was copied.")
        return

    # Save the filtered manifest next to the copied files.
    out_manifest = dest_root / "manifest_confirmed.csv"
    dest_root.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(confirmed)
    print(f"Filtered manifest:     {out_manifest}")

    if args.verify:
        bad = 0
        for path, expected in copied_images:
            if not path.is_file():
                bad += 1
                print(f"  not at destination: {path}")
            elif expected and sha256_of(path) != expected:
                bad += 1
                print(f"  hash mismatch:      {path}")
        print(f"Verification:          {len(copied_images) - bad} ok, {bad} bad")
        failed = failed or bad > 0

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
