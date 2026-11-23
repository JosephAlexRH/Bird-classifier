# Dataset and deployment utilities

The `scripts/` directory contains desktop dataset tools and Raspberry Pi data
management helpers. Run the examples below from the repository root unless a
section says otherwise.

Some utilities copy, move, archive, or erase capture data. Read their `--help`
output and take a verified backup before using a destructive option.

## Bounding-box and label review

### `bbox_viewer.py`

`bbox_viewer.py` is a Tkinter/Pillow multi-box editor for:

- a detector or bounding-box manifest CSV;
- an image directory, with or without an existing annotation CSV; or
- an unmodified `CUB_200_2011` directory.

Human decisions (`boxed`, `no_bird`, `unusable`, or `skipped`) are atomically
saved to a resumable CSV. Use `unusable` for ambiguous or noisy images that
should be excluded even when they may contain a bird. A boxed image is held as
a list of boxes in memory and written as one CSV row per box, repeating the
image's source ID and metadata.

Install Pillow through the server requirements, then run:

```powershell
python -m pip install -r .\server\requirements.txt
python .\scripts\bbox_viewer.py `
  ".\Notebooks\Outputs\iNaturalist Outputs\Bounding Box Outputs\inaturalist_bird_bbox_manifest.csv" `
  --image-root "E:\Datasets\iNaturalist\train_mini\train_mini"
```

If the paths stored in the manifest are valid on the current machine,
`--image-root` can be omitted. Reviews default to
`<manifest-name>_ground_truth.csv` beside the source manifest; use
`--ground-truth` to select another file.

Machine-generated boxes are deliberately not copied into a new review draft.
Cyan boxes are saved or newly drawn human-review boxes. Drag empty image space
to add a box, drag inside a box to select or move it, and drag a selected corner
to resize it. Shift-drag adds a box that overlaps an existing box. Delete
removes the selected box.

**Delete Picture + JSON** (or Shift+Delete) immediately deletes the current
image and its same-stem JSON sidecar without a confirmation dialog, then moves
to the next image. Use this action carefully.

```powershell
python .\scripts\bbox_viewer.py --help
```

Quote paths containing spaces. In PowerShell, a backtick (`` ` ``), not a
backslash, continues a command on the next line. For the untouched iNaturalist
mini training set, `--image-root` must be the directory directly containing the
numbered class folders.

### `image_labeler.py`

`image_labeler.py` is a compatibility entry point for the same combined label
and bounding-box editor. It is convenient when starting from a capture folder:

```powershell
python .\scripts\image_labeler.py E:\Datasets\ESP32_environment
```

If the directory has no annotation manifest, the editor discovers its images
and creates `bird_annotations.csv` on the first save.

## Confirmed-capture transfer

### `copy_confirmed.py`

Copies or moves reviewed captures, acquisition JSON, and matching inference,
motion, and review sidecars. Review state can come from `bird_annotations.csv`
or from the older `manual_label.bird_present` field.

Start with a dry run:

```powershell
python .\scripts\copy_confirmed.py `
  C:\Users\Joe\Pictures\ESP32Cam `
  E:\ESP32Cam_confirmed `
  --annotations `
  --dry-run
```

`--move` deletes source files after a successful Robocopy operation.
`--update-dataset` also updates the destination annotation and inventory files.
Use `--verify` to re-hash destination images.

## Dataset migration

### `merge_esp32_datasets.py`

Converts legacy 160×120 ESP32 annotations into the current tight-box schema and
merges them with a newer annotation CSV. It requires NumPy and pandas.

```powershell
python .\scripts\merge_esp32_datasets.py `
  --old .\legacy_annotations.csv `
  --new .\bird_annotations.csv `
  --old-image-dir E:\Datasets\legacy_captures `
  --out .\combined_annotations.csv
```

The script verifies the legacy context crops, recomputes normalized geometry,
and assigns time-based split groups to reduce train/validation leakage.

## Raspberry Pi backup and archival tools

These scripts are intended for the deployment workflows in `deploy/` and have
strong assumptions about the Pi user, mount point, service, and USB UUID. Do not
run them against a different deployment without reviewing the source.

### `backup_pi_usb.py`

Creates a verified incremental copy of a live `bird_images` tree on a specific
mounted USB volume. It preserves replaced destination versions and never deletes
source files. This is a live backup, not a consistent end-of-run snapshot.

```bash
python scripts/backup_pi_usb.py \
  --source ~/bird_images \
  --mount /mnt/bird-backup \
  --uuid YOUR-USB-UUID
```

Use `--check-only` to validate the mounted volume without copying data.

### `archive_pi_deployment.py`

Builds SHA-256 inventories, verifies copied trees, seals archive directories,
and provides a limited HTTP rescue mode when SSH access is unavailable.

```bash
python scripts/archive_pi_deployment.py inventory \
  --source ~/bird_images \
  --output ~/deployment_exports/run-archive \
  --project ~/birdserver
```

The HTTP rescue cannot claim to be a complete filesystem archive when the old
API cannot return every original sidecar, crop, profile, manifest, and run log.
Prefer the SSH archive workflow driven by `deploy/archive_pi_deployment.ps1`
when that script is present in the working tree.

### `clean_redeploy_pi.py`

This is a destructive, deployment-specific reset tool. It erases archived Pi
capture trees only after every target matches supplied SHA-256 inventories,
then starts a fresh deployment while preserving camera and dashboard settings.
It currently expects the configured project user, `birdhub.service`, the
collection-mode override, `/mnt/bird-backup`, and the USB UUID embedded in the
script. Review and test the complete archival workflow before using it.

## Related documentation

- [Project overview](../README.md)
- [Raspberry Pi server and deployment guide](../server/README.md)
- [Camera firmware guide](../docs/camera-firmware.md)