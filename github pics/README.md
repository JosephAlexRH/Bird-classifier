# Bird capture annotation exporter

`render_bird_captures.py` makes a clean, presentation-ready copy of each
capture. It draws only:

- every valid bird bounding box in the matching `.inference.json` sidecar;
- that box's own predicted species; and
- that box's own species confidence.

It does not add a card, title, footer, capture metadata, motion crop, configured
ROI, or capture ID. The output keeps the capture's original pixel dimensions.

The script pairs files by stem:

```text
capture_....jpg
capture_....inference.json
```

Modern sidecars use `species.boxes`. Each label comes from the same box's
`top_k[0]`, so captures with multiple birds can show different predictions and
confidence values. If an older sidecar has only `localizer.detections`, the
script can still draw those detections with their recorded score.

## Requirements

The exporter requires Pillow:

```powershell
python -m pip install -r .\server\requirements.txt
```

## Basic use

From the `github pics` directory:

```powershell
# Export every capture to .\exported\
python .\render_bird_captures.py --overwrite

# Export one capture.
python .\render_bird_captures.py .\capture_20261120T173411_907031Z_6f224845.jpg --overwrite

# Use the larger sample collection.
python .\render_bird_captures.py ..\sample --output .\sample_exports --overwrite
```

## Change colors and styles

Colors accept Pillow/CSS names, hex colors, and alpha hex colors such as
`#081218e6`. Use `transparent` or `none` for a transparent color.

```powershell
# Cyan dashed boxes with pill labels.
python .\render_bird_captures.py ..\sample `
  --box-color "#00e5ff" --box-style dashed --box-width 4 `
  --dash-length 12 --dash-gap 7 `
  --label-style pill --label-background-color "#001820dd" `
  --label-border-color "#00e5ff" --overwrite

# Minimal white corner marks and plain text with a black stroke.
python .\render_bird_captures.py ..\sample `
  --box-color white --box-style corners --box-width 3 `
  --label-style text --label-text-color white `
  --text-stroke-width 2 --text-stroke-color black --overwrite

# Custom font, placement, text format, and PNG output.
python .\render_bird_captures.py ..\sample `
  --font "C:\Windows\Fonts\arial.ttf" --font-size 20 `
  --label-position below `
  --label-template "{prediction} ({confidence})" `
  --confidence-decimals 0 --format png --overwrite
```

Available label-template fields are:

- `{prediction}` — species predicted for this box;
- `{confidence}` — species confidence formatted as a percentage;
- `{confidence_value}` — species confidence as a decimal value; and
- `{detector_confidence}` — localizer confidence formatted as a percentage.

Run the following for every option and its default:

```powershell
python .\render_bird_captures.py --help
```

Return to the [project overview](../README.md).