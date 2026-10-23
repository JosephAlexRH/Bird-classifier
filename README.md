# ESP32-CAM Firmware

Firmware and local-network receiver tools for a bird-feeder monitoring system. The ESP32-CAM watches for motion using low-resolution luminance samples decoded from JPEG frames, captures a full-resolution JPEG when something moves, and pushes it to the included laptop or Raspberry Pi receiver.

Motion ownership defaults to the Raspberry Pi. In this mode the Pi polls fresh
JPEGs from the ESP32 and owns motion decisions; the ESP32 remains a manual,
exclusive camera endpoint and does not run its local detector or flush legacy
buffered motion uploads. Set `motion_owner` to `firmware` only as an explicit
fallback to the on-device detector.

The Pi keeps a per-camera running median of confident not-bird grayscale frames.
It removes global brightness shift, thresholds the full-resolution residual,
and uses the largest unambiguous changed component. The final crop receives 20%
total context and is at least 160×160 pixels; cold, noisy, broad, or ambiguous
frames fall back to the capture-time static ROI. Quiet polls remain memory-only.
Clear-blob events and birds are stored with immutable `.motion.json` and
`.inference.json` provenance beside the JPEG.

## Responsibilities

- Connects the ESP32-CAM to Wi-Fi
- Lets the Raspberry Pi poll fresh full-resolution JPEGs and detect/localize motion
- Retains the low-resolution firmware detector, burst upload, and SD buffering as a manual fallback
- Classifies selected Pi crops while keeping the species model on the whole frame
- Saves the JPEG under `/birds` on microSD only when the Pi explicitly confirms a bird
- Exposes HTTP endpoints for on-demand capture, live config, and status
- Loads and persists configuration on a microSD card, so settings survive reflashing
- Includes a standalone laptop job for one-shot or interval environment-dataset captures

## Hardware

- AI Thinker ESP32-CAM
- OV2640 camera
- ESP32-CAM-MB programmer (or an FTDI adapter wired to the UART pins)
- FAT32-formatted microSD card
- Stable 5V power source (brownouts during Wi-Fi TX are the most common cause of random resets)

## Setup

1. Install [PlatformIO](https://platformio.org/) (VS Code extension or CLI).
2. Clone the repository.
3. Copy `data/config.example.json` to the root of a FAT32 microSD card as
   `config.json`. Set `wifi_ssid` and `wifi_password` for a **2.4 GHz** network,
   and set `server_ip` to the LAN address of the machine running the receiver.
   The example contains placeholders only; do not commit your populated file.
4. Optional fallback: copy `include/secrets.example.h` to `include/secrets.h`
   and fill in the same Wi-Fi credentials. `secrets.h` is gitignored. The SD
   card values are intended for normal provisioning so changing networks does
   not require rebuilding the firmware.
5. Insert the card and connect the ESP32-CAM to the programmer.
6. Build and upload:
   ```bash
   pio run --target upload
   ```
7. Open the serial monitor to confirm it connects and grabs an IP address:
   ```bash
   pio device monitor
   ```

If upload fails, hold the ESP32-CAM's `IO0` pin low (or hold the programmer's reset/boot buttons per its silkscreen) while it starts, then release once the upload begins.

For scheduled background/environment collection, including a one-frame-every-30-minutes
example and Windows Task Scheduler setup, see
[`server/README.md`](server/README.md#scheduled-environment-negative-candidates).

## Bounding-box review tool

`bbox_viewer.py` is a desktop Tkinter/Pillow multi-box editor for the
generated iNaturalist/CUB manifest CSVs and an unmodified `CUB_200_2011`
directory. It never edits the source manifest. Human decisions (`boxed`,
`no_bird`, `unusable`, or `skipped`) are atomically saved to a separate
resumable CSV. Use `unusable` for ambiguous/noisy images that should be excluded
even when they may contain a bird. A
boxed image is held as a list of boxes in memory and written as one CSV row per
box (repeating the image's `source_id` and metadata). The custom `BBoxDataset`
and its source detection records remain unchanged.

Install Pillow (it is also listed in `server/requirements.txt`), then run:

```powershell
python .\bbox_viewer.py `
  ".\Notebooks\Outputs\iNaturalist Outputs\Bounding Box Outputs\inaturalist_bird_bbox_manifest.csv" `
  --image-root "E:\Datasets\iNaturalist\train_mini\train_mini"
```

If paths stored in the manifest are valid on the current machine,
`--image-root` can be omitted. By default, reviews are written beside the
manifest as `<manifest-name>_ground_truth.csv`; use `--ground-truth` to choose a
different file. Machine/source boxes are deliberately neither displayed nor
copied into the draft, preventing them from biasing a blind review. Cyan boxes
are saved or newly drawn human-review boxes. Drag empty image space to add a
box, drag inside a box to select/move it, and drag a selected corner to resize
it. Shift-drag to add a box overlapping an existing box. The Delete key removes
only the selected box. Run
`python .\bbox_viewer.py --help` for all options and shortcuts.

Quote paths containing spaces. In PowerShell, a backtick (`` ` ``), not a
backslash (`\`), continues a command on the next line. For the untouched
iNaturalist mini training set, `--image-root` must be the directory directly
containing the numbered class folders; the viewer joins it to each manifest
row's `photo_relative_path`.

## Configuration

Settings live in `config.json` on the SD card and are re-read every 10 seconds,
so most changes don't require a reflash. Wi-Fi credential changes take effect
when the device next connects; reboot after changing them. If no card is
available, the firmware can use the SPIFFS copy uploaded from `data/config.json`.

| Field | Type | Description |
|---|---|---|
| `wifi_ssid` | string | 2.4 GHz Wi-Fi network name (at most 32 bytes); the SD/SPIFFS value takes precedence over the optional compiled fallback |
| `wifi_password` | string | Wi-Fi passphrase (at most 64 bytes); it is write-only in the config API, so keep the populated configuration private and never print or commit it |
| `motion_threshold` | int | Minimum global-light-normalized block luminance residual (0-255) required before a block counts as changed |
| `motion_sensitivity` | float | Lower fraction of valid changed blocks required to trigger a motion event |
| `motion_check_ms` | int | Milliseconds between motion checks |
| `motion_enabled` | bool | Arms the local detector only when `motion_owner` is `firmware`; must be `false` under Pi ownership |
| `motion_owner` | string | `pi` (default) gives polling/detection ownership to the Pi; `firmware` explicitly enables the legacy on-device fallback |
| `motion_roi` | object | Normalized `{x,y,width,height}` rectangle used for motion analysis; defaults to the full frame and must rasterize to at least 16×12 pixels in the 80×60 analysis image |
| `motion_exposure_settle_frames` | int | Consecutive stable exposure/gain product samples required before the detector arms (2-20) |
| `motion_exposure_settle_tolerance` | float | Maximum `(window max / window min) - 1` allowed while exposure is settling (0-1) |
| `motion_exposure_shift_tolerance` | float | Maximum relative exposure/gain product shift between compared frames; a larger shift skips triggering and refreshes the reference |
| `motion_block_columns`, `motion_block_rows` | int | ROI block grid dimensions, limited to 64 total blocks |
| `motion_ratio_threshold` | float | Minimum proportional deviation of a block ratio from the median global ratio |
| `motion_clip_low`, `motion_clip_high` | int | Luminance boundaries used to identify clipped pixels |
| `motion_max_clipped_fraction` | float | Maximum clipped fraction allowed in both versions of a usable block |
| `motion_min_valid_blocks` | int | Minimum non-clipped blocks required to make a motion decision |
| `motion_max_changed_fraction` | float | Upper changed-block fraction; broader changes are treated as lighting/scene changes and refresh the reference without triggering |
| `jpeg_quality` | int | JPEG quality passed to the camera driver (0 = best, 63 = worst) |
| `framesize` | int | `framesize_t` enum value used for full-resolution captures (8 = VGA) |
| `server_ip` | string | IP address of the receiving server |
| `server_port` | int | Port the receiving server listens on |
| `save_birds_to_sd` | bool | Save classifier-confirmed bird JPEGs under `/birds` on the physical SD card |
| `bird_confidence_threshold` | float | Minimum classifier confidence (0-1) required for an SD save |
| `wait_for_classification` | bool | `true`: wait for the receiver's classification after each upload (needed for SD bird saves). `false`: send `X-ESP32-Classification: deferred`, let the receiver queue inference, and re-arm motion immediately; nothing is saved to SD |
| `burst_count` | int | JPEGs captured and uploaded per motion trigger (1-10) |
| `burst_interval_ms` | int | Pause between burst JPEGs in milliseconds (0-5000) |
| `keep_reference_after_trigger` | bool | `true`: keep the pre-trigger background after a burst, so a subject that stays in view keeps triggering. `false`: relearn the background after every trigger |
| `max_consecutive_triggers` | int | Back-to-back triggers (1-100) after which the background is relearned anyway, so a permanent scene change cannot trigger forever |
| `buffer_uploads_to_sd` | bool | Write motion JPEGs that cannot be delivered to `/pending` on the SD card and upload them later, one every 2 s, once the receiver answers again |

You can also read or update these live over HTTP — see below.

## HTTP API

The device runs a small unauthenticated HTTP server on port 80. All endpoints assume a trusted local network.

| Method | Path | Description |
|---|---|---|
| `GET` | `/capture` | Returns a full-resolution JPEG on demand |
| `GET` | `/monitor-frame` | Returns a fresh full-resolution JPEG for Pi motion polling, with metadata source `pi_motion_poll` |
| `GET` | `/stream` | Runs an exclusive multipart MJPEG stream until the client disconnects |
| `GET` | `/metadata` | Returns sensor identity, current processing state, capture settings, and motion settings as JSON |
| `GET` | `/motion` | Returns `{"motion": true/false}` and clears the flag on read |
| `GET` | `/status` | Returns current motion flag, config summary, and SD/reference-frame readiness |
| `GET` | `/config` | Returns the current configuration as JSON |
| `POST` | `/config` | Accepts a partial JSON body to update and persist any of the fields above |
| `POST` | `/mode` | Selects `{"mode":"motion"}` or `{"mode":"idle"}` and persists the choice |

### Camera operating modes

The camera has two persistent operating modes plus one temporary exclusive state.
Local motion mode is available only when `motion_owner` is `firmware`:

- **Idle** leaves continuous capture disabled while retaining manual capture and
  configuration access.
- **Motion** runs the ESP32 software detector using JPEG frames decoded to an
  80×60 luminance reference. QQVGA, QVGA, and VGA capture configurations monitor
  at their configured resolution so a trigger does not restart AEC/AGC just
  before the first burst frame; other capture sizes use QQVGA for monitoring. It
  reports `motion_arming` until the observed
  AEC-lines × nominal-gain product remains stable for the configured number of
  frames and a reference exists, then `motion_armed`. A trigger captures and
  uploads a JPEG before automatically re-arming the detector.
- **Streaming** is entered by `GET /stream`. The sensor remains in JPEG mode and
  emits `multipart/x-mixed-replace` frames without reconnecting for every JPEG.
  The firmware is intentionally single-client while this response is open, so
  motion checks, uploads, configuration polling, and other camera endpoints are
  suspended. Disconnecting the viewer restores idle or re-arms motion according
  to the previously selected mode.

The OV2640 is not providing a hardware motion interrupt here. Motion detection
is performed by the ESP32 inside the configured ROI. The ROI is divided into
blocks and each current/reference mean-luminance ratio is calculated. Their
median estimates global illumination change; only blocks that deviate from this
median by `motion_ratio_threshold` and exceed `motion_threshold` count as local
motion. Blocks with too many black/white clipped pixels are ignored. A fraction
above `motion_max_changed_fraction` is treated as a broad lighting/scene change,
not a bird, and refreshes the reference. The legacy
`GET /motion` endpoint only consumes the last detected-event flag; use
`POST /mode` to arm or disarm detection.

`GET /status` includes `selected_mode`, `runtime_state`, `stream_active`, motion
event count and changed fraction, exposure-settle/product state, global ratio,
valid/clipped/changed block counts, effective ROI pixels, a decision reason,
last capture/upload results, upload cooldown, and the last camera error. The runtime state can be `idle`, `motion_arming`,
`motion_armed`, `motion_triggered`, `capturing`, `uploading`, `upload_cooldown`,
`streaming`, or `error`.

`GET /capture` includes the same capture-specific JSON in the
`X-ESP32-Camera-Metadata` response header. Motion-triggered `POST /upload`
requests include that header too. Camera metadata schema version 5 separates
requested capture settings from observed frame, driver, and OV2640 state. It
keeps frame size, pixel format, and JPEG quality in only the intentional
`requested.capture` and `observed.capture` locations. Actual dimensions,
`framebuffer_bytes`, and the frame sequence occur only under observed capture.
The receiver's top-level `bytes` remains separate because it verifies the JPEG
length actually received and stored. Bird-classification and retention settings
are grouped under `camera.requested.pipeline`. The active normalized motion ROI
is recorded as `{x,y,width,height}` under `camera.requested.motion.roi` for every
profile, preview, and motion-capture metadata snapshot.

Crop-aware is-bird models declare `roi_crop_then_letterbox` in their inference
contract. The receiver then expands the ROI by 20% total (10% per side), clips
and outward-rounds it to pixels, crops the original decoded frame, and only then
letterboxes it. Missing or mismatched ROI provenance is an inference error, not
a full-frame fallback. The species model remains whole-frame letterbox. Captures
made before metadata schema 5 have no recorded ROI and are excluded by the
crop-aware training notebook rather than assigned the current camera setting.

`GET /monitor-frame` has the same fresh-frame and metadata behavior as
`GET /capture`, but labels metadata `source` as `pi_motion_poll`. Metadata's
`device.camera_id` and `/status`'s top-level `camera_id` are the same stable
`esp32cam-<12 hex digits>` identifier derived from the chip's factory eFuse MAC.
It does not change across boots; `boot_id` remains random per startup.

Each record includes a random per-startup `boot_id`, exact active config-file
SHA-256 when a file exists, internal heap/PSRAM statistics, and the ESP32's
uncalibrated on-die temperature (not ambient temperature). SNTP starts
asynchronously after Wi-Fi connects. `device_time_utc` is ISO-8601 UTC when the
clock has synchronized and `null` beforehand; capture never waits for NTP.
The receivers combine that wall-clock sample with `device_time_boot_us` and
`frame_boot_us` and store `camera.timing.frame_utc_estimate` in each sidecar.
The estimate is `null` without a valid synchronized clock sample and describes
the framebuffer timestamp, not exposure start.

`frame_boot_us` is the camera driver's framebuffer timestamp and
`register_read_boot_us` is sampled immediately before the first SCCB register
read. Their difference can be used to estimate queue/register lag. It is not an
exposure-start measurement: `fb_count`, `grab_mode`, queued frames, and the
OV2640 AEC/AGC control pipeline can make the returned image one or more control
cycles older than the post-frame register sample. Register keys expose the real
bank and physical address, for example `sensor_bank_0x04_mask_0x03` rather than
the driver's virtual address `0x104`.

The assembled AEC line count is accompanied by `aec_max_lines` (1200 for the
bundled driver's supported AEC range) and a qualified `ae_saturated` derivation.
This is not a hardware saturation status bit. It is only valid when automatic
exposure is enabled and the repeated exposure-register reads are consistent.

`camera.timing.settings_revision` tracks runtime sensor-mode changes, while
`camera.register_dump.revision` identifies the persistent capture configuration
used for a cached register dump. The first capture or profile after startup or a persistent frame-size/JPEG-
quality change includes a compact base64 best-effort sweep of both OV2640
register banks. The canonical dump is SHA-256 addressed; later captures carry
only the hash until another relevant settings change. The sweep is non-atomic
and is a diagnostic catch-all, not the exact register state of one frame. The
laptop and Raspberry Pi receivers retain all of this metadata in JSON sidecars.

The Raspberry Pi upload response must contain an explicit classification for
the firmware to retain the frame on its own SD card:

```json
{
  "capture_id": "capture_20260929T143212_123456Z_a1b2c3d4",
  "classification": {
    "is_bird": true,
    "label": "sparrow",
    "confidence": 0.92
  }
}
```

Only `is_bird: true` results at or above `bird_confidence_threshold` are written
to `/birds/<capture_id>.jpg`. Missing or malformed classifications, classifier
errors, HTTP errors, low-confidence results, and non-birds are not written. A
motion JPEG must still exist briefly in RAM and be uploaded so the Pi can decide
what it contains. SPIFFS remains a configuration fallback and is never used for
image storage.

The camera uses `fb_count: 1` with `CAMERA_GRAB_WHEN_EMPTY` and remains in JPEG
pixel format because the camera HAL configures its DMA mode during initialization.
With this configuration, one completed frame can wait indefinitely while idle.
Manual capture and stream startup therefore discard exactly one queued frame and
use the following frame. Camera-setting and motion/full-resolution transitions
use the same drain operation after allowing the new sensor mode to settle. This
keeps low-resolution motion checks active while ensuring manual JPEGs do not
predate their request by the camera's idle duration. The
`frames_since_boot` counter includes motion-analysis and deliberately discarded
transition frames as well as retained JPEGs.

`exposure_gain_product` is `AEC lines × nominal analog gain`. It is a relative
exposure/sensitivity proxy when timing and camera mode are unchanged, not a lux
measurement or direct JPEG-brightness value. A darker scene normally makes an
automatic camera increase this product. DSP registers `0xCC` through `0xCE`
are retained as raw white-balance controls; public OV2640 documentation does
not establish them as readable live AWB channel decisions, so they are not
reported as such.

## Security note

This firmware has no authentication and is meant for a trusted home LAN only. Do not expose port 80 on this device to the public internet — anyone on the same network can pull camera frames or change its configuration.

## Troubleshooting

- **Wi-Fi does not connect** — double-check `wifi_ssid` and `wifi_password` in
  the SD card's `/config.json`; the board only supports 2.4 GHz Wi-Fi. If those
  fields are absent, check the optional `include/secrets.h` fallback.
- **Camera init failed** — usually a power issue; use a dedicated 5V supply rather than a laptop USB port.
- **`config.json not found`** — confirm the card is FAT32-formatted and the file sits at the SD card root, not in a subfolder.
- **No images under `/birds`** — check `/status` for the last classification and SD result, then check the Pi's `/health` response for model readiness, queue depth, and the latest inference. Remember that `wait_for_classification: false` deliberately returns before a verdict, so the ESP32 cannot make its own `/birds` copy in deferred mode.
- **Random reboots during capture/upload** — almost always insufficient power; add a capacitor across the 5V/GND pins or use a beefier supply.
- **Laptop live view times out** — use the IP printed by the ESP32 serial monitor
  as the laptop server's `--camera-url`. The ESP32's `server_ip` is the reverse
  direction and must be the laptop's LAN address. Temporarily set
  `motion_enabled` to `false` to isolate preview traffic from motion uploads.
- **ESP32 stops answering when the receiver is offline** — failed motion uploads
  use bounded timeouts and a 15-second retry cooldown. Check that `server_ip`
  still matches the receiver's current DHCP address.

## Commands

```bash
pio run                  # build
pio run --target upload  # build and flash
pio device monitor       # serial monitor at 115200 baud
```

The Pi receiver starts active polling whenever `--camera-url` is supplied:

```bash
python3 server/raspberry_pi_server.py \
  --camera-url http://192.168.1.84 \
  --motion-poll-seconds 0.25 \
  --motion-timeout-seconds 2
```

`/health` reports poll counts, background readiness/epoch, persistence count,
last crop selection, and the latest polling error. OpenCV is used automatically
when installed; a tested NumPy fallback keeps the service functional without an
additional Pi wheel. Existing `letterbox` and static `roi_crop_then_letterbox`
models keep their original inputs. Retrain with the generated `.motion.json`
records before deploying a `selected_crop_then_letterbox` model.

## License

[MIT](LICENSE)
