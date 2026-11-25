# ESP32-CAM firmware

The firmware is the camera-side component of the bird-classifier system. It
connects an AI Thinker ESP32-CAM to Wi-Fi, exposes capture and configuration
endpoints, sends JPEGs and metadata to the laptop or Raspberry Pi receiver, and
can optionally retain classifier-confirmed birds on microSD.

Pi-owned motion detection is the default. In that mode the receiver polls fresh
full-resolution frames and owns background modelling, motion selection, and
inference. The camera's lower-cost onboard detector remains available as a
fallback for deployments that need it.

## Supported hardware

| Board | PlatformIO env | Notes |
|---|---|---|
| AI Thinker ESP32-CAM | `esp32cam` | Original fleet board; flashed through an ESP32-CAM-MB or USB-to-UART adapter |
| Freenove ESP32-S3 WROOM CAM (N8R8) | `freenove_s3` | 8 MB flash, 8 MB octal PSRAM; flash through the USB-C port wired to the CH343 UART bridge. See the [S3 bring-up and field guide](esp32-s3-bringup.md) |

Both boards take an OV2640 or OV5640 sensor and need a stable 5 V supply and a
FAT32 microSD card for configuration, optional bird retention, and optional
failed-upload buffering. Pin maps for each board are in
[`include/board_pins.h`](../include/board_pins.h); `pio run` with no `-e`
builds both.

## Setup

### 1. Install PlatformIO

Install the PlatformIO extension for Visual Studio Code or the PlatformIO CLI,
then clone or copy this repository.

### 2. Prepare `config.json`

Copy [`data/config.example.json`](../data/config.example.json) to the root of a
FAT32 microSD card as `config.json`. At minimum, set:

- `wifi_ssid` and `wifi_password` for a 2.4 GHz network;
- `server_host`, normally `birdhub.local`;
- `server_ip` to the laptop or Pi LAN address as a fallback; and
- the correct OV2640 and OV5640 sensor profiles for the deployment.

Do not commit a populated configuration file. The example credentials are
placeholders.

If no SD card is available, the firmware can read an SPIFFS copy from
`data/config.json`. That device-specific file is ignored by Git. An optional
compile-time Wi-Fi fallback can also be created by copying
[`include/secrets.example.h`](../include/secrets.example.h) to
`include/secrets.h` and replacing its placeholders.

### 3. Build and upload

Connect the programmer and run from the repository root:

```powershell
pio run -e esp32cam
pio run -e esp32cam -t upload
pio device monitor
```

The serial monitor runs at 115200 baud and reports the detected sensor, Wi-Fi
connection, camera address, hub discovery, and runtime errors. If upload fails,
put the board into its bootloader mode according to the programmer's IO0/BOOT
and RESET controls, then retry.

### 4. Upload the SPIFFS fallback when needed

After creating or changing `data/config.json`, upload the filesystem image
separately:

```powershell
pio run -e esp32cam -t uploadfs
```

A normal firmware upload does not update SPIFFS. When a physical SD card is
present, its `/config.json` takes precedence.

### 5. Verify the camera

Open the address printed in the serial monitor through the laptop server or
query the camera directly on the trusted LAN:

```text
http://CAMERA_IP/status
http://CAMERA_IP/capture
```

For dashboard setup, multi-camera registration, Pi deployment, and inference,
continue with the [server guide](../server/README.md).

## Configuration

Camera settings are found in `config.json` on the SD card. Wi-Fi credential changes take effect
when the device next connects; reboot after changing them. If no card is
available, the firmware can use the SPIFFS copy uploaded from `data/config.json`.
Camera frame-size changes safely reinitialize the driver so framebuffer storage
is allocated for the new dimensions; JPEG-quality changes apply live. An
`xclk_hz` change is saved immediately but requires a restart because the camera
driver selects XCLK during initialization. `GET /config` and `GET /status`
report `camera_restart_required` while the configured clock differs from the
running clock.

| Field | Type | Description |
|---|---|---|
| `wifi_ssid` | string | 2.4 GHz Wi-Fi network name (at most 32 bytes); the SD/SPIFFS value takes precedence over the optional compiled fallback |
| `wifi_password` | string | Wi-Fi passphrase (at most 64 bytes); it is write-only in the config API, so keep the populated configuration private and never print or commit it |
| `motion_threshold` | int | Minimum global-light-normalized block luminance residual (0-255) required before a block counts as changed |
| `motion_sensitivity` | float | Lower fraction of valid changed blocks required to trigger a motion event |
| `motion_check_ms` | int | Milliseconds between motion checks |
| `motion_enabled` | bool | Arms the local detector only when `onboard_motion_detection` is `true`; keep it `false` for laptop/Pi-owned detection |
| `onboard_motion_detection` | bool | `false` (default) gives motion ownership to laptop/Pi workers; `true` selects the ESP32 detector |
| `motion_owner` | string | Legacy alias: `pi` maps to `onboard_motion_detection: false`, and `firmware` maps to `true` |
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
| `sensors` | object | Required camera profiles keyed by sensor identity. Keep both `0x2642` (OV2640) and `0x5640` (OV5640) entries |
| `sensors.<key>.frame_size` | string or int | Capture size from `QQVGA` through the sensor maximum, such as `VGA`; numeric `framesize_t` values remain accepted for migration |
| `sensors.<key>.jpeg_quality` | int | Sensor-specific JPEG quality (0 = best, 63 = worst) |
| `sensors.<key>.xclk_hz` | int | Sensor-specific boot clock in Hz: 6-24 MHz for OV2640 and 6-27 MHz for OV5640; changing it requires a restart |
| `server_host` | string | Name of the receiving server, looked up on the local network (default `birdhub.local`); see the [server guide](../server/README.md) |
| `server_ip` | string | IP address of the receiving server; fallback when `server_host` cannot be found |
| `server_port` | int | Port the receiving server listens on |
| `save_birds_to_sd` | bool | Save classifier-confirmed bird JPEGs under `/birds` on the physical SD card |
| `pipeline.bird_confidence_threshold` | float | Minimum classifier confidence (0-1) required for an SD save |
| `wait_for_classification` | bool | `true`: wait for the receiver's classification after each upload (needed for SD bird saves). `false`: send `X-ESP32-Classification: deferred`, let the receiver queue inference, and re-arm motion immediately; nothing is saved to SD |
| `burst_count` | int | JPEGs captured and uploaded per motion trigger (1-10) |
| `burst_interval_ms` | int | Pause between burst JPEGs in milliseconds (0-5000) |
| `status_led` | string | When the board's status light is lit: `boot` (default, first 5 minutes after power-up), `on` (always, including a flicker per capture) or `off`. Freenove ESP32-S3 only; see [Status light](#status-light) |
| `idle_power_down_s` | int | Seconds with no frame requests before the sensor is powered down (default 30; 0 = never, 0-86400). The first frame after a power-down takes about 2-3 s while the sensor restarts and exposure settles. On the Freenove ESP32-S3 the sensor has no power-down line, so this only stops its clock and DMA |
| `keep_reference_after_trigger` | bool | `true`: keep the pre-trigger background after a burst, so a subject that stays in view keeps triggering. `false`: relearn the background after every trigger |
| `max_consecutive_triggers` | int | Back-to-back triggers (1-100) after which the background is relearned anyway, so a permanent scene change cannot trigger forever |
| `buffer_uploads_to_sd` | bool | Write motion JPEGs that cannot be delivered to `/pending` on the SD card and upload them later, one every 2 s, once the receiver answers again |

Live streaming over HTTP is available.
The ESP32 camera driver reports OV2640 as PID `0x26` and version `0x42`; the
firmware combines those bytes into the configuration key `0x2642`. OV5640 is
reported directly as PID/configuration key `0x5640`; the pinned driver does not
populate its version or manufacturer-ID fields, so metadata marks those fields
unavailable instead of treating zero as a measured value. The raw identity and
selected configuration key remain separate in capture metadata and `/status`.

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
Local motion mode is available only when `onboard_motion_detection` is `true`;
the legacy `motion_owner: "firmware"` value maps to the same setting:

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

Neither supported sensor provides a hardware motion interrupt here. Motion detection
is performed by the ESP32 inside the configured ROI. The ROI is divided into
blocks and each current/reference mean-luminance ratio is calculated. Their
median estimates global illumination change; only blocks that deviate from this
median by `motion_ratio_threshold` and exceed `motion_threshold` count as local
motion. Blocks with too many black/white clipped pixels are ignored. A fraction
above `motion_max_changed_fraction` is treated as a broad lighting/scene change,
not a bird, and refreshes the reference. The legacy
`GET /motion` endpoint only consumes the last detected-event flag; use
`POST /mode` to arm or disarm detection.

`GET /status` includes detected camera identity, `camera_restart_required`,
`selected_mode`, `runtime_state`, `stream_active`, motion
event count and changed fraction, exposure-settle/product state, global ratio,
valid/clipped/changed block counts, effective ROI pixels, a decision reason,
last capture/upload results, upload cooldown, and the last camera error. The runtime state can be `idle`, `motion_arming`,
`motion_armed`, `motion_triggered`, `capturing`, `uploading`, `upload_cooldown`,
`streaming`, or `error`.

`GET /capture` includes the same capture-specific JSON in the
`X-ESP32-Camera-Metadata` response header. Motion-triggered `POST /upload`
requests include that header too. Camera metadata schema version 5 separates
requested capture settings from observed frame, driver, and OV5640 state. It
keeps frame size, pixel format, and JPEG quality in only the intentional
`requested.capture` and `observed.capture` locations. Actual dimensions,
`framebuffer_bytes`, and the frame sequence occur only under observed capture.
The receiver's top-level `bytes` remains separate because it verifies the JPEG
length actually received and stored. Bird-classification and retention settings
are grouped under `camera.requested.pipeline`. The active normalized motion ROI
is recorded as `{x,y,width,height}` under `camera.requested.motion.roi` for every
profile, preview, and motion-capture metadata snapshot.

The deployed float detector declares `selected_crop_then_letterbox` in its
`DetectorContract`. The receiver uses the motion selector's exact
`final_bounds_px` crop with no additional inference margin, letterboxes it to
224x224 with TensorFlow-compatible antialiased resize and pad value 114, then
decodes and maps the top box back to full-frame pixels. Missing or mismatched
crop provenance is an inference error, not a full-frame fallback. The species
model remains whole-frame letterbox. Captures made before metadata schema 5 have
no recorded ROI and are excluded by crop-aware training rather than assigned the
current camera setting.

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
OV5640 AEC/AGC control pipeline can make the returned image one or more control
cycles older than the post-frame register sample. Exposure is read from
`0x3500` as sixteenths of a line, analog gain from `0x350A`, and frame length
from `0x380E`; these post-frame reads are not an atomic exposure snapshot.

`camera.timing.settings_revision` tracks runtime sensor-mode changes, while
`camera.register_dump.revision` identifies the persistent capture configuration
used for a cached register dump. The first capture or profile after startup or a
persistent frame-size/JPEG-quality change includes a base64 best-effort register
snapshot: the OV2640 uses a full two-bank dump and the OV5640 uses a compact
sparse dump. The canonical dump is SHA-256 addressed; later captures carry
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

Only `is_bird: true` results at or above `pipeline.bird_confidence_threshold` are written
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
automatic camera increase this product.

Every receiver sidecar and `manifest.jsonl` row denormalizes the canonical
`camera.sensor.model` into top-level `camera_model`. Pi motion and inference
records also contain `camera_provenance` with the model and SHA-256 of the exact
capture sidecar. Treat historical OV2640 and new OV5640 captures as separate
evaluation/training cohorts: do not backfill missing legacy camera models or
pool sensor metrics without explicitly accounting for the camera-induced data
shift.

## Status light

The Freenove ESP32-S3 board's RGB LED (GPIO 48) shows the camera's state. The
AI Thinker ESP32-CAM has no RGB LED, so `status_led` has no effect there. The
light runs at about 10 % brightness.

At boot it flashes once or more, in this order:

| Flash | Meaning |
|---|---|
| White, once | Booted (power-on or reset) |
| Orange, twice | The previous restart was a crash, watchdog reset or brownout; check the dashboard's Fleet tab for the crash record |
| Purple, once | No usable SD card, so the configuration came from SPIFFS |

After that it shows the highest-priority current state:

| Pattern | Meaning |
|---|---|
| Red, fast blink | Camera failed to initialize; the board reboots within 60 s (see the serial log's `camera bus:` lines) |
| Cyan, solid | Streaming |
| White, brief flicker | A frame was just captured (only when `status_led` is `on`) |
| Blue, slow blink | Joining Wi-Fi, or waiting for the first heartbeat reply |
| Yellow, slow blink | On Wi-Fi but the hub is not answering heartbeats (wrong network, hub down, or out of range). A single slow hub reply can show this until the next heartbeat 15 s later |
| Green, slow fade | All good; the camera sensor is asleep after `idle_power_down_s` |
| Green, short blink every 3 s | All good: Wi-Fi up, hub answering, camera ready |

`status_led` chooses when it is lit:

- `boot` (default): lit for the first 5 minutes after power-up, then dark. Use
  this in the field: you can check a camera on site after plugging it in, and it
  stays dark afterwards so it doesn't startle birds, reflect into images or
  draw attention.
- `on`: always lit, including the capture flicker. For bench work.
- `off`: never lit (the LED is switched off at boot).

The setting can be changed from the dashboard's camera settings card and takes
effect immediately; the 5-minute window always counts from power-up.

## Security note

This firmware has no authentication and is meant for a trusted home LAN only. Do not expose port 80 on this device to the public internet — anyone on the same network can pull camera frames or change its configuration.

## Troubleshooting

- **Wi-Fi does not connect** — double-check `wifi_ssid` and `wifi_password` in
  the SD card's `/config.json`; the board only supports 2.4 GHz Wi-Fi. If those
  fields are absent, check the optional `include/secrets.h` fallback. The serial
  log prints the name it uses in brackets with its length
  (`Connecting to [BirdHub] (7 chars)`); stray spaces, tabs and line breaks
  around `wifi_ssid`, `server_ip` and `server_host` are trimmed on load (only
  tabs and line breaks around `wifi_password`, where a space can be real). If it
  is still not connected after 20 s, the firmware scans once and lists every
  network it can see with channel, signal and security (`=>` marks the
  configured one, `~~` a name that differs only in capitalisation), then resumes
  connecting. The disconnect reason narrows it down:
  - `201 NO_AP_FOUND` and the network absent from the scan: hub off, out of range
    or antenna shielded.
  - `201 NO_AP_FOUND` but the scan lists it as `WPA`: the network offers only WPA1,
    which the ESP32 ignores. Set the access point to WPA2 (AES/CCMP); on the Pi,
    `sudo bash deploy/pi_hotspot.sh on` applies the right settings.
  - `2 AUTH_EXPIRE` with the scan showing `WPA2/WPA3`: turn off protected management
    frames (WPA2 only) on the access point.
  - `15` or `204`: wrong password.
- **Camera init failed** — usually a power issue; use a dedicated 5V supply rather than a laptop USB port.
- **`config.json not found`** — confirm the card is FAT32-formatted and the file sits at the SD card root, not in a subfolder.
- **No images under `/birds`** — check `/status` for the last classification and SD result, then check the Pi's `/health` response for model readiness, queue depth, and the latest inference. Remember that `wait_for_classification: false` deliberately returns before a verdict, so the ESP32 cannot make its own `/birds` copy in deferred mode.
- **Random reboots during capture/upload** — almost always insufficient power; add a capacitor across the 5V/GND pins or use a beefier supply.
- **Laptop live view times out** — use the IP printed by the ESP32 serial monitor
  as the laptop server's `--camera-url`. The ESP32's `server_ip` is the reverse
  direction and must be the laptop's LAN address. Temporarily set
  `motion_enabled` to `false` to isolate preview traffic from motion uploads.
- **Status light blinks yellow after a field power-up** — the camera booted
  before the hub was answering. Power the hub first and the camera about two
  minutes later, or press RST once the dashboard loads; the hub log
  (`journalctl -u birdhub`) should show its heartbeats every ~15 s.
- **`/sdcard/pending does not exist` at boot** — harmless when `buffer_uploads_to_sd`
  is `false`; create an empty `pending` folder at the card root to silence it.
- **ESP32 stops answering when the receiver is offline** — failed motion uploads
  use bounded timeouts and a 15-second retry cooldown. Check that `server_ip`
  still matches the receiver's current DHCP address.

## Common commands

```powershell
pio run -e esp32cam             # build
pio run -e esp32cam -t upload   # build and flash firmware
pio run -e esp32cam -t uploadfs # upload data/config.json to SPIFFS
pio device monitor              # serial monitor at 115200 baud

pio run -e freenove_s3 -t upload --upload-port COMx   # Freenove ESP32-S3
pio device monitor -e freenove_s3 --port COMx
```

Return to the [project overview](../README.md).