# ESP32-CAM servers

The firmware sends raw JPEG bytes to `POST /upload` with the content type
`image/jpeg`. Capture metadata is sent in the optional
`X-ESP32-Camera-Metadata` JSON header.

## Multiple cameras

Both `laptop_test_server.py` and `raspberry_pi_server.py` maintain a persistent
camera registry, and cameras add themselves to it. Every ESP32 posts a heartbeat
to `/api/heartbeat` every 15 s carrying its stable hardware `camera_id` and its
current IP, and the server keeps the registry in step with that:

- **New camera**: added automatically as `New camera <last 4 of its ID>`, enabled,
  and flagged `needs_name`. The dashboards show a **name this camera** button;
  naming it (or any rename) clears the flag.
- **Known camera, new IP** (router handed out a different address): the stored
  URL is rewritten in place. The name, registry ID, captures and settings are
  untouched, and on the Pi the camera's motion worker restarts on the new address.
- **Camera you already added by hand** at that address: the hardware ID is
  attached to your entry instead of creating a duplicate.
- **Address still claimed by a different registered camera that has not checked
  in**: nothing changes and the server prints one notice. Remove the retired
  camera from the dashboard and the new one is added on its next heartbeat.

Cameras find the server by name too (see **Finding the hub by name** below), so
neither side needs an IP address typed in. A camera that is removed
from the dashboard while still powered on will re-add itself on its next
heartbeat; use **Disable** to keep a live camera out of use. Pass
`--no-auto-register` to either server to manage the registry purely by hand.

### Finding the hub by name

Both servers answer on the local network for `birdhub.local` (mDNS). Each camera
looks that name up just before its heartbeat and sends heartbeats and uploads to
whatever address answers, so every camera can carry the same `config.json`:

```json
"server_host": "birdhub.local",
"server_ip": "192.168.1.100"
```

- `server_host` is tried first. It may also be a plain IPv4 address or a normal
  DNS name. Set it to `""` to switch name lookup off.
- `server_ip` is the fallback, used until the name has resolved once. Cameras
  with an older `config.json` (no `server_host`) get `birdhub.local` by default
  and keep their `server_ip` as that fallback.
- If the hub's own address changes, a camera notices at its next failed
  heartbeat and looks the name up again (at most every 30 s).
- Each camera's `/health` and heartbeat report `hub.host`, `hub.address` and
  `hub.found_by_name`, and its event log notes `hub: birdhub.local is at ...`.

Server side this needs the optional `zeroconf` package (now in
`requirements.txt`; `pip install zeroconf`). At startup the server prints
`Hub name: cameras can find this server as birdhub.local (...)`. Options:

| Option | Purpose |
|---|---|
| `--hub-name garden` | Answer for `garden.local` instead (set the cameras' `server_host` to match) |
| `--no-advertise` | Do not answer for any name; cameras use `server_ip` |

Only one server on a network should use a given name. If the laptop and the Pi
both run with the default, cameras may reach either one; start one of them with
`--no-advertise` or a different `--hub-name`. On Windows, allow Python through
the firewall on private networks when prompted (mDNS uses UDP port 5353). A
Raspberry Pi whose own hostname is `birdhub` already answers for `birdhub.local`
through Raspberry Pi OS, with or without `zeroconf`.

Adding by address remains as a fallback: use **Camera fleet** to add an
ESP32-CAM by IP address, for example `192.168.1.84`. A complete base URL such as
`http://192.168.1.84` is also accepted. The optional name is used in the
dashboard and capture sidecars.

Each registered camera is shown as its own dashboard tab. Preview/stream,
manual capture, status, metadata, configuration, mode, ROI, profile, and latest
saved-result actions are sent with that tab's registry ID, so no **Set default**
step is required. Tab-local stream, ROI, editor, and result state is preserved
when switching cameras. The default flag remains only for the legacy unscoped
`/camera/...` routes and `--camera-url` compatibility.

The registry stores each camera's URL, enabled/default state, stable ESP32
hardware `camera_id`, and detected sensor identity. OV2640 and OV5640 cameras
can run together without sharing camera locks or Pi motion backgrounds. Each
camera has an independent operation lock, so one stream or failed camera does
not block the others.

By default the command-line servers store their registry at
`~/.esp32-cameras/cameras.json`. Override this with `--registry-file`. The
existing `--camera-url` option remains supported and creates or selects the
default registry entry.

Camera management API:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/cameras` | List definitions, sensor data, health, and Pi worker state |
| `POST` | `/api/cameras` | Add `{"address":"192.168.1.84","name":"Backyard"}` |
| `PATCH` | `/api/cameras/<id>` | Rename, enable/disable, or set the default |
| `DELETE` | `/api/cameras/<id>` | Remove a camera and stop its Pi worker |
| `POST` | `/api/cameras/<id>/poll` | Refresh status/config/metadata and identity |
| `GET` | `/api/cameras/<id>/capture` | Proxy a selected-camera capture |
| `GET`/`POST` | `/api/cameras/<id>/config` | Read or update selected-camera config |
| `POST` | `/api/cameras/<id>/mode` | Set selected camera to `motion` or `idle` |

The laptop also exposes `/api/cameras/<id>/stream`,
`/api/cameras/<id>/captures`, `/api/cameras/<id>/profiles`, and
`/api/cameras/<id>/latest.jpg`. The Pi exposes
`/api/cameras/<id>/latest` and `/api/cameras/<id>/latest.jpg`. Existing unscoped
`/camera/...` routes target the configured default or sole camera.

Set `onboard_motion_detection` in each ESP32's `config.json` or camera config
editor. Use `false` for laptop/Pi-owned motion detection across multiple cameras
and `true` for the ESP32's local detector. `motion_enabled` separately arms that
local detector, so it should normally remain `false` when the boolean is false.
The older `motion_owner` value remains accepted as a compatibility alias.

On the Pi, every enabled camera gets its own `MotionService` and running-median
background. TFLite inference remains serialized through a shared lock. Capture
records include `camera_id`, `camera_registry_id`, `camera_name`, and
`camera_model`. Filter with `/api/captures?camera_id=esp32cam-...` or
`/api/captures?camera_registry_id=<registry-id>`.



The Raspberry Pi server validates its classifier output and returns this
contract to the ESP32:

```json
{
  "classification": {
    "is_bird": true,
    "label": "sparrow",
    "confidence": 0.92
  }
}
```

`is_bird` must be a boolean and `confidence` must be a number from 0 through 1.
`label` is optional. Invalid output or an inference exception produces a null
classification plus `classification_error`, which deliberately prevents an SD
save on the ESP32.


## Testing configuration without an SD card

The firmware uses the SD card's `/config.json` when a card is available. Start
from the sanitized `data/config.example.json`, copy it to the card root as
`config.json`, and set `wifi_ssid`, `wifi_password`, `server_ip`, and
`server_port`. The Wi-Fi credentials must be for a 2.4 GHz network. If no SD
card is detected, the firmware falls back to the ESP32-CAM's internal SPIFFS
storage; `data/config.json` is the ignored, device-specific development copy.

After changing `data/config.json`, upload the firmware and the SPIFFS image:

```powershell
pio run -e esp32cam -t upload
pio run -e esp32cam -t uploadfs
```

The filesystem image is separate from the firmware, so a normal firmware
upload does not copy `data/config.json`. The `/status` response reports
`"config_storage":"SPIFFS"` when the fallback is active and `"SD card"` when
the physical card is active. Configuration changes made through `POST /config`
are saved to the active storage backend.

`GET /config` never returns the Wi-Fi password. It reports the SSID and whether
a password/fallback is configured. If `wifi_ssid` is absent or empty, an
optional `include/secrets.h` made from `include/secrets.example.h` supplies a
compile-time fallback. Do not commit either a populated `config.json` or
`secrets.h`.

## Scheduled environment-negative candidates

`camera_capture_job.py` is a small standalone collector for building an
environment dataset without adding scheduling responsibilities to the laptop
dashboard. It calls the ESP32's existing `/capture` endpoint directly, so the
Flask test server does not need to be running.

Take one picture:

```powershell
python .\server\camera_capture_job.py `
  --camera-url http://192.168.1.84 `
  --save-dir E:\Datasets\ESP32_environment
```

Or keep the command open and take the first picture immediately, then one every
30 minutes:

```powershell
python .\server\camera_capture_job.py `
  --camera-url http://192.168.1.84 `
  --save-dir E:\Datasets\ESP32_environment `
  --interval-minutes 30
```

Each successful request uses the normal `CaptureStore` layout: a JPEG, JSON
sidecar, and `manifest.jsonl` row. Records have `source: "environment"` and a
`collection` object whose purpose is `environment_negative_candidate` and whose
review status is `unreviewed`. The collector deliberately does **not** add a
`manual_label`: a scheduled frame can still contain a bird.

Keep these candidates in their own directory and review them before training:

```powershell
python .\image_labeler.py E:\Datasets\ESP32_environment
```

`image_labeler.py` now opens the combined label and bounding-box editor (the
same application as `bbox_viewer.py`). A directory does not need an existing
manifest: the editor discovers its images and creates `bird_annotations.csv`
on the first save. You can also open the current detector annotation CSV
directly to edit it in place:

```powershell
python .\bbox_viewer.py .\Models\Multi_Object_Detector\runs\RUN_NAME\joined_bird_detector_annotations.csv
```

For one species across all boxes, enter the shared species (for example
`Blue Tit`). Enable **Mixed species — label each box** only when individual
boxes need different `species_name` values. The manifest stores `is_Bird`,
`mixed_species`, and `bird_count` for every image; `bird_count` is kept equal
to the number of accepted bird boxes.

The **Context around boxes (%)** control defaults to 0, so the context rectangle
initially matches the tight box. Positive values add that total percentage to
the box width and height, shifted or clipped to remain inside the image. Cyan
rectangles remain the tight detector annotations; dashed orange rectangles
preview the context. Detector CSVs preserve the tight `xmin`/`ymin`/`xmax`/
`ymax` values and write `context_percent` plus `context_xmin`/`context_ymin`/
`context_xmax`/`context_ymax` alongside them.

For unattended collection on Windows, the recommended setup is a Task Scheduler
task that invokes the one-picture command every 30 minutes. Set the action's
**Program/script** to the path of `python.exe`, put the script and its arguments
in **Add arguments**, set **Start in** to this project directory, and configure
the task not to start a second instance when one is already running. This is
more restart-safe than leaving the interval process open.

The laptop must be awake and on the same network as the ESP32. An active MJPEG
stream occupies the firmware's single HTTP client and prevents captures; stop
streaming before collection. A brief motion upload can also delay a request.
One-shot mode exits nonzero on failure so Task Scheduler can report it; interval
mode logs an individual failure and tries again at the next fixed-rate slot.

## Laptop test server

Install the dependency from the PlatformIO project directory:

```powershell
python -m pip install -r .\server\requirements.txt
```

Find the laptop's LAN IPv4 address:

```powershell
ipconfig
```

Put that address and port `5000` in the SD card's `config.json`:

```json
{
  "server_ip": "192.168.1.100",
  "server_port": 5000
}
```

Replace `192.168.1.84` below with the IP printed by the ESP32-CAM serial
monitor, then start the server:

```powershell
python .\server\laptop_test_server.py --camera-url http://192.168.1.84
```

Startup prints separate local and LAN dashboard addresses, the URL the ESP32
must use for uploads, and the URL the laptop uses to reach the camera. It also
probes the camera's `/status` endpoint. An unsuccessful probe is an actionable
warning and does not prevent the receiver from accepting uploads. Before Flask
starts, the script checks whether the requested listening address and port are
already occupied and reports a duplicate-server diagnostic instead of a generic
bind failure.

Open <http://127.0.0.1:5000> on the laptop. The page provides:

- **Start stream** and **Stop stream** controls using the ESP32's continuous
  multipart MJPEG response instead of polling one JPEG at a time;
- **Arm motion** and **Idle** controls plus a visible camera runtime-state badge;
- **Capture and save** for a fresh manual image;
- calibration categories for normal, dark-frame, flat-field, gray-card,
  color-chart, and resolution-target captures, plus optional labels and notes;
- **Save camera profile**, which saves `/metadata`, `/config`, and `/status`
  without taking a picture;
- the latest saved manual or motion-triggered image;
- camera status, motion, metadata, and configuration responses;
- an editor that sends JSON to the camera's `POST /config` endpoint;
- a mouse/touch drag-to-draw motion ROI editor. **Load current ROI** reads the
  nested `motion_roi` object, **Reset full frame** selects the whole image, and
  **Save ROI** sends only a partial update such as
  `{"motion_roi":{"x":0.1,"y":0.2,"width":0.7,"height":0.6}}`.

Streaming is deliberately exclusive on the ESP32. While the stream is active,
the Flask proxy immediately returns HTTP 409 for conflicting camera operations
instead of leaving requests queued behind the long-lived connection. The page
stops the stream before changing camera mode. Closing the stream restores the
ESP32's selected mode; if motion was selected, it builds a fresh reference frame
and returns to `motion_armed`. Status polling resumes after the stream closes.

The **Motion** control arms the ESP32's software luminance frame-difference
detector. It does not use an OV2640 hardware motion interrupt. Every valid
motion-triggered JPEG uploaded to the laptop or Pi receiver is saved there. The
ESP32 microSD copy remains restricted to classifier-confirmed birds meeting
`pipeline.bird_confidence_threshold`.

Images are saved to `Pictures\ESP32Cam` by default. Override this with:

```powershell
python .\server\laptop_test_server.py --camera-url http://192.168.1.84 --save-dir .\received_images
```

Allow Python through Windows Firewall on **Private networks** when prompted.
Both the laptop and ESP32-CAM must be on the same network. The ESP32 cannot
use `127.0.0.1`; its `server_ip` must be the laptop's LAN address.

The two addresses are intentionally different: `--camera-url` points from the
laptop to the ESP32 address printed in the serial monitor, while `server_ip` in
the ESP32 configuration points back to the laptop. The dashboard probes the
camera's `/status` endpoint at startup and reports whether that configured URL
is reachable. If preview requests time out, verify the serial-monitor IP before
increasing timeouts; a stale DHCP address will not be fixed by waiting longer.

## Dataset layout and ML preprocessing

Each successful manual or motion capture creates three related records in the
selected save directory:

```text
capture_20260928T120000_000000Z_a1b2c3d4.jpg
capture_20260928T120000_000000Z_a1b2c3d4.json
manifest.jsonl
```

The JSON sidecar and corresponding JSONL row contain:

- schema version 3 and a unique capture ID;
- UTC server receive time, source (`manual` or `motion`), category, label, and
  note;
- JPEG byte count and SHA-256 hash;
- firmware metadata schema version 4, flattened into the receiver envelope as
  top-level `device` and `camera` objects without a redundant `camera.camera`;
- per-boot identity, optional NTP-synchronized device UTC, config-file SHA-256,
  ESP32 heap/PSRAM health, and qualified internal die temperature;
- canonical requested and observed capture settings, dimensions, and source
  framebuffer byte count. The envelope byte count independently verifies the
  payload received and stored;
- frame and first-register-read boot timestamps, a derived
  `camera.timing.frame_utc_estimate`, successful acquisitions, the runtime
  settings revision, the register-dump revision, and time since the last
  frame-affecting change;
- observed live OV2640 values including assembled AEC lines, the bundled
  driver's 1200-line limit, qualified derived AE saturation, nominal analog
  gain, and relative exposure/gain product;
- content-addressed full banked register dumps. The first sidecar/profile after
  startup or a persistent capture-settings change contains the base64 payload;
  subsequent records reference the same SHA-256 without repeating it;
- white-balance controls, image corrections, and orientation supplied by the
  firmware;
- bird classification/retention configuration grouped under
  `camera.requested.pipeline`;
- `metadata_error` when an older client omitted metadata or sent malformed
  metadata.

Camera profiles are stored under `profiles/`. `manifest.jsonl` can be read one
JSON object per line by Python, pandas, PyTorch, TensorFlow, or another dataset
loader. Image and sidecar files are written through temporary files and
atomically renamed; manifest writes are serialized and flushed before the API
reports success.

The firmware decodes live exposure from sensor-bank registers `0x45[5:0]`,
`0x10`, and `0x04[1:0]`, and live analog gain from register `0x00`. Metadata keys
name the physical bank/address; the driver's virtual `0x104` therefore becomes
`sensor_bank_0x04_mask_0x03`. `register_read_boot_us` is recorded before the
first SCCB read. The multi-register read and full dump are not atomic hardware
snapshots, and `CAMERA_GRAB_WHEN_EMPTY` plus sensor AEC/AGC latency means the
frame can predate the registers by more than the measured software interval.
Manual capture drains the single queued idle framebuffer before retaining the
next frame, preventing idle time from appearing as capture/register lag.
`frame_utc_estimate` is computed as `device_time_utc + (frame_boot_us -
device_time_boot_us)` and is `null` if those inputs are unavailable or invalid.
It estimates the driver's framebuffer timestamp rather than exposure start and
can inherit an SNTP clock step.
The `ov2640_banked_register_dump_v1` payload is 578 canonical bytes: ASCII
`OVD1`, DSP-bank values for physical addresses `0x00` through `0xFE`, a 32-byte
DSP validity bitmap, then the equivalent sensor-bank values and validity bitmap.
Bits in each validity bitmap correspond to ascending physical addresses.

The stored `exposure_gain_product` is AEC lines multiplied by the nominal
analog-gain multiplier. It is useful for relative comparisons under the same
sensor timing and mode, but is not calibrated scene brightness, lux, or ISO.
Automatic exposure generally raises this value as the scene gets darker.

DSP registers `0xCC`, `0xCD`, and `0xCE` are stored as raw white-balance
controls. Espressif uses these registers for fixed white-balance presets, but
the public OV2640 interface does not verify that they are readable live AWB
channel results. The dataset therefore does not mislabel them as automatic
per-frame red, green, and blue gains.

Calibration categories describe the intended capture; they cannot verify the
physical target. Cover the lens for a dark frame, use a uniformly illuminated
neutral target for a flat field, use a neutral gray card for gray-card captures,
use a known color chart for color-chart captures, and use an appropriate chart
for resolution-target captures. New API requests accept exactly `normal`,
`dark_frame`, `flat_field`, `gray_card`, `color_chart`, and
`resolution_target`; existing older sidecars are not rewritten.

## Raspberry Pi receiver and models

Copy the `server` directory and the two selected TFLite model directories to
the Pi. Create an environment and install the server, image-processing, and
TFLite runtime dependencies appropriate to the Pi's Python/OS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r server/requirements.txt
python3 -m pip install ai-edge-litert
```

If `ai-edge-litert` has no wheel for the Pi image, install `tflite-runtime`
instead. Full TensorFlow is also supported but is normally unnecessary on the
Pi. NumPy, Pillow, and `simplejpeg` are included in `server/requirements.txt`;
only the platform-specific inference runtime is installed separately.

### Models

The default bird-presence model is the float32 split-head detector in
`Models/pi_detector_float_stride16_224`. Copy that whole directory so
`bird_detector_motion_crop_stride16_float32.tflite`, `detector_contract.json`,
`anchors.json`, and `tflite_tensor_details.json` remain together. The deployed
export is 224x224 (the exact model in
`keras_motion_crop_detector_stride16_split_heads_224`), not 288x288. The species
model is still the float32 `letterbox_224_20261009_131631/model_float32.tflite`,
with `labels.txt` beside it.

```bash
python3 server/raspberry_pi_server.py \
  --camera-url http://192.168.1.84 \
  --detector-model /opt/bird-models/detector/bird_detector_motion_crop_stride16_float32.tflite \
  --species-model /opt/bird-models/species/model_float32.tflite
```

`detector_contract.json` beside the detector is the authority for deployment.
At startup the receiver verifies the exact exported tensor-details file and its
SHA-256, model SHA-256, float32 input signature, both split output signatures,
anchor-file SHA-256, and anchor values. It also rejects unsupported geometry or
decode formulas. A mismatched model, metadata, or anchor file stops the server
instead of quietly mis-scoring birds. Use `--detector-contract-file` if the
contract lives elsewhere.

Detector preprocessing takes the Pi motion record's exact
`selection.final_bounds_px` half-open crop, adds no runtime margin, then uses the
existing `letterbox_array` implementation with pad value 114. It reproduces the
notebook's `tf.io.decode_jpeg` +
`tf.image.resize_with_pad(..., antialias=True)` exactly, without TensorFlow:
JPEGs are decoded with `simplejpeg` using TensorFlow's default fast integer
IDCT, EXIF orientation is ignored (as TensorFlow does), and the resize repeats
TensorFlow's float32 geometry and accumulation order. Receiver pixels and model
inputs match the notebook's geometry. If `simplejpeg` is missing the receiver
falls back to Pillow and prints a warning: Pillow's accurate IDCT shifts pixels
by about one level, which moved the uint8 output by up to 22 steps on borderline
test images. The species model still receives a separate black-padded letterbox
of the whole original frame.

The receiver decodes centres with `sigmoid(t) * 3 - 1`, applies the contract's
anchor/size formulas, runs NMS at IoU 0.45 with at most five boxes, and maps the
surviving boxes back to full-frame pixels. The crop score is the top box score;
the float deployment decision is `top_score >= 0.5`. Use
`--detector-threshold` only for an intentional runtime override. Every completed
`.inference.json` records the detector model SHA-256, threshold, top score,
full-frame top box, and decision. Detector failures retain the hash and threshold
with null result fields.

Both detector and species models run for every accepted JPEG. The species result
is therefore an advisory guess even when `is_bird` is false.

Set `server_ip` in the ESP32's `config.json` to the Raspberry Pi's LAN address
when moving from laptop testing to hardware deployment. Images are saved under
`~/bird_images` by default. Each completed inference is written separately as
`<capture_id>.inference.json`; the acquisition JSON sidecar is not rewritten.

The Pi receiver uses the same JPEG, JSON sidecar, and `manifest.jsonl` layout as
the laptop server, so field images can be moved into the same preprocessing
pipeline without conversion.

`--camera-url` is optional: without it, upload, classification, health, and
latest-image APIs continue to work, while camera proxy calls return HTTP 503.
When supplied, the Pi root page exposes camera capture preview, config/status,
motion/idle mode controls, and the same normalized drag-to-draw ROI editor as
the laptop dashboard. Browser requests go through the Pi's `/camera/capture`,
`/camera/config`, `/camera/status`, and `/camera/mode` proxy endpoints.

### Unattended field runs: service, time and run log

**Hub service.** Install the Pi server as a service so it starts at power-on and
restarts itself after any failure. Run once, at home, with internet:

```bash
sudo bash deploy/install_pi_hub.sh
# or, with your own arguments (models, a USB drive for captures, ...):
sudo bash deploy/install_pi_hub.sh -- --save-dir /mnt/usb/bird_images
```

Arguments live in `/etc/default/birdhub`; edit that file and run
`sudo systemctl restart birdhub`. Follow the output with
`journalctl -u birdhub -f`. The service does not wait for internet, so it comes
up at a site that has none. It replaces the hand-written `bird-pi.service`
below (the installer disables that one if it is enabled).

**Wi-Fi hotspot (no router at the site).** The Pi can be its own Wi-Fi network
that the cameras join directly:

```bash
sudo bash deploy/pi_hotspot.sh on        # field mode; stays on across reboots
sudo bash deploy/pi_hotspot.sh status    # network name, password, connected devices
sudo bash deploy/pi_hotspot.sh off       # back to your normal Wi-Fi
```

The first `on` creates a network called `BirdHub` with a random password and
prints both (choose your own with `--ssid` and `--password`). The hub is then
always at `10.42.0.1`. Put the network name and password in each camera's
`config.json` as `wifi_ssid` / `wifi_password`, and set `server_ip` to
`10.42.0.1` as the fallback for `server_host`. The Pi has a single Wi-Fi radio,
so it is either the hotspot or on your home Wi-Fi, not both: switching modes
drops a Wi-Fi SSH session (use Ethernet, or rejoin through the hotspot). With an
Ethernet cable plugged in, the hotspot also shares that internet connection.
To view the dashboard or give the hub the time, join `BirdHub` from your phone
and open `http://10.42.0.1:5000`. Needs Raspberry Pi OS Bookworm or newer.

**Time without internet or a clock module.** A Pi has no battery clock, and the
cameras normally take their time from internet servers. In the field:

- Cameras set their clocks from the hub: every heartbeat reply carries
  `server_time_unix_ms`. Camera and hub timestamps therefore always agree. The
  heartbeat reports `time_source` (`ntp`, `hub` or `none`).
- Any phone or laptop that opens the dashboard offers its own clock to the hub
  (`POST /api/time`). If the hub has no network time and disagrees by more than
  two seconds, the hub's clock is corrected and the offset is written to the
  run log, so timestamps taken before the visit can be fixed afterwards.
  **Open the dashboard from your phone when you set the kit up at the site**:
  that one visit gives the whole power-up correct time.
- The hub saves the latest time every minute and never lets its clock run
  backwards after a power cut. After a cut it resumes from the last saved time,
  so it runs behind by however long the power was off until a phone visits.
  The dashboard shows `HUB CLOCK UNVERIFIED` until then, and every run-log line
  records the clock state (`ntp`, `system`, `reference`, `offset`, `unverified`).

With a clock module fitted later, none of this changes; the phone visit simply
finds nothing to correct.

**Run log.** Everything the fleet page shows only briefly is kept permanently in
`<save-dir>/runlog/run-YYYY-MM-DD.jsonl`, one JSON object per line: every
heartbeat, each camera boot (with its reset reason), camera event-log lines,
crashes, cameras added or re-addressed, clock corrections, server starts, and a
hub status line every 5 minutes (free disk, CPU temperature, and the Pi's
under-voltage flag, which is the one to watch on a battery). Each line also
carries a monotonic time and the hub power-up it belongs to, so durations stay
right even where the wall clock was wrong. Expect very roughly 10 MB per camera per day.

Summarise a run afterwards:

```bash
python server/run_report.py ~/bird_images/runlog
```

It prints, per camera, the share of time it was reachable, the gaps, restarts by
cause, crashes, Wi-Fi signal and memory lows, and for the hub its restarts,
clock corrections, power warnings and lowest free disk.

### Start the Pi receiver at boot

After verifying the command interactively, create `/etc/systemd/system/bird-pi.service`
and adjust the user, working directory, camera address, and model paths:

```ini
[Unit]
Description=ESP32-CAM bird inference receiver
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/esp32-classifier
ExecStart=/opt/esp32-classifier/.venv/bin/python server/raspberry_pi_server.py --host 0.0.0.0 --port 5000 --camera-url http://192.168.1.84 --save-dir /home/pi/bird_images --detector-model /opt/bird-models/detector/bird_detector_motion_crop_stride16_float32.tflite --species-model /opt/bird-models/species/model_float32.tflite --num-threads 1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable it with:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bird-pi.service
sudo systemctl status bird-pi.service
```

## Laptop polling monitor for a deployed Pi

Use `pi_monitor.py` on a laptop to watch the Pi receiver without mounting or
writing to its filesystem. `--pi-url` is the Pi receiver's base URL, while
`--mirror-dir` is a directory on the laptop:

```powershell
python .\server\pi_monitor.py --pi-url http://192.168.1.50:5000 --mirror-dir .\pi_mirror
```

Open <http://127.0.0.1:5050> for the monitor dashboard. It polls every two
seconds by default; use `--poll-seconds` to change the interval. The default
mirror directory is `~/Pictures/BirdPiMirror`, or pass `--no-mirror` to display
results without retaining local copies.

The monitor mirrors JPEGs and completed `.inference.json` records. It verifies
an image SHA-256 when the Pi summary provides one, tracks pending inference, and
stores a `.pi-monitor-cursor` so the same mirror directory resumes after a
restart. It is a pull client: do not set the ESP32's `server_ip` to the laptop
when using it. Keep `server_ip` pointed at the Pi, leave the Pi receiver running,
and allow laptop-to-Pi TCP port 5000 on the trusted LAN.

The Pi exposes these read-only polling endpoints to the monitor and other local
clients:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Receiver, queue, latest-inference, and loaded-model status |
| `GET` | `/api/captures?after=CAPTURE_ID&limit=50` | Capture summaries in ID order; returns `next_after` and `has_more` for paging (`limit` is clamped to 1-200) |
| `GET` | `/api/captures/<capture_id>` | One capture summary and its image/inference URLs |
| `GET` | `/api/captures/<capture_id>/image` | Stored JPEG bytes |
| `GET` | `/api/captures/<capture_id>/inference` | Derived inference JSON, or HTTP 202 with `status: "pending"` until inference is available |

These endpoints, like the rest of the receiver, have no authentication. Keep
them on a trusted LAN and do not forward port 5000 from the public internet.

## Automated checks

### Detector-only field collection and Pi review

Use `--collection-mode` on the Pi for dataset collection. This skips loading and
running the species model. Camera acquisition runs independently from detector
inference at `--motion-poll-seconds` (two seconds is a starting point, not a
guaranteed rate). Every acquired JPEG and its original acquisition sidecar are
retained, including empty, uncertain and failed-inference examples. A camera has
one waiting inference job: if processing falls behind, the newest waiting frame
replaces it, but the superseded original stays on disk with `status: skipped`.
This bounds inference memory without silently losing the evidence of short visits.
Capture pauses below 256 MiB of free disk space; it does not delete originals.

```bash
source ~/.venv/bin/activate
cd ~/birdserver
LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1 \
python server/raspberry_pi_server.py --collection-mode \
  --deployment-id garden-feeder-20261116 --num-threads 1 \
  --motion-poll-seconds 2 --motion-timeout-seconds 10
```

The existing camera registry and heartbeats supply camera addresses; omit a
fixed `--camera-url` when cameras already register with the Pi. Keep
`onboard_motion_detection: false` on each camera. Use a new deployment ID when
the scene changes. Camera ID plus deployment ID becomes the manifest split
group, so frames from the same scene stay together. No random train/test split
is assigned automatically. The Pi Zero still has limited processing capacity:
use measured capture rates and skipped-inference counters when adding cameras.

From Windows, the prepared updater backs up Pi server code, copies the changed
files, and enables a systemd override using the **existing** `~/.venv` and models:

```powershell
.\deploy\update_pi_collection.ps1 -PiAddress 192.168.7.128 -DeploymentId garden-feeder-20261116
```

It requires the existing `birdhub.service`, SSH authentication and sudo. It does
not install Python packages. The service override is
`/etc/systemd/system/birdhub.service.d/collection.conf`; removing that override
and reloading/restarting systemd restores the previous service arguments.
The code backup path is printed during transfer.

The Pi root page is the unified **Bird hub**. Live & capture, Review data and
Settings share one camera selection and visual layout. Old `/fleet` and `/legacy`
bookmarks display the same interface (`/legacy` initially selects Settings).
The laptop proxy also uses this interface when connected to a Pi. On the laptop, launch:

No laptop server is needed in the field: join the Pi's hotspot and open
`http://10.42.0.1:5000`. The same hub page runs locally on the Pi
without internet or external web assets. Live & capture includes previews,
manual/calibration capture, ROI editing and MJPEG framing. Streaming holds the
camera's operation lock and pauses its acquisition; close the stream to resume.
Sensor settings and hub-wide processing switches are in the Settings tab.
The updater also copies `deploy/pi_hotspot.sh`, which can enable the existing
NetworkManager hotspot setup. Cameras must join that hotspot too; keep
`server_host: birdhub.local` and use `server_ip: 10.42.0.1` as fallback.
Opening the dashboard offers the laptop/phone time through the hub clock API;
when the OS clock cannot be set, the existing clock-offset logging applies.

```powershell
python .\server\laptop_test_server.py --pi-url http://192.168.7.128:5000 --no-advertise
```

Open the Pi section on the laptop overview, then **Open collection monitor &
review**, or visit `/pi`. The connection URL is saved in the laptop capture
directory's `pi_monitor.json`. This monitor reads the Pi APIs rather than
starting a second camera poller. `--no-advertise` avoids competing with the Pi's
`birdhub.local` name. Avoid registering Pi-owned cameras in a second receiver;
close direct camera streams after framing because streaming blocks acquisition.

Review the complete original frame before **Confirm empty**. Draw/remove boxes
to tightly enclose every visible bird, then **Confirm bird boxes**. A species
name is optional and applies to all boxes in that frame; leave it blank for
unknown or mixed species. Reviews and their history are saved separately in
`.review.json`; acquisition sidecars and model predictions are not rewritten.
Model crops in `crops/` and human-corrected crops in `review_crops/` are native
resolution PNGs with half-open original-frame pixel coordinates, no padding,
and no resizing. They include hashes and retain links to the original image.
The detector's proposed boxes can be poor: exported candidates are not ground truth.

**Export dataset CSV** uses the existing detector manifest's column order,
followed by camera, deployment, time and model provenance. Only human-reviewed
`boxed` and `no_bird` rows have `eligible_for_training=True`. Empty rows have
`is_Bird=FALSE`, `bird_count=0`, and blank box/species fields. Unreviewed model
negatives and skipped frames are never exported as confirmed empties. CSV image
paths are relative to the original capture directory; copy the original JPEGs
and sidecars as well, and point the existing bbox editor at that directory.
Legacy schema-4 captures lacking an ROI retain that missing provenance; no
current ROI or hardware identity is invented for them.

For an optional local data mirror, the separate `pi_monitor.py` additionally
downloads original metadata and available crops with checksum verification.
Reviews made after a historical frame was mirrored need to be re-synced; the
live collection review page always reads current Pi records.

From the project root:

```powershell
python -m unittest discover -s test -v
python -m py_compile .\server\capture_storage.py .\server\roi_control.py .\server\laptop_test_server.py .\server\raspberry_pi_server.py .\server\bird_inference.py .\server\pi_monitor.py
pio run -e esp32cam
```

## Useful checks

Laptop or Pi server health:

```text
http://SERVER_IP:5000/health
```

Latest received motion image:

```text
http://SERVER_IP:5000/latest.jpg
```
