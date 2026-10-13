# ESP32-CAM servers

The firmware sends raw JPEG bytes to `POST /upload` with the content type
`image/jpeg`. Capture metadata is sent in the optional
`X-ESP32-Camera-Metadata` JSON header.


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
`bird_confidence_threshold`.

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
Pi. NumPy and Pillow are included in `server/requirements.txt`; only the
platform-specific inference runtime is installed separately.

Deploy the **float32** `model_float32.tflite` from the binary is-bird run and a
**float32** species `model_float32.tflite`. Keep `labels.txt` beside the species
model and `threshold.json` beside the is-bird model. Use `--species-labels` or
`--threshold-file` if either companion file is stored elsewhere. Then start the
receiver, using paths on the Pi:

```bash
python3 server/raspberry_pi_server.py \
  --camera-url http://192.168.1.84 \
  --is-bird-model /opt/bird-models/is-bird/model_float32.tflite \
  --species-model /opt/bird-models/species/model_float32.tflite
```

The two model files are not interchangeable. Both expect a 224×224 RGB image
with letterboxing; scaling is embedded in the model. Do not substitute the
`model_int8.tflite` exports for this deployment unless you have deliberately
validated the quantized pipeline.

For this baseline, both models run for every accepted JPEG. The species result
is therefore an advisory guess even when `is_bird` is false; this deliberately
keeps the complete architecture observable while the binary model is being
retrained and recalibrated.

> **Threshold warning:** the bird decision threshold is part of the trained
> is-bird artifact, not a generic `0.5`. Deploy its matching `threshold.json`.
> `--bird-threshold` deliberately overrides that artifact value; changing it
> changes which uploads the ESP32 may retain as birds, so validate an override
> against the intended field data before deployment.
>
> This is an architecture baseline, not a production accuracy claim. At `0.818`,
> the selected model's recorded held-out ESP32 subset detected 0 of 1,245 labelled
> bird images (while rejecting all 620 labelled non-birds). Keep the raw
> `bird_probability` in each inference record visible, collect new field results,
> and retrain or recalibrate before relying on the boolean decision.

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
ExecStart=/opt/esp32-classifier/.venv/bin/python server/raspberry_pi_server.py --host 0.0.0.0 --port 5000 --camera-url http://192.168.1.84 --save-dir /home/pi/bird_images --is-bird-model /opt/bird-models/is-bird/model_float32.tflite --species-model /opt/bird-models/species/model_float32.tflite
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