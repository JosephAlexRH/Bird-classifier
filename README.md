# ESP32 Bird Classifier

An edge-based bird-feeder monitor that captures images with ESP32-S3 devices,
detects and classifies birds on a Raspberry Pi, groups related captures into
visits, and presents the results in a local review dashboard.

The system is designed to run on a trusted local network and can operate at a
field site without internet access. Multiple cameras can register with one hub,
and capture metadata is retained alongside each image for later review,
training, and deployment analysis.

## Demonstration



## How it works

1. One or more ESP32-S3/ESP32-CAM devices provide fresh JPEG frames over Wi-Fi.
2. The Raspberry Pi maintains a per-camera background and looks for local motion.
3. Motion crops are passed to the bird detector; accepted captures can also be
   passed to a species classifier.
4. Images, acquisition metadata, motion records, and inference results are saved
   without rewriting the original evidence.
5. Captures from the same camera and time window are grouped into visits and can
   be reviewed from the local Bird Hub dashboard.

The firmware can also run its own lightweight motion detector as a fallback,
but Pi-owned motion detection is the default configuration.

## Features

- Multi-camera registration, health monitoring, and camera-specific controls
- Support for OV2640 and OV5640 camera sensors
- Colour status light on the ESP32-S3 board for on-site checks (Wi-Fi, hub link, camera faults)
- Local motion selection, bird detection, species prediction, and visit grouping
- Offline Raspberry Pi hotspot for field deployments, chosen at boot by a jumper on the hub
- Browser-based live view, camera configuration, ROI editing, and data review
- Immutable JPEG acquisition sidecars plus separate motion, inference, and review records
- Dataset collection, bounding-box review, export, backup, and deployment tools

## Hardware

The firmware builds for two boards, each with its own PlatformIO environment:
`esp32cam` (AI Thinker ESP32-CAM) and `freenove_s3` (Freenove ESP32-S3 WROOM
CAM). Either board detects and configures an OV2640 or OV5640 sensor, and both
can run side by side as one fleet.

A typical deployment uses:

- One or more AI Thinker ESP32-CAM or Freenove ESP32-S3 WROOM CAM boards
- An OV2640 or OV5640 camera module
- A Raspberry Pi hub, such as a Raspberry Pi Zero 2 W, with a USB drive for
  captures and optionally a PiRTC clock module
- A stable 5 V supply for each camera
- A FAT32 microSD card for camera configuration and optional buffering
- For the ESP32-CAM, an ESP32-CAM-MB programmer or USB-to-UART adapter (the
  Freenove S3 has USB-C on board)

See the [ESP32-S3 bring-up and field guide](docs/esp32-s3-bringup.md) for
flashing, benchmarking and the pre-deployment checklist.

## Quick start

### 1. Prepare and flash a camera

1. Install PlatformIO.
2. Copy `data/config.example.json` to the root of a FAT32 microSD card as
   `config.json`.
3. Set the 2.4 GHz Wi-Fi credentials and the Raspberry Pi or laptop receiver
   address.
4. Build and upload the firmware:

   ```powershell
   pio run -e esp32cam -t upload      # AI Thinker ESP32-CAM
   pio run -e freenove_s3 -t upload   # Freenove ESP32-S3 WROOM CAM
   pio device monitor
   ```

See the [camera firmware guide](docs/camera-firmware.md) for SPIFFS setup,
configuration fields, HTTP endpoints, operating modes, and troubleshooting.

### 2. Test with a laptop

From the repository root:

```powershell
python -m pip install -r .\server\requirements.txt
python .\server\laptop_test_server.py --camera-url http://CAMERA_IP
```

Open `http://127.0.0.1:5000`. The laptop server can verify camera connectivity,
preview frames, edit settings, draw a motion ROI, and save test captures.

### 3. Run the Raspberry Pi hub

Copy or clone the repository onto the Pi, install an appropriate TFLite runtime,
and run the supplied service installer:

```bash
sudo bash deploy/install_pi_hub.sh
```

The hub dashboard listens on port `5000`. Cameras advertise themselves by
heartbeat and normally find the hub as `birdhub.local`; a configured
`server_ip` remains the fallback.

For a site with no router, the Pi runs its own `BirdHub` Wi-Fi network (hub at
`192.168.4.1`). A jumper on GPIO pins 39-40 selects field mode at boot, so
deploying is: jumper on, USB drive in, power the hub, then the cameras:

```bash
sudo bash deploy/pi_hotspot.sh on && sudo bash deploy/pi_hotspot.sh off
sudo bash deploy/birdhub_mode.sh install
```

See the [server and deployment guide](server/README.md) for model files,
multi-camera operation, collection mode, hotspot setup, monitoring, and data
layout.

## Documentation

- [Camera firmware and configuration](docs/camera-firmware.md)
- [Laptop server, Raspberry Pi hub, models, and deployment](server/README.md)
- [Dataset, review, backup, and archive scripts](scripts/README.md)
- [Annotated-capture presentation exporter](<github pics/README.md>)

## Repository layout

| Path | Purpose |
|---|---|
| `src/`, `include/`, `data/` | Camera firmware (ESP32-CAM and ESP32-S3) and example configuration |
| `server/` | Laptop receiver, Raspberry Pi hub, dashboards, inference, and monitoring |
| `deploy/` | Raspberry Pi service, hotspot, collection, backup, and update helpers |
| `Models/` | Deployed model exports, contracts, labels, manifests, and training outputs |
| `scripts/` | Dataset review, migration, backup, and archive utilities |
| `Notebooks/` | Data preparation, training, and analysis notebooks |
| `test/` | Python unit and integration tests |

## Security

The camera and hub APIs do not provide authentication. Keep them on a trusted
LAN or on the private Raspberry Pi hotspot. Do not forward their HTTP ports to
the public internet.

## License

[MIT](LICENSE)