#!/usr/bin/env bash
# One-time setup of a NEW 64-bit hub (Pi Zero 2 W, Pi 3/4/5) from a fresh
# Raspberry Pi OS Lite (64-bit). Run from the checkout at ~/birdserver, with
# internet, and with the bird USB drive plugged in if you use one:
#
#   sudo bash deploy/setup_new_pi.sh
#
# Steps: system packages -> ~/.venv with ai-edge-litert -> model smoke test ->
# USB capture storage (if the drive is present) -> birdhub.service.
# Afterwards start collection with: sudo bash deploy/enable_pi_collection.sh <deployment-id>
set -euo pipefail
[[ ${EUID} -eq 0 ]] || { echo 'Run with sudo bash.' >&2; exit 1; }
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-jrussell}"
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
VENV="${RUN_HOME}/.venv"
UUID="${BIRDHUB_USB_UUID:-FC1B-D5B3}"

if [[ "$(uname -m)" != aarch64 ]]; then
  echo "This is $(uname -m), not aarch64. ai-edge-litert only ships 64-bit ARM wheels:" >&2
  echo "reflash with Raspberry Pi OS Lite (64-bit)." >&2
  exit 1
fi

echo "== System packages"
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git >/dev/null

echo "== Python environment ${VENV}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  sudo -u "${RUN_USER}" python3 -m venv --system-site-packages "${VENV}"
fi
sudo -u "${RUN_USER}" "${VENV}/bin/python" -m pip install --quiet --upgrade pip
sudo -u "${RUN_USER}" "${VENV}/bin/python" -m pip install --quiet -r "${REPO_DIR}/server/requirements.txt"

echo "== Model smoke test"
sudo -u "${RUN_USER}" "${VENV}/bin/python" - "${REPO_DIR}" <<'PY'
import os, sys, time
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]); sys.path.insert(0, str(root / "server"))
from bird_inference import _interpreter_class
Interpreter, runtime = _interpreter_class()
threads = max(1, (os.cpu_count() or 1) - 1)
for model in [root / "Models/pi_detector_float_stride16_224/bird_detector_motion_crop_stride16_float32.tflite",
              root / "Models/runs/letterbox_224_20261009_131631/model_float32.tflite"]:
    if not model.exists():
        print(f"  missing {model.relative_to(root)}"); continue
    it = Interpreter(model_path=str(model), num_threads=threads); it.allocate_tensors()
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], np.zeros(inp["shape"], dtype=inp["dtype"]))
    it.invoke()
    start = time.perf_counter()
    for _ in range(5): it.invoke()
    ms = (time.perf_counter() - start) / 5 * 1000
    print(f"  {model.name}: {ms:.0f} ms/frame ({runtime}, {threads} threads)")
PY

if [[ -e "/dev/disk/by-uuid/${UUID}" ]]; then
  echo "== USB capture storage"
  BIRDHUB_USB_UUID="${UUID}" bash "${REPO_DIR}/deploy/adopt_usb_storage.sh"
else
  echo "== USB drive ${UUID} not plugged in; captures will go to the SD card."
  echo "   Plug it in later and run: sudo bash deploy/adopt_usb_storage.sh"
fi

echo "== Hub service"
BIRDHUB_VENV="${VENV}" bash "${REPO_DIR}/deploy/install_pi_hub.sh"

cat <<DONE

New hub ready. Next:
  sudo bash deploy/enable_pi_collection.sh <deployment-id>
  sudo bash deploy/pi_hotspot.sh on        # create the BirdHub hotspot (192.168.4.1)
  sudo bash deploy/pi_hotspot.sh off       #   ...then hand boot-time choice back to:
  sudo bash deploy/birdhub_mode.sh install # field-mode jumper on GPIO pins 39-40
Swap check (512 MB RAM): swapon --show
DONE
