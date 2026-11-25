#!/usr/bin/env bash
# Enable collection using the existing working venv and registry. No pip changes.
set -euo pipefail
if [[ ${EUID} -ne 0 ]]; then
  echo "Run with sudo bash deploy/enable_pi_collection.sh [deployment-id]" >&2
  exit 1
fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-jrussell}"
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
DEPLOYMENT_ID="${1:-feeder-test}"
if [[ ! "${DEPLOYMENT_ID}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Deployment ID must use only letters, numbers, underscores and hyphens." >&2
  exit 1
fi
if [[ ! -x "${RUN_HOME}/.venv/bin/python" ]]; then
  echo "Existing runtime not found at ${RUN_HOME}/.venv/bin/python" >&2
  exit 1
fi
if ! systemctl cat birdhub.service >/dev/null 2>&1; then
  echo "Install the birdhub service first. This script only updates an existing service." >&2
  exit 1
fi
# Syntax-check before restarting a working field service.
sudo -u "${RUN_USER}" "${RUN_HOME}/.venv/bin/python" -m py_compile \
  "${REPO_DIR}/server/bird_inference.py" "${REPO_DIR}/server/pi_motion.py" \
  "${REPO_DIR}/server/collection.py" "${REPO_DIR}/server/collection_dashboard.py" \
  "${REPO_DIR}/server/camera_tools.py" \
  "${REPO_DIR}/server/hub_dashboard.py" "${REPO_DIR}/server/hub_settings.py" \
  "${REPO_DIR}/server/scene_background.py" "${REPO_DIR}/server/motion_crop.py" \
  "${REPO_DIR}/server/raspberry_pi_server.py"
mkdir -p /etc/systemd/system/birdhub.service.d
OVERRIDE=/etc/systemd/system/birdhub.service.d/collection.conf
if [[ -f "${OVERRIDE}" ]]; then
  cp "${OVERRIDE}" "${OVERRIDE}.backup-$(date +%Y%m%dT%H%M%S)"
fi
# The Pi Zero W (ARMv6) needs libatomic preloaded for its hand-built tflite-runtime;
# 64-bit boards such as the Pi Zero 2 W use ai-edge-litert and need nothing.
PRELOAD_LINE=""
if [[ "$(uname -m)" == armv6l && -f /usr/lib/arm-linux-gnueabihf/libatomic.so.1 ]]; then
  PRELOAD_LINE="Environment=LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1"
fi
# Leave one core for camera acquisition and the dashboard (override with BIRDHUB_THREADS).
CORES="$(nproc)"
NUM_THREADS="${BIRDHUB_THREADS:-$(( CORES > 1 ? CORES - 1 : 1 ))}"
cat >"${OVERRIDE}" <<EOF
[Service]
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
Environment=PYTHONUNBUFFERED=1
${PRELOAD_LINE}
ExecStart=
ExecStart=${RUN_HOME}/.venv/bin/python server/raspberry_pi_server.py --host 0.0.0.0 --port 5000 --save-dir ${RUN_HOME}/bird_images --collection-mode --deployment-id ${DEPLOYMENT_ID} --num-threads ${NUM_THREADS} --motion-poll-seconds 2 --motion-timeout-seconds 10
EOF
# No fixed camera IP: retain the registry and follow camera heartbeats after DHCP changes.
systemctl daemon-reload
systemctl enable birdhub.service
systemctl restart birdhub.service
systemctl --no-pager --full status birdhub.service
echo "Hub updated for ${DEPLOYMENT_ID}. Open http://$(hostname -I | awk '{print $1}'):5000"
echo "Saved dashboard settings take precedence over CLI defaults. Use the dataset collection preset to start a run."
echo "Follow startup: journalctl -u birdhub -n 40 --no-pager"
