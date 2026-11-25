#!/usr/bin/env bash
# Run only AFTER verifying the completed run's laptop archive. No data deletion.
set -euo pipefail
[[ ${EUID} -eq 0 ]] || { echo 'Run with sudo bash.' >&2; exit 1; }
DEPLOYMENT_ID="${1:?Supply a new deployment ID}"
[[ "${DEPLOYMENT_ID}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid deployment ID.' >&2; exit 1; }
RUN_USER="${SUDO_USER:-jrussell}"
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
MOUNT=/mnt/bird-backup
TARGET="${MOUNT}/birdhub-backup/bird_images"
RETAINED="${TARGET}.completed-${DEPLOYMENT_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
[[ -L "${RUN_HOME}/bird_images" && "$(readlink -f "${RUN_HOME}/bird_images")" == "${TARGET}" ]]
[[ -d "${TARGET}" && ! -L "${TARGET}" && ! -e "${RETAINED}" ]]
sudo -u "${RUN_USER}" /usr/bin/python3 "${RUN_HOME}/birdserver/scripts/backup_pi_usb.py" \
  --mount "${MOUNT}" --uuid FC1B-D5B3 --check-only
systemctl stop birdhub.service
trap 'echo "Fresh-run setup failed. All captures are retained; inspect the error before restarting." >&2' ERR
sudo -u "${RUN_USER}" mv -- "${TARGET}" "${RETAINED}"
sudo -u "${RUN_USER}" mkdir -- "${TARGET}"
# Carry camera identities, ROI/configuration, and dashboard settings forward;
# do not carry capture indexes, results, reviews, crops, or old run logs.
for FILE in cameras.json fleet.json hub_settings.json; do
  if [[ -f "${RETAINED}/${FILE}" ]]; then
    sudo -u "${RUN_USER}" cp -- "${RETAINED}/${FILE}" "${TARGET}/${FILE}"
  fi
done
OVERRIDE=/etc/systemd/system/birdhub.service.d/collection.conf
[[ -f "${OVERRIDE}" ]]
cp "${OVERRIDE}" "${OVERRIDE}.before-fresh-$(date -u +%Y%m%dT%H%M%SZ)"
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
WorkingDirectory=${RUN_HOME}/birdserver
Environment=PYTHONUNBUFFERED=1
${PRELOAD_LINE}
ExecStart=
ExecStart=${RUN_HOME}/.venv/bin/python server/raspberry_pi_server.py --host 0.0.0.0 --port 5000 --save-dir ${RUN_HOME}/bird_images --collection-mode --deployment-id ${DEPLOYMENT_ID} --num-threads ${NUM_THREADS} --motion-poll-seconds 2 --motion-timeout-seconds 10
EOF
systemctl daemon-reload
systemctl start birdhub.service
systemctl --no-pager --full status birdhub.service
echo "New deployment: ${DEPLOYMENT_ID}"
echo "Completed data retained on USB: ${RETAINED}"
echo 'Old SD copy is also untouched. No capture files were deleted.'
