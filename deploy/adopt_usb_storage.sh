#!/usr/bin/env bash
# Use an existing bird USB drive as capture storage on a NEW Pi (for example after
# moving from the Pi Zero W to a Pi Zero 2 W). The drive already holds
# birdhub-backup/bird_images from the old hub, so nothing is copied:
#   - adds a persistent, boot-safe fstab mount (nofail, so the Pi still boots without it)
#   - links ~/bird_images to the drive
#   - makes birdhub.service wait for the drive and refuse to run on the wrong one
# Never formats, copies or deletes anything.
#
#   sudo bash deploy/adopt_usb_storage.sh
#   sudo BIRDHUB_USB_UUID=ABCD-1234 bash deploy/adopt_usb_storage.sh   # a different drive
set -euo pipefail
[[ ${EUID} -eq 0 ]] || { echo 'Run with sudo bash.' >&2; exit 1; }
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-jrussell}"
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
RUN_UID="$(id -u "${RUN_USER}")"
RUN_GID="$(id -g "${RUN_USER}")"
MOUNT=/mnt/bird-backup
UUID="${BIRDHUB_USB_UUID:-FC1B-D5B3}"
TARGET="${MOUNT}/birdhub-backup/bird_images"
LINK="${RUN_HOME}/bird_images"
SCRIPT="${REPO_DIR}/scripts/backup_pi_usb.py"
OVERRIDE=/etc/systemd/system/birdhub.service.d/usb-storage.conf

[[ -f "${SCRIPT}" ]] || { echo "Missing ${SCRIPT}; run this from the birdserver checkout." >&2; exit 1; }
[[ -e "/dev/disk/by-uuid/${UUID}" ]] || { echo "USB drive ${UUID} is not plugged in (see: lsblk -f)." >&2; exit 1; }
if [[ -L "${LINK}" ]]; then
  [[ "$(readlink -f "${LINK}")" == "${TARGET}" ]] || { echo "${LINK} already points somewhere else; stopping." >&2; exit 1; }
elif [[ -e "${LINK}" ]]; then
  echo "${LINK} is a real folder on this SD card. Move it aside (it may hold captures) and re-run." >&2
  exit 1
fi

# Persistent mount. Keep an existing matching line; refuse a conflicting one.
if awk -v m="${MOUNT}" '$1 !~ /^#/ && $2 == m {found=1} END {exit !found}' /etc/fstab; then
  awk -v u="UUID=${UUID}" -v m="${MOUNT}" '$1 == u && $2 == m && $3 == "vfat" {found=1} END {exit !found}' /etc/fstab || {
    echo "An fstab entry for ${MOUNT} exists but is for a different drive; leaving it alone." >&2; exit 1;
  }
else
  cp /etc/fstab "/etc/fstab.before-bird-usb-$(date +%Y%m%dT%H%M%S)"
  printf '\nUUID=%s %s vfat uid=%s,gid=%s,umask=0077,flush,nofail,x-systemd.device-timeout=30s 0 2\n' \
    "${UUID}" "${MOUNT}" "${RUN_UID}" "${RUN_GID}" >> /etc/fstab
  echo "Added ${MOUNT} to /etc/fstab"
fi
mkdir -p "${MOUNT}"
systemctl daemon-reload
findmnt --mountpoint "${MOUNT}" >/dev/null || mount "${MOUNT}"
python3 "${SCRIPT}" --mount "${MOUNT}" --uuid "${UUID}" --check-only

if [[ ! -d "${TARGET}" ]]; then
  echo "No birdhub-backup/bird_images on this drive yet; creating an empty one."
  sudo -u "${RUN_USER}" mkdir -p -- "${TARGET}"
fi
[[ -L "${LINK}" ]] || sudo -u "${RUN_USER}" ln -s -- "${TARGET}" "${LINK}"

MOUNT_UNIT="$(systemd-escape --path --suffix=mount "${MOUNT}")"
mkdir -p "$(dirname "${OVERRIDE}")"
cat >"${OVERRIDE}" <<EOF
[Unit]
RequiresMountsFor=${MOUNT}
BindsTo=${MOUNT_UNIT}
After=${MOUNT_UNIT}

[Service]
ExecStartPre=/usr/bin/python3 ${SCRIPT} --mount ${MOUNT} --uuid ${UUID} --check-only
EOF
systemctl daemon-reload
if systemctl is-active --quiet birdhub.service; then systemctl restart birdhub.service; fi

echo "Captures: ${LINK} -> ${TARGET}"
echo "Existing files there: $(find "${TARGET}" -type f | wc -l)"
df -h "${MOUNT}"
