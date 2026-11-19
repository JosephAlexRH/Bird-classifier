#!/usr/bin/env bash
# Specific to Joe's mounted MEMORYSAVER drive. Never deletes capture files.
set -euo pipefail
[[ ${EUID} -eq 0 ]] || { echo 'Run with sudo bash.' >&2; exit 1; }
RUN_USER="${SUDO_USER:-jrussell}"
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
SOURCE="${RUN_HOME}/bird_images"
MOUNT=/mnt/bird-backup
UUID=FC1B-D5B3
TARGET="${MOUNT}/birdhub-backup/bird_images"
SCRIPT="${RUN_HOME}/birdserver/scripts/backup_pi_usb.py"
RETAINED="${RUN_HOME}/bird_images.before-usb-$(date -u +%Y%m%dT%H%M%SZ)"
OVERRIDE=/etc/systemd/system/birdhub.service.d/usb-storage.conf
[[ -d "${SOURCE}" && ! -L "${SOURCE}" && ! -e "${RETAINED}" && ! -e "${OVERRIDE}" ]] || {
  echo 'Already migrated, or paths conflict; nothing changed.' >&2; exit 1;
}
sudo -u "${RUN_USER}" /usr/bin/python3 "${SCRIPT}" --mount "${MOUNT}" --uuid "${UUID}" --check-only
awk '$1 == "UUID=FC1B-D5B3" && $2 == "/mnt/bird-backup" && $3 == "vfat" {found=1} END {exit !found}' /etc/fstab || {
  echo 'Install the persistent USB mount first with enable_pi_usb_backup.sh.' >&2; exit 1;
}
# Do not proceed if a full additional copy cannot fit (conservative).
NEEDED="$(du -sb "${SOURCE}" | cut -f1)"
AVAILABLE="$(df -B1 --output=avail "${MOUNT}" | tail -1 | tr -d ' ')"
(( AVAILABLE > NEEDED + 268435456 )) || { echo 'Insufficient USB space.' >&2; exit 1; }
systemctl disable --now birdhub-usb-backup.timer
systemctl stop birdhub-usb-backup.service
systemctl stop birdhub.service
trap 'echo "Migration interrupted. Originals are preserved. Capture may remain stopped; inspect the error before restarting." >&2' ERR
# All writers are now stopped; report must verify before switching paths.
sudo -u "${RUN_USER}" /usr/bin/python3 "${SCRIPT}" --source "${SOURCE}" --mount "${MOUNT}" --uuid "${UUID}"
sync
# Retain the full old directory on SD. No source cleanup is automatic.
sudo -u "${RUN_USER}" mv -- "${SOURCE}" "${RETAINED}"
sudo -u "${RUN_USER}" ln -s -- "${TARGET}" "${SOURCE}"
MOUNT_UNIT="$(systemd-escape --path --suffix=mount "${MOUNT}")"
mkdir -p /etc/systemd/system/birdhub.service.d
cat >"${OVERRIDE}" <<EOF
[Unit]
RequiresMountsFor=${MOUNT}
BindsTo=${MOUNT_UNIT}
After=${MOUNT_UNIT}

[Service]
ExecStartPre=/usr/bin/python3 ${SCRIPT} --mount ${MOUNT} --uuid ${UUID} --check-only
EOF
systemctl daemon-reload
systemctl start birdhub.service
systemctl --no-pager --full status birdhub.service
echo "Active data: ${TARGET} (via ${SOURCE})"
echo "Retained SD copy: ${RETAINED}"
echo 'USB backup timer disabled: USB is now primary storage, not an independent backup.'
echo 'Keep the retained copy until a verified laptop archive exists.'
df -h "${SOURCE}"
