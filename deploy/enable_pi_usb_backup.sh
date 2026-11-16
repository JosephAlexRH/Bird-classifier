#!/usr/bin/env bash
# Existing FAT USB only. Does not format storage or modify capture service.
set -euo pipefail
[[ ${EUID} -eq 0 ]] || { echo 'Run this script with sudo.' >&2; exit 1; }
RUN_USER="${SUDO_USER:-jrussell}"
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
RUN_UID="$(id -u "${RUN_USER}")"
RUN_GID="$(id -g "${RUN_USER}")"
MOUNT=/mnt/bird-backup
UUID=FC1B-D5B3
SCRIPT="${RUN_HOME}/birdserver/scripts/backup_pi_usb.py"
[[ -f "${SCRIPT}" && -d "${RUN_HOME}/bird_images" ]]
# Check current mount before changing fstab or installing anything.
ACTUAL="$(findmnt --noheadings --raw --mountpoint "${MOUNT}" --output UUID)"
[[ "${ACTUAL^^}" == "${UUID}" ]] || { echo 'Wrong USB volume; stopping.' >&2; exit 1; }
if awk '$1 !~ /^#/ && $2 == "/mnt/bird-backup" {found=1} END {exit !found}' /etc/fstab; then
  awk '$1 == "UUID=FC1B-D5B3" && $2 == "/mnt/bird-backup" && $3 == "vfat" {found=1} END {exit !found}' /etc/fstab || {
    echo 'Existing fstab entry conflicts; leave it unchanged and inspect it.' >&2; exit 1;
  }
else
  cp /etc/fstab "/etc/fstab.before-bird-backup-$(date +%Y%m%dT%H%M%S)"
  printf '\nUUID=%s %s vfat uid=%s,gid=%s,umask=0077,nofail,x-systemd.device-timeout=5s 0 0\n' \
    "${UUID}" "${MOUNT}" "${RUN_UID}" "${RUN_GID}" >> /etc/fstab
fi
cat >/etc/systemd/system/birdhub-usb-backup.service <<EOF
[Unit]
Description=Verified incremental bird capture backup to USB
RequiresMountsFor=${MOUNT}

[Service]
Type=oneshot
User=${RUN_USER}
ExecStart=/usr/bin/python3 ${SCRIPT} --source ${RUN_HOME}/bird_images --mount ${MOUNT} --uuid ${UUID}
Nice=15
IOSchedulingClass=idle
TimeoutStartSec=infinity
EOF
cat >/etc/systemd/system/birdhub-usb-backup.timer <<'EOF'
[Unit]
Description=Back up bird capture data every 15 minutes

[Timer]
OnBootSec=2min
OnUnitInactiveSec=15min
Unit=birdhub-usb-backup.service

[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now birdhub-usb-backup.timer
systemctl start birdhub-usb-backup.service
systemctl --no-pager status birdhub-usb-backup.timer
journalctl -u birdhub-usb-backup.service -n 25 --no-pager
