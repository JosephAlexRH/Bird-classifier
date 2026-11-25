#!/usr/bin/env bash
# Field/home Wi-Fi switch for the Bird Hub, chosen by a jumper on the GPIO header.
#
#   Jumper ON  pins 39 (GND) + 40 (GPIO21)  -> field mode: always start the BirdHub hotspot
#   Jumper OFF                               -> home mode: join home Wi-Fi; hotspot only if it's absent
#
#   sudo bash deploy/birdhub_mode.sh install     # install + enable the boot-time switch
#   sudo bash deploy/birdhub_mode.sh status      # what the jumper reads right now, and the Wi-Fi state
#   sudo bash deploy/birdhub_mode.sh uninstall
#
# Options (install): --pin N (BCM GPIO, default 21)  --ap NAME (default birdhub-ap)
#                    --home NAME (home Wi-Fi profile; default: auto-detected)
# The mode is read once at boot. To switch, power off, move the jumper, power on.
set -euo pipefail

PIN=21
AP=birdhub-ap
HOME_CON=""
BIN=/usr/local/sbin/birdhub-mode
UNIT=/etc/systemd/system/birdhub-mode.service

ACTION="${1:-}"; [[ -n "${ACTION}" ]] && shift || { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pin) PIN="$2"; shift 2 ;;
    --ap) AP="$2"; shift 2 ;;
    --home) HOME_CON="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ ${EUID} -eq 0 ]] || { echo "Please run with sudo." >&2; exit 1; }
command -v pinctrl >/dev/null || { echo "pinctrl not found (sudo apt install raspi-utils)." >&2; exit 1; }
command -v nmcli >/dev/null || { echo "This needs NetworkManager." >&2; exit 1; }

detect_home() {
  # First Wi-Fi client profile that isn't the hotspot.
  local name mode
  while IFS=: read -r name type; do
    [[ "${type}" == "802-11-wireless" && "${name}" != "${AP}" ]] || continue
    mode="$(nmcli -g 802-11-wireless.mode con show "${name}" 2>/dev/null || true)"
    [[ "${mode}" == "ap" ]] && continue
    echo "${name}"; return
  done < <(nmcli -t -f NAME,TYPE con show)
}

case "${ACTION}" in
install)
  nmcli -t -f NAME con show | grep -Fxq "${AP}" || { echo "Hotspot profile '${AP}' not found." >&2; exit 1; }
  [[ -n "${HOME_CON}" ]] || HOME_CON="$(detect_home)"
  [[ -n "${HOME_CON}" ]] || { echo "No home Wi-Fi profile found; pass --home NAME." >&2; exit 1; }
  echo "Hotspot: ${AP}   Home Wi-Fi: ${HOME_CON}   Jumper: GPIO${PIN} to GND"

  cat > "${BIN}" <<EOF
#!/usr/bin/env bash
# Installed by deploy/birdhub_mode.sh. Reads the field-mode jumper once at boot.
PIN=${PIN}; AP='${AP}'; HOME_CON='${HOME_CON}'
log() { logger -t birdhub-mode "\$*"; echo "\$*"; }
pinctrl set "\${PIN}" ip pu        # input with pull-up: jumper to GND reads low
sleep 0.2
if pinctrl get "\${PIN}" | grep -q '| lo'; then
  log "jumper ON (GPIO\${PIN} low): field mode, starting hotspot \${AP}"
  for attempt in 1 2 3 4 5 6; do
    nmcli con up "\${AP}" && { log "hotspot up"; exit 0; }
    log "hotspot attempt \${attempt} failed; retrying in 5 s"; sleep 5
  done
  log "hotspot failed to start; leaving Wi-Fi to NetworkManager"; exit 1
else
  log "jumper OFF: home mode (\${HOME_CON} preferred, \${AP} only if it is out of range)"
  nmcli con mod "\${HOME_CON}" autoconnect yes || true
fi
EOF
  chmod 755 "${BIN}"

  cat > "${UNIT}" <<EOF
[Unit]
Description=Bird Hub field/home Wi-Fi mode (GPIO${PIN} jumper)
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=oneshot
ExecStart=${BIN}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
  # Home Wi-Fi back on autoconnect; hotspot stays the away-from-home fallback.
  nmcli con mod "${HOME_CON}" autoconnect yes
  nmcli con mod "${AP}" autoconnect yes connection.autoconnect-priority -10
  systemctl daemon-reload
  systemctl enable birdhub-mode.service
  echo "Installed. Jumper reads: $(pinctrl get "${PIN}" 2>/dev/null | grep -q '| lo' && echo ON/field || echo OFF/home) (takes effect on next boot)."
  ;;
status)
  pinctrl set "${PIN}" ip pu; sleep 0.2
  pinctrl get "${PIN}" | grep -q '| lo' && echo "Jumper: ON (field mode at next boot)" || echo "Jumper: OFF (home mode at next boot)"
  echo "Wi-Fi now: $(nmcli -t -f NAME,DEVICE con show --active | grep ':wlan0$' | cut -d: -f1 || echo none)"
  journalctl -t birdhub-mode -b --no-pager | tail -5 || true
  ;;
uninstall)
  systemctl disable --now birdhub-mode.service 2>/dev/null || true
  rm -f "${UNIT}" "${BIN}"; systemctl daemon-reload
  echo "Removed. NetworkManager priorities are unchanged."
  ;;
*) echo "Unknown action: ${ACTION}" >&2; exit 1 ;;
esac
