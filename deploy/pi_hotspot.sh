#!/usr/bin/env bash
# Turn the Raspberry Pi into its own Wi-Fi network for sites with no router.
# The cameras join the Pi directly; the hub is always at 10.42.0.1.
#
#   sudo bash deploy/pi_hotspot.sh on                      # field mode (stays on across reboots)
#   sudo bash deploy/pi_hotspot.sh on --ssid BirdHub --password "choose-a-password"
#   sudo bash deploy/pi_hotspot.sh off                     # back to your normal Wi-Fi
#   sudo bash deploy/pi_hotspot.sh status                  # mode, network name, password, who is connected
#
# The Pi has one Wi-Fi radio, so it is either a hotspot or on your home Wi-Fi,
# never both. If you are connected to the Pi over Wi-Fi (SSH), switching modes
# cuts that connection: reconnect by joining the hotspot, or use Ethernet.
#
# Needs Raspberry Pi OS "Bookworm" or newer (NetworkManager).
set -euo pipefail

PROFILE=birdhub-hotspot
HUB_ADDRESS=10.42.0.1
SSID=""
PASSWORD=""
ASSUME_YES=0

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

ACTION="${1:-}"
[[ -n "${ACTION}" ]] && shift || usage 1
case "${ACTION}" in -h|--help|help) usage 0 ;; esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssid) SSID="${2:-}"; shift 2 ;;
    --password) PASSWORD="${2:-}"; shift 2 ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

if [[ ${EUID} -ne 0 ]]; then
  echo "Please run with sudo: sudo bash $0 ${ACTION}" >&2
  exit 1
fi
if ! command -v nmcli >/dev/null 2>&1 || ! systemctl is-active --quiet NetworkManager; then
  echo "This script needs NetworkManager (Raspberry Pi OS Bookworm or newer)." >&2
  echo "On an older system, update Raspberry Pi OS or set up hostapd by hand." >&2
  exit 1
fi

WIFI_DEVICE="$(nmcli -t -f DEVICE,TYPE device | awk -F: '$2 == "wifi" { print $1; exit }')"
if [[ -z "${WIFI_DEVICE}" ]]; then
  echo "No Wi-Fi adapter found." >&2
  exit 1
fi

profile_exists() { nmcli -t -f NAME connection show | grep -Fxq "${PROFILE}"; }
profile_get() { nmcli -s -g "$1" connection show "${PROFILE}" 2>/dev/null || true; }
is_active() { nmcli -t -f NAME connection show --active | grep -Fxq "${PROFILE}"; }

warn_about_remote_session() {
  if [[ -n "${SSH_CONNECTION:-}" && ${ASSUME_YES} -eq 0 ]]; then
    echo "You are connected over the network. If that is Wi-Fi, this session will drop in 8 seconds."
    echo "Press Ctrl+C now to cancel (or pass --yes to skip this pause)."
    sleep 8
  fi
}

show_status() {
  if ! profile_exists; then
    echo "Hotspot: not set up yet. Run: sudo bash $0 on"
    return
  fi
  local name password
  name="$(profile_get 802-11-wireless.ssid)"
  password="$(profile_get 802-11-wireless-security.psk)"
  if is_active; then
    echo "Hotspot: ON (field mode)"
  else
    echo "Hotspot: off (normal Wi-Fi mode)"
  fi
  echo "  Network name:  ${name}"
  echo "  Password:      ${password}"
  echo "  Hub address:   ${HUB_ADDRESS}   (dashboard: http://${HUB_ADDRESS}:5000 or http://birdhub.local:5000)"
  echo "  Starts at boot: $(profile_get connection.autoconnect)"
  if is_active; then
    echo "  Connected devices:"
    if command -v iw >/dev/null 2>&1; then
      iw dev "${WIFI_DEVICE}" station dump | awk '/^Station/ { print "    " $2 }'
    fi
    local leases="/var/lib/NetworkManager/dnsmasq-${WIFI_DEVICE}.leases"
    [[ -r "${leases}" ]] && awk '{ print "    " $3 "  " $4 }' "${leases}"
  fi
}

case "${ACTION}" in
  on)
    if profile_exists; then
      [[ -n "${SSID}" ]] && nmcli connection modify "${PROFILE}" 802-11-wireless.ssid "${SSID}"
      if [[ -n "${PASSWORD}" ]]; then
        [[ ${#PASSWORD} -ge 8 ]] || { echo "The password must be at least 8 characters." >&2; exit 1; }
        nmcli connection modify "${PROFILE}" 802-11-wireless-security.psk "${PASSWORD}"
      fi
    else
      SSID="${SSID:-BirdHub}"
      if [[ -z "${PASSWORD}" ]]; then
        PASSWORD="$(tr -dc 'a-z0-9' </dev/urandom | head -c 12 || true)"
      fi
      [[ ${#PASSWORD} -ge 8 ]] || { echo "The password must be at least 8 characters." >&2; exit 1; }
      # 2.4 GHz only (the ESP32 has no 5 GHz radio), WPA2, and protected
      # management frames off because some ESP32 firmware cannot join with them on.
      nmcli connection add type wifi ifname "${WIFI_DEVICE}" con-name "${PROFILE}" ssid "${SSID}" \
        802-11-wireless.mode ap \
        802-11-wireless.band bg \
        802-11-wireless.channel 6 \
        802-11-wireless.powersave 2 \
        802-11-wireless-security.key-mgmt wpa-psk \
        802-11-wireless-security.proto rsn \
        802-11-wireless-security.pairwise ccmp \
        802-11-wireless-security.pmf 1 \
        802-11-wireless-security.psk "${PASSWORD}" \
        ipv4.method shared \
        ipv4.addresses "${HUB_ADDRESS}/24" \
        ipv6.method disabled >/dev/null
    fi
    if command -v iw >/dev/null 2>&1 && iw reg get 2>/dev/null | grep -q "country 00"; then
      echo "NOTE: no Wi-Fi country is set, which can stop the hotspot from starting."
      echo "      Set it once with: sudo raspi-config nonint do_wifi_country US"
    fi
    # Higher priority than your home Wi-Fi so the hotspot wins at every boot.
    nmcli connection modify "${PROFILE}" connection.autoconnect yes connection.autoconnect-priority 100
    warn_about_remote_session
    nmcli connection up "${PROFILE}" >/dev/null
    echo
    show_status
    cat <<NOTE

Point each camera at this network (config.json on its SD card, or the config editor):
  "wifi_ssid":     the network name above
  "wifi_password": the password above
  "server_ip":     "${HUB_ADDRESS}"      (fallback; "server_host": "birdhub.local" finds it by name)
The hotspot stays on across reboots until you run: sudo bash $0 off
NOTE
    ;;
  off)
    if ! profile_exists; then
      echo "Hotspot was never set up; nothing to turn off."
      exit 0
    fi
    warn_about_remote_session
    nmcli connection modify "${PROFILE}" connection.autoconnect no
    nmcli connection down "${PROFILE}" >/dev/null 2>&1 || true
    nmcli device wifi rescan ifname "${WIFI_DEVICE}" >/dev/null 2>&1 || true
    sleep 3
    nmcli device connect "${WIFI_DEVICE}" >/dev/null 2>&1 || \
      echo "Could not rejoin a known Wi-Fi network yet; it will be retried automatically."
    echo "Hotspot off. Current Wi-Fi: $(nmcli -t -f ACTIVE,SSID device wifi | awk -F: '$1 == "yes" { print $2 }')"
    ;;
  status)
    show_status
    ;;
  *)
    usage 1
    ;;
esac
