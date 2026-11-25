#!/usr/bin/env bash
# Install the bird camera hub as a service on a Raspberry Pi, so it starts at
# power-on and restarts itself if it ever stops. Run it once, at home, with
# internet available (it may need to download Python packages):
#
#   sudo bash deploy/install_pi_hub.sh
#   sudo bash deploy/install_pi_hub.sh -- --save-dir /mnt/usb/bird_images
#
# Anything after "--" is passed to server/raspberry_pi_server.py. The arguments
# are stored in /etc/default/birdhub; edit that file and run
# "sudo systemctl restart birdhub" to change them later. Re-running this script
# is safe: it keeps an existing /etc/default/birdhub unless you pass new arguments.
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Please run with sudo: sudo bash $0" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-pi}"
if ! id "${RUN_USER}" >/dev/null 2>&1; then
  echo "User '${RUN_USER}' does not exist; run this with sudo from your normal account." >&2
  exit 1
fi
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"

EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  [[ $1 == "--" ]] && shift
  EXTRA_ARGS=("$@")
fi

# --- Python environment -----------------------------------------------------
# BIRDHUB_VENV lets setup_new_pi.sh share ~/.venv with the collection scripts.
VENV_DIR="${BIRDHUB_VENV:-${REPO_DIR}/.venv}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating Python environment in ${VENV_DIR} ..."
  # System packages stay visible so an apt-installed TFLite runtime keeps working.
  sudo -u "${RUN_USER}" python3 -m venv --system-site-packages "${VENV_DIR}"
fi
echo "Installing Python packages ..."
if ! sudo -u "${RUN_USER}" "${VENV_DIR}/bin/python" -m pip install --quiet -r "${REPO_DIR}/server/requirements.txt"; then
  echo "WARNING: could not install every package (no internet?). The service is still set up;" >&2
  echo "         re-run this script when the Pi is online." >&2
fi

# --- Arguments --------------------------------------------------------------
ARGS_FILE=/etc/default/birdhub
if [[ ${#EXTRA_ARGS[@]} -gt 0 || ! -f "${ARGS_FILE}" ]]; then
  ARGS="--host 0.0.0.0 --port 5000"
  if [[ " ${EXTRA_ARGS[*]-} " != *" --save-dir "* ]]; then
    ARGS+=" --save-dir ${RUN_HOME}/bird_images"
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    ARGS+=" ${EXTRA_ARGS[*]}"
  fi
  {
    echo "# Arguments for server/raspberry_pi_server.py (see: python server/raspberry_pi_server.py --help)."
    echo "# After editing: sudo systemctl restart birdhub"
    echo "BIRDHUB_ARGS=\"${ARGS}\""
  } >"${ARGS_FILE}"
  echo "Wrote ${ARGS_FILE}"
else
  echo "Keeping existing ${ARGS_FILE}"
fi

# --- Let the service set the clock -------------------------------------------
# The Pi has no battery clock. The hub corrects its time from a phone or laptop
# that opens the dashboard; this rule allows exactly that one command.
DATE_BIN="$(command -v date)"
SUDOERS_FILE=/etc/sudoers.d/birdhub-clock
echo "${RUN_USER} ALL=(root) NOPASSWD: ${DATE_BIN} -u -s @*" >"${SUDOERS_FILE}.new"
chmod 0440 "${SUDOERS_FILE}.new"
if visudo -cf "${SUDOERS_FILE}.new" >/dev/null; then
  mv "${SUDOERS_FILE}.new" "${SUDOERS_FILE}"
else
  rm -f "${SUDOERS_FILE}.new"
  echo "WARNING: could not install the clock rule; the hub will log time offsets instead of fixing its clock." >&2
fi

# --- Service ------------------------------------------------------------------
cat >/etc/systemd/system/birdhub.service <<UNIT
[Unit]
Description=Bird camera hub (ESP32-CAM receiver and classifier)
# Deliberately not network-online.target: a field site has no internet to wait for.
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${ARGS_FILE}
Environment=PYTHONUNBUFFERED=1
ExecStart=${VENV_DIR}/bin/python server/raspberry_pi_server.py \$BIRDHUB_ARGS
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

if systemctl is-enabled --quiet bird-pi.service 2>/dev/null; then
  echo "NOTE: the older bird-pi.service is enabled and would compete for the same port. Disabling it."
  systemctl disable --now bird-pi.service || true
fi

systemctl daemon-reload
systemctl enable --now birdhub.service
sleep 3
systemctl --no-pager --lines=15 status birdhub.service || true

cat <<DONE

Done. The hub now starts at power-on and restarts itself after a failure.
  Follow its output:   journalctl -u birdhub -f
  Change arguments:    sudo nano ${ARGS_FILE} && sudo systemctl restart birdhub
  Stop / start:        sudo systemctl stop birdhub   /   sudo systemctl start birdhub
  Remove:              sudo systemctl disable --now birdhub
DONE
