#!/usr/bin/env bash
# Setup a one-shot systemd service to run register_testnet.sh once and stop.
# Usage: sudo bash install_register_service.sh

set -euo pipefail

SERVICE_NAME="taos-register"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
WORKDIR="/home/ocean/Draven/sn79-miner"
ENV_FILE="${WORKDIR}/miner.env"
LOG_FILE="${WORKDIR}/pow.log"
USER_NAME="ocean"

cat <<EOF | sudo tee "${SERVICE_FILE}" >/dev/null
[Unit]
Description=TAOS Testnet Registration (one-shot)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=${USER_NAME}
WorkingDirectory=${WORKDIR}
EnvironmentFile=${ENV_FILE}
Environment=LOG_FILE=${LOG_FILE}
ExecStart=${WORKDIR}/register_testnet.sh
TimeoutStartSec=1800

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo "Service installed: ${SERVICE_NAME}"
echo "Start:   sudo systemctl start ${SERVICE_NAME}"
echo "Status:  sudo systemctl status ${SERVICE_NAME}"
echo "Logs:    journalctl -u ${SERVICE_NAME} -f"
echo "File log: tail -f ${LOG_FILE}"
echo "Enable (optional on boot): sudo systemctl enable ${SERVICE_NAME}"

