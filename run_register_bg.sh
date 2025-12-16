#!/usr/bin/env bash
# Run register_testnet.sh in background with logging (for environments without systemd)
# Usage:
#   LOG_FILE=/home/ocean/Draven/sn79-miner/pow.log ./run_register_bg.sh
#   tail -f /home/ocean/Draven/sn79-miner/pow.log
# To stop:
#   kill $(cat /home/ocean/Draven/sn79-miner/register.pid)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/miner.env}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/pow.log}"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/register.pid}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

mkdir -p "$(dirname "${LOG_FILE}")"

# NOTE: register_testnet.sh handles logging to $LOG_FILE.
# Don't redirect here, otherwise you'll double-write logs (because register_testnet.sh tees/appends too).
nohup env LOG_FILE="${LOG_FILE}" "${SCRIPT_DIR}/register_testnet.sh" >/dev/null 2>&1 &
echo $! > "${PID_FILE}"

echo "Started register_testnet.sh in background."
echo "PID: $(cat "${PID_FILE}")"
echo "Log: ${LOG_FILE}"

