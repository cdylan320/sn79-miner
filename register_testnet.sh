#!/usr/bin/env bash
# Helper script to restore a wallet from JSON and register on testnet (netuid 366).
# Requirements: the bittensor v10 venv at $HOME/btcli-latest and your exported bittensor.json.
set -euo pipefail

# ---- Load env file if present -----------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/miner.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # export all vars defined in the env file
  set -a
  source "${ENV_FILE}"
  set +a
fi

# ---- User-configurable vars -------------------------------------------------
VENV="${VENV:-$SCRIPT_DIR/.venv}"
JSON_PATH="${JSON_PATH:-/home/ocean/Draven/sn79-miner/bittensor.json}"
JSON_PASSWORD="${JSON_PASSWORD:-}"
WALLET_NAME="${WALLET_NAME:-cold_draven}"
HOTKEY_NAME="${HOTKEY_NAME:-miner}"
WALLET_PATH="${WALLET_PATH:-$HOME/.bittensor/wallets}"
NETUID="${NETUID:-366}"
NETWORK="${NETWORK:-test}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

if [[ -z "${JSON_PASSWORD}" ]]; then
  echo "Set JSON_PASSWORD to your bittensor.json password before running." >&2
  exit 1
fi

# ---- Environment limits to avoid OpenBLAS thread spawn issues ---------------
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Verbose bittensor logging/progress (can override or unset)
export BT_LOGGING="${BT_LOGGING:-DEBUG}"
export BT_PROGRESS="${BT_PROGRESS:-1}"
# Ensure wallet path and override bittensor default location
mkdir -p "${WALLET_PATH}"
export BT_WALLET_PATH="${WALLET_PATH}"
# If HOME resolves to /root, force it to the current user to avoid permission issues
if [[ "${HOME}" == "/root" ]]; then
  export HOME="/home/ocean"
fi

# ---- Activate venv ----------------------------------------------------------
if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "Venv not found at ${VENV}; create it and install bittensor first." >&2
  exit 1
fi
source "${VENV}/bin/activate"

# ---- Restore wallet and register -------------------------------------------
python "${SCRIPT_DIR}/register_pow.py"

