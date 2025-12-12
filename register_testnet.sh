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
VENV="${VENV:-$HOME/btcli-latest}"
JSON_PATH="${JSON_PATH:-/home/ocean/Draven/sn79-miner/bittensor.json}"
JSON_PASSWORD="${JSON_PASSWORD:-}"
WALLET_NAME="${WALLET_NAME:-cold_draven}"
HOTKEY_NAME="${HOTKEY_NAME:-miner}"
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

# ---- Activate venv ----------------------------------------------------------
if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "Venv not found at ${VENV}; create it and install bittensor first." >&2
  exit 1
fi
source "${VENV}/bin/activate"

# ---- Restore wallet and register -------------------------------------------
python - <<'PY'
import os
import time
from pathlib import Path
from bittensor_wallet import Wallet
from bittensor.core.subtensor import Subtensor

json_path = Path(os.environ["JSON_PATH"]).expanduser()
json_password = os.environ["JSON_PASSWORD"]
wallet_name = os.environ["WALLET_NAME"]
hotkey_name = os.environ["HOTKEY_NAME"]
wallet_path = Path("~/.bittensor/wallets").expanduser()
network = os.environ["NETWORK"]
netuid = int(os.environ["NETUID"])
num_processes = int(os.environ.get("NUM_PROCESSES", 1))

print(f"[info] Using env: json_path={json_path}, wallet={wallet_name}/{hotkey_name}, "
      f"network={network}, netuid={netuid}, num_processes={num_processes}, "
      f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}, "
      f"BT_LOGGING={os.environ.get('BT_LOGGING')}, BT_PROGRESS={os.environ.get('BT_PROGRESS')}")

if not json_path.is_file():
    raise FileNotFoundError(f"JSON file not found at {json_path}")

json_data = json_path.read_text()

w = Wallet(name=wallet_name, hotkey=hotkey_name, path=str(wallet_path))
w.regenerate_coldkey(json=(json_data, json_password), use_password=False, overwrite=True, suppress=True)
w.regenerate_hotkey(json=(json_data, json_password), use_password=False, overwrite=True, suppress=True)
print(f"Restored wallet {wallet_name}/{hotkey_name} at {wallet_path}")

sub = Subtensor(network=network)
print("[info] Starting register() ...")
t0 = time.perf_counter()
resp = sub.register(
    wallet=w,
    netuid=netuid,
    mev_protection=False,
    wait_for_inclusion=True,
    wait_for_finalization=True,
    raise_error=True,
    num_processes=num_processes,
)
elapsed = time.perf_counter() - t0
print(f"[info] register() completed in {elapsed:.1f}s")
print("Registration response:", resp)
PY

