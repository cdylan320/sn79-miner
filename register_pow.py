#!/usr/bin/env python3
"""Register a wallet on Bittensor subnet using PoW."""
import os
import time
import threading
from pathlib import Path
from bittensor_wallet import Wallet
from bittensor.core.subtensor import Subtensor

def main():
    json_path = Path(os.environ["JSON_PATH"]).expanduser()
    json_password = os.environ["JSON_PASSWORD"]
    wallet_name = os.environ["WALLET_NAME"]
    hotkey_name = os.environ["HOTKEY_NAME"]
    wallet_path = Path(os.environ["WALLET_PATH"]).expanduser()
    network = os.environ["NETWORK"]
    netuid = int(os.environ["NETUID"])
    num_processes = int(os.environ.get("NUM_PROCESSES", 1))
    use_torch = bool(int(os.environ.get("USE_TORCH", "0")))

    print(f"[info] Using env: json_path={json_path}, wallet={wallet_name}/{hotkey_name}, "
          f"network={network}, netuid={netuid}, num_processes={num_processes}, use_torch={use_torch}, "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}, "
          f"BT_LOGGING={os.environ.get('BT_LOGGING')}, BT_PROGRESS={os.environ.get('BT_PROGRESS')}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    json_data = json_path.read_text()

    w = Wallet(name=wallet_name, hotkey=hotkey_name, path=str(wallet_path))
    w.regenerate_coldkey(json=(json_data, json_password), use_password=False, overwrite=True, suppress=True)
    w.regenerate_hotkey(json=(json_data, json_password), use_password=False, overwrite=True, suppress=True)
    print(f"Restored wallet {wallet_name}/{hotkey_name} at {wallet_path}")

    sub = Subtensor(network=network)

    stop_flag = [False]  # Use list to allow mutation in nested function
    def heartbeat(start):
        while not stop_flag[0]:
            elapsed = time.perf_counter() - start
            print(f"[heartbeat] register running... elapsed={elapsed:.1f}s", flush=True)
            time.sleep(30)

    print("[info] Starting register() ...")
    t0 = time.perf_counter()
    hb = threading.Thread(target=heartbeat, args=(t0,), daemon=True)
    hb.start()
    
    try:
        resp = sub.register(
            wallet=w,
            netuid=netuid,
            mev_protection=False,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            raise_error=True,
            num_processes=num_processes,
            cuda=use_torch,
        )
    finally:
        stop_flag[0] = True
        hb.join(timeout=2)

    elapsed = time.perf_counter() - t0
    print(f"[info] register() completed in {elapsed:.1f}s")
    print("Registration response:", resp)

if __name__ == "__main__":
    main()

