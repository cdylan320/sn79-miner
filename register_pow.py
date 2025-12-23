#!/usr/bin/env python3
"""Register a wallet on Bittensor subnet using PoW."""
import os
import time
import threading
import shutil
import subprocess
import logging
import re
from pathlib import Path

# Set wallet path before importing Bittensor to avoid permission issues
user_home = os.path.expanduser('~')
os.environ.setdefault("BT_WALLET_PATH", os.path.join(user_home, '.bittensor', 'wallets'))
# Override HOME to ensure wallet directory is created in the right place
os.environ["HOME"] = user_home

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

    print(f"[info] pid={os.getpid()} python={os.sys.executable}")
    print(f"[info] Using env: json_path={json_path}, wallet={wallet_name}/{hotkey_name}, "
          f"network={network}, netuid={netuid}, num_processes={num_processes}, use_torch={use_torch}, "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}, "
          f"BT_LOGGING={os.environ.get('BT_LOGGING')}, BT_PROGRESS={os.environ.get('BT_PROGRESS')}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if use_torch:
        try:
            import torch  # type: ignore
            print(
                "[info] torch:",
                {
                    "version": getattr(torch, "__version__", None),
                    "cuda_is_available": bool(torch.cuda.is_available()),
                    "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                    "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None,
                },
            )
        except Exception as e:
            print(f"[warn] USE_TORCH=1 but torch/cuda probe failed: {type(e).__name__}: {e}")

    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    json_data = json_path.read_text()

    # Set up custom logging to capture PoW details
    pow_stats = {"target": None, "best_hash": float('inf'), "best_nonce": None, "iterations": 0}

    class PowLogHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)

            # Parse difficulty from rich-formatted logs like "Registration Difficulty: [bold white]1.23k[/bold white]"
            difficulty_match = re.search(r'Registration Difficulty:\s*\[bold white\]([^[\]]+)\[/bold white\]', msg)
            if difficulty_match and pow_stats["target"] is None:
                difficulty_str = difficulty_match.group(1)
                # Parse millified numbers (like "1.23k" -> 1230)
                try:
                    if 'k' in difficulty_str.lower():
                        pow_stats["target"] = int(float(difficulty_str.lower().replace('k', '')) * 1000)
                    elif 'm' in difficulty_str.lower():
                        pow_stats["target"] = int(float(difficulty_str.lower().replace('m', '')) * 1000000)
                    else:
                        pow_stats["target"] = int(difficulty_str.replace(',', ''))

                    print(f"[pow] 🎯 Target difficulty: {pow_stats['target']:,} ({difficulty_str})", flush=True)
                except ValueError:
                    pass

            # Parse hash rate information for progress tracking
            if 'Iters' in msg and 'H/s' in msg:
                # Extract hash rates to show we're actively computing
                hash_match = re.search(r'Iters[^:]*:\s*\[bold white\]([^/]+)', msg)
                if hash_match:
                    rate_str = hash_match.group(1).strip()
                    pow_stats["iterations"] += 1
                    if pow_stats["iterations"] % 10 == 0:  # Log every 10 updates to avoid spam
                        print(f"[pow] Hash rate: {rate_str}/s (computing...)", flush=True)

    # Add our custom handler to capture console output
    import sys
    from io import StringIO

    # Capture stdout to parse rich-formatted messages
    original_stdout = sys.stdout
    captured_output = StringIO()

    class CapturingStdout:
        def write(self, text):
            # Check for difficulty and hash rate info in console output
            if 'Registration Difficulty:' in text:
                difficulty_match = re.search(r'Registration Difficulty:\s*\[bold white\]([^[\]]+)\[/bold white\]', text)
                if difficulty_match and pow_stats["target"] is None:
                    difficulty_str = difficulty_match.group(1)
                    try:
                        if 'k' in difficulty_str.lower():
                            pow_stats["target"] = int(float(difficulty_str.lower().replace('k', '')) * 1000)
                        elif 'm' in difficulty_str.lower():
                            pow_stats["target"] = int(float(difficulty_str.lower().replace('m', '')) * 1000000)
                        else:
                            pow_stats["target"] = int(difficulty_str.replace(',', ''))

                        print(f"[pow] 🎯 Target difficulty: {pow_stats['target']:,} ({difficulty_str})", flush=True)
                    except ValueError:
                        pass

            if 'Iters' in text and 'H/s' in text:
                hash_match = re.search(r'Iters[^:]*:\s*\[bold white\]([^/]+)', text)
                if hash_match:
                    rate_str = hash_match.group(1).strip()
                    pow_stats["iterations"] += 1
                    if pow_stats["iterations"] % 5 == 0:  # Log every 5 updates
                        print(f"[pow] Current hash rate: {rate_str}/s", flush=True)

            # Write to both captured buffer and original stdout
            captured_output.write(text)
            original_stdout.write(text)

        def flush(self):
            original_stdout.flush()

    # Redirect stdout
    sys.stdout = CapturingStdout()

    # Also patch the Console.log method to capture PoW progress
    from bittensor.utils.registration.pow import Console
    original_console_log = Console.log

    def patched_console_log(text: str):
        # Parse difficulty from rich-formatted logs
        difficulty_match = re.search(r'Registration Difficulty:\s*\[bold white\]([^[\]]+)\[/bold white\]', text)
        if difficulty_match:
            difficulty_str = difficulty_match.group(1)
            print(f"[debug] Found difficulty string: '{difficulty_str}'", flush=True)
            try:
                if 'k' in difficulty_str.lower():
                    new_target = int(float(difficulty_str.lower().replace('k', '')) * 1000)
                elif 'm' in difficulty_str.lower():
                    new_target = int(float(difficulty_str.lower().replace('m', '')) * 1000000)
                else:
                    new_target = int(difficulty_str.replace(',', ''))

                print(f"[debug] Parsed target: {new_target} (0x{new_target:064x})", flush=True)
                if pow_stats["target"] is None or pow_stats["target"] != new_target:
                    pow_stats["target"] = new_target
                    print(f"[pow] 🎯 Target difficulty updated: {pow_stats['target']:,} ({difficulty_str})", flush=True)
            except ValueError as e:
                print(f"[debug] Failed to parse difficulty '{difficulty_str}': {e}", flush=True)

        # Parse hash rate information
        if 'Iters' in text and 'H/s' in text:
            hash_match = re.search(r'Iters[^:]*:\s*\[bold white\]([^/]+)', text)
            if hash_match:
                rate_str = hash_match.group(1).strip()
                pow_stats["iterations"] += 1
                if pow_stats["iterations"] % 2 == 0:  # Log more frequently
                    print(f"[pow] Current hash rate: {rate_str}/s", flush=True)

        # Call original log
        original_console_log(text)

    Console.log = patched_console_log

    w = Wallet(name=wallet_name, hotkey=hotkey_name, path=str(wallet_path))
    w.regenerate_coldkey(json=(json_data, json_password), use_password=False, overwrite=True, suppress=True)
    w.regenerate_hotkey(json=(json_data, json_password), use_password=False, overwrite=True, suppress=True)
    hotkey_pubkey = w.hotkey.ss58_address
    print(f"Restored wallet {wallet_name}/{hotkey_name} at {wallet_path} (hotkey: {hotkey_pubkey})")

    sub = Subtensor(network=network)

    # Query and display current registration difficulty
    try:
        hyperparams = sub.get_subnet_hyperparameters(netuid=netuid)
        difficulty = hyperparams.difficulty
        print(f"[pow] 🎯 Current registration difficulty for netuid {netuid}: {difficulty:,}", flush=True)
        print(f"[pow] Difficulty hex: {difficulty:#x}", flush=True)
        pow_stats["target"] = difficulty
    except Exception as e:
        print(f"[warn] Could not query difficulty: {e}", flush=True)

    stop_flag = [False]  # Use list to allow mutation in nested function
    def heartbeat(start):
        while not stop_flag[0]:
            elapsed = time.perf_counter() - start
            msg = f"[heartbeat] register running... elapsed={elapsed:.1f}s"

            # Show current PoW stats in heartbeat
            if pow_stats["target"] is not None:
                print(f"[debug] pow_stats target: {pow_stats['target']} (type: {type(pow_stats['target'])})", flush=True)
                target_str = f"{pow_stats['target']:#066x}"
                best_hash_str = f"{pow_stats['best_hash']:#066x}" if pow_stats['best_hash'] != float('inf') else "none"
                msg += f" target={target_str} best_hash={best_hash_str}"

            # Optional: show GPU utilization if available (helps confirm GPU PoW is actually active).
            if use_torch and shutil.which("nvidia-smi"):
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total", "--format=csv,noheader,nounits"],
                        text=True,
                        timeout=5,
                    ).strip()
                    if out:
                        msg += f" gpu[{out}]"
                except Exception:
                    pass
            print(msg, flush=True)
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
            log_verbose=True,  # Enable verbose logging for PoW details
            update_interval=10000,  # Update every 10k nonces for progress
        )
    finally:
        stop_flag[0] = True
        hb.join(timeout=2)

    elapsed = time.perf_counter() - t0
    print(f"[info] register() completed in {elapsed:.1f}s")
    print("Registration response:", resp)

if __name__ == "__main__":
    main()

