#!/usr/bin/env python3
"""Query current registration difficulty for Bittensor subnet."""
import os
from pathlib import Path

# Set wallet path before importing Bittensor
os.environ.setdefault("BT_WALLET_PATH", "/home/ocean/.bittensor/wallets")
os.environ["HOME"] = "/home/ocean"

from bittensor.core.subtensor import Subtensor

def main():
    network = os.environ.get("NETWORK", "test")
    netuid = int(os.environ.get("NETUID", "366"))

    print(f"Querying difficulty for network={network}, netuid={netuid}")

    try:
        sub = Subtensor(network=network)
        hyperparams = sub.get_subnet_hyperparameters(netuid=netuid)
        difficulty = hyperparams.difficulty

        print(f"🎯 Current registration difficulty: {difficulty:,}")
        print(f"💡 This is the target value you need to beat with hash(nonce + hotkey + block) < target")

        # Estimate time based on difficulty (rough approximation)
        if difficulty < 1000:
            time_est = "< 1 minute"
        elif difficulty < 10000:
            time_est = "1-5 minutes"
        elif difficulty < 100000:
            time_est = "5-30 minutes"
        elif difficulty < 1000000:
            time_est = "30 minutes - 2 hours"
        else:
            time_est = "2+ hours (very difficult)"

        print(f"⏱️  Estimated time with good hardware: {time_est}")

    except Exception as e:
        print(f"❌ Error querying difficulty: {e}")

if __name__ == "__main__":
    main()
