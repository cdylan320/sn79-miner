#!/usr/bin/env python3
"""Test PoW computation manually."""
import os
import time
from pathlib import Path

# Set wallet path
os.environ.setdefault("BT_WALLET_PATH", "/home/ocean/.bittensor/wallets")
os.environ["HOME"] = "/home/ocean"

from bittensor_wallet import Wallet
from bittensor.core.subtensor import Subtensor
import hashlib

def test_info():
    # Get block info
    sub = Subtensor(network="test")
    netuid = 366

    # Get current block
    try:
        block_info = sub.substrate.get_block()
        block_number = block_info['header']['number']
        block_hash = block_info['header']['hash']
        print(f"Block: {block_number}, Hash: {block_hash}")

        # Get difficulty from subnet
        hyperparams = sub.get_subnet_hyperparameters(netuid)
        difficulty = hyperparams.difficulty
        print(f"Difficulty: {difficulty}")
        print(f"Difficulty (hex): {difficulty:#x}")
        print(f"Max uint64: {(2**64)-1:#x}")
        print(f"Is difficulty max uint64? {difficulty == (2**64)-1}")

    except Exception as e:
        print(f"Error getting block info: {e}")
        return

if __name__ == "__main__":
    test_info()
