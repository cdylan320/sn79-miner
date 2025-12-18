#!/bin/bash
# Wrapper script to test full agent pipeline with correct environment variables

cd /home/ocean/Draven/sn79-miner
source venv/bin/activate

# Set environment variables before running Python
export HOME="/home/ocean"
export BT_WALLET_PATH="/home/ocean/.bittensor/wallets"

# Ensure wallets directory exists
mkdir -p "/home/ocean/.bittensor/wallets"

echo "🏠 HOME: $HOME"
echo "💰 BT_WALLET_PATH: $BT_WALLET_PATH"

# Run the full test
python test_agent_locally.py

