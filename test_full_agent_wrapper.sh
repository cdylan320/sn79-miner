#!/bin/bash
# Wrapper script to test full agent pipeline with correct environment variables

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

# Set environment variables before running Python
export BT_WALLET_PATH="$HOME/.bittensor/wallets"

# Ensure wallets directory exists
mkdir -p "$HOME/.bittensor/wallets"

echo "🏠 HOME: $HOME"
echo "💰 BT_WALLET_PATH: $BT_WALLET_PATH"

# Run the full test
python test_agent_locally.py

