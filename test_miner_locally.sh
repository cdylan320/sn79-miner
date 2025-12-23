#!/bin/bash
# Test script to run the TAOS miner locally in mock mode

echo "🧪 Testing TAOS Miner Locally (Mock Mode)"
echo "========================================"

# Set environment variables
export BT_WALLET_PATH="$HOME/.bittensor/wallets"
export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$HOME/sn79-miner:$PYTHONPATH"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
cd "$SCRIPT_DIR"
source venv/bin/activate

# Install/update taos package
pip install -e . --quiet

# Create agent directory and copy agent
mkdir -p "$HOME/.taos/agents"
if [ ! -f "$HOME/.taos/agents/SimpleRegressorAgent.py" ]; then
    if [ -f "agents/SimpleRegressorAgent.py" ]; then
        cp "agents/SimpleRegressorAgent.py" "$HOME/.taos/agents/"
        echo "✅ Copied SimpleRegressorAgent.py"
    fi
fi

echo "🚀 Starting miner in MOCK mode..."
echo "This will test your agent logic without connecting to the network"
echo ""

# Run miner in mock mode with a timeout (so it doesn't run forever)
cd taos/im/neurons
timeout 30 python miner.py \
    --netuid 366 \
    --mock \
    --wallet.path "$HOME/.bittensor/wallets" \
    --wallet.name cold_draven \
    --wallet.hotkey miner \
    --axon.port 8092 \
    --logging.debug \
    --agent.path "$HOME/.taos/agents" \
    --agent.name SimpleRegressorAgent \
    --agent.params min_quantity=0.1 max_quantity=1.0 expiry_period=200 model=PassiveAggressiveRegressor signal_threshold=0.0025 \
    --logging.info

echo ""
echo "✅ Local test completed!"
echo "If you see agent processing logs above, your miner is working correctly!"
echo ""
echo "📊 To run longer tests, remove 'timeout 30' from the command"
