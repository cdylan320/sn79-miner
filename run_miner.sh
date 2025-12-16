#!/bin/bash
ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
WALLET_PATH=/home/ocean/.bittensor/wallets/
WALLET_NAME=cold_draven
HOTKEY_NAME=miner
NETUID=79
AXON_PORT=8091
AGENT_PATH=/home/ocean/.taos/agents
AGENT_NAME=SimpleRegressorAgent
AGENT_PARAMS=(min_quantity=0.1 max_quantity=1.0 expiry_period=200 model=PassiveAggressiveRegressor signal_threshold=0.0025)
LOG_LEVEL=info
while getopts e:p:w:h:u:a:g:n:m:l: flag
do
    case "${flag}" in
        e) ENDPOINT=${OPTARG};;
        p) WALLET_PATH=${OPTARG};;
        w) WALLET_NAME=${OPTARG};;
        h) HOTKEY_NAME=${OPTARG};;
        u) NETUID=${OPTARG};;
        a) AXON_PORT=${OPTARG};;
        g) AGENT_PATH=${OPTARG};;
        n) AGENT_NAME=${OPTARG};;
        m) AGENT_PARAMS=${OPTARG};;
        l) LOG_LEVEL=${OPTARG};;
    esac
done
echo "ENDPOINT: $ENDPOINT"
echo "WALLET_PATH: $WALLET_PATH"
echo "WALLET_NAME: $WALLET_NAME"
echo "HOTKEY_NAME: $HOTKEY_NAME"
echo "NETUID: $NETUID"
echo "AXON_PORT: $AXON_PORT"
echo "AGENT_PATH: $AGENT_PATH"
echo "AGENT_NAME: $AGENT_NAME"
echo "AGENT_PARAMS: ${AGENT_PARAMS[@]}"

# Update code
git pull

# Use btcli virtual environment instead of local pip install
export HOME="/home/ocean"
export BT_WALLET_PATH="/home/ocean/.bittensor/wallets"
source /home/ocean/btcli-latest/bin/activate

# Install/update taos package in the btcli environment
pip install -e . --quiet

# Create agent directory if it doesn't exist
mkdir -p "$AGENT_PATH"

# Copy agent if it doesn't exist
if [ ! -f "$AGENT_PATH/$AGENT_NAME.py" ]; then
    if [ -f "agents/$AGENT_NAME.py" ]; then
        cp "agents/$AGENT_NAME.py" "$AGENT_PATH/"
        echo "Copied $AGENT_NAME.py to $AGENT_PATH/"
    fi
fi

# Run miner directly (without pm2 for now) - ensure we stay in the virtual environment
cd taos/im/neurons
echo "Starting miner..."
echo "Command: python miner.py --netuid $NETUID --subtensor.chain_endpoint $ENDPOINT --wallet.path $WALLET_PATH --wallet.name $WALLET_NAME --wallet.hotkey $HOTKEY_NAME --axon.port $AXON_PORT --logging.debug --agent.path $AGENT_PATH --agent.name $AGENT_NAME --agent.params ${AGENT_PARAMS[@]} --logging.$LOG_LEVEL"

# Ensure environment variables are set for the python process
export HOME="/home/ocean"
export BT_WALLET_PATH="/home/ocean/.bittensor/wallets"

# Use exec to replace the shell with python, ensuring it stays in the virtual environment
exec python miner.py --netuid $NETUID --subtensor.chain_endpoint $ENDPOINT --wallet.path $WALLET_PATH --wallet.name $WALLET_NAME --wallet.hotkey $HOTKEY_NAME --axon.port $AXON_PORT --logging.debug --agent.path $AGENT_PATH --agent.name $AGENT_NAME --agent.params "${AGENT_PARAMS[@]}" --logging.$LOG_LEVEL