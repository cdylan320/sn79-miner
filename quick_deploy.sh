#!/bin/bash
# Quick deployment script for τaos Subnet 79 miners
# This script helps you deploy optimized mining agents quickly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║        τaos Subnet 79 - Quick Deployment Script              ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}ℹ ${1}${NC}"
}

print_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ ${1}${NC}"
}

print_error() {
    echo -e "${RED}✗ ${1}${NC}"
}

# Main menu
show_menu() {
    echo ""
    echo "Select deployment option:"
    echo ""
    echo "1) Deploy OptimizedMarketMaker (Recommended for beginners)"
    echo "2) Deploy AdaptiveHybridAgent (Advanced)"
    echo "3) Deploy SimpleRegressorAgent (Built-in example)"
    echo "4) Custom parameters for OptimizedMarketMaker"
    echo "5) View current miners"
    echo "6) Stop all miners"
    echo "7) Exit"
    echo ""
    read -p "Enter choice [1-7]: " choice
}

# Deploy function
deploy_miner() {
    local agent_name=$1
    local params=$2
    local network=$3
    local netuid=$4
    local wallet_name=$5
    local hotkey_name=$6
    local port=$7

    print_info "Deploying ${agent_name} on ${network}..."

    # Check if agent file exists
    if [ ! -f ~/.taos/agents/${agent_name}.py ]; then
        if [ -f agents/${agent_name}.py ]; then
            print_info "Copying agent to ~/.taos/agents/"
            mkdir -p ~/.taos/agents
            cp agents/${agent_name}.py ~/.taos/agents/
        else
            print_error "Agent ${agent_name}.py not found!"
            return 1
        fi
    fi

    # Deploy
    ./run_miner.sh \
        -e "$network" \
        -u "$netuid" \
        -w "$wallet_name" \
        -h "$hotkey_name" \
        -a "$port" \
        -n "$agent_name" \
        -m "$params"

    print_success "Deployed ${agent_name}!"
    print_info "Monitor with: pm2 logs miner"
    
    if [ "$network" == "test" ]; then
        print_info "Dashboard: https://testnet.simulate.trading"
    else
        print_info "Dashboard: https://taos.simulate.trading"
    fi
}

# Get deployment parameters
get_params() {
    print_info "Deployment Configuration"
    echo ""
    
    # Network
    read -p "Deploy to (1) Testnet or (2) Mainnet? [1]: " net_choice
    net_choice=${net_choice:-1}
    
    if [ "$net_choice" == "2" ]; then
        NETWORK="finney"
        NETUID=79
        print_warning "Deploying to MAINNET (costs TAO for registration)"
    else
        NETWORK="test"
        NETUID=366
        print_success "Deploying to testnet (safe for testing)"
    fi
    
    # Wallet
    read -p "Wallet name [taos]: " wallet
    WALLET_NAME=${wallet:-taos}
    
    read -p "Hotkey name [miner]: " hotkey
    HOTKEY_NAME=${hotkey:-miner}
    
    # Port
    read -p "Axon port [8091]: " port
    PORT=${port:-8091}
    
    # Check if wallet exists
    if [ ! -d ~/.bittensor/wallets/${WALLET_NAME} ]; then
        print_warning "Wallet ${WALLET_NAME} not found!"
        read -p "Create wallet now? (y/n) [y]: " create_wallet
        create_wallet=${create_wallet:-y}
        if [ "$create_wallet" == "y" ]; then
            btcli wallet create --wallet.name ${WALLET_NAME} --wallet.hotkey ${HOTKEY_NAME}
        else
            print_error "Cannot proceed without wallet"
            exit 1
        fi
    fi
    
    # Check registration
    print_info "Make sure your wallet is registered on netuid ${NETUID}"
    if [ "$NETWORK" == "test" ]; then
        print_info "Get testnet TAO from: https://discord.com/channels/799672011265015819/1389370202327748629"
        print_info "Register with: btcli subnet register --netuid 366 --subtensor.network test --wallet.name ${WALLET_NAME} --wallet.hotkey ${HOTKEY_NAME}"
    else
        print_warning "Register with: btcli subnet register --netuid 79 --subtensor.network finney --wallet.name ${WALLET_NAME} --wallet.hotkey ${HOTKEY_NAME}"
    fi
    
    read -p "Press Enter when ready to continue..."
}

# Main script
print_header

# Check if installation completed
if [ ! -f "taos/im/neurons/miner.py" ]; then
    print_error "Installation incomplete!"
    print_info "Please run: ./install_miner.sh"
    exit 1
fi

print_success "Installation found!"
echo ""

# Get common parameters
get_params

# Show menu
show_menu

case $choice in
    1)
        print_info "OptimizedMarketMaker - Balanced market making strategy"
        print_info "Expected Sharpe: 2.0-2.5"
        echo ""
        deploy_miner "OptimizedMarketMaker" \
            "base_quantity=1.0 min_spread=0.0005 max_spread=0.002 expiry_period=60000000000" \
            "$NETWORK" "$NETUID" "$WALLET_NAME" "$HOTKEY_NAME" "$PORT"
        ;;
    2)
        print_info "AdaptiveHybridAgent - Multi-strategy with adaptive weighting"
        print_info "Expected Sharpe: 2.2-3.0 (more complex)"
        echo ""
        deploy_miner "AdaptiveHybridAgent" \
            "base_quantity=1.0 mm_spread=0.001 mr_threshold=0.005 adaptation_rate=0.1" \
            "$NETWORK" "$NETUID" "$WALLET_NAME" "$HOTKEY_NAME" "$PORT"
        ;;
    3)
        print_info "SimpleRegressorAgent - Built-in ML-based example"
        print_info "Expected Sharpe: 1.5-2.0"
        echo ""
        deploy_miner "SimpleRegressorAgent" \
            "quantity=1.0 expiry_period=120000000000 model=PassiveAggressiveRegressor signal_threshold=0.0025" \
            "$NETWORK" "$NETUID" "$WALLET_NAME" "$HOTKEY_NAME" "$PORT"
        ;;
    4)
        print_info "Custom OptimizedMarketMaker parameters"
        echo ""
        echo "Available parameters:"
        echo "  base_quantity: Order size (default: 1.0)"
        echo "  min_spread: Minimum spread from mid (default: 0.0005 = 0.05%)"
        echo "  max_spread: Maximum spread from mid (default: 0.002 = 0.2%)"
        echo "  expiry_period: Order expiry in nanoseconds (default: 60000000000 = 60s)"
        echo "  max_inventory_pct: Max inventory % (default: 0.1 = 10%)"
        echo ""
        read -p "Enter parameters (or press Enter for defaults): " custom_params
        
        if [ -z "$custom_params" ]; then
            custom_params="base_quantity=1.0 min_spread=0.0005 max_spread=0.002 expiry_period=60000000000"
        fi
        
        deploy_miner "OptimizedMarketMaker" "$custom_params" \
            "$NETWORK" "$NETUID" "$WALLET_NAME" "$HOTKEY_NAME" "$PORT"
        ;;
    5)
        print_info "Current miners:"
        pm2 list
        ;;
    6)
        print_warning "Stopping all miners..."
        pm2 stop all
        print_success "All miners stopped"
        ;;
    7)
        print_info "Exiting..."
        exit 0
        ;;
    *)
        print_error "Invalid choice!"
        exit 1
        ;;
esac

echo ""
print_success "Deployment complete!"
echo ""
print_info "Useful commands:"
echo "  pm2 list                    - View all processes"
echo "  pm2 logs miner              - View miner logs"
echo "  pm2 restart miner           - Restart miner"
echo "  pm2 stop miner              - Stop miner"
echo "  pm2 monit                   - Resource monitor"
echo ""
print_info "Documentation:"
echo "  docs/README.md              - Complete guide"
echo "  docs/01_QUICK_START_GUIDE.md - Quick start"
echo "  MINING_GUIDE.md             - Mining guide"
echo ""
print_info "Support:"
echo "  Discord: https://discord.com/channels/799672011265015819/1353733356470276096"
echo "  Dashboard: https://${NETWORK}.simulate.trading"
echo ""


