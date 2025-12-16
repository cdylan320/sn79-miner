#!/bin/bash
# TAOS Miner Service Script
# Runs miner as background service with logging

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/miner_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "==========================================" | tee -a "$LOG_FILE"
echo "TAOS MINER SERVICE STARTED" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Function to cleanup on exit
cleanup() {
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "TAOS MINER SERVICE STOPPED" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Change to script directory
cd "$SCRIPT_DIR"

# Run miner with logging
echo "Starting miner..." | tee -a "$LOG_FILE"
./run_miner.sh -e test -w cold_draven -h miner -u 366 2>&1 | tee -a "$LOG_FILE"
