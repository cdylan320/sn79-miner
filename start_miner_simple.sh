#!/bin/bash
# Simple TAOS Miner Starter (without systemd)

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/miner.pid"
LOG_FILE="$LOG_DIR/miner_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "==========================================" | tee "$LOG_FILE"
echo "TAOS MINER STARTED (Simple Mode)" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "PID: $$" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Save PID
echo $$ > "$PID_FILE"

# Function to cleanup
cleanup() {
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "TAOS MINER STOPPED" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 0
}

# Set trap
trap cleanup SIGINT SIGTERM

# Change to script directory
cd "$SCRIPT_DIR"

# Start miner in background with logging
echo "Starting miner in background..." | tee -a "$LOG_FILE"
nohup ./run_miner.sh -e test -w cold_draven -h miner -u 366 >> "$LOG_FILE" 2>&1 &

MINER_PID=$!
echo "Miner PID: $MINER_PID" | tee -a "$LOG_FILE"

# Wait for miner process
wait $MINER_PID
