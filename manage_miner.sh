#!/bin/bash
# TAOS Miner Management Script

SERVICE_FILE="/home/ocean/Draven/sn79-miner/taos-miner.service"
SCRIPT_DIR="/home/ocean/Draven/sn79-miner"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/miner.pid"

# Check if systemd is available
check_systemd() {
    if [ -d "/run/systemd/system" ] && [ -n "$(pgrep -f systemd)" ]; then
        return 0  # systemd available
    else
        return 1  # systemd not available (container)
    fi
}

case "$1" in
    start)
        echo "🚀 Starting TAOS Miner..."

        if check_systemd; then
            echo "Using systemd service..."
            sudo cp "$SERVICE_FILE" /etc/systemd/system/ 2>/dev/null || {
                echo "❌ Failed to copy service file. Try with simple method."
                echo "Run: ./manage_miner.sh start_simple"
                exit 1
            }
            sudo systemctl daemon-reload
            sudo systemctl start taos-miner
            sudo systemctl enable taos-miner
            echo "✅ Systemd service started! Use './manage_miner.sh status' to check"
        else
            echo "Systemd not available, using background process..."
            # Create log directory
            mkdir -p "$LOG_DIR"
            LOG_FILE="$LOG_DIR/miner_$(date +%Y%m%d_%H%M%S).log"

            # Check if already running
            if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
                echo "❌ Miner already running (PID: $(cat $PID_FILE))"
                exit 1
            fi

            # Start miner in background
            echo "Starting miner in background..." | tee "$LOG_FILE"
            nohup "$SCRIPT_DIR/run_miner.sh" -e test -w cold_draven -h miner -u 366 >> "$LOG_FILE" 2>&1 &
            MINER_PID=$!
            echo $MINER_PID > "$PID_FILE"

            echo "✅ Background service started (PID: $MINER_PID)"
            echo "📄 Logs: $LOG_FILE"
            echo "Use './manage_miner.sh status' to check status"
        fi
        ;;

    stop)
        echo "🛑 Stopping TAOS Miner..."

        if check_systemd; then
            sudo systemctl stop taos-miner
            sudo systemctl disable taos-miner
            echo "✅ Systemd service stopped"
        else
            if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
                MINER_PID=$(cat "$PID_FILE")
                echo "Stopping miner (PID: $MINER_PID)..."
                kill $MINER_PID 2>/dev/null || kill -9 $MINER_PID 2>/dev/null
                rm -f "$PID_FILE"
                echo "✅ Background service stopped"
            else
                echo "❌ No miner process found running"
            fi
        fi
        ;;

    remove)
        echo "🗑️ Removing TAOS Miner Service..."

        # First stop the service
        if check_systemd; then
            echo "Stopping systemd service..."
            sudo systemctl stop taos-miner 2>/dev/null || true
            sudo systemctl disable taos-miner 2>/dev/null || true

            echo "Removing systemd service file..."
            sudo rm -f /etc/systemd/system/taos-miner.service
            sudo systemctl daemon-reload
            echo "✅ Systemd service completely removed"
        else
            echo "Stopping background process..."
            if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
                MINER_PID=$(cat "$PID_FILE")
                echo "Stopping miner (PID: $MINER_PID)..."
                kill $MINER_PID 2>/dev/null || kill -9 $MINER_PID 2>/dev/null
                rm -f "$PID_FILE"
                echo "✅ Background process stopped"
            else
                echo "No running process found"
            fi
        fi

        # Ask about cleaning up logs
        echo ""
        echo "🧹 Clean up options:"
        echo "1. Keep logs (recommended for debugging)"
        echo "2. Remove all logs"
        echo "3. Remove old logs only (keep last 3)"
        read -p "Choose cleanup option (1-3) [1]: " cleanup_choice

        case "$cleanup_choice" in
            2)
                echo "Removing all log files..."
                rm -rf "$LOG_DIR"
                echo "✅ All logs removed"
                ;;
            3)
                echo "Removing old log files (keeping last 3)..."
                if [ -d "$LOG_DIR" ]; then
                    cd "$LOG_DIR"
                    ls -t *.log 2>/dev/null | tail -n +4 | xargs -r rm -f
                    echo "✅ Old logs cleaned"
                fi
                ;;
            *)
                echo "Keeping all logs"
                ;;
        esac

        echo ""
        echo "🎯 Service removal complete!"
        echo "To restart the service later, run: ./manage_miner.sh start"
        ;;

    restart)
        echo "🔄 Restarting TAOS Miner..."
        $0 stop
        sleep 2
        $0 start
        ;;

    status)
        echo "📊 TAOS Miner Status:"

        if check_systemd; then
            sudo systemctl status taos-miner --no-pager -l
            echo ""
            echo "🔍 Recent Logs:"
            sudo journalctl -u taos-miner -n 10 --no-pager
        else
            if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
                MINER_PID=$(cat "$PID_FILE")
                echo "✅ Miner running (PID: $MINER_PID)"
                echo "📅 Started: $(ps -p $MINER_PID -o lstart=)"
                echo "💾 Memory: $(ps -p $MINER_PID -o pmem= | tr -d ' ')"
            else
                echo "❌ Miner not running"
                rm -f "$PID_FILE" 2>/dev/null
            fi

            echo ""
            echo "🔍 Recent Logs:"
            if [ -d "$LOG_DIR" ]; then
                LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
                if [ -f "$LATEST_LOG" ]; then
                    echo "Latest log: $LATEST_LOG"
                    echo "Last 5 lines:"
                    tail -5 "$LATEST_LOG"
                else
                    echo "No log files found"
                fi
            else
                echo "Log directory not found"
            fi
        fi
        ;;

    logs)
        echo "📄 TAOS Miner Logs:"
        if [ -d "$LOG_DIR" ]; then
            LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
            if [ -f "$LATEST_LOG" ]; then
                echo "Latest log file: $LATEST_LOG"
                echo "File size: $(du -h "$LATEST_LOG" | cut -f1)"
                echo "Last 20 lines:"
                tail -20 "$LATEST_LOG"
            else
                echo "No log files found in $LOG_DIR"
                echo "Start the miner first: ./manage_miner.sh start"
            fi
        else
            echo "Log directory $LOG_DIR not found"
            mkdir -p "$LOG_DIR"
            echo "Created log directory. Start miner to create logs."
        fi
        ;;

    journal)
        echo "📋 Miner Logs (follow mode):"
        if check_systemd; then
            sudo journalctl -u taos-miner -n 20 --no-pager -f
        else
            if [ -d "$LOG_DIR" ]; then
                LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
                if [ -f "$LATEST_LOG" ]; then
                    echo "Following: $LATEST_LOG"
                    echo "Press Ctrl+C to stop following"
                    tail -f "$LATEST_LOG"
                else
                    echo "No log files found. Start miner first."
                fi
            else
                echo "Log directory not found. Start miner first."
            fi
        fi
        ;;

    clean)
        echo "🧹 Cleaning old log files (keeping last 5)..."
        if [ -d "$LOG_DIR" ]; then
            cd "$LOG_DIR"
            ls -t *.log 2>/dev/null | tail -n +6 | xargs -r rm -f
            echo "✅ Old logs cleaned"
        else
            echo "Log directory not found"
        fi
        ;;

    start_simple)
        echo "🚀 Starting TAOS Miner (Simple Background Mode)..."
        mkdir -p "$LOG_DIR"
        LOG_FILE="$LOG_DIR/miner_$(date +%Y%m%d_%H%M%S).log"

        # Check if already running
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "❌ Miner already running (PID: $(cat $PID_FILE))"
            exit 1
        fi

        # Start miner
        echo "Starting miner..." | tee "$LOG_FILE"
        nohup "$SCRIPT_DIR/run_miner.sh" -e test -w cold_draven -h miner -u 366 >> "$LOG_FILE" 2>&1 &
        MINER_PID=$!
        echo $MINER_PID > "$PID_FILE"

        echo "✅ Miner started (PID: $MINER_PID)"
        echo "📄 Logs: $LOG_FILE"
        echo "Use './manage_miner.sh status' to check"
        ;;

    *)
        echo "🎯 TAOS Miner Management Script"
        echo ""
        echo "Usage: $0 {start|start_simple|stop|remove|restart|status|logs|journal|clean}"
        echo ""
        echo "Commands:"
        echo "  start        - Auto-detect: systemd or background mode"
        echo "  start_simple - Force background mode (for containers)"
        echo "  stop         - Stop miner service/process"
        echo "  remove       - Stop and completely remove service/logs"
        echo "  restart      - Restart miner"
        echo "  status       - Show status and recent logs"
        echo "  logs         - Show latest log file contents"
        echo "  journal      - Follow logs in real-time"
        echo "  clean        - Remove old log files (keep last 5)"
        echo ""
        echo "Environment: $(check_systemd && echo 'Systemd detected' || echo 'Container mode - using background processes')"
        echo "Log files: $LOG_DIR"
        ;;
esac
