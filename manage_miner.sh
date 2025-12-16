#!/bin/bash
# TAOS Miner Management Script

SERVICE_FILE="/home/ocean/Draven/sn79-miner/taos-miner.service"
SCRIPT_DIR="/home/ocean/Draven/sn79-miner"
LOG_DIR="$SCRIPT_DIR/logs"

case "$1" in
    start)
        echo "🚀 Starting TAOS Miner Service..."
        sudo cp "$SERVICE_FILE" /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl start taos-miner
        sudo systemctl enable taos-miner
        echo "✅ Service started! Use './manage_miner.sh status' to check"
        ;;

    stop)
        echo "🛑 Stopping TAOS Miner Service..."
        sudo systemctl stop taos-miner
        sudo systemctl disable taos-miner
        echo "✅ Service stopped"
        ;;

    restart)
        echo "🔄 Restarting TAOS Miner Service..."
        sudo systemctl restart taos-miner
        echo "✅ Service restarted"
        ;;

    status)
        echo "📊 TAOS Miner Service Status:"
        sudo systemctl status taos-miner --no-pager -l
        echo ""
        echo "🔍 Recent Logs:"
        sudo journalctl -u taos-miner -n 10 --no-pager
        ;;

    logs)
        echo "📄 TAOS Miner Logs:"
        if [ -d "$LOG_DIR" ]; then
            LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
            if [ -f "$LATEST_LOG" ]; then
                echo "Latest log file: $LATEST_LOG"
                echo "Last 20 lines:"
                tail -20 "$LATEST_LOG"
            else
                echo "No log files found in $LOG_DIR"
            fi
        else
            echo "Log directory $LOG_DIR not found"
        fi
        ;;

    journal)
        echo "📋 System Journal Logs (last 50 lines):"
        sudo journalctl -u taos-miner -n 50 --no-pager -f
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

    *)
        echo "🎯 TAOS Miner Management Script"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|journal|clean}"
        echo ""
        echo "Commands:"
        echo "  start   - Start miner as system service"
        echo "  stop    - Stop miner service"
        echo "  restart - Restart miner service"
        echo "  status  - Show service status and recent logs"
        echo "  logs    - Show latest log file contents"
        echo "  journal - Follow system journal logs"
        echo "  clean   - Remove old log files (keep last 5)"
        echo ""
        echo "Log files are saved in: $LOG_DIR"
        ;;
esac
