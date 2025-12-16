# 🚀 TAOS Miner Service Setup

Run your TAOS miner as a background service with automatic logging.

## 📁 Files Created

- `run_miner_service.sh` - Service wrapper script
- `manage_miner.sh` - Service management (start/stop/status)
- `start_miner_simple.sh` - Simple background starter
- `taos-miner.service` - Systemd service file
- `logs/` - Directory for all log files

## 🎯 Quick Start

### Option 1: Systemd Service (Recommended)

```bash
# Start miner as system service
./manage_miner.sh start

# Check status
./manage_miner.sh status

# Stop miner
./manage_miner.sh stop

# View logs
./manage_miner.sh logs
```

### Option 2: Simple Background (No systemd)

```bash
# Start miner in background
./start_miner_simple.sh

# Check if running
ps aux | grep python | grep miner

# Stop miner
kill $(cat miner.pid)
```

## 📊 Log Files

Logs are automatically saved to `logs/` directory:

```
logs/
├── miner_20251216_143000.log
├── miner_20251216_150000.log
└── miner_20251216_153000.log (latest)
```

Each log file contains:
- ✅ Registration checks
- ✅ Block updates
- ✅ Trading activities (when they happen)
- ✅ Performance metrics
- ✅ Error messages

## 🛠️ Management Commands

```bash
# Service management
./manage_miner.sh start    # Start service
./manage_miner.sh stop     # Stop service
./manage_miner.sh restart  # Restart service
./manage_miner.sh status   # Show status

# Log management
./manage_miner.sh logs     # Show latest logs
./manage_miner.sh journal  # Follow system logs
./manage_miner.sh clean    # Clean old logs
```

## 📈 Monitoring Your Miner

### Real-time Status
```bash
./manage_miner.sh status
```

### View Latest Activity
```bash
./manage_miner.sh logs
```

### Follow Live Logs
```bash
./manage_miner.sh journal
```

### Check Trading Activity
```bash
# Look for these patterns in logs:
# "BOOK BTC/USD: PREDICTION..."
# "VALIDATOR abc123...: RETURN"
# "Training Progress"
```

## 🔧 Troubleshooting

### Service Won't Start
```bash
# Check systemd status
sudo systemctl status taos-miner

# Check permissions
ls -la run_miner_service.sh manage_miner.sh
```

### Logs Not Appearing
```bash
# Check log directory
ls -la logs/

# Check latest log
tail -20 logs/$(ls -t logs/ | head -1)
```

### Miner Not Responding
```bash
# Restart service
./manage_miner.sh restart

# Check network connection
./btcli subnets list --network test | grep 366
```

## 📋 Log Content Examples

### Normal Operation:
```
2025-12-16 17:45:56.980 | DEBUG | Key cold_draven.miner is registered
2025-12-16 17:45:57.225 | DEBUG | Current Block: 6051265
```

### Trading Activity (when validators query):
```
BOOK BTC/USD: PREDICTION LogReturn: 0.0234 | SIGNAL 0.0345
VALIDATOR abc123...: RETURN 0.0123 | ACC: 0.67
```

### Training Progress:
```
BOOK BTC/USD: Training Progress 45/60
```

## 🎯 Best Practices

1. **Monitor Daily**: Check `./manage_miner.sh status` daily
2. **Clean Logs**: Run `./manage_miner.sh clean` weekly
3. **Backup Logs**: Keep important trading logs for analysis
4. **Restart Weekly**: Restart service weekly for stability

## 🚨 Emergency Stop

```bash
# Immediate stop
./manage_miner.sh stop

# Force kill if needed
pkill -f "python miner.py"
```

---

**Your miner will now run 24/7 with complete logging!** 🎉
