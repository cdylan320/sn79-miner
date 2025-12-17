# τaos Subnet 79 - Quick Start Guide

## Overview

**τaos (Subnet 79)** is a Bittensor subnet focused on algorithmic trading in simulated markets. Miners develop trading strategies that are evaluated on **risk-adjusted performance** (Sharpe ratio), trading volume, and consistency across multiple orderbooks.

## Key Facts

- **No GPU Required**: This is CPU-based trading strategy development, not deep learning
- **40 Orderbooks**: You must trade on ALL orderbooks simultaneously
- **3 Second Timeout**: You have 3 seconds to respond to each state update
- **Scoring**: Based on Sharpe ratio, trading volume, and consistency
- **Update Schedule**: Every Wednesday at 17:00 UTC for mainnet

## Installation

### For Miners

```bash
# Clone the repository
git clone https://github.com/taos-im/sn-79
cd sn-79

# Install (as root if possible)
./install_miner.sh

# Re-open your shell after installation
```

### For Validators

```bash
# Clone the repository
git clone https://github.com/taos-im/sn-79
cd sn-79

# Install (must run as root)
./install_validator.sh
```

**Note**: Installation takes 2+ hours on Ubuntu 22.04 due to compiling g++ and cmake from source.

## Register Your Miner

### Create Wallet (if needed)

```bash
btcli w create --wallet-name taos --wallet-hotkey miner
```

### Register on Testnet First

```bash
# Get testnet TAO from Bittensor Discord
# https://discord.com/channels/799672011265015819/1389370202327748629

# Register on testnet (netuid 366)
btcli s register --netuid 366 --network test --wallet-name taos --wallet-hotkey miner
```

### Register on Mainnet

```bash
# Register on mainnet (netuid 79) - costs TAO
btcli s register --netuid 79 --network finney --wallet-name taos --wallet-hotkey miner
```

## Run Your Miner

### Testnet

```bash
./run_miner.sh -e test -u 366
```

### Mainnet

```bash
./run_miner.sh -e finney -w taos -h miner -u 79
```

## Check Wallet & Miner Status

### Wallet Balance
```bash
./btcli wallet balance --wallet-name taos --network test
./btcli wallet balance --wallet-name taos --network finney
```

### Miner Registration Status
```bash
# Check specific miner
./btcli wallet overview --wallet-name taos --wallet-hotkey miner --netuid 366 --network test

# Check all wallets on subnet
./btcli wallet overview --all --netuid 366 --network test
```

### Wallet List
```bash
./btcli wallet list
```

## Monitor Performance

- **Mainnet**: https://taos.simulate.trading
- **Testnet**: https://testnet.simulate.trading

Click on your UID in the "Agents" table to see detailed metrics.

## Key Metrics to Watch

1. **Sharpe Score**: Your risk-adjusted performance
2. **Trading Volume**: Are you trading enough?
3. **Request Success Rate**: Are you timing out?
4. **Outlier Penalty**: Are you consistent across books?
5. **Inventory Value**: Is it growing steadily?

## Common Issues

### Issue: Timeouts

**Solution**: 
- Optimize your code for speed
- Use faster hardware
- Simplify your strategy
- Consider server location near validators

### Issue: Low Score Despite Trading

**Solution**:
- Check Sharpe ratio (need positive returns)
- Ensure trading on ALL books
- Check for failed requests
- Review outlier penalty

### Issue: Hit Volume Cap

**Solution**:
- Reduce trading frequency
- Implement volume budgeting
- Monitor `traded_volume` in account state
- Trade more strategically, not more frequently

## Next Steps

1. **Read**: [Strategy Development Guide](02_STRATEGY_DEVELOPMENT.md)
2. **Read**: [Understanding Scoring](03_SCORING_MECHANICS.md)
3. **Test**: Deploy on testnet and monitor for 24-48 hours
4. **Optimize**: Iterate based on dashboard data
5. **Deploy**: Move to mainnet when consistently profitable on testnet

## Important Reminders

- ⚠️ GPU will NOT be used - this is CPU-based
- ⚠️ Updates every Wednesday - follow Discord for announcements
- ⚠️ Test on testnet before mainnet deployment
- ⚠️ Monitor your miner daily via dashboard
- ⚠️ Maintain sufficient TAO for re-registration if needed

## Resources

- **Website**: https://taos.im
- **Whitepaper**: https://simulate.trading/taos-im-paper
- **Dashboard**: https://taos.simulate.trading
- **Discord**: https://discord.com/channels/799672011265015819/1353733356470276096
- **GitHub**: https://github.com/taos-im/sn-79


