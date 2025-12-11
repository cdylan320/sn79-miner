# τaos Subnet 79 - Mining Guide

## 🎯 Quick Links

- **📚 Complete Documentation**: [docs/README.md](docs/README.md)
- **🚀 Quick Start**: [docs/01_QUICK_START_GUIDE.md](docs/01_QUICK_START_GUIDE.md)
- **📊 Dashboard (Mainnet)**: https://taos.simulate.trading
- **📊 Dashboard (Testnet)**: https://testnet.simulate.trading
- **💬 Discord Support**: https://discord.com/channels/799672011265015819/1353733356470276096

## ⚡ Ultra Quick Start

### 1. Install (5 minutes + 2 hours compilation)

```bash
git clone https://github.com/taos-im/sn-79
cd sn-79
./install_miner.sh
```

### 2. Register on Testnet (2 minutes)

```bash
# Create wallet (if needed)
btcli wallet create --wallet.name taos --wallet.hotkey miner

# Get testnet TAO from Discord
# https://discord.com/channels/799672011265015819/1389370202327748629

# Register
btcli subnet register --netuid 366 --subtensor.network test --wallet.name taos --wallet.hotkey miner
```

### 3. Deploy Optimized Agent (1 minute)

```bash
./run_miner.sh -e test -u 366 -n OptimizedMarketMaker
```

### 4. Monitor

- Logs: `pm2 logs miner`
- Dashboard: https://testnet.simulate.trading
- Find your UID and watch your Sharpe ratio!

## 🤖 Available Optimized Agents

### OptimizedMarketMaker (Recommended)
**Best for**: Beginners, stable performance, learning

```bash
./run_miner.sh -e test -u 366 -n OptimizedMarketMaker
```

**Features**:
- Fast response time (< 0.5s)
- Stable Sharpe ratio (2.0-2.5)
- Intelligent inventory management
- Fee-aware trading

### AdaptiveHybridAgent (Advanced)
**Best for**: Experienced traders, maximum returns

```bash
./run_miner.sh -e test -u 366 -n AdaptiveHybridAgent
```

**Features**:
- 4-strategy combination
- Market regime detection
- Adaptive weighting
- Higher potential returns (2.2-3.0 Sharpe)

## 📈 Success Metrics

| Metric | Target | Excellent |
|--------|--------|-----------|
| Sharpe Ratio | > 2.0 | > 2.5 |
| Request Success | > 95% | > 99% |
| Response Time | < 2s | < 1s |
| Volume Usage | 70-90% | 80-85% |
| Outlier Penalty | < 0.2 | < 0.1 |

## ⚠️ Common Mistakes to Avoid

1. ❌ **Deploying to mainnet without testnet testing**
   - ✅ Test on testnet for 24+ hours first

2. ❌ **Expecting GPU to be used**
   - ✅ This is CPU-based algorithmic trading

3. ❌ **Running multiple identical miners**
   - ✅ Only run multiple if using DIFFERENT strategies

4. ❌ **Not monitoring daily**
   - ✅ Check dashboard and logs every day

5. ❌ **Ignoring Wednesday updates**
   - ✅ Follow Discord, update code weekly

## 🚀 5-Day Plan

### Day 1: Setup
- Install miner
- Read documentation
- Register on testnet

### Day 2: Deploy
- Deploy OptimizedMarketMaker
- Monitor performance
- Learn dashboard

### Day 3: Optimize
- Analyze metrics
- Adjust parameters
- Improve Sharpe ratio

### Day 4: Advanced
- Try AdaptiveHybridAgent
- Compare strategies
- Choose best

### Day 5: Mainnet
- Register mainnet
- Deploy best strategy
- Monitor and iterate

## 📚 Full Documentation

For detailed guides, see [docs/README.md](docs/README.md):

1. **[Quick Start Guide](docs/01_QUICK_START_GUIDE.md)** - Installation, registration, deployment
2. **[Strategy Development](docs/02_STRATEGY_DEVELOPMENT.md)** - Understanding strategies, code examples
3. **[Scoring Mechanics](docs/03_SCORING_MECHANICS.md)** - How miners are scored, optimization
4. **[Multiple Miners](docs/06_MULTI_MINER_DEPLOYMENT.md)** - Running multiple miners (advanced)

## 🆘 Getting Help

**Issue**: Miner timing out
- **Fix**: Optimize code, upgrade CPU, simplify strategy

**Issue**: Low Sharpe ratio
- **Fix**: Use OptimizedMarketMaker, improve risk management

**Issue**: Hitting volume cap
- **Fix**: Reduce order sizes, increase expiry times

**Issue**: High outlier penalty
- **Fix**: Use more robust strategy across all books

**More help**: [Discord Support](https://discord.com/channels/799672011265015819/1353733356470276096)

## 🎓 Key Concepts

### What is τaos?
Algorithmic trading in simulated markets. Miners develop strategies evaluated on risk-adjusted returns (Sharpe ratio).

### How are miners scored?
```
Score = Sharpe Ratio × Volume Weight × (1 - Activity Decay) × (1 - Outlier Penalty)
```

### What hardware do I need?
- **CPU**: 4+ cores (GPU not used!)
- **RAM**: 4GB+
- **Storage**: 100GB
- **Network**: 200+ Mbps

### Do I need ML/AI knowledge?
**No**. Simple algorithmic strategies often outperform complex ML models. Focus on:
- Fast execution
- Stable inventory
- Risk management
- Volume balance

## 🔧 Quick Commands

```bash
# Installation
./install_miner.sh

# Testnet deployment
./run_miner.sh -e test -u 366 -n OptimizedMarketMaker

# Mainnet deployment
./run_miner.sh -e finney -u 79 -n OptimizedMarketMaker

# Monitor
pm2 list
pm2 logs miner
pm2 restart miner
pm2 stop miner

# Update code (Wednesdays)
git pull
pip install -e .
pm2 restart miner
```

## 📊 Parameter Tuning

### Conservative (Stable Sharpe, Lower Volume)
```bash
-n OptimizedMarketMaker -m "base_quantity=0.5 min_spread=0.001 max_spread=0.003"
```

### Balanced (Recommended)
```bash
-n OptimizedMarketMaker -m "base_quantity=1.0 min_spread=0.0005 max_spread=0.002"
```

### Aggressive (Higher Volume, More Risk)
```bash
-n OptimizedMarketMaker -m "base_quantity=1.5 min_spread=0.0003 max_spread=0.0015"
```

## 💡 Pro Tips

1. **Start Simple**: Use OptimizedMarketMaker first
2. **Test Everything**: Testnet before mainnet, always
3. **Monitor Daily**: Check dashboard and logs
4. **Iterate**: Adjust parameters based on data
5. **Stay Updated**: Follow Discord for Wednesday updates
6. **Be Patient**: Building good strategy takes time
7. **Focus on Sharpe**: Not just raw returns, but risk-adjusted
8. **Trade All Books**: Must be active on all 40 books

## 🎯 Success Checklist

### Before Mainnet Deployment
- [ ] Tested on testnet for 24+ hours
- [ ] Sharpe ratio > 2.0 consistently
- [ ] Request success rate > 95%
- [ ] Trading on all 40 books
- [ ] No timeout errors in logs
- [ ] Understand why strategy works

### After Mainnet Deployment
- [ ] Monitor first 24 hours closely
- [ ] Compare testnet vs mainnet performance
- [ ] Adjust if needed
- [ ] Set up daily monitoring routine
- [ ] Plan for Wednesday updates

## 📞 Resources

- **Complete Docs**: [docs/README.md](docs/README.md)
- **Main README**: [README.md](README.md) (original project README)
- **FAQ**: [FAQ.md](FAQ.md) (subnet FAQ)
- **Whitepaper**: https://simulate.trading/taos-im-paper
- **Discord**: https://discord.com/channels/799672011265015819/1353733356470276096

---

## 🚀 Ready? Let's Go!

```bash
# Step 1: Install
./install_miner.sh

# Step 2: Register testnet
btcli subnet register --netuid 366 --subtensor.network test

# Step 3: Deploy
./run_miner.sh -e test -u 366 -n OptimizedMarketMaker

# Step 4: Monitor
pm2 logs miner
```

**Dashboard**: https://testnet.simulate.trading

**Good luck! May your Sharpe ratios be high! 📈**


