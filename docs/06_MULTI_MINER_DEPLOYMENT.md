# Multiple Miner Deployment Guide

## Can You Run Multiple Miners?

**Yes**, but with important considerations and diminishing returns.

## Key Facts

### What You Need

- ✅ Different hotkeys for each miner
- ✅ Different axon ports for each miner
- ✅ Sufficient CPU resources (4+ cores per miner)
- ✅ Sufficient RAM (4GB+ per miner)
- ✅ TAO for registration of each UID

### What You DON'T Need

- ❌ Different IP addresses (same IP is fine)
- ❌ GPU (not used at all)
- ❌ Different server locations

## Economic Reality

### Registration Costs

Each miner requires:
- Separate UID registration (costs TAO)
- Registration fee varies by network demand
- On mainnet, can be 0.1 - 1+ TAO per registration

### Income Reality

**NOT LINEAR SCALING**

```
1 miner with optimized strategy: 100% of your potential
2 identical miners: NOT 200% income
3 identical miners: NOT 300% income
```

**Why?**
- Same strategy = same performance = similar rank
- Registration costs eat into profits
- More maintenance overhead
- Reward distribution based on relative performance, not absolute

### When Multiple Miners Make Sense

✅ **Good Reasons:**
- Testing different strategies simultaneously
- Comparing strategy variants
- Diversifying risk across approaches
- One in testnet, one in mainnet

❌ **Bad Reasons:**
- Trying to "multiply" income with same strategy
- Hoping for linear scaling
- Gaming the system (won't work)

## Technical Setup

### Server Requirements

For N miners on same VPS:

```
CPU: 4N cores (4 cores per miner)
RAM: 4N GB (4GB per miner)
Storage: 100GB + (50GB per additional miner)
Network: 200+ Mbps bandwidth
```

### Example: 3 Miners

```
CPU: 12 cores minimum
RAM: 12GB minimum
Storage: 200GB
Network: 200 Mbps
```

## Configuration

### Step 1: Create Multiple Hotkeys

```bash
# Create wallet with multiple hotkeys
btcli wallet create --wallet.name taos --wallet.hotkey miner1
btcli wallet create --wallet.name taos --wallet.hotkey miner2
btcli wallet create --wallet.name taos --wallet.hotkey miner3

# Or use different coldkeys
btcli wallet create --wallet.name taos1 --wallet.hotkey miner
btcli wallet create --wallet.name taos2 --wallet.hotkey miner
btcli wallet create --wallet.name taos3 --wallet.hotkey miner
```

### Step 2: Register Each Hotkey

```bash
# Testnet
btcli subnet register --netuid 366 --subtensor.network test --wallet.name taos --wallet.hotkey miner1
btcli subnet register --netuid 366 --subtensor.network test --wallet.name taos --wallet.hotkey miner2
btcli subnet register --netuid 366 --subtensor.network test --wallet.name taos --wallet.hotkey miner3

# Mainnet (costs TAO each)
btcli subnet register --netuid 79 --subtensor.network finney --wallet.name taos --wallet.hotkey miner1
btcli subnet register --netuid 79 --subtensor.network finney --wallet.name taos --wallet.hotkey miner2
btcli subnet register --netuid 79 --subtensor.network finney --wallet.name taos --wallet.hotkey miner3
```

### Step 3: Create Launch Scripts

**run_miner1.sh**
```bash
#!/bin/sh
cd /home/user/sn-79/taos/im/neurons

pm2 delete miner1
pm2 start --name=miner1 "python miner.py \
    --netuid 79 \
    --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
    --wallet.path ~/.bittensor/wallets \
    --wallet.name taos \
    --wallet.hotkey miner1 \
    --axon.port 8091 \
    --logging.info \
    --agent.path ~/.taos/agents \
    --agent.name StrategyA \
    --agent.params 'param1=value1 param2=value2'"

pm2 save
```

**run_miner2.sh**
```bash
#!/bin/sh
cd /home/user/sn-79/taos/im/neurons

pm2 delete miner2
pm2 start --name=miner2 "python miner.py \
    --netuid 79 \
    --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
    --wallet.path ~/.bittensor/wallets \
    --wallet.name taos \
    --wallet.hotkey miner2 \
    --axon.port 8092 \
    --logging.info \
    --agent.path ~/.taos/agents \
    --agent.name StrategyB \
    --agent.params 'param1=value1 param2=value2'"

pm2 save
```

**run_miner3.sh**
```bash
#!/bin/sh
cd /home/user/sn-79/taos/im/neurons

pm2 delete miner3
pm2 start --name=miner3 "python miner.py \
    --netuid 79 \
    --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
    --wallet.path ~/.bittensor/wallets \
    --wallet.name taos \
    --wallet.hotkey miner3 \
    --axon.port 8093 \
    --logging.info \
    --agent.path ~/.taos/agents \
    --agent.name StrategyC \
    --agent.params 'param1=value1 param2=value2'"

pm2 save
```

### Step 4: Launch All Miners

```bash
chmod +x run_miner1.sh run_miner2.sh run_miner3.sh

./run_miner1.sh
./run_miner2.sh
./run_miner3.sh

# View all running miners
pm2 list

# Monitor specific miner
pm2 logs miner1
pm2 logs miner2
pm2 logs miner3
```

## Monitoring Multiple Miners

### PM2 Dashboard

```bash
# See all processes
pm2 list

# Monitor resources
pm2 monit

# View logs
pm2 logs --lines 100
```

### Performance Dashboard

Visit https://taos.simulate.trading

- Find each of your UIDs in the Agents table
- Compare performance side-by-side
- Analyze which strategy performs better

### Resource Monitoring

```bash
# Install htop if not already
apt-get install htop

# Monitor CPU/RAM usage
htop

# Check per-process
htop -p $(pgrep -f "miner1") -p $(pgrep -f "miner2") -p $(pgrep -f "miner3")
```

## Strategy Diversification

### Recommended Approach

Don't run identical strategies. Instead:

**Miner 1: Conservative Market Making**
```python
- Tight spreads
- Small order sizes
- Frequent updates
- Low inventory targets
```

**Miner 2: Aggressive Mean Reversion**
```python
- Wider thresholds
- Larger positions
- Hold for longer
- Higher volatility tolerance
```

**Miner 3: Statistical Arbitrage**
```python
- ML-based signals
- Quick in/out
- Multiple small trades
- Feature-driven
```

### A/B Testing Framework

```python
# agents/StrategyTest.py

class StrategyTest(FinanceSimulationAgent):
    def initialize(self):
        # Load variant from params
        self.variant = self.config.variant  # 'A', 'B', or 'C'
        
        if self.variant == 'A':
            self.spread = 0.001
            self.quantity = 1.0
        elif self.variant == 'B':
            self.spread = 0.002
            self.quantity = 2.0
        elif self.variant == 'C':
            self.spread = 0.003
            self.quantity = 3.0
    
    def respond(self, state):
        # Use self.variant to adjust behavior
        ...
```

Launch with different params:
```bash
--agent.params "variant=A"
--agent.params "variant=B"
--agent.params "variant=C"
```

## Cost-Benefit Analysis

### Example Scenario

**Single Miner**
```
Initial: 0.2 TAO registration
Monthly: ~2 TAO earnings (hypothetical)
Cost: 0.2 TAO
ROI: 1900%
```

**Three Identical Miners**
```
Initial: 0.6 TAO registration (3×)
Monthly: ~2.5 TAO earnings (NOT 6 TAO!)
Cost: 0.6 TAO
ROI: 317%
```

**Why the difference?**
- Same strategy = similar performance
- Compete against each other
- Share similar rank
- NO linear scaling of rewards

**Three DIFFERENT Miners**
```
Initial: 0.6 TAO registration
Monthly: ~4 TAO earnings (better diversification)
Cost: 0.6 TAO
ROI: 567%
```

**Why better?**
- Different strategies capture different opportunities
- Hedge against single strategy failure
- One might excel in current conditions
- True diversification benefit

## Common Issues

### Issue 1: Port Conflicts

```
Error: Address already in use: 8091
```

**Solution**: Ensure each miner uses unique port
```bash
miner1: --axon.port 8091
miner2: --axon.port 8092
miner3: --axon.port 8093
```

### Issue 2: PM2 Name Conflicts

```
Error: Script already launched
```

**Solution**: Use unique names
```bash
pm2 start --name=miner1 ...
pm2 start --name=miner2 ...
pm2 start --name=miner3 ...
```

### Issue 3: Resource Exhaustion

```
System slowing down, timeouts increasing
```

**Solution**: 
- Reduce number of miners
- Upgrade server specs
- Optimize code efficiency
- Monitor with `htop`

### Issue 4: All Miners Timing Out

```
All miners showing high timeout rates
```

**Solution**:
- CPU bottleneck - upgrade or reduce miners
- Network bottleneck - check bandwidth
- Code inefficiency - optimize strategy
- Server location - relocate near validators

## Advanced: Load Balancing

### Split Books Across Miners

**Miner 1: Books 0-19**
```python
def respond(self, state):
    response = FinanceAgentResponse(agent_id=self.uid)
    
    for book_id in range(0, 20):
        if book_id in state.books:
            # Full strategy
            self.full_strategy(response, state.books[book_id])
    
    for book_id in range(20, 40):
        if book_id in state.books:
            # Minimal activity to avoid decay
            self.minimal_activity(response, state.books[book_id])
    
    return response
```

**Miner 2: Books 20-39**
```python
def respond(self, state):
    response = FinanceAgentResponse(agent_id=self.uid)
    
    for book_id in range(0, 20):
        if book_id in state.books:
            # Minimal activity
            self.minimal_activity(response, state.books[book_id])
    
    for book_id in range(20, 40):
        if book_id in state.books:
            # Full strategy
            self.full_strategy(response, state.books[book_id])
    
    return response
```

**Caution**: This is advanced and risky:
- Still need activity on all books (both miners)
- Outlier penalty might hurt both
- Coordination overhead
- Might be better to just use one better miner

## Recommended Approach

### For Most Users

**Start with ONE miner**
- Perfect your strategy
- Achieve consistent profitability
- Understand the system deeply

### For Advanced Users

**Run 2-3 miners with DIFFERENT strategies**
- Diversify approaches
- A/B test variants
- Hedge risk
- Learn what works

### For Experts

**Consider multiple miners if:**
- You have proven strategies
- Registration costs are negligible
- You have excess resources
- You're testing at scale
- You understand diminishing returns

## Summary

### Key Takeaways

1. ✅ Multiple miners are technically possible
2. ⚠️ Income does NOT scale linearly
3. ✅ Best for testing different strategies
4. ⚠️ Requires significant resources
5. ✅ Can provide diversification benefit
6. ⚠️ Maintenance overhead increases
7. ✅ Useful for A/B testing
8. ⚠️ Registration costs add up

### Decision Framework

**Run 1 Miner If:**
- You're just starting
- Resources are limited
- Still developing strategy
- Want simplicity

**Run 2-3 Miners If:**
- Have proven strategies
- Want to test variants
- Have sufficient resources
- Understand the system well

**Don't Run Multiple If:**
- Same strategy on all
- Limited resources
- Just trying to "multiply" income
- Don't understand why

### Your RTX 3090

⚠️ **Will NOT be used for any of this**

This subnet is CPU-based algorithmic trading. Your GPU will sit idle. Consider using it for other projects or subnets that actually need GPU compute.

## Next Steps

1. **Start with one miner** on testnet
2. **Perfect your strategy** until profitable
3. **Deploy to mainnet** with single miner
4. **Monitor for 1-2 weeks** to establish baseline
5. **Consider 2nd miner** only if testing new strategy
6. **Compare performance** and decide if worth it

## Further Reading

- [Performance Tuning](05_PERFORMANCE_TUNING.md)
- [Advanced Strategies](04_ADVANCED_STRATEGIES.md)
- [FAQ](../FAQ.md) - Questions 6, 17


