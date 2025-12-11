# Parameters Reference Guide

Complete reference for tuning OptimizedMarketMaker and AdaptiveHybridAgent parameters.

## OptimizedMarketMaker Parameters

### Core Trading Parameters

#### `base_quantity` (default: 1.0)
**What it does**: Base order size for limit orders

**Impact**:
- **Higher** → More volume, more inventory risk
- **Lower** → Less volume, more stable inventory

**Recommended ranges**:
- Conservative: 0.5 - 0.8
- Balanced: 0.8 - 1.2
- Aggressive: 1.2 - 2.0

**Adjust when**:
- Hitting volume cap → decrease
- Insufficient volume → increase
- High volatility → decrease
- Low volatility → increase

```bash
# Example
-m "base_quantity=1.0"
```

---

#### `min_spread` (default: 0.0005 = 0.05%)
**What it does**: Minimum spread from midquote for order placement

**Impact**:
- **Higher** → Wider spreads, less fills, more stable inventory
- **Lower** → Tighter spreads, more fills, more volume

**Recommended ranges**:
- Tight: 0.0003 - 0.0005 (0.03% - 0.05%)
- Medium: 0.0005 - 0.001 (0.05% - 0.1%)
- Wide: 0.001 - 0.002 (0.1% - 0.2%)

**Adjust when**:
- Too many fills → increase
- Too few fills → decrease
- High Sharpe, low volume → decrease
- Low Sharpe, high volume → increase

```bash
# Example
-m "min_spread=0.0005"
```

---

#### `max_spread` (default: 0.002 = 0.2%)
**What it does**: Maximum spread from midquote

**Impact**:
- **Higher** → Allows wider spreads when inventory skewed
- **Lower** → Keeps orders tighter to market

**Recommended ranges**:
- Tight: 0.0015 - 0.002 (0.15% - 0.2%)
- Medium: 0.002 - 0.003 (0.2% - 0.3%)
- Wide: 0.003 - 0.005 (0.3% - 0.5%)

**Relationship**: Should be 2-4x `min_spread`

```bash
# Example
-m "max_spread=0.002"
```

---

#### `spread_adjustment_factor` (default: 0.5)
**What it does**: How much inventory affects spread

**Impact**:
- **Higher** → Spreads widen more with inventory imbalance
- **Lower** → Spreads stay tighter regardless of inventory

**Recommended ranges**:
- Low sensitivity: 0.2 - 0.4
- Medium: 0.4 - 0.6
- High sensitivity: 0.6 - 1.0

```bash
# Example
-m "spread_adjustment_factor=0.5"
```

---

### Order Management

#### `expiry_period` (default: 60e9 = 60 seconds)
**What it does**: How long orders stay on book before expiring (in nanoseconds)

**Impact**:
- **Higher** → Orders stay longer, less churn, may get stale
- **Lower** → Orders refresh often, more responsive, more API calls

**Recommended ranges**:
- Very short: 30e9 (30 seconds)
- Short: 45e9 - 60e9 (45-60 seconds)
- Medium: 60e9 - 90e9 (60-90 seconds)
- Long: 90e9 - 120e9 (90-120 seconds)

**Adjust when**:
- High cancel/replace ratio → increase
- Stale orders getting filled badly → decrease
- Volume too high → increase

```bash
# Example
-m "expiry_period=60000000000"  # 60 seconds in nanoseconds
```

---

#### `max_open_orders` (default: 4)
**What it does**: Maximum number of open orders per book

**Impact**:
- **Higher** → More orders on book, more flexibility
- **Lower** → Fewer orders, simpler management

**Recommended ranges**:
- Minimal: 2 (1 per side)
- Standard: 4 (2 per side)
- Multiple: 6-8 (3-4 per side)

**Note**: Simulator has a hard limit (check config)

```bash
# Example
-m "max_open_orders=4"
```

---

### Risk Management

#### `max_inventory_pct` (default: 0.1 = 10%)
**What it does**: Maximum inventory as % of initial wealth

**Impact**:
- **Higher** → Can accumulate more inventory, higher risk
- **Lower** → Stays more neutral, lower risk

**Recommended ranges**:
- Conservative: 0.05 - 0.08 (5% - 8%)
- Balanced: 0.08 - 0.12 (8% - 12%)
- Aggressive: 0.12 - 0.2 (12% - 20%)

**Adjust when**:
- High volatility → decrease
- Low volatility → increase
- Excessive inventory accumulation → decrease

```bash
# Example
-m "max_inventory_pct=0.1"
```

---

#### `inventory_target` (default: 0.0 = neutral)
**What it does**: Target inventory position (0 = 50/50 base/quote)

**Impact**:
- **Positive** → Bias towards holding base
- **Zero** → Stay neutral
- **Negative** → Bias towards holding quote

**Recommended**: Usually keep at 0.0 for neutrality

```bash
# Example
-m "inventory_target=0.0"
```

---

#### `max_fee_rate` (default: 0.003 = 0.3%)
**What it does**: Maximum maker fee rate acceptable

**Impact**:
- **Higher** → Trade even with high fees
- **Lower** → Only trade with low fees/rebates

**Recommended ranges**:
- Fee-sensitive: 0.001 - 0.002 (0.1% - 0.2%)
- Balanced: 0.002 - 0.003 (0.2% - 0.3%)
- Fee-agnostic: 0.003 - 0.005 (0.3% - 0.5%)

**Note**: DIS makes fees dynamic - adjust accordingly

```bash
# Example
-m "max_fee_rate=0.003"
```

---

### Volume Management

#### `volume_target_pct` (default: 0.8 = 80%)
**What it does**: Target % of volume cap to use

**Impact**:
- **Higher** → More aggressive volume usage
- **Lower** → More conservative, safer

**Recommended ranges**:
- Conservative: 0.6 - 0.7 (60% - 70%)
- Balanced: 0.7 - 0.85 (70% - 85%)
- Aggressive: 0.85 - 0.95 (85% - 95%)

**Warning**: Don't set too high or you'll hit cap

```bash
# Example
-m "volume_target_pct=0.8"
```

---

#### `volume_safety_margin` (default: 0.9 = 90%)
**What it does**: When to stop trading due to volume cap

**Impact**:
- **Higher** → Trade closer to cap limit (risky)
- **Lower** → Stop earlier (safer)

**Recommended**: 0.85 - 0.95

```bash
# Example
-m "volume_safety_margin=0.9"
```

---

## AdaptiveHybridAgent Parameters

### Base Parameters

#### `base_quantity` (default: 1.0)
Same as OptimizedMarketMaker - see above

---

#### `expiry_period` (default: 60e9)
Same as OptimizedMarketMaker - see above

---

### Strategy Parameters

#### `mm_spread` (default: 0.001 = 0.1%)
**What it does**: Market making spread

**Impact**: Similar to `min_spread` in OptimizedMarketMaker

**Recommended**: 0.0005 - 0.002

```bash
# Example
-m "mm_spread=0.001"
```

---

#### `mr_threshold` (default: 0.005 = 0.5%)
**What it does**: Threshold for mean reversion signal

**Impact**:
- **Higher** → Only trade large deviations (fewer trades)
- **Lower** → Trade smaller deviations (more trades)

**Recommended ranges**:
- Sensitive: 0.003 - 0.005 (0.3% - 0.5%)
- Medium: 0.005 - 0.008 (0.5% - 0.8%)
- Conservative: 0.008 - 0.015 (0.8% - 1.5%)

```bash
# Example
-m "mr_threshold=0.005"
```

---

#### `imb_threshold` (default: 0.1 = 10%)
**What it does**: Orderbook imbalance threshold

**Impact**:
- **Higher** → Only trade extreme imbalances
- **Lower** → Trade smaller imbalances

**Recommended ranges**:
- Sensitive: 0.05 - 0.1 (5% - 10%)
- Medium: 0.1 - 0.15 (10% - 15%)
- Conservative: 0.15 - 0.25 (15% - 25%)

```bash
# Example
-m "imb_threshold=0.1"
```

---

#### `mom_threshold` (default: 0.003 = 0.3%)
**What it does**: Momentum signal threshold

**Impact**:
- **Higher** → Only follow strong trends
- **Lower** → Follow weak trends

**Recommended ranges**:
- Sensitive: 0.002 - 0.003 (0.2% - 0.3%)
- Medium: 0.003 - 0.005 (0.3% - 0.5%)
- Conservative: 0.005 - 0.01 (0.5% - 1.0%)

```bash
# Example
-m "mom_threshold=0.003"
```

---

### Lookback Periods

#### `ma_period` (default: 20)
**What it does**: Moving average period for mean reversion

**Impact**:
- **Higher** → Smoother MA, slower to react
- **Lower** → More responsive, noisier

**Recommended**: 15 - 30

```bash
# Example
-m "ma_period=20"
```

---

#### `vol_period` (default: 20)
**What it does**: Period for volatility calculation

**Impact**: Similar to `ma_period`

**Recommended**: 15 - 30

```bash
# Example
-m "vol_period=20"
```

---

#### `regime_period` (default: 50)
**What it does**: Lookback for regime detection

**Impact**:
- **Higher** → Slower regime changes
- **Lower** → Faster regime changes

**Recommended**: 30 - 100

```bash
# Example
-m "regime_period=50"
```

---

### Adaptation

#### `adaptation_rate` (default: 0.1)
**What it does**: How fast strategy weights adapt

**Impact**:
- **Higher** → Adapt quickly (0.5 = 50% new, 50% old)
- **Lower** → Adapt slowly (0.1 = 10% new, 90% old)

**Recommended ranges**:
- Slow: 0.05 - 0.1
- Medium: 0.1 - 0.2
- Fast: 0.2 - 0.4

```bash
# Example
-m "adaptation_rate=0.1"
```

---

### Risk Management

#### `max_inventory_pct` (default: 0.15 = 15%)
**What it does**: Maximum inventory % (higher than OptimizedMarketMaker)

**Recommended**: 0.1 - 0.2

```bash
# Example
-m "max_inventory_pct=0.15"
```

---

#### `max_fee_rate` (default: 0.003 = 0.3%)
Same as OptimizedMarketMaker

---

## Parameter Presets

### OptimizedMarketMaker Presets

#### Conservative (Stable, Low Risk)
```bash
./run_miner.sh -n OptimizedMarketMaker -m "
base_quantity=0.5
min_spread=0.001
max_spread=0.003
expiry_period=90000000000
max_inventory_pct=0.08
volume_target_pct=0.7
"
```

**Expected**: Sharpe 2.0-2.3, Very stable

---

#### Balanced (Recommended)
```bash
./run_miner.sh -n OptimizedMarketMaker -m "
base_quantity=1.0
min_spread=0.0005
max_spread=0.002
expiry_period=60000000000
max_inventory_pct=0.1
volume_target_pct=0.8
"
```

**Expected**: Sharpe 2.0-2.5, Good balance

---

#### Aggressive (Higher Volume, More Risk)
```bash
./run_miner.sh -n OptimizedMarketMaker -m "
base_quantity=1.5
min_spread=0.0003
max_spread=0.0015
expiry_period=45000000000
max_inventory_pct=0.15
volume_target_pct=0.85
"
```

**Expected**: Sharpe 1.8-2.3, Higher volume

---

### AdaptiveHybridAgent Presets

#### Conservative
```bash
./run_miner.sh -n AdaptiveHybridAgent -m "
base_quantity=0.8
mm_spread=0.0015
mr_threshold=0.008
imb_threshold=0.15
mom_threshold=0.005
adaptation_rate=0.05
"
```

**Expected**: Sharpe 2.2-2.6, Stable adaptation

---

#### Balanced (Recommended)
```bash
./run_miner.sh -n AdaptiveHybridAgent -m "
base_quantity=1.0
mm_spread=0.001
mr_threshold=0.005
imb_threshold=0.1
mom_threshold=0.003
adaptation_rate=0.1
"
```

**Expected**: Sharpe 2.2-3.0, Good adaptation

---

#### Aggressive
```bash
./run_miner.sh -n AdaptiveHybridAgent -m "
base_quantity=1.2
mm_spread=0.0008
mr_threshold=0.003
imb_threshold=0.08
mom_threshold=0.002
adaptation_rate=0.2
"
```

**Expected**: Sharpe 2.0-3.2, Fast adaptation, higher risk

---

## Tuning Based on Observations

### If Sharpe Ratio Too Low
**Problem**: High volatility in inventory value

**Solutions**:
- Decrease `base_quantity`
- Increase `min_spread` and `max_spread`
- Decrease `max_inventory_pct`
- Increase `expiry_period`

```bash
# Conservative tuning
-m "base_quantity=0.6 min_spread=0.001 max_inventory_pct=0.08"
```

---

### If Volume Too Low
**Problem**: Not reaching volume target, low activity factor

**Solutions**:
- Increase `base_quantity`
- Decrease `min_spread`
- Decrease `expiry_period`
- Increase `volume_target_pct`

```bash
# Volume boost
-m "base_quantity=1.3 min_spread=0.0004 expiry_period=45000000000"
```

---

### If Hitting Volume Cap
**Problem**: Locked out from trading

**Solutions**:
- Decrease `base_quantity`
- Increase `expiry_period`
- Decrease `volume_target_pct`
- Trade less aggressively

```bash
# Volume reduction
-m "base_quantity=0.7 expiry_period=90000000000 volume_target_pct=0.7"
```

---

### If High Outlier Penalty
**Problem**: Inconsistent across books

**Solutions**:
- Use more robust parameters
- Avoid extreme values
- Test across all books
- Consider OptimizedMarketMaker (more consistent)

```bash
# Robust settings
-m "base_quantity=1.0 min_spread=0.0008 max_spread=0.0025"
```

---

### If Timeout Errors
**Problem**: Response too slow

**Solutions**:
- Simplify strategy (use OptimizedMarketMaker)
- Upgrade hardware
- Optimize code
- For AdaptiveHybridAgent: increase thresholds

```bash
# Less computation
-m "base_quantity=1.0 mr_threshold=0.008 imb_threshold=0.15"
```

---

## Testing Parameter Changes

### A/B Testing Framework

1. **Baseline**: Deploy with default parameters
2. **Monitor**: 24 hours minimum
3. **Record**: Sharpe, volume, outliers
4. **Adjust**: Change ONE parameter
5. **Monitor**: Another 24 hours
6. **Compare**: Did it improve?
7. **Iterate**: Repeat

### Example Test Plan

```bash
# Day 1-2: Baseline
./run_miner.sh -n OptimizedMarketMaker

# Day 3-4: Test spread
./run_miner.sh -n OptimizedMarketMaker -m "min_spread=0.001 max_spread=0.003"

# Day 5-6: Test quantity
./run_miner.sh -n OptimizedMarketMaker -m "base_quantity=1.2"

# Day 7-8: Test both (if both improved individually)
./run_miner.sh -n OptimizedMarketMaker -m "base_quantity=1.2 min_spread=0.001"
```

---

## Quick Reference

### Most Important Parameters

1. **`base_quantity`** - Order size (most direct impact on volume)
2. **`min_spread`** - Tightness to market (affects fills and Sharpe)
3. **`max_inventory_pct`** - Risk limit (affects Sharpe stability)
4. **`expiry_period`** - Order lifetime (affects responsiveness)
5. **`volume_target_pct`** - Volume usage (safety margin)

### Parameter Interactions

- `base_quantity` ↑ → Volume ↑, Inventory Risk ↑
- `min_spread` ↑ → Sharpe ↑, Volume ↓
- `max_inventory_pct` ↑ → Flexibility ↑, Risk ↑
- `expiry_period` ↑ → Churn ↓, Staleness Risk ↑

---

## Support

Need help tuning? Check:
- [Strategy Development Guide](02_STRATEGY_DEVELOPMENT.md)
- [Scoring Mechanics](03_SCORING_MECHANICS.md)
- [Discord](https://discord.com/channels/799672011265015819/1353733356470276096)


