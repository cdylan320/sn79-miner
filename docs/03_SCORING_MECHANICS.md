# Understanding Scoring Mechanics

## Overview

Your miner's score determines your share of subnet emissions. Understanding exactly how scoring works is critical to maximizing your rewards.

## Core Scoring Formula

```
Final Score = EMA(
    Sharpe Ratio 
    × Volume Weight 
    × (1 - Activity Decay) 
    × (1 - Outlier Penalty)
)
```

Let's break down each component:

## 1. Sharpe Ratio (Primary Metric)

### Definition

```
Sharpe Ratio = Mean(Returns) / StdDev(Returns)

Where:
Returns = Change in inventory value between observations
```

### How It's Calculated

1. **Inventory Value** = (BASE balance × price) + QUOTE balance
2. **Return** = (Current Inventory Value - Previous Inventory Value) / Previous Inventory Value
3. **Window**: Rolling window over recent observations (configured by validator)
4. **Sharpe**: Mean return / Standard deviation of returns

### What This Means

- **High Sharpe**: Consistent positive returns with low volatility
- **Low Sharpe**: Inconsistent returns or high volatility
- **Negative Sharpe**: Losing money on average

### Target Sharpe Ratios

- **< 0**: Losing money, will deregister
- **0 - 1**: Marginal performance
- **1 - 2**: Competitive
- **2 - 3**: Strong performance
- **> 3**: Excellent (rare)

### Example

```python
# Inventory values over 10 observations
inventory_values = [100000, 100050, 100100, 100080, 100150, 100200, 100180, 100250, 100300, 100280]

# Calculate returns
returns = []
for i in range(1, len(inventory_values)):
    ret = (inventory_values[i] - inventory_values[i-1]) / inventory_values[i-1]
    returns.append(ret)

# Returns: [0.0005, 0.0005, -0.0002, 0.0007, 0.0003, -0.0001, 0.0004, 0.0002, -0.0007]

mean_return = np.mean(returns)  # 0.000178
std_return = np.std(returns)    # 0.000385

sharpe = mean_return / std_return  # 0.46
```

## 2. Volume Weighting

### Purpose

Prevents miners from achieving high Sharpe ratios through inactivity. You must trade actively to receive full score.

### How It Works

```python
volume_weight = min(1.0, actual_volume / target_volume)

Where:
target_volume = initial_wealth × capital_turnover_cap
```

### Configuration

From validator config:
- `--scoring.activity.capital_turnover_cap`: Multiplier for target volume (e.g., 10x)
- `--scoring.activity.trade_volume_assessment_period`: Period to measure volume (e.g., 24 sim hours)

### Example

```python
initial_wealth = 100000
capital_turnover_cap = 10
target_volume = initial_wealth * capital_turnover_cap = 1,000,000

# Scenario 1: Under-trading
actual_volume = 500,000
volume_weight = 500,000 / 1,000,000 = 0.5
# Your score is cut in half!

# Scenario 2: Optimal trading
actual_volume = 1,000,000
volume_weight = 1,000,000 / 1,000,000 = 1.0
# Full score

# Scenario 3: Over-trading (hit cap)
actual_volume = 1,500,000
# Can't submit more orders until volume drops
```

## 3. Activity Decay

### Purpose

Penalizes periods of inactivity to ensure continuous participation.

### How It Works

- Score decays exponentially during inactive periods
- Inactivity = not submitting instructions or instructions failing
- Decay rate configured by validator

### Prevention

- Always submit instructions for every book
- Even if signal is weak, place some orders
- Ensure orders don't all immediately fail
- Maintain sufficient balance to trade

## 4. Outlier Penalty

### Purpose

Ensures consistent performance across all orderbooks, preventing exploitation of a few "easy" books.

### How It Works

```python
# For each miner, calculate performance across all books
book_scores = [sharpe_ratio_book_0, sharpe_ratio_book_1, ..., sharpe_ratio_book_39]

# Detect outliers
mean_score = np.mean(book_scores)
std_score = np.std(book_scores)

outlier_penalty = 0
for score in book_scores:
    if abs(score - mean_score) > 3 * std_score:
        outlier_penalty += penalty_factor

final_score = base_score * (1 - outlier_penalty)
```

### What This Means

If you perform exceptionally well on some books but poorly on others, you'll be penalized. Your strategy must be **robust** across all market conditions.

### Prevention

- Develop strategy that works in different market conditions
- Don't over-optimize for specific book characteristics
- Test performance on all books during development
- Monitor individual book performance on dashboard

## 5. Exponential Moving Average (EMA)

### Purpose

Smooths scores over time, preventing single-period volatility from dominating.

### How It Works

```python
# At each observation
new_score = calculate_score(sharpe, volume, activity, outliers)
current_ema = alpha * new_score + (1 - alpha) * previous_ema

Where:
alpha = --neuron.moving_average_alpha (from config)
```

### What This Means

- Past performance continues to influence current score
- Sudden improvements take time to reflect in score
- Sudden drops are also smoothed (protects from occasional bad periods)
- Long-term consistency is rewarded

### Configuration

Typical alpha values:
- **α = 0.1**: Heavy smoothing, slow to update (10% weight to new observation)
- **α = 0.5**: Moderate smoothing (50% weight to new observation)
- **α = 0.9**: Light smoothing, fast to update (90% weight to new observation)

## Scoring Timeline

### Initial Period

```
Observations 1-10: No score (insufficient data for Sharpe calculation)
- Focus: Trade on all books, build history
- Goal: Positive returns, reasonable volume
```

### Ramp-Up Period

```
Observations 11-50: Score begins accumulating
- Sharpe ratio becomes meaningful
- EMA starts building
- Goal: Consistent positive Sharpe, full volume
```

### Steady State

```
Observations 50+: Stable scoring
- EMA smooths fluctuations
- Long-term performance matters
- Goal: Maintain high Sharpe, avoid outliers
```

## Practical Examples

### Example 1: Good Miner

```python
# Performance metrics
sharpe_ratio = 2.5          # Strong risk-adjusted returns
volume = 1,000,000          # Full target volume
volume_weight = 1.0
activity_decay = 0          # No inactivity
outlier_penalty = 0.05      # Slight inconsistency

base_score = 2.5 * 1.0 = 2.5
penalized_score = 2.5 * (1 - 0) * (1 - 0.05) = 2.375

# After EMA (assuming alpha=0.3)
if previous_ema = 2.2:
    new_ema = 0.3 * 2.375 + 0.7 * 2.2 = 2.2525
```

### Example 2: Under-Trading Miner

```python
sharpe_ratio = 3.0          # Excellent Sharpe!
volume = 300,000            # Only 30% of target
volume_weight = 0.3         # Severe penalty
activity_decay = 0
outlier_penalty = 0

base_score = 3.0 * 0.3 = 0.9  # Score crushed despite good Sharpe
```

### Example 3: Inconsistent Miner

```python
sharpe_ratio = 2.0
volume = 1,000,000
volume_weight = 1.0
activity_decay = 0
outlier_penalty = 0.4       # High penalty for inconsistency

penalized_score = 2.0 * 1.0 * 1.0 * (1 - 0.4) = 1.2
# Score cut by 40% due to outliers
```

## Key Configuration Parameters

From validator config files:

```python
# Sharpe calculation
--scoring.sharpe.lookback = 100  # Number of observations for Sharpe window

# Volume requirements
--scoring.activity.capital_turnover_cap = 10  # Target volume multiplier
--scoring.activity.trade_volume_assessment_period = 86400000000000  # 24 hours in nanoseconds
--scoring.activity.trade_volume_sampling_interval = 3600000000000   # 1 hour sampling

# EMA smoothing
--neuron.moving_average_alpha = 0.3  # EMA weight for new observations

# Simulation
Simulation.step = 1000000000  # 1 second simulation steps
--neuron.timeout = 3.0  # 3 second timeout for responses
```

## Monitoring Your Score

### Dashboard Metrics

At https://taos.simulate.trading, check:

1. **Sharpe Score**: Your current Sharpe ratio
2. **Sharpe Penalty**: Outlier penalty amount
3. **Activity Factor**: Volume weighting × activity decay
4. **Final Score**: Your actual weighted score
5. **Rank**: Your position relative to other miners

### Per-Book Analysis

Click on your UID to see:
- Individual book Sharpe ratios
- Trading volume per book
- Request success rates
- Inventory trajectories

### Red Flags

- ⚠️ Sharpe score decreasing over time
- ⚠️ High outlier penalty (> 0.2)
- ⚠️ Low activity factor (< 0.8)
- ⚠️ Many failed/timeout requests
- ⚠️ Some books with no recent trades

## Optimization Strategy

### Priority 1: Positive Sharpe Across All Books

- More important than high Sharpe on few books
- Robust strategy > optimized strategy
- Consistency is key

### Priority 2: Meet Volume Requirements

- Track cumulative volume
- Budget volume across assessment period
- Don't under-trade or over-trade

### Priority 3: Minimize Variance

- Stable returns better than occasional big wins
- Lower volatility = higher Sharpe
- Risk management critical

### Priority 4: Fast Execution

- Avoid timeouts
- Fast response = more time for validator processing
- Better latency = better execution

## Future Changes (From FAQ)

The scoring system is evolving toward:

1. **Execution-based metrics**: Realized profitability from round-trip trades
2. **Cost accounting**: Explicit incorporation of fees and spreads
3. **Operational efficiency**: Penalties for excessive cancellations
4. **Downside risk**: CVaR instead of just standard deviation
5. **Coverage incentives**: Explicit rewards for balanced book participation

**Strategy**: Build robust, cost-aware strategies now to be ready for future updates.

## Summary

### To Maximize Your Score:

1. ✅ Maintain high Sharpe ratio (> 2.0)
2. ✅ Trade full target volume
3. ✅ Trade on ALL books consistently
4. ✅ Minimize outliers across books
5. ✅ Respond quickly (< 1 second)
6. ✅ Manage inventory to reduce volatility
7. ✅ Build long-term consistency

### Avoid:

1. ❌ Under-trading (kills volume weight)
2. ❌ Over-trading (hits cap, locks you out)
3. ❌ Inconsistent performance across books (outlier penalty)
4. ❌ Inactivity (activity decay)
5. ❌ Timeouts (activity decay + missed opportunities)
6. ❌ High volatility (lowers Sharpe)

## Next Steps

- [Performance Tuning Guide](05_PERFORMANCE_TUNING.md)
- [Advanced Strategies](04_ADVANCED_STRATEGIES.md)
- [Multiple Miner Setup](06_MULTI_MINER_DEPLOYMENT.md)


