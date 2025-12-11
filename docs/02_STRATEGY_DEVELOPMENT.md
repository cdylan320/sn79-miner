# Strategy Development Guide

## Understanding the Environment

### Market Structure

- **40 Independent Orderbooks**: Each book evolves independently
- **Background Agents**: ~1000 agents per book creating market dynamics
- **Limit Order Books**: Full Level 3 (market-by-order) data available
- **Simulation Time**: Progresses slower than real-time due to query overhead

### State Updates

Every few seconds, you receive a state update containing:

1. **Current orderbook state**: Bids, asks, recent trades
2. **Your account state**: Balances, positions, open orders
3. **Recent events**: Trades, order placements, cancellations
4. **Market statistics**: Volume, price movements, imbalances

### Your Response

You have **3 seconds** to:

1. Analyze market state for all 40 books
2. Calculate trading signals
3. Generate instructions (place orders, cancel orders)
4. Return response to validator

## Strategy Types

### 1. Market Making (Recommended for Beginners)

**Concept**: Provide liquidity by placing limit orders on both sides of the spread

**Pros**:
- Relatively simple to implement
- Earns rebates (negative fees)
- Lower risk if inventory managed well

**Cons**:
- Need fast cancellation/replacement
- Inventory risk if market moves
- Must maintain tight spreads

**Basic Implementation**:

```python
def market_making_strategy(book, account):
    bestBid = book.bids[0].price
    bestAsk = book.asks[0].price
    midquote = (bestBid + bestAsk) / 2
    
    # Calculate target prices with small spread
    spread = 0.002  # 0.2%
    buy_price = midquote * (1 - spread/2)
    sell_price = midquote * (1 + spread/2)
    
    # Adjust for inventory
    inventory_bias = account.base_balance.total - initial_base
    buy_price -= inventory_bias * 0.001
    sell_price -= inventory_bias * 0.001
    
    # Place orders
    response.limit_order(book.id, OrderDirection.BUY, quantity, buy_price, ...)
    response.limit_order(book.id, OrderDirection.SELL, quantity, sell_price, ...)
```

### 2. Mean Reversion

**Concept**: Prices tend to revert to their average after deviations

**Pros**:
- Works well in ranging markets
- Can be highly profitable
- Relatively low complexity

**Cons**:
- Fails in trending markets
- Requires good signal detection
- Risk of accumulating losing positions

**Basic Implementation**:

```python
def mean_reversion_strategy(book, price_history):
    current_price = (book.bids[0].price + book.asks[0].price) / 2
    moving_average = np.mean(price_history[-20:])  # 20-period MA
    
    deviation = (current_price - moving_average) / moving_average
    threshold = 0.005  # 0.5%
    
    if deviation > threshold:
        # Price too high, sell
        response.limit_order(book.id, OrderDirection.SELL, quantity, book.bids[0].price, ...)
    elif deviation < -threshold:
        # Price too low, buy
        response.limit_order(book.id, OrderDirection.BUY, quantity, book.asks[0].price, ...)
```

### 3. Orderbook Imbalance

**Concept**: Trade based on buy/sell pressure in the orderbook

**Pros**:
- Uses real orderbook data
- Can predict short-term movements
- Fast to calculate

**Cons**:
- Sensitive to calculation depth
- May be noisy
- Requires recent history

**Basic Implementation**:

```python
def imbalance_strategy(book, depth=10):
    # Calculate imbalance at specified depth
    bid_volume = sum(level.quantity for level in book.bids[:depth])
    ask_volume = sum(level.quantity for level in book.asks[:depth])
    
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    threshold = 0.1  # 10% imbalance
    
    if imbalance > threshold:
        # More buy pressure, price likely to rise
        response.limit_order(book.id, OrderDirection.BUY, quantity, book.asks[0].price, ...)
    elif imbalance < -threshold:
        # More sell pressure, price likely to fall
        response.limit_order(book.id, OrderDirection.SELL, quantity, book.bids[0].price, ...)
```

### 4. Statistical Arbitrage (Advanced)

**Concept**: Use simple ML models to predict short-term price movements

**Pros**:
- Can capture complex patterns
- Adapts to market conditions
- Potentially higher Sharpe ratio

**Cons**:
- More complex to implement
- Requires careful feature engineering
- Risk of overfitting
- Must be very fast

**Basic Implementation**:

```python
from sklearn.linear_model import PassiveAggressiveRegressor

def statistical_arbitrage(book, features, model):
    # Features: recent returns, volume, imbalance, etc.
    X = calculate_features(book, recent_history)
    
    # Predict next return
    predicted_return = model.predict(X.reshape(1, -1))[0]
    
    threshold = 0.001  # 0.1%
    if predicted_return > threshold:
        # Predict price increase
        response.limit_order(book.id, OrderDirection.BUY, quantity, price, ...)
    elif predicted_return < -threshold:
        # Predict price decrease
        response.limit_order(book.id, OrderDirection.SELL, quantity, price, ...)
```

## Critical Requirements

### 1. Trade on ALL Books

```python
def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
    response = FinanceAgentResponse(agent_id=self.uid)
    
    # MUST iterate over ALL books
    for book_id, book in state.books.items():
        # Your strategy logic for this book
        # Even if you don't have a strong signal, place some orders
        # Inactivity leads to score decay and potential deregistration
        ...
    
    return response
```

### 2. Speed Optimization

```python
import time

def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
    start = time.time()
    response = FinanceAgentResponse(agent_id=self.uid)
    
    # Parallel processing for 40 books
    # Use vectorized operations (numpy)
    # Pre-compute reusable values
    # Avoid heavy calculations per book
    
    elapsed = time.time() - start
    bt.logging.info(f"Response generated in {elapsed:.2f}s")
    
    # Target: < 1 second for 40 books
    return response
```

### 3. Risk Management

```python
class RiskManager:
    def __init__(self):
        self.max_inventory = 100
        self.max_order_size = 5
        self.target_inventory = 0
    
    def can_trade(self, direction, quantity, current_inventory):
        new_inventory = current_inventory + (quantity if direction == OrderDirection.BUY else -quantity)
        
        # Check inventory limits
        if abs(new_inventory) > self.max_inventory:
            return False
        
        # Check order size
        if quantity > self.max_order_size:
            return False
        
        return True
    
    def inventory_adjustment(self, current_inventory):
        # Bias orders to move inventory back to target
        bias = (current_inventory - self.target_inventory) / self.max_inventory
        return bias  # Use this to adjust order prices
```

### 4. Volume Management

```python
def check_volume_limit(self, book_id, account):
    # Monitor traded volume to avoid hitting cap
    traded_volume = account.traded_volume
    initial_wealth = 100000  # From simulation config
    cap_multiplier = 10  # capital_turnover_cap from config
    
    volume_limit = initial_wealth * cap_multiplier
    volume_used_pct = (traded_volume / volume_limit) * 100
    
    if volume_used_pct > 90:
        bt.logging.warning(f"Book {book_id}: Volume at {volume_used_pct:.1f}% of cap!")
        # Reduce trading aggressiveness
        return True
    
    return False
```

## Performance Optimization Tips

### CPU Optimization

```python
import numpy as np
from numba import jit

# Use numpy for vectorized operations
prices = np.array([book.bids[0].price for book in state.books.values()])
mean_price = np.mean(prices)

# JIT compile hot paths
@jit(nopython=True)
def calculate_signals(prices, features):
    # Fast calculation
    return signals
```

### Memory Efficiency

```python
# Don't store unnecessary history
self.history = self.history[-100:]  # Keep only recent data

# Use generators for large iterations
def process_books(state):
    for book_id, book in state.books.items():
        yield process_single_book(book)
```

### Response Time Target

- **Excellent**: < 0.5 seconds
- **Good**: 0.5 - 1.0 seconds
- **Acceptable**: 1.0 - 2.0 seconds
- **Risky**: 2.0 - 3.0 seconds (may timeout occasionally)
- **Failing**: > 3.0 seconds (will timeout)

## Testing Your Strategy

### Local Testing (Proxy Validator)

```bash
# See agents/proxy/README.md for details
# Test against background model locally without network
```

### Testnet Testing

```bash
# Deploy to testnet first
./run_miner.sh -e test -u 366 -n YourAgent

# Monitor at https://testnet.simulate.trading
# Look for:
# - Positive Sharpe ratio
# - No timeouts
# - Activity on all books
# - Growing inventory value
```

### Metrics to Track

1. **Sharpe Ratio**: Target > 2.0 for competitive performance
2. **Win Rate**: % of profitable books
3. **Max Drawdown**: Largest inventory value drop
4. **Volume Efficiency**: Volume / Sharpe ratio
5. **Response Time**: Average time to generate response

## Common Mistakes to Avoid

1. ❌ **Using GPU/Deep Learning**: Won't help, will slow you down
2. ❌ **Complex Models**: Simple, fast strategies often win
3. ❌ **Ignoring Books**: Must trade on all books
4. ❌ **Over-Trading**: Will hit volume cap
5. ❌ **Under-Trading**: Will get activity decay penalty
6. ❌ **No Risk Management**: Will accumulate excessive inventory
7. ❌ **Slow Code**: Will timeout and fail requests
8. ❌ **Inconsistent Strategy**: Will get outlier penalty

## Example Agent Structure

```python
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *

class OptimizedAgent(FinanceSimulationAgent):
    def initialize(self):
        # Initialize strategy parameters
        self.risk_manager = RiskManager()
        self.signal_generator = SignalGenerator()
        
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        
        # Process all books quickly
        for book_id, book in state.books.items():
            account = self.accounts[book_id]
            
            # Generate signal
            signal = self.signal_generator.calculate(book, account)
            
            # Check risk
            if self.risk_manager.can_trade(signal, account):
                # Place orders
                self.place_orders(response, book_id, book, signal, account)
        
        return response
    
    def place_orders(self, response, book_id, book, signal, account):
        # Implement your order placement logic
        ...
```

## Next Steps

1. **Choose a strategy type** from the options above
2. **Implement basic version** without complexity
3. **Test locally** with proxy validator
4. **Deploy to testnet** and monitor for 24 hours
5. **Iterate based on data** from dashboard
6. **Deploy to mainnet** when consistently profitable

## Further Reading

- [Understanding Scoring](03_SCORING_MECHANICS.md)
- [Advanced Strategies](04_ADVANCED_STRATEGIES.md)
- [Performance Tuning](05_PERFORMANCE_TUNING.md)


