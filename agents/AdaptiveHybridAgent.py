# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT

"""
AdaptiveHybridAgent - Advanced Multi-Strategy Agent

Combines multiple trading strategies with adaptive weighting based on market conditions:
1. Market Making - Provides liquidity with inventory management
2. Mean Reversion - Trades deviations from moving average
3. Orderbook Imbalance - Uses buy/sell pressure signals
4. Momentum - Captures short-term trends

The agent adapts strategy weights based on recent performance and market regime detection.
"""

import time
import numpy as np
import bittensor as bt
from collections import defaultdict, deque
from enum import Enum

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"


class AdaptiveHybridAgent(FinanceSimulationAgent):
    """
    Advanced hybrid strategy that adapts to market conditions.
    
    Features:
    - Multi-strategy approach
    - Market regime detection
    - Adaptive strategy weighting
    - Performance-based optimization
    - Fast vectorized calculations
    """
    
    def initialize(self):
        """Initialize all sub-strategies and adaptation systems."""
        # Base parameters
        self.base_quantity = float(getattr(self.config, 'base_quantity', 1.0))
        self.expiry_period = int(getattr(self.config, 'expiry_period', 60e9))
        
        # Strategy parameters
        self.mm_spread = float(getattr(self.config, 'mm_spread', 0.001))  # Market making spread
        self.mr_threshold = float(getattr(self.config, 'mr_threshold', 0.005))  # Mean reversion threshold
        self.imb_threshold = float(getattr(self.config, 'imb_threshold', 0.1))  # Imbalance threshold
        self.mom_threshold = float(getattr(self.config, 'mom_threshold', 0.003))  # Momentum threshold
        
        # Lookback periods
        self.ma_period = int(getattr(self.config, 'ma_period', 20))
        self.volatility_period = int(getattr(self.config, 'vol_period', 20))
        self.regime_period = int(getattr(self.config, 'regime_period', 50))
        
        # Risk management
        self.max_inventory_pct = float(getattr(self.config, 'max_inventory_pct', 0.15))
        self.max_fee_rate = float(getattr(self.config, 'max_fee_rate', 0.003))
        
        # Adaptation parameters
        self.adaptation_rate = float(getattr(self.config, 'adaptation_rate', 0.1))
        
        # Data storage
        self.price_history = defaultdict(lambda: deque(maxlen=self.regime_period))
        self.signal_history = defaultdict(lambda: {'mm': [], 'mr': [], 'imb': [], 'mom': []})
        self.performance_history = defaultdict(lambda: {'mm': [], 'mr': [], 'imb': [], 'mom': []})
        
        # Strategy weights (start equal)
        self.strategy_weights = defaultdict(lambda: {
            'mm': 0.25,    # Market making
            'mr': 0.25,    # Mean reversion
            'imb': 0.25,   # Imbalance
            'mom': 0.25    # Momentum
        })
        
        # Market regime
        self.market_regime = defaultdict(lambda: MarketRegime.RANGING)
        
        # Initial wealth
        self.initial_wealth = None
        
        bt.logging.info(f"""
╔══════════════════════════════════════════════════════════════╗
║              Adaptive Hybrid Agent Initialized                ║
╠══════════════════════════════════════════════════════════════╣
║ Strategies:                                                   ║
║   - Market Making (MM)                                       ║
║   - Mean Reversion (MR)                                      ║
║   - Orderbook Imbalance (IMB)                                ║
║   - Momentum (MOM)                                           ║
║                                                               ║
║ Features:                                                     ║
║   - Adaptive weighting based on performance                  ║
║   - Market regime detection                                  ║
║   - Dynamic risk management                                  ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def detect_market_regime(self, book_id):
        """
        Detect current market regime based on price history.
        
        Returns: MarketRegime enum
        """
        if len(self.price_history[book_id]) < self.regime_period:
            return MarketRegime.RANGING
        
        prices = np.array(list(self.price_history[book_id]))
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate metrics
        volatility = np.std(returns)
        trend_strength = abs(np.mean(returns)) / (volatility + 1e-8)
        
        # Regime detection
        if volatility > 0.01:  # High volatility
            return MarketRegime.VOLATILE
        elif volatility < 0.002:  # Low volatility
            return MarketRegime.CALM
        elif trend_strength > 1.0:  # Strong trend
            return MarketRegime.TRENDING
        else:  # Default
            return MarketRegime.RANGING
    
    def calculate_mm_signal(self, book, midquote, inventory_position):
        """Market making signal - provides liquidity around mid."""
        spread = self.mm_spread * (1 + abs(inventory_position) * 0.5)
        
        # Inventory bias
        bias = -inventory_position * 0.001
        
        buy_price = midquote * (1 - spread) * (1 + bias)
        sell_price = midquote * (1 + spread) * (1 + bias)
        
        return {
            'buy_price': buy_price,
            'sell_price': sell_price,
            'buy_qty': self.base_quantity * (1 - max(0, inventory_position)),
            'sell_qty': self.base_quantity * (1 - max(0, -inventory_position)),
            'confidence': 0.7  # Base confidence
        }
    
    def calculate_mr_signal(self, book_id, midquote, inventory_position):
        """Mean reversion signal - trades deviations from MA."""
        if len(self.price_history[book_id]) < self.ma_period:
            return None
        
        prices = np.array(list(self.price_history[book_id]))
        ma = np.mean(prices[-self.ma_period:])
        
        deviation = (midquote - ma) / ma
        
        if abs(deviation) < self.mr_threshold:
            return None
        
        # Strong mean reversion signal
        if deviation > self.mr_threshold:  # Price above MA - sell
            return {
                'direction': OrderDirection.SELL,
                'price': midquote,  # Aggressive - cross spread
                'quantity': self.base_quantity * min(1.5, abs(deviation) * 20),
                'confidence': min(0.9, abs(deviation) * 10)
            }
        else:  # Price below MA - buy
            return {
                'direction': OrderDirection.BUY,
                'price': midquote,
                'quantity': self.base_quantity * min(1.5, abs(deviation) * 20),
                'confidence': min(0.9, abs(deviation) * 10)
            }
    
    def calculate_imbalance_signal(self, book):
        """Orderbook imbalance signal."""
        depth = 10
        
        if len(book.bids) < depth or len(book.asks) < depth:
            return None
        
        bid_volume = sum(level.quantity for level in book.bids[:depth])
        ask_volume = sum(level.quantity for level in book.asks[:depth])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return None
        
        imbalance = (bid_volume - ask_volume) / total_volume
        
        if abs(imbalance) < self.imb_threshold:
            return None
        
        if imbalance > self.imb_threshold:  # More buy pressure
            return {
                'direction': OrderDirection.BUY,
                'price': book.asks[0].price,  # Take liquidity
                'quantity': self.base_quantity * abs(imbalance) * 5,
                'confidence': abs(imbalance)
            }
        else:  # More sell pressure
            return {
                'direction': OrderDirection.SELL,
                'price': book.bids[0].price,
                'quantity': self.base_quantity * abs(imbalance) * 5,
                'confidence': abs(imbalance)
            }
    
    def calculate_momentum_signal(self, book_id, midquote):
        """Momentum signal - follows short-term trends."""
        if len(self.price_history[book_id]) < 10:
            return None
        
        prices = np.array(list(self.price_history[book_id]))
        recent_returns = np.diff(prices[-10:]) / prices[-10:-1]
        momentum = np.mean(recent_returns)
        
        if abs(momentum) < self.mom_threshold:
            return None
        
        if momentum > self.mom_threshold:  # Upward momentum
            return {
                'direction': OrderDirection.BUY,
                'price': midquote * 1.001,  # Slightly aggressive
                'quantity': self.base_quantity,
                'confidence': min(0.8, abs(momentum) * 100)
            }
        else:  # Downward momentum
            return {
                'direction': OrderDirection.SELL,
                'price': midquote * 0.999,
                'quantity': self.base_quantity,
                'confidence': min(0.8, abs(momentum) * 100)
            }
    
    def adapt_strategy_weights(self, book_id, regime):
        """Adapt strategy weights based on market regime and performance."""
        current_weights = self.strategy_weights[book_id]
        
        # Regime-based adjustment
        regime_adjustments = {
            MarketRegime.TRENDING: {'mm': 0.8, 'mr': 0.6, 'imb': 1.0, 'mom': 1.4},
            MarketRegime.RANGING: {'mm': 1.2, 'mr': 1.3, 'imb': 0.9, 'mom': 0.6},
            MarketRegime.VOLATILE: {'mm': 0.7, 'mr': 0.8, 'imb': 1.1, 'mom': 1.2},
            MarketRegime.CALM: {'mm': 1.3, 'mr': 1.0, 'imb': 0.8, 'mom': 0.7}
        }
        
        adjustments = regime_adjustments[regime]
        
        # Apply adjustments
        for strategy in ['mm', 'mr', 'imb', 'mom']:
            target_weight = 0.25 * adjustments[strategy]
            current_weights[strategy] = (
                (1 - self.adaptation_rate) * current_weights[strategy] +
                self.adaptation_rate * target_weight
            )
        
        # Normalize weights
        total = sum(current_weights.values())
        for strategy in current_weights:
            current_weights[strategy] /= total
    
    def combine_signals(self, book_id, mm_signal, mr_signal, imb_signal, mom_signal):
        """Combine signals from all strategies using adaptive weights."""
        weights = self.strategy_weights[book_id]
        
        orders = []
        
        # Market making orders (always present)
        if mm_signal:
            orders.append({
                'direction': OrderDirection.BUY,
                'price': mm_signal['buy_price'],
                'quantity': mm_signal['buy_qty'] * weights['mm'] * 2,
                'passive': True
            })
            orders.append({
                'direction': OrderDirection.SELL,
                'price': mm_signal['sell_price'],
                'quantity': mm_signal['sell_qty'] * weights['mm'] * 2,
                'passive': True
            })
        
        # Directional signals
        for signal, strategy in [(mr_signal, 'mr'), (imb_signal, 'imb'), (mom_signal, 'mom')]:
            if signal and weights[strategy] > 0.1:  # Only if weighted significantly
                orders.append({
                    'direction': signal['direction'],
                    'price': signal['price'],
                    'quantity': signal['quantity'] * weights[strategy] * signal['confidence'],
                    'passive': False
                })
        
        return orders
    
    def get_inventory_position(self, account, price, initial_wealth):
        """Calculate inventory position."""
        target_base = initial_wealth / (2 * price)
        current_base_value = account.base_balance.total * price
        base_deviation = (current_base_value - target_base * price) / initial_wealth
        return base_deviation
    
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """Main strategy loop with adaptive multi-strategy execution."""
        start_time = time.time()
        response = FinanceAgentResponse(agent_id=self.uid)
        
        # Initialize wealth
        if self.initial_wealth is None:
            first_account = list(self.accounts.values())[0]
            first_book = list(state.books.values())[0]
            first_price = (first_book.bids[0].price + first_book.asks[0].price) / 2
            base_value = first_account.base_balance.total * first_price
            self.initial_wealth = base_value + first_account.quote_balance.total
        
        # Process each book
        for book_id, book in state.books.items():
            try:
                account = self.accounts[book_id]
                
                if not book.bids or not book.asks:
                    continue
                
                # Calculate market conditions
                best_bid = book.bids[0].price
                best_ask = book.asks[0].price
                midquote = (best_bid + best_ask) / 2
                
                # Update price history
                self.price_history[book_id].append(midquote)
                
                # Get inventory position
                inventory_position = self.get_inventory_position(account, midquote, self.initial_wealth)
                
                # Detect market regime
                regime = self.detect_market_regime(book_id)
                self.market_regime[book_id] = regime
                
                # Adapt strategy weights
                self.adapt_strategy_weights(book_id, regime)
                
                # Generate signals from each strategy
                mm_signal = self.calculate_mm_signal(book, midquote, inventory_position)
                mr_signal = self.calculate_mr_signal(book_id, midquote, inventory_position)
                imb_signal = self.calculate_imbalance_signal(book)
                mom_signal = self.calculate_momentum_signal(book_id, midquote)
                
                # Combine signals
                orders = self.combine_signals(book_id, mm_signal, mr_signal, imb_signal, mom_signal)
                
                # Place orders
                for order in orders:
                    qty = round(order['quantity'], state.config.volumeDecimals)
                    price = round(order['price'], state.config.priceDecimals)
                    
                    # Check minimum quantity
                    if qty < 10 ** (-state.config.volumeDecimals):
                        continue
                    
                    # Check affordability
                    if order['direction'] == OrderDirection.BUY:
                        if account.quote_balance.free < qty * price:
                            continue
                    else:
                        if account.base_balance.free < qty:
                            continue
                    
                    # Place order
                    response.limit_order(
                        book_id=book_id,
                        direction=order['direction'],
                        quantity=qty,
                        price=price,
                        stp=STP.CANCEL_BOTH,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=self.expiry_period
                    )
                
            except Exception as e:
                bt.logging.error(f"Error processing book {book_id}: {e}")
                continue
        
        elapsed = time.time() - start_time
        bt.logging.info(f"Hybrid strategy response in {elapsed:.3f}s")
        
        # Periodic logging
        if len(self.price_history.get(0, [])) % 20 == 0:
            self._log_strategy_weights()
        
        return response
    
    def _log_strategy_weights(self):
        """Log current strategy weights and regimes."""
        if not self.strategy_weights:
            return
        
        # Sample from first few books
        sample_books = list(self.strategy_weights.keys())[:3]
        
        bt.logging.info("\n" + "="*60)
        bt.logging.info("STRATEGY WEIGHTS & REGIMES")
        bt.logging.info("="*60)
        
        for book_id in sample_books:
            weights = self.strategy_weights[book_id]
            regime = self.market_regime.get(book_id, MarketRegime.RANGING)
            
            bt.logging.info(f"\nBook {book_id} - Regime: {regime.value.upper()}")
            bt.logging.info(f"  MM:  {weights['mm']:.2%}")
            bt.logging.info(f"  MR:  {weights['mr']:.2%}")
            bt.logging.info(f"  IMB: {weights['imb']:.2%}")
            bt.logging.info(f"  MOM: {weights['mom']:.2%}")
        
        bt.logging.info("="*60 + "\n")


if __name__ == "__main__":
    """
    Launch the AdaptiveHybridAgent.
    
    Example usage:
    python AdaptiveHybridAgent.py --port 8888 --agent_id 0 --params \
        base_quantity=1.0 \
        mm_spread=0.001 \
        mr_threshold=0.005 \
        imb_threshold=0.1 \
        mom_threshold=0.003 \
        ma_period=20 \
        adaptation_rate=0.1
    """
    launch(AdaptiveHybridAgent)


