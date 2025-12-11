# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT

"""
OptimizedMarketMaker Agent

A high-performance market-making agent optimized for τaos Subnet 79 scoring mechanics.
This strategy focuses on:
1. Fast execution (< 1 second response time)
2. Stable inventory management (low volatility = high Sharpe)
3. Consistent performance across all books
4. Intelligent volume management
5. Adaptive spreads based on market conditions

Key Features:
- Vectorized operations for speed
- Dynamic spread adjustment
- Inventory-aware order placement
- Fee-sensitive trading
- Volume budget management
"""

import time
import numpy as np
import bittensor as bt
from collections import defaultdict

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse


class OptimizedMarketMaker(FinanceSimulationAgent):
    """
    High-performance market-making agent optimized for maximum Sharpe ratio
    while maintaining required trading volume and consistency across books.
    """
    
    def initialize(self):
        """Initialize strategy parameters and risk management systems."""
        # Base trading parameters
        self.base_quantity = float(getattr(self.config, 'base_quantity', 1.0))
        self.min_spread = float(getattr(self.config, 'min_spread', 0.0005))  # 0.05%
        self.max_spread = float(getattr(self.config, 'max_spread', 0.002))    # 0.2%
        self.spread_adjustment_factor = float(getattr(self.config, 'spread_adjustment', 0.5))
        
        # Order management
        self.expiry_period = int(getattr(self.config, 'expiry_period', 60e9))  # 60 seconds
        self.max_open_orders = int(getattr(self.config, 'max_open_orders', 4))  # 2 per side
        
        # Risk management
        self.max_inventory_pct = float(getattr(self.config, 'max_inventory_pct', 0.1))  # 10% of wealth
        self.inventory_target = float(getattr(self.config, 'inventory_target', 0.0))  # Stay neutral
        self.max_fee_rate = float(getattr(self.config, 'max_fee_rate', 0.003))  # 0.3%
        
        # Volume management
        self.volume_target_pct = float(getattr(self.config, 'volume_target_pct', 0.8))  # Use 80% of cap
        self.volume_safety_margin = float(getattr(self.config, 'volume_safety', 0.9))  # Stop at 90%
        
        # Performance tracking
        self.initial_wealth = None
        self.book_prices = defaultdict(list)
        self.last_midquote = {}
        self.inventory_values = defaultdict(list)
        
        bt.logging.info(f"""
╔══════════════════════════════════════════════════════════════╗
║           Optimized Market Maker Initialized                  ║
╠══════════════════════════════════════════════════════════════╣
║ Base Quantity:        {self.base_quantity:>10.2f}                           ║
║ Min Spread:           {self.min_spread * 100:>10.3f}%                         ║
║ Max Spread:           {self.max_spread * 100:>10.3f}%                         ║
║ Expiry Period:        {self.expiry_period / 1e9:>10.1f} seconds                  ║
║ Max Inventory:        {self.max_inventory_pct * 100:>10.1f}% of wealth            ║
║ Volume Target:        {self.volume_target_pct * 100:>10.1f}% of cap               ║
║ Max Fee Rate:         {self.max_fee_rate * 100:>10.3f}%                         ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def calculate_inventory_value(self, account, price):
        """Calculate total inventory value in QUOTE currency."""
        base_value = account.base_balance.total * price
        quote_value = account.quote_balance.total
        return base_value + quote_value
    
    def get_inventory_position(self, account, price, initial_wealth):
        """Calculate inventory position as % deviation from neutral."""
        target_base = initial_wealth / (2 * price)
        target_quote = initial_wealth / 2
        
        current_base_value = account.base_balance.total * price
        current_quote_value = account.quote_balance.total
        
        # Deviation from balanced 50/50 position
        base_deviation = (current_base_value - target_base * price) / initial_wealth
        
        return base_deviation  # Positive = long base, Negative = long quote
    
    def calculate_optimal_spread(self, book, account, inventory_position):
        """
        Calculate optimal spread based on market conditions and inventory.
        
        Key considerations:
        - Wider spread when volatile
        - Tighter spread when inventory is balanced
        - Adjust based on orderbook depth
        """
        # Base spread
        spread = self.min_spread
        
        # Adjust for inventory - widen spread on side we're long
        inventory_adjustment = abs(inventory_position) * self.spread_adjustment_factor
        spread += inventory_adjustment
        
        # Adjust for orderbook depth
        if len(book.bids) > 5 and len(book.asks) > 5:
            # Deep book = can use tighter spreads
            depth_factor = min(book.bids[0].quantity, book.asks[0].quantity) / 10.0
            spread *= max(0.7, 1.0 - depth_factor * 0.01)
        else:
            # Thin book = use wider spreads
            spread *= 1.2
        
        # Cap spread
        spread = max(self.min_spread, min(self.max_spread, spread))
        
        return spread
    
    def check_volume_limit(self, book_id, account, initial_wealth):
        """Check if we're approaching volume cap."""
        # Volume cap calculation (from FAQ Q9)
        capital_turnover_cap = 10  # From validator config
        volume_limit = initial_wealth * capital_turnover_cap
        
        current_volume = account.traded_volume
        volume_used_pct = current_volume / volume_limit
        
        if volume_used_pct > self.volume_safety_margin:
            bt.logging.warning(
                f"Book {book_id}: Volume at {volume_used_pct * 100:.1f}% of cap! Reducing activity."
            )
            return True
        
        return False
    
    def should_place_orders(self, book_id, account, fees):
        """Determine if we should place new orders based on various factors."""
        # Check fee rates (DIS policy)
        if fees and fees.maker_fee_rate > self.max_fee_rate:
            bt.logging.debug(
                f"Book {book_id}: Maker fee too high ({fees.maker_fee_rate * 100:.3f}% > {self.max_fee_rate * 100:.3f}%)"
            )
            return False
        
        # Check if we have too many open orders
        if len(account.orders) >= self.max_open_orders:
            return False
        
        return True
    
    def calculate_order_quantity(self, direction, account, price, inventory_position, initial_wealth):
        """Calculate optimal order quantity considering inventory and risk."""
        base_qty = self.base_quantity
        
        # Adjust quantity based on inventory position
        if direction == OrderDirection.BUY:
            # Reduce buy quantity if already long base
            if inventory_position > 0:
                base_qty *= (1.0 - abs(inventory_position))
        else:  # SELL
            # Reduce sell quantity if already short base (long quote)
            if inventory_position < 0:
                base_qty *= (1.0 - abs(inventory_position))
        
        # Check affordability
        if direction == OrderDirection.BUY:
            max_affordable = account.quote_balance.free / price
            base_qty = min(base_qty, max_affordable * 0.9)  # Use 90% of available
        else:
            max_affordable = account.base_balance.free
            base_qty = min(base_qty, max_affordable * 0.9)
        
        # Check against max inventory
        max_inventory_value = initial_wealth * self.max_inventory_pct
        max_inventory_qty = max_inventory_value / price
        
        if direction == OrderDirection.BUY:
            current_base = account.base_balance.total
            available_capacity = max(0, max_inventory_qty - current_base)
            base_qty = min(base_qty, available_capacity)
        
        return base_qty
    
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """
        Main strategy loop - optimized for speed and efficiency.
        
        Target: < 0.5 seconds for 40 books
        """
        start_time = time.time()
        response = FinanceAgentResponse(agent_id=self.uid)
        
        # Initialize wealth tracking on first run
        if self.initial_wealth is None:
            # Estimate from first book's account
            first_account = list(self.accounts.values())[0]
            first_book = list(state.books.values())[0]
            first_price = (first_book.bids[0].price + first_book.asks[0].price) / 2
            self.initial_wealth = self.calculate_inventory_value(first_account, first_price)
            bt.logging.info(f"Initial wealth estimated: {self.initial_wealth:.2f} QUOTE")
        
        # Process all books
        for book_id, book in state.books.items():
            try:
                account = self.accounts[book_id]
                
                # Calculate current market conditions
                if not book.bids or not book.asks:
                    continue
                
                best_bid = book.bids[0].price
                best_ask = book.asks[0].price
                midquote = (best_bid + best_ask) / 2
                
                # Track price for analytics
                self.book_prices[book_id].append(midquote)
                self.book_prices[book_id] = self.book_prices[book_id][-100:]  # Keep last 100
                
                # Calculate inventory position
                inventory_position = self.get_inventory_position(account, midquote, self.initial_wealth)
                
                # Check volume limits
                if self.check_volume_limit(book_id, account, self.initial_wealth):
                    # Approaching volume cap - only place minimal orders
                    continue
                
                # Check if we should place orders
                if not self.should_place_orders(book_id, account, account.fees):
                    # Cancel most aggressive orders if fee rate too high
                    if account.fees and account.fees.maker_fee_rate > self.max_fee_rate:
                        self._cancel_aggressive_orders(response, book_id, account)
                    continue
                
                # Calculate optimal spread
                spread = self.calculate_optimal_spread(book, account, inventory_position)
                
                # Calculate order prices with inventory bias
                # If long base (inventory_position > 0), bias towards selling (lower both prices)
                # If short base (inventory_position < 0), bias towards buying (raise both prices)
                inventory_bias = -inventory_position * 0.001  # Small bias factor
                
                buy_price = midquote * (1 - spread) * (1 + inventory_bias)
                sell_price = midquote * (1 + spread) * (1 + inventory_bias)
                
                # Round prices
                buy_price = round(buy_price, state.config.priceDecimals)
                sell_price = round(sell_price, state.config.priceDecimals)
                
                # Calculate quantities
                buy_qty = self.calculate_order_quantity(
                    OrderDirection.BUY, account, buy_price, inventory_position, self.initial_wealth
                )
                sell_qty = self.calculate_order_quantity(
                    OrderDirection.SELL, account, sell_price, inventory_position, self.initial_wealth
                )
                
                # Round quantities
                buy_qty = round(buy_qty, state.config.volumeDecimals)
                sell_qty = round(sell_qty, state.config.volumeDecimals)
                
                # Place buy order if quantity is sufficient
                if buy_qty >= 10 ** (-state.config.volumeDecimals):
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=buy_qty,
                        price=buy_price,
                        stp=STP.CANCEL_BOTH,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=self.expiry_period
                    )
                
                # Place sell order if quantity is sufficient
                if sell_qty >= 10 ** (-state.config.volumeDecimals):
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=sell_qty,
                        price=sell_price,
                        stp=STP.CANCEL_BOTH,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=self.expiry_period
                    )
                
                # Track performance
                inv_value = self.calculate_inventory_value(account, midquote)
                self.inventory_values[book_id].append(inv_value)
                self.inventory_values[book_id] = self.inventory_values[book_id][-100:]
                
            except Exception as e:
                bt.logging.error(f"Error processing book {book_id}: {e}")
                continue
        
        elapsed = time.time() - start_time
        bt.logging.info(f"Response generated in {elapsed:.3f}s for {len(state.books)} books")
        
        # Log summary every 10 updates
        if len(self.book_prices.get(0, [])) % 10 == 0:
            self._log_performance_summary(state)
        
        return response
    
    def _cancel_aggressive_orders(self, response, book_id, account):
        """Cancel orders closest to market when fees are too high."""
        # Cancel most aggressive bid
        bids = [o for o in account.orders if o.side == OrderDirection.BUY]
        if bids:
            most_aggressive_bid = max(bids, key=lambda o: o.price)
            response.cancel_order(book_id, most_aggressive_bid.id)
        
        # Cancel most aggressive ask
        asks = [o for o in account.orders if o.side == OrderDirection.SELL]
        if asks:
            most_aggressive_ask = min(asks, key=lambda o: o.price)
            response.cancel_order(book_id, most_aggressive_ask.id)
    
    def _log_performance_summary(self, state):
        """Log summary of performance across all books."""
        try:
            total_books = len(state.books)
            books_with_data = sum(1 for v in self.inventory_values.values() if len(v) > 1)
            
            if books_with_data == 0:
                return
            
            # Calculate simple metrics
            total_inventory = sum(
                vals[-1] if vals else self.initial_wealth
                for vals in self.inventory_values.values()
            )
            avg_inventory = total_inventory / total_books
            
            # Calculate returns where we have data
            returns = []
            for book_id, values in self.inventory_values.items():
                if len(values) >= 2:
                    ret = (values[-1] - values[0]) / values[0]
                    returns.append(ret)
            
            if returns:
                mean_return = np.mean(returns) * 100
                std_return = np.std(returns) * 100
                sharpe = mean_return / std_return if std_return > 0 else 0
                
                bt.logging.info(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Performance Summary                        ║
╠══════════════════════════════════════════════════════════════╣
║ Books Active:         {books_with_data:>3} / {total_books:<3}                            ║
║ Avg Inventory Value:  {avg_inventory:>15,.2f}                    ║
║ Mean Return:          {mean_return:>10.4f}%                         ║
║ Return StdDev:        {std_return:>10.4f}%                         ║
║ Est. Sharpe Ratio:    {sharpe:>10.3f}                            ║
╚══════════════════════════════════════════════════════════════╝
                """)
        except Exception as e:
            bt.logging.debug(f"Error logging summary: {e}")


if __name__ == "__main__":
    """
    Launch the OptimizedMarketMaker agent.
    
    Example usage:
    python OptimizedMarketMaker.py --port 8888 --agent_id 0 --params \
        base_quantity=1.0 \
        min_spread=0.0005 \
        max_spread=0.002 \
        expiry_period=60000000000 \
        max_inventory_pct=0.1 \
        volume_target_pct=0.8
    """
    launch(OptimizedMarketMaker)


