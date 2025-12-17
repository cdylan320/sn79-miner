# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import numpy as np
import traceback
from loky.backend.context import set_start_method
set_start_method('forkserver', force=True)
from loky import get_reusable_executor

from taos.im.utils import normalize


def sharpe(uid, inventory_values, realized_pnl_values, lookback, norm_min, norm_max, 
           min_lookback, min_realized_observations, grace_period, deregistered_uids) -> dict:
    """
    Calculates both unrealized and realized Sharpe ratios.
    
    Unrealized: Based on inventory value changes (mark-to-market)
    Realized: Based on actual P&L from completed round-trip trades
    
    Both use the SAME observation window and timestamps for consistency.
    
    Args:
        uid: Miner UID
        inventory_values: Dict of {timestamp: {book_id: inventory_value}}
        realized_pnl_values: Dict of {timestamp: {book_id: realized_pnl}}
        lookback: Number of periods to look back
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        min_lookback: Minimum required periods for valid unrealized Sharpe calculation
        min_realized_observations: Minimum required non-zero trades for valid realized Sharpe calculation
        grace_period: Time threshold for detecting simulation changeovers
        deregistered_uids: List of UIDs that are deregistered
        
    Returns:
        Dict containing unrealized and realized Sharpe metrics, or None on error
    """
    try:
        num_values = len(inventory_values)
        if uid in deregistered_uids or num_values < min(min_lookback, lookback):
            return None
        
        timestamps = list(inventory_values.keys())
        book_ids = list(next(iter(inventory_values.values())).keys())
        
        # ===== UNREALIZED SHARPE =====
        np_inventory_values = np.array([
            [inventory_values[ts][book_id] for book_id in book_ids]
            for ts in timestamps
        ], dtype=np.float64).T
        
        # Detect changeover periods (simulation restarts)
        changeover_mask = None
        if grace_period > 0:
            ts_array = np.array(timestamps, dtype=np.int64)
            time_diffs = np.diff(ts_array)
            changeover_indices = np.where(time_diffs >= grace_period)[0]
            
            if len(changeover_indices) > 0:
                changeover_mask = np.ones(len(timestamps) - 1, dtype=bool)
                changeover_mask[changeover_indices] = False
        
        # Calculate unrealized returns (period-over-period changes)
        returns = np.diff(np_inventory_values, axis=1)
        
        # Apply changeover mask to exclude restart periods
        if changeover_mask is not None:
            returns = returns[:, changeover_mask]
        
        # Vectorized unrealized Sharpe calculation: sqrt(n) * (mean / std)
        means = returns.mean(axis=1)
        stds = returns.std(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            sharpe_ratios = np.where(stds != 0.0, means / stds, 0.0)
        
        # ===== REALIZED SHARPE =====
        sharpe_ratios_realized = np.full(len(book_ids), np.nan)

        if realized_pnl_values and len(realized_pnl_values) > 0:
            np_realized_pnl = np.array([
                [realized_pnl_values.get(ts, {}).get(book_id, 0.0) for book_id in book_ids]
                for ts in timestamps
            ], dtype=np.float64).T
            
            # Drop first timestamp to align with returns (after diff)
            realized_returns = np_realized_pnl[:, 1:]
            
            if changeover_mask is not None:
                realized_returns = realized_returns[:, changeover_mask]
            
            # Vectorized realized Sharpe calculation (per book)
            for book_idx in range(len(book_ids)):
                book_returns = realized_returns[book_idx, :]
                non_zero_count = np.count_nonzero(book_returns)
                if non_zero_count >= min_realized_observations:
                    # Use ALL returns (including zeros) for Sharpe calculation
                    # This treats each timestamp as an observation period
                    realized_mean = book_returns.mean()
                    realized_std = book_returns.std()                    
                    if realized_std != 0.0:
                        sharpe_ratios_realized[book_idx] = (realized_mean / realized_std)
                    else:
                        # Zero std means constant returns (all zeros or all same value)
                        # If mean is positive, perfect consistency; if zero, no activity
                        sharpe_ratios_realized[book_idx] = np.nan if realized_mean == 0.0 else 0.0
                else:
                    # Insufficient trading activity
                    sharpe_ratios_realized[book_idx] = np.nan
        
        sharpe_values = {
            'books': {book_id: float(sharpe_ratios[i]) for i, book_id in enumerate(book_ids)},
            'books_realized': {
                book_id: (float(sharpe_ratios_realized[i]) if not np.isnan(sharpe_ratios_realized[i]) else None)
                for i, book_id in enumerate(book_ids)
            }
        }
        
        # Aggregate values across books (only for unrealized)
        sharpe_values['average'] = float(sharpe_ratios.mean())
        sharpe_values['median'] = float(np.median(sharpe_ratios))
        
        # Aggregate realized values (only if we have valid data)
        valid_realized = sharpe_ratios_realized[~np.isnan(sharpe_ratios_realized)]
        if len(valid_realized) > 0:
            sharpe_values['average_realized'] = float(valid_realized.mean())
            sharpe_values['median_realized'] = float(np.median(valid_realized))
        else:
            sharpe_values['average_realized'] = None
            sharpe_values['median_realized'] = None
        
        # ===== TOTAL PORTFOLIO SHARPE (UNREALIZED) =====
        total_inventory = np_inventory_values.sum(axis=0)
        total_returns = np.diff(total_inventory)
        
        if changeover_mask is not None:
            total_returns = total_returns[changeover_mask]
        
        total_std = total_returns.std()
        total_mean = total_returns.mean()
        sharpe_values['total'] = float(total_mean / total_std if total_std != 0.0 else 0.0)
        
        # ===== TOTAL PORTFOLIO SHARPE (REALIZED) =====
        if realized_pnl_values and len(realized_pnl_values) > 0:
            total_realized_pnl = np_realized_pnl.sum(axis=0)[1:]
            if changeover_mask is not None:
                total_realized_pnl = total_realized_pnl[changeover_mask]
            
            non_zero_total = total_realized_pnl[total_realized_pnl != 0.0]
            if len(non_zero_total) >= min_realized_observations:
                realized_total_std = non_zero_total.std()
                realized_total_mean = non_zero_total.mean()
                sharpe_values['total_realized'] = float(realized_total_mean / realized_total_std if realized_total_std != 0.0 else 0.0)
            else:
                sharpe_values['total_realized'] = None  # Insufficient observations
        else:
            sharpe_values['total_realized'] = None  # No realized P&L data
        
        # ===== NORMALIZE ALL VALUES =====
        sharpe_values['normalized_average'] = normalize(norm_min, norm_max, sharpe_values['average'])
        sharpe_values['normalized_median'] = normalize(norm_min, norm_max, sharpe_values['median'])
        sharpe_values['normalized_total'] = normalize(norm_min, norm_max, sharpe_values['total'])
        
        # Normalize realized values (only if defined)
        sharpe_values['normalized_average_realized'] = (
            normalize(norm_min, norm_max, sharpe_values['average_realized'])
            if sharpe_values['average_realized'] is not None else None
        )
        sharpe_values['normalized_median_realized'] = (
            normalize(norm_min, norm_max, sharpe_values['median_realized'])
            if sharpe_values['median_realized'] is not None else None
        )
        sharpe_values['normalized_total_realized'] = (
            normalize(norm_min, norm_max, sharpe_values['total_realized'])
            if sharpe_values['total_realized'] is not None else None
        )
        
        return sharpe_values
        
    except Exception as ex:
        print(f"Failed to calculate Sharpe for UID {uid}: {traceback.format_exc()}")
        return None


def sharpe_batch(inventory_values, realized_pnl_values, lookback, norm_min, norm_max, 
                 min_lookback, min_realized_observations, grace_period, deregistered_uids):
    """
    Process a batch of UIDs for Sharpe calculation with realized P&L.
    
    Args:
        inventory_values: Dict of {uid: {timestamp: {book_id: value}}}
        realized_pnl_values: Dict of {uid: {timestamp: {book_id: pnl}}}
        lookback: Number of periods to look back
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        min_lookback: Minimum required periods for unrealized Sharpe
        min_realized_observations: Minimum required non-zero trades for realized Sharpe
        grace_period: Time threshold for changeover detection
        deregistered_uids: List of deregistered UIDs
        
    Returns:
        Dict of {uid: sharpe_values}
    """
    return {
        uid: sharpe(uid, inventory_value, realized_pnl_values.get(uid, {}), 
                   lookback, norm_min, norm_max, min_lookback, min_realized_observations, 
                   grace_period, deregistered_uids) 
        for uid, inventory_value in inventory_values.items()
    }


def batch_sharpe(inventory_values, realized_pnl_values, batches, lookback, norm_min, norm_max, 
                 min_lookback, min_realized_observations, grace_period, deregistered_uids):
    """
    Parallel processing of Sharpe calculations with realized P&L.
    
    Uses loky for process-based parallelism to avoid GIL limitations
    during NumPy computations.
    
    Args:
        inventory_values: Dict of {uid: {timestamp: {book_id: value}}}
        realized_pnl_values: Dict of {uid: {timestamp: {book_id: pnl}}}
        batches: List of UID batches for parallel processing
        lookback: Number of periods to look back
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        min_lookback: Minimum required periods for unrealized Sharpe
        min_realized_observations: Minimum required non-zero trades for realized Sharpe
        grace_period: Time threshold for changeover detection
        deregistered_uids: List of deregistered UIDs
        
    Returns:
        Dict of {uid: sharpe_values} for all UIDs
    """
    pool = get_reusable_executor(max_workers=len(batches))
    
    tasks = [
        pool.submit(
            sharpe_batch,
            {uid: inventory_values[uid] for uid in batch},
            {uid: realized_pnl_values.get(uid, {}) for uid in batch},
            lookback, norm_min, norm_max, min_lookback, min_realized_observations,
            grace_period, deregistered_uids
        )
        for batch in batches
    ]
    
    result = {}
    for task in tasks:
        batch_result = task.result()
        for k, v in batch_result.items():
            result[int(k)] = v
    
    return result