# Bittensor Miner Data Columns Explained (Using Miner 148)

## Detailed Explanation of Bittensor Miner Data (Using Miner 148)

Let me explain each column using **Miner 148** as our example. This miner is actively trading and shows good performance metrics.

### 🔍 Basic Identification

| Column | Value (Miner 148) | Meaning |
|--------|-------------------|---------|
| **Time** | 2025-12-17 13:03:51.000 | When this data snapshot was taken |
| **id** | 148 | Your miner ID (UID) in the subnet |
| **agent_id** | 148 | Same as id - internal identifier |
| **hotkey** | 5E7RbxVyLz681r9tF9a9bfB9ZuBQa4ZARbJ8HApjRruGaNtY | Your miner's public key address |
| **wallet** | 5E7RbxVyLz681r9tF9a9bfB9ZuBQa4ZARbJ8HApjRruGaNtY | Same as hotkey |
| **netuid** | 366 | Subnet number (TAOS testnet) |
| **exported_netuid** | 366 | Same as netuid |
| **miner_gauge_name** | miners | Type of miner (trading miner) |
| **job** | bittensor | The network/protocol |
| **task** | validator_metrics | What type of requests this miner handles |

### 🏢 Network & Infrastructure

| Column | Value (Miner 148) | Meaning |
|--------|-------------------|---------|
| **instance** | 3.148.166.166:9001 | Server IP + port where miner runs |
| **placement** | 2 | Server rack/location identifier |

### 💰 Trading Performance (Live Trading)

| Column | Value (Miner 148) | Meaning |
|--------|-------------------|---------|
| **activity_factor** | 0.961 | **How active this miner is** (0-1 scale)<br>0.961 = Very active (96.1% uptime) |
| **activity_factor_realized** | 0.902 | **Realized activity** (confirmed activity)<br>0.902 = 90.2% confirmed uptime |

### 💼 Inventory & Positions (Trading Book)

| Column | Value (Miner 148) | Meaning |
|--------|-------------------|---------|
| **base_balance** | 7101 | **Base currency** (BTC) balance in inventory |
| **base_collateral** | 288 | BTC held as collateral/security |
| **base_loan** | 802 | BTC borrowed (leverage) |
| **quote_balance** | 18298 | **Quote currency** (USDT) balance |
| **quote_collateral** | 0 | USDT held as collateral |
| **quote_loan** | 9.02 | USDT borrowed |

**What this means for Miner 148:**
- Has 7101 BTC in inventory but owes 802 BTC (net position: 6299 BTC)
- Has 18298 USDT available
- Using leverage (borrowing) to trade

### 📊 Profit & Loss (P&L)

| Column | Value (Miner 148) | Meaning |
|--------|-------------------|---------|
| **pnl** | 68377 | **Total profit/loss** in quote currency<br>+68,377 USDT profit! |
| **pnl_change** | -2.03 | Change in P&L this period |
| **inventory_value** | -10064 | Current value of all positions |
| **inventory_value_change** | -2.03 | Change in inventory value |

### 📈 Risk & Performance Metrics

| Column | Value (Miner 148) | Meaning |
|--------|-------------------|---------|
| **score** | 0.00880 | **Overall performance score** (0-1)<br>0.00880 = Good performance |
| **unnormalized_score** | 0 | Raw score before normalization |

### 🎯 Sharpe Ratios (Risk-Adjusted Returns)

Sharpe ratio measures return per unit of risk. Higher = better.

| Column | Value (Miner 148) | Meaning |
|--------|-------------------|---------|
| **sharpe** | 0 | Overall Sharpe ratio |
| **sharpe_penalty** | -0.0810 | Penalty for risk/volatility |
| **sharpe_realized** | 0.333 | Sharpe from completed trades |
| **sharpe_realized_penalty** | 0.140 | Risk penalty on realized trades |
| **sharpe_realized_score** | 0.339 | Final realized Sharpe score |
| **sharpe_score** | 0.539 | Combined Sharpe performance |
| **sharpe_unrealized_score** | 0 | Sharpe from open positions |

### 📊 Volume & Trading Activity

| Column | Value (Miner 148) | Meaning |
|--------|-------------------|---------|
| **total_daily_volume** | 55513064 | Total trading volume today<br>$55.5M traded! |
| **total_realized_pnl** | -17438 | P&L from closed positions |
| **total_roundtrip_volume** | 3018379 | Volume from complete trade cycles |
| **min_daily_volume** | 576584 | Minimum daily volume requirement |
| **min_roundtrip_volume** | 0 | Minimum roundtrip volume |

### ⏰ Timing & Technical

| Column | Value (Miner 148) | Meaning |
|--------|-------------------|---------|
| **timestamp** | 2741000000000 | Internal timestamp |
| **timestamp_str** | 00:45:41.000000000 | Human-readable time |
| **Value** | 1.00000 | Status indicator (1 = active) |

---

## 🎯 What Makes Miner 148 Successful:

### ✅ Strengths:
- **High Activity**: 96.1% active (vs your 0%)
- **Good P&L**: +$68K profit
- **Strong Score**: 0.00880 performance rating
- **High Volume**: $55M daily trading volume
- **Smart Leverage**: Using borrowing effectively

### 📈 Performance Analysis:
1. **Consistent Trading**: High activity factor shows reliable uptime
2. **Risk Management**: Using Sharpe ratios to measure risk-adjusted returns
3. **Volume Leader**: Handling massive trading volume ($55M+)
4. **Profit Focus**: Strong P&L despite market volatility

### 🔧 Technical Excellence:
- **Server**: Running on 3.148.166.166:9001
- **Network**: Properly registered on subnet 366
- **Infrastructure**: Good server placement (rack 2)

---

## 📋 Column Categories Summary:

| Category | Key Columns | Purpose |
|----------|-------------|---------|
| **Identity** | id, hotkey, netuid | Who you are |
| **Activity** | activity_factor, activity_factor_realized | How active you are |
| **Inventory** | base_balance, quote_balance, collateral | What you hold |
| **Performance** | pnl, score, sharpe ratios | How well you perform |
| **Risk** | sharpe_*, penalties | Risk-adjusted returns |
| **Volume** | total_*, min_* volumes | Trading activity scale |
| **Technical** | instance, timestamp | Infrastructure details |

**Miner 148 is a prime example of a successful Bittensor miner - high activity, strong profits, and excellent risk management!** 🚀

*Your miner (147) shows 0 activity because validators aren't selecting it yet. Keep it running - it will eventually get requests!*
