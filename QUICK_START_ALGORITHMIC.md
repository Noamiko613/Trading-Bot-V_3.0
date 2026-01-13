# Quick Start: Algorithmic Trading System

## Quick Start

1. **Start the system with dashboard:**
   ```bash
   python start_algorithmic_trading.py --pairs BTCUSDT ETHUSDT --dashboard
   ```

2. **Start in background (no dashboard):**
   ```bash
   python start_algorithmic_trading.py --pairs BTCUSDT ETHUSDT
   ```

3. **Monitor multiple pairs:**
   ```bash
   python start_algorithmic_trading.py --pairs BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT ADAUSDT --dashboard
   ```

## What It Does

1. **Discovers Algorithms**: Every hour, generates and tests new trading algorithms
2. **Validates Algorithms**: Only saves algorithms that meet:
   - 66-75% win rate (configurable)
   - 2:1-3:1 risk/reward ratio (configurable)
   - Minimum 100-150 trades in backtest
3. **Trades Automatically**: Uses validated algorithms to generate and execute trades
4. **Isolated**: Completely separate from RL training - won't interfere

## Files Created

- `algorithmic_trading/` - All system data (databases, algorithms, status)
- `logs/algorithmic.log` - System logs
- `logs/algorithmic_errors.log` - Error logs

## Stopping

Press `Ctrl+C` to stop gracefully. All state is saved automatically.

## Running Alongside RL Training

The algorithmic trading system can run in a separate terminal/window alongside RL training. They are completely isolated:
- Different databases
- Different paper trading balances
- Different algorithm storage
- No shared state

## Example Output

```
================================================================================
ALGORITHMIC TRADING SYSTEM
================================================================================
Pairs: BTCUSDT, ETHUSDT
Mode: spot
Starting Balance per Pair: $10,000.00
Target Win Rate: 66.0%
Target R:R: 2.0:1
Min Trades for Validation: 100
================================================================================

🚀 Starting system...
```

