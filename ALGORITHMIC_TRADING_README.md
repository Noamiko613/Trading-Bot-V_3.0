# Algorithmic Trading System

## Overview

The Algorithmic Trading System is an independent trading instance that:
- Uses the same live data from CoinEx as the main system
- Has its own isolated paper trading instance with separate balance and trades
- Monitors specified trading pairs independently
- Discovers and tests profitable trading algorithms (target: 66-75% win rate, 2:1-3:1 R:R)
- Saves validated algorithms when they meet criteria (100-150 trades minimum)
- Runs concurrently with RL training but is completely isolated

## Key Features

### Isolation
- **Separate Database**: Uses `algorithmic_trading/` directory for all data
- **Isolated Paper Trading**: Each pair has its own balance and trade history
- **Independent Algorithms**: Algorithm storage separate from RL models
- **No Interference**: Does not share state with RL training or main trading system

### Algorithm Discovery
- **Multiple Templates**: Moving Average Crossover, RSI, Bollinger Bands, Momentum
- **Random Parameter Generation**: Tests various parameter combinations
- **Backtesting**: Evaluates algorithms on historical data before validation
- **Quality Gates**: Only saves algorithms meeting win rate and R:R criteria

### Trading
- **Paper Trading Only**: No real money trading (safety enforced)
- **Risk Management**: 1% risk per trade by default
- **Signal Generation**: Checks for signals every minute
- **Algorithm Cooldown**: 5-minute cooldown between signals from same algorithm

## Usage

### Basic Usage

```bash
# Start with default settings
python start_algorithmic_trading.py --pairs BTCUSDT ETHUSDT --dashboard

# Start in background mode (no dashboard)
python start_algorithmic_trading.py --pairs BTCUSDT ETHUSDT

# Custom settings
python start_algorithmic_trading.py \
    --pairs BTCUSDT ETHUSDT SOLUSDT \
    --mode spot \
    --balance 10000 \
    --target-win-rate 70 \
    --target-rr 2.5 \
    --min-trades 150 \
    --dashboard
```

### Command Line Options

- `--pairs`: Trading pairs to monitor (required, space-separated)
- `--mode`: Trading mode: `spot` or `futures` (default: `spot`)
- `--balance`: Starting balance per pair in USDT (default: 10000)
- `--min-trades`: Minimum trades for algorithm validation (default: 100)
- `--target-win-rate`: Target win rate percentage (default: 66.0)
- `--target-rr`: Target risk/reward ratio (default: 2.0)
- `--discovery-interval`: Algorithm discovery interval in seconds (default: 3600 = 1 hour)
- `--signal-interval`: Signal check interval in seconds (default: 60 = 1 minute)
- `--dashboard`: Show real-time dashboard
- `--no-dashboard`: Run without dashboard (background mode)

## Configuration

Edit `config/algorithmic_trading.json` to change default settings:

```json
{
  "pairs": ["BTCUSDT", "ETHUSDT"],
  "mode": "spot",
  "starting_balance_per_pair": 10000.0,
  "discovery_interval_seconds": 3600,
  "signal_check_interval_seconds": 60,
  "min_trades_for_validation": 100,
  "target_win_rate": 66.0,
  "target_rr": 2.0,
  "risk_per_trade_pct": 0.01,
  "algorithm_cooldown_seconds": 300,
  "min_rr_for_trade": 2.0
}
```

## Directory Structure

```
algorithmic_trading/
├── paper_trading_BTCUSDT.db          # Per-pair paper trading database
├── paper_trading_ETHUSDT.db
├── algorithms/                        # Saved validated algorithms
│   ├── MovingAverageCrossover_1234567890.json
│   └── RSIStrategy_1234567891.json
└── BTCUSDT/                          # Per-pair status files
    ├── status.json
    └── algorithm_status.json
```

## Dashboard

The dashboard shows:
- **Account Info**: Balance, equity per pair
- **Trading Stats**: Open/closed trades, win rate, PnL, profit factor
- **Algorithm Stats**: Active algorithms, validation status
- **Per-Algorithm Performance**: Win rate and PnL for each algorithm

## Algorithm Validation Criteria

An algorithm is considered valid and saved when it meets:
- **Minimum Trades**: At least 100-150 trades in backtest
- **Win Rate**: ≥ 66% (configurable, target 66-75%)
- **Risk/Reward**: Average R:R ≥ 2.0 (configurable, target 2:1-3:1)

## Integration with Main System

### Shared Resources
- **Data Source**: Uses same `CoinExDataFetcher` for live market data
- **Indicators**: Uses same indicator functions from `core.indicators`
- **Logging**: Uses same logging system (separate log file: `logs/algorithmic.log`)

### Isolation Guarantees
- **Separate Databases**: No shared SQLite databases
- **Separate Directories**: All files in `algorithmic_trading/` directory
- **Separate Balance**: Each pair has independent paper trading balance
- **No State Sharing**: No shared variables or state with RL training

## Monitoring

### Logs
- Main log: `logs/algorithmic.log`
- Error log: `logs/algorithmic_errors.log`

### Status Files
- Per-pair status: `algorithmic_trading/{PAIR}/status.json`
- Algorithm status: `algorithmic_trading/{PAIR}/algorithm_status.json`

### Database
- Per-pair database: `algorithmic_trading/paper_trading_{PAIR}.db`
- Contains: account balance, all trades (open and closed)

## Stopping the System

Press `Ctrl+C` to gracefully stop the system. All trades and state are saved automatically.

## Notes

- **Paper Trading Only**: The system enforces paper trading mode and will not trade with real money
- **Concurrent Operation**: Can run alongside RL training without interference
- **Resource Usage**: Each pair runs independently, so multiple pairs increase resource usage
- **Algorithm Discovery**: New algorithms are discovered every hour by default (configurable)

