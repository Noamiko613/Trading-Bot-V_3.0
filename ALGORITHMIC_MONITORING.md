# Algorithmic Trading System - Market Monitoring & Algorithm Testing

## Continuous Market Monitoring

The system **actively monitors the market** and **continuously tests algorithms** to find profitable strategies.

### Market Data Updates

- **Continuous Updates**: Market data is refreshed every 30 seconds
- **Fresh Data for Testing**: Algorithms are tested on current market conditions
- **Real-time Signals**: Trading signals use the latest market data

### Algorithm Discovery & Testing

The system **actively searches** for profitable algorithms:

1. **Discovery Frequency**: Tests new algorithms every hour (configurable)
2. **Multiple Tests**: Tests 3 algorithms per discovery cycle for faster results
3. **Live Market Testing**: All algorithms are tested on current market data
4. **Validation Criteria**: Only algorithms meeting these criteria are saved:
   - Win Rate: ≥ 66% (configurable, target 66-75%)
   - Risk/Reward: ≥ 2:1 (configurable, target 2:1-3:1)
   - Minimum Trades: 100-150 trades in backtest

### How It Works

```
┌─────────────────────────────────────────────────┐
│  Market Data Loop (every 30 seconds)           │
│  - Updates market data for all pairs            │
│  - Keeps data fresh for testing                  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Algorithm Discovery Loop (every hour)          │
│  - Generates 3 random algorithms                 │
│  - Tests each on current market data             │
│  - Saves algorithms that meet criteria           │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Signal Check Loop (every minute)                │
│  - Updates market data                           │
│  - Checks active algorithms for signals          │
│  - Executes trades when signals found           │
└─────────────────────────────────────────────────┘
```

### What Gets Tested

The system tests 4 types of algorithm templates:

1. **Moving Average Crossover**: Fast/slow MA crossovers
2. **RSI Strategy**: Mean reversion using RSI
3. **Bollinger Bands**: Mean reversion using BB
4. **Momentum Strategy**: Breakout trading on momentum

Each template is tested with **random parameters** to find optimal settings.

### Monitoring Status

The system tracks:
- **Algorithms Tested**: Total number of algorithms tested
- **Algorithms Validated**: Number that met criteria
- **Algorithms Active**: Currently trading algorithms
- **Market Data Points**: Amount of data available
- **Last Discovery**: When last algorithm was tested
- **Last Market Update**: When market data was last refreshed

### Example Output

```
[INFO] algorithm_tested | symbol=BTCUSDT | algorithm_id=RSIStrategy_1234567890 | 
       reason=does_not_meet_criteria | win_rate=58.5 | avg_rr=1.8 | total_trades=120

[INFO] algorithm_validated | symbol=BTCUSDT | algorithm_id=MovingAverageCrossover_1234567891 | 
       win_rate=72.3 | avg_rr=2.5 | total_trades=145
```

The system is **always active** - it doesn't just sit idle. It continuously:
- ✅ Monitors market data
- ✅ Tests new algorithms
- ✅ Finds profitable strategies
- ✅ Trades with validated algorithms

