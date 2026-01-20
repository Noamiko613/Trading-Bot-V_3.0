# Historical Pre-Training Guide

## Overview

The trading bot now supports a **two-phase training approach** that follows best practices for RL in trading:

1. **Phase 1: Historical Pre-Training** - Train on extensive historical data (2019-2024) to build a robust baseline policy
2. **Phase 2: Live Paper Trading Fine-Tuning** - Fine-tune on live market data for adaptation to current conditions

This approach is superior to training directly on live data because:
- Historical data provides diverse market regimes (bull, bear, flash crashes)
- Builds a robust baseline before facing real-time noise
- Reduces the "reality gap" through realistic simulation
- Better sample efficiency and faster convergence

## Configuration

### Environment Variables

Set these environment variables to configure historical pre-training:

```bash
# Enable/disable historical pre-training (default: enabled)
USE_HISTORICAL_PRETRAINING=1

# Historical data date range
HISTORICAL_START_DATE=2019-01-01
HISTORICAL_END_DATE=2024-12-31

# Number of timesteps for historical pre-training
HISTORICAL_PRETRAIN_TIMESTEPS=500000
```

### Default Behavior

By default, historical pre-training is **enabled** with:
- Date range: 2019-01-01 to 2024-12-31 (5+ years of data)
- Target timesteps: 500,000
- Automatically transitions to live paper trading after completion

## Usage

### Basic Training (with historical pre-training)

```bash
python rl_training.py
```

This will:
1. Load historical data from 2019-2024
2. Pre-train the model for 500,000 timesteps
3. Automatically transition to live paper trading fine-tuning

### Custom Historical Date Range

```bash
HISTORICAL_START_DATE=2020-01-01 \
HISTORICAL_END_DATE=2023-12-31 \
python rl_training.py
```

### Disable Historical Pre-Training (legacy behavior)

```bash
USE_HISTORICAL_PRETRAINING=0 python rl_training.py
```

### Resume Training

The system automatically tracks whether historical pre-training is completed. If you restart training:
- If historical pre-training is incomplete, it will resume from where it left off
- If historical pre-training is complete, it will skip directly to live fine-tuning

## How It Works

### Phase 1: Historical Pre-Training

1. **Data Loading**: Fetches historical OHLCV data from CoinEx for the specified date range
2. **Chronological Replay**: Replays historical data step-by-step, simulating real-time conditions
3. **Training**: PPO model learns from historical market patterns, crashes, and trends
4. **Checkpointing**: Saves progress periodically for resume capability

### Phase 2: Live Fine-Tuning

1. **Transition**: Automatically switches to live market data after historical phase completes
2. **Fine-Tuning**: Model adapts to current market conditions while retaining historical knowledge
3. **Paper Trading**: All trades are simulated (no real money at risk)

## Benefits

### 1. Robust Baseline Policy
- Learns from diverse market conditions before facing live data
- Understands bull markets, bear markets, and flash crashes
- Better generalization to unseen market regimes

### 2. Faster Convergence
- Historical data provides high-quality, diverse samples
- Model starts with good policy instead of random initialization
- Reduces training time on live data

### 3. Reduced Reality Gap
- Historical simulation closely mimics live trading
- Includes fees, slippage, and order book dynamics
- Better transfer from training to live trading

### 4. Sample Efficiency
- Historical data is "free" (no waiting for live market events)
- Can train on years of data in hours
- More diverse experience than live-only training

## Technical Details

### Historical Data Replay

The `HistoricalDataReplay` class:
- Loads all historical data upfront for efficiency
- Replays data chronologically, candle-by-candle
- Maintains lookback windows for indicators
- Supports multiple timeframes simultaneously

### Environment Integration

The `TradingEnv` class:
- Detects historical replay mode automatically
- Uses historical data instead of live API calls
- Maintains same interface for seamless transition
- Supports both historical and live modes

### Training State Management

The system tracks:
- Historical pre-training completion status
- Current training phase
- Checkpoint paths for resume
- Progress metrics

## Troubleshooting

### Historical Data Not Loading

If you see errors loading historical data:
1. Check your internet connection (data is fetched from CoinEx)
2. Verify date range is valid (CoinEx may not have data for very old dates)
3. Check symbol format (use CCXT format like BTC/USDT)

### Out of Memory

If you run out of memory:
1. Reduce date range (e.g., 2020-2024 instead of 2019-2024)
2. Reduce number of timeframes
3. Use smaller lookback window

### Slow Historical Loading

Historical data loading can take several minutes:
- This is normal for large date ranges
- Data is cached after first load
- Consider using smaller date ranges for testing

## Best Practices

1. **Start with Recent Data**: For initial testing, use 2022-2024 to reduce load time
2. **Full Range for Production**: Use 2019-2024 for production training (covers multiple crypto cycles)
3. **Monitor Progress**: Watch training metrics during historical phase
4. **Checkpoint Frequently**: Historical training can take hours, use checkpoints
5. **Validate Before Live**: Ensure model performs well on historical validation set

## Example Training Session

```
[RL] ====================================================
[RL] RL TRAINING MODE - PAPER SIMULATION ONLY
[RL] ===================================================

================================================================================
PHASE 1: HISTORICAL PRE-TRAINING
================================================================================
Training on historical data: 2019-01-01 to 2024-12-31
Target timesteps: 500,000
================================================================================

[HistoricalReplay] Loading historical data for BTCUSDT from 2019-01-01 to 2024-12-31
[HistoricalReplay] Loading 1h timeframe...
[HistoricalReplay] Loaded 43800 candles for 1h timeframe
[RL] Historical pre-training progress: 50,000/500,000 (10.0%)
[RL] Historical pre-training progress: 100,000/500,000 (20.0%)
...
[RL] ✅ Historical pre-training completed: 500,000 timesteps
[RL] Model ready for live paper trading fine-tuning

================================================================================
PHASE 2: LIVE PAPER TRADING FINE-TUNING
================================================================================
Fine-tuning on live market data for adaptation to current conditions
================================================================================

[RL] Starting training...
[RL] Progress: 500,000 timesteps - Trades: 1,234
```

## Conclusion

The two-phase training approach provides a solid foundation for RL trading models. By pre-training on historical data, the model learns robust strategies before adapting to live conditions, resulting in better performance and faster convergence.
