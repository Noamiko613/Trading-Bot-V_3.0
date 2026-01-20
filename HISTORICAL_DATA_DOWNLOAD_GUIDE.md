# Historical Data Download Guide

## Overview

The historical data download system allows you to pre-download and cache all historical market data before running RL training. This provides several benefits:

1. **Faster Training**: Data is loaded from local cache instead of fetching from CoinEx API
2. **Reliability**: Data is cached, so training can resume even if CoinEx API is temporarily unavailable
3. **Efficiency**: Download once, train multiple times without re-downloading

## Quick Start

### Option 1: Automatic Download + Training (Recommended)

Use the wrapper script that downloads data first, then runs training:

```bash
python run_rl_training.py
```

This will:
1. Download and cache all historical data for all training symbols/timeframes
2. Run RL training using the cached data

### Option 2: Manual Download First

Download data separately, then run training:

```bash
# Step 1: Download data
python download_historical_data.py

# Step 2: Run training (will use cached data)
python rl_training.py
```

## Download Script Usage

### Basic Usage

```bash
python download_historical_data.py
```

Downloads data for:
- **Symbols**: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
- **Timeframes**: 15m, 1h, 4h, 6h, 12h, 1d
- **Date Range**: 2019-12-01 to today

### Custom Options

```bash
# Custom symbols
python download_historical_data.py --symbols BTCUSDT,ETHUSDT

# Custom timeframes
python download_historical_data.py --timeframes 1h,4h,1d

# Custom date range
python download_historical_data.py --start-date 2020-01-01 --end-date 2024-12-31

# Force re-download (ignore cache)
python download_historical_data.py --no-cache

# Custom cache directory
python download_historical_data.py --cache-dir /path/to/cache
```

### Full Example

```bash
python download_historical_data.py \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --timeframes 1h,4h,1d \
    --start-date 2020-01-01 \
    --end-date 2024-12-31
```

## Training Runner Usage

### Basic Usage

```bash
python run_rl_training.py
```

### Skip Download (Use Existing Cache)

```bash
python run_rl_training.py --skip-download
```

### Download Only (Don't Train)

```bash
python run_rl_training.py --download-only
```

### Force Re-download

```bash
python run_rl_training.py --force-download
```

### Pass Arguments to Training

All arguments are passed through to `rl_training.py`:

```bash
python run_rl_training.py --timesteps 1000000 --symbol BTCUSDT
```

## Cache Location

Cached data is stored in:
```
data/historical_cache/
```

Each symbol/timeframe combination is saved as a CSV file:
```
data/historical_cache/BTCUSDT_1h.csv
data/historical_cache/BTCUSDT_4h.csv
data/historical_cache/ETHUSDT_1h.csv
...
```

Metadata is stored in:
```
data/historical_cache/metadata.json
```

## How It Works

### 1. Download Phase

The `download_historical_data.py` script:
- Checks if data is already cached
- If cached and up-to-date, skips download
- If not cached or outdated, downloads from CoinEx
- Saves data to CSV files in the cache directory
- Updates metadata with date ranges and download info

### 2. Training Phase

The `HistoricalDataReplay` class:
- First checks for cached data
- If found, loads from cache (much faster)
- If not found, falls back to fetching from CoinEx API
- Uses the same data format regardless of source

### 3. Cache Validation

The system checks:
- If cache file exists
- If cached data covers the requested date range
- If cache is recent (within 7 days)

If all checks pass, cached data is used. Otherwise, fresh data is downloaded.

## Benefits

1. **Speed**: Loading from CSV is much faster than API calls
2. **Reliability**: Training can continue even if CoinEx API is down
3. **Cost**: Reduces API rate limit issues
4. **Reproducibility**: Same data for multiple training runs
5. **Offline Training**: Can train without internet after initial download

## Troubleshooting

### Cache Not Being Used

If training still fetches from API:
1. Check that cache files exist in `data/historical_cache/`
2. Verify cache files are not corrupted
3. Check that date ranges match
4. Try `--force-download` to refresh cache

### Out of Date Cache

To refresh cache:
```bash
python download_historical_data.py --no-cache
```

### Clear Cache

To start fresh:
```bash
rm -rf data/historical_cache/
python download_historical_data.py
```

## Integration with RL Training

The system is fully integrated with `rl_training.py`:

- `HistoricalDataReplay` automatically checks cache first
- Falls back to API if cache is missing
- No changes needed to existing training code
- Works with both historical pre-training and live fine-tuning phases

## File Structure

```
data/
└── historical_cache/
    ├── metadata.json          # Cache metadata
    ├── BTCUSDT_1h.csv         # Cached data files
    ├── BTCUSDT_4h.csv
    ├── ETHUSDT_1h.csv
    └── ...
```

## Notes

- Cache files are CSV format for easy inspection
- Metadata tracks download dates and date ranges
- Cache is automatically used when available
- No manual intervention needed - just run the scripts!
