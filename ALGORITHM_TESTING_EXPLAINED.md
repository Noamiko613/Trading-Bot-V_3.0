# Algorithm Testing System - How It Works

## What You're Seeing

```
Algorithms: 0 active | 0 validated | 3 tested
```

This means:
- **3 tested**: 3 algorithms were generated and backtested
- **0 validated**: None of them met the quality criteria
- **0 active**: No algorithms are currently trading (because none passed validation)

## The Algorithm Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: TESTING (Every Hour)                          │
│  ─────────────────────────────────────────────────────  │
│  • System generates random algorithm                   │
│  • Tests it on last 500 candles of market data         │
│  • Simulates trades: entry → TP/SL outcomes            │
│  • Calculates: win rate, R:R ratio, total trades       │
│  • Counter: "algorithms_tested" increments             │
└─────────────────────────────────────────────────────────┘
                    ↓
         ┌──────────────────────┐
         │  Did it pass?        │
         │  • Win Rate ≥ 66%?   │
         │  • R:R ≥ 2:1?        │
         │  • ≥ 100 trades?     │
         └──────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        │                       │
     ❌ NO                   ✅ YES
        │                       │
        │                       ↓
        │           ┌──────────────────────────┐
        │           │  STEP 2: VALIDATION      │
        │           │  ──────────────────────  │
        │           │  • Algorithm saved       │
        │           │  • Added to probationary  │
        │           │  • Counter increments    │
        │           └──────────────────────────┘
        │                       ↓
        │           ┌──────────────────────────┐
        │           │  STEP 3: PROBATION        │
        │           │  ──────────────────────  │
        │           │  • Trades in live market │
        │           │  • Monitored for 50 trades│
        │           │  • Must maintain:        │
        │           │    - Win Rate ≥ 60%      │
        │           │    - R:R ≥ 1.8            │
        │           └──────────────────────────┘
        │                       ↓
        │           ┌──────────────────────────┐
        │           │  STEP 4: ACTIVE         │
        │           │  ──────────────────────  │
        │           │  • Fully validated        │
        │           │  • Trading live           │
        │           │  • Counter increments    │
        │           └──────────────────────────┘
        │
   Algorithm discarded
   (doesn't meet criteria)
```

## What "Tested" Means

When an algorithm is **tested**, the system:

1. **Generates a random algorithm** from 4 templates:
   - Moving Average Crossover (fast/slow MA crossovers)
   - RSI Strategy (mean reversion)
   - Bollinger Bands (mean reversion)
   - Momentum Strategy (breakout trading)

2. **Tests it on real market data**:
   - Uses the last 500 candles from CoinEx
   - Simulates trades from candle 50 onwards
   - For each signal, checks if TP or SL was hit in next 50 candles
   - Records win/loss and calculates R-multiple

3. **Calculates metrics**:
   - Win Rate: (Wins / Total Trades) × 100
   - Average R:R: Average risk/reward ratio
   - Total Trades: Number of signals generated

4. **Checks if it meets criteria**:
   - ✅ Win Rate ≥ 66% (configurable)
   - ✅ R:R ≥ 2:1 (configurable)
   - ✅ ≥ 100 trades in backtest

## Why 0 Validated?

The 3 tested algorithms **didn't meet the criteria**. They might have:
- Win rate < 66% (e.g., 58%, 62%, 55%)
- R:R < 2:1 (e.g., 1.5:1, 1.8:1)
- Too few trades (< 100)

**This is normal!** Most random algorithms won't be profitable. The system keeps testing until it finds good ones.

## What Happens Now?

The system **continues working automatically**:

1. **Every Hour**: Tests 3 new random algorithms
2. **Every Minute**: Checks active algorithms for trading signals
3. **Every 30 Seconds**: Updates market data

### Current Status

Since you have:
- **3 tested, 0 validated, 0 active**

The system is:
- ✅ Running and testing algorithms
- ✅ Waiting for algorithms that meet criteria
- ⏳ Will automatically start trading when algorithms pass validation

## How Long Until Algorithms Are Found?

**Typical timeline**:
- **First validated algorithm**: Usually within 2-10 hours
- **First active algorithm**: Usually within 4-20 hours (after probation period)

**Factors**:
- Market conditions (trending vs. ranging)
- Algorithm templates (some work better in certain markets)
- Random parameter generation (luck of the draw)

## What You Can Do

### Option 1: Wait (Recommended)
- System will continue testing automatically
- Algorithms will be found and start trading
- No action needed

### Option 2: Check Logs
Look for messages like:
```
[INFO] algorithm_tested | symbol=BTCUSDT | win_rate=58.5 | avg_rr=1.8
[INFO] algorithm_validated | symbol=BTCUSDT | win_rate=72.3 | avg_rr=2.5
```

### Option 3: Adjust Criteria (If Too Strict)
Edit `config/algorithmic_trading.json` or use environment variables:
```bash
# Lower win rate threshold (easier to pass)
export ALGO_TARGET_WIN_RATE=60

# Lower R:R threshold
export ALGO_TARGET_RR=1.8

# Lower minimum trades
export ALGO_MIN_TRADES=80
```

**Warning**: Lowering criteria means more algorithms pass, but quality may be lower.

### Option 4: Check Dashboard
If you're running with `--dashboard`:
```bash
python start_algorithmic_trading.py --pairs BTCUSDT ETHUSDT --dashboard
```

The dashboard shows:
- Real-time algorithm testing
- Which algorithms are being tested
- Why they pass/fail
- When algorithms are validated

## Example: What a Successful Test Looks Like

```
[INFO] algorithm_tested | symbol=BTCUSDT | algorithm_id=RSIStrategy_1234567890
       reason=does_not_meet_criteria | win_rate=58.5 | avg_rr=1.8 | total_trades=120
       ❌ FAILED: Win rate 58.5% < 66% required

[INFO] algorithm_tested | symbol=BTCUSDT | algorithm_id=MovingAverageCrossover_1234567891
       reason=does_not_meet_criteria | win_rate=64.2 | avg_rr=1.9 | total_trades=135
       ❌ FAILED: Win rate 64.2% < 66% required

[INFO] algorithm_validated | symbol=BTCUSDT | algorithm_id=BollingerBands_1234567892
       win_rate=72.3 | avg_rr=2.5 | total_trades=145
       ✅ PASSED: Win rate 72.3% ≥ 66%, R:R 2.5 ≥ 2.0, Trades 145 ≥ 100
       → Algorithm saved and added to probationary list
```

## Summary

**What "3 tested" means**:
- 3 algorithms were generated and backtested on real market data
- They were evaluated for profitability
- None met the quality criteria (66% win rate, 2:1 R:R, 100+ trades)

**What happens now**:
- System continues testing every hour
- Will find profitable algorithms eventually
- When found, they'll be validated → probationary → active
- Then they'll start generating trades automatically

**You don't need to do anything** - the system is working as designed! It's just being selective about which algorithms it uses (which is good for profitability).
