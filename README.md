# Multi-Timeframe Pattern Detection Trading Bot

A sophisticated cryptocurrency trading bot system featuring multi-timeframe analysis, machine learning pattern scoring, session-aware trading, comprehensive daily trading strategies, and advanced risk management.

## 🚀 Quick Start (3 Steps)

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Configure symbols in `symbols_config.json`
```json
{
  "trading_settings": {
    "auto_train_ml": true,
    "operating_mode": "hybrid",
    "moderation_mode": "balanced",
    "trading_mode": "balanced"
  },
  "symbols": [
    { "symbol": "BTC-USDT", "enabled": true, "timeframes": ["5min", "15min", "1h", "4h"], "risk_per_trade_pct": 0.3 }
  ]
}
```

3) Start trading and optionally monitor
```bash
python auto_trader.py
# In another terminal (optional):
python examples/performance_monitor.py
```

---

## 📋 Table of Contents
- [Features](#-features)
- [Installation](#️-installation)
- [Configuration](#️-configuration)
- [Usage](#-usage)
- [Daily Trading Strategies](#-daily-trading-strategies)
- [Pattern Detection](#-pattern-detection)
- [Confidence Scoring System](#-confidence-scoring-system)
- [Session Awareness](#-session-awareness)
- [Risk Management](#️-risk-management)
- [Automation](#-automation)
- [Monitoring & Analytics](#-monitoring--analytics)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)
- [Risk Warnings](#️-risk-warnings)

---

## ✨ Features

### Core Capabilities
- **Multi-Timeframe Analysis**: 1min → 1month timeframes
- **15+ Technical Patterns**: Verified with ML-enhanced confidence scoring
- **6 Daily Trading Strategies**: Scalping, Momentum, Range, Swing, Arbitrage, Session-Based
- **Session-Aware Trading**: Adaptive to US/London/Asia market sessions
- **Dynamic Risk Management**: Confidence-based position sizing with kill-switch
- **Multi-Symbol Automation**: Automated trading across multiple symbols
- **Real-Time Analytics**: Performance tracking and pattern analysis
- **Machine Learning Integration**: Pattern scoring with bounded ML adjustments

### Advanced Features
- **Weighted Confidence Scoring**: 7-component system with normalized weights
- **Dynamic Position Sizing**: Scales risk based on confidence (0.1-0.3% per trade)
- **Dynamic Reward-to-Risk**: Adaptive RR targets based on confidence levels
- **Account-Level Protections**: Max trades, exposure limits, daily loss limits
- **Kill Switch System**: Soft alert at 15% DD, hard stop at 25% DD
- **Directional MTF Confirmation**: Relaxed to trend alignment only (not full pattern match)
- **Session Multipliers**: Confidence and position size adjustments by session quality

---

## 🛠️ Installation

### Requirements
- Python 3.8+
- Dependencies: `pip install -r requirements.txt`

### Environment Setup
Create/update `.env` file with optimized defaults:
```env
# Trading Mode
TRADE_LIVE=0
DEFAULT_OPERATING_MODE=hybrid
DEFAULT_MODERATION_MODE=balanced
DEFAULT_TRADING_MODE=balanced

# Trading Logic Parameters (LAYER 1)
REQUIRE_MTF_CONFIRMATION=1
MIN_CONFIDENCE=55
MIN_RR=1.3
APPLY_FILTERS_TO_DAILY=1
MAX_RISK_PCT=0.003          # 0.3% per trade
RISK_PER_TRADE_PCT=0.002    # 0.2% base risk

# ML Configuration (LAYER 4)
ML_NUDGE_LIMIT=0.05         # ±5% ML adjustment

# Account-Level Protections (LAYER 6)
MAX_CONCURRENT_TRADES=6
MAX_TOTAL_EXPOSURE_PCT=0.02  # 2% equity cap
DAILY_LOSS_LIMIT_PCT=0.03    # -3% daily loss limit
KILL_SWITCH_HARD_DD=0.25     # 25% hard drawdown limit
KILL_SWITCH_SOFT_DD=0.15      # 15% soft drawdown alert
```

### Setup Helper
Run the setup script to create default configurations:
```bash
python setup_enhancements.py
```

---

## ⚙️ Configuration

### Symbols Configuration (`symbols_config.json`)
```json
{
  "trading_settings": {
    "auto_train_ml": true,
    "min_trades_for_ml": 50,
    "operating_mode": "hybrid",
    "moderation_mode": "balanced",
    "trading_mode": "balanced"
  },
  "symbols": [
    {
      "symbol": "BTC-USDT",
      "enabled": true,
      "timeframes": ["5min", "15min", "1h", "4h"],
      "risk_per_trade_pct": 0.3,  # 0.3% for BTC
      "max_concurrent_trades": 3,
      "custom_settings": {
        "confidence_threshold": 55,
        "volume_threshold": 0.8
      }
    }
  ]
}
```

### Configuration Files (`config/`)
- **`trading.json`**: Trading mode and timeframes
- **`risk.json`**: Risk management parameters, kill switch settings, dynamic RR
- **`sessions.json`**: Session multipliers, confidence adjustments, position multipliers
- **`indicators.json`**: Technical indicator settings
- **`patterns.json`**: Pattern detection parameters
- **`modes.json`**: Trading mode configurations

---

## 🎮 Usage

### Basic Examples
```bash
# Multi-symbol automation
python auto_trader.py

# Manual run with specific symbol/timeframes
python main.py --symbol ETH-USDT --timeframes 5min 15min 1h --mode spot --paper-trading
```

### Strategy Modes
- **Operating Mode**: `pattern` | `session` | `hybrid` | `auto`
- **Moderation Mode**: `strict` | `balanced` | `aggressive`
- **Trading Mode**: `safe` | `balanced` | `aggressive`

---

## 📈 Daily Trading Strategies

The bot implements a comprehensive daily trading system with 6 distinct strategies that adapt to market sessions and conditions.

### Baseline Daily Trading Structure

**Time Segmentation (UTC)**:
- **Tokyo (Asian)**: 00:00 – 09:00 UTC → Low volatility, range setups
- **London**: 07:00 – 16:00 UTC → Breakout + momentum session
- **New York**: 12:00 – 21:00 UTC → High volatility, reversals, trend continuations
- **Overlap (London–NY)**: 12:00–16:00 UTC → Most volume and breakouts

**Market Context Classification**:
- Identifies market structure (trend or range)
- Measures session volatility (quiet or expanding)
- Tracks key levels (previous day's high/low, session highs/lows, liquidity zones)

**Base Trade Rules**:
- Risk per trade: 0.5–2% max (configurable per symbol)
- Reward-to-risk: Minimum 1.5:1 (dynamic based on confidence)
- Always trade with volume, avoid low-liquidity hours
- Avoid trading during major news releases

### 1. ⚡ Scalping Logic

**Goal**: Small, frequent profits during high liquidity (1m–5m timeframes)

**Entry Logic**:
- Focus on micro-trends using EMA(9) & EMA(20)
- Confirm with RSI(3–5) or MACD crossover
- Entry trigger: Price breaks micro-range or VWAP level with volume surge

**Exit Logic**:
- Target: 0.2–0.5% per trade
- Tight SL: 5–10 pips or 0.1–0.3%
- Trail stop-loss behind micro structure

**Best Sessions**: London open (07:00–09:00 UTC) and NY open (12:00–14:00 UTC)

### 2. 🚀 Momentum Trading Logic

**Goal**: Capture strong directional moves caused by large players or news

**Entry Logic**:
- Identify momentum ignition candles (large volume, long-bodied candle)
- Confirm with volume > 1.5× average, expanding MACD histogram, RSI crossing 50
- Enter on retest of breakout zone or continuation candle

**Exit Logic**:
- First target: Previous major swing
- Partial close at 1.5R, move stop to breakeven
- Full close when RSI diverges or volume fades

**Best Sessions**: London → NY overlap (12:00–16:00 UTC)

### 3. 📉 Range Trading Logic

**Goal**: Trade between well-defined support & resistance levels

**Entry Logic**:
- Identify clear horizontal range (2+ touches on top and bottom)
- Timeframe: 15m–1h
- Enter long near support when RSI < 30
- Enter short near resistance when RSI > 70
- Confirm with candle rejection or wick pattern

**Exit Logic**:
- Exit at opposite boundary (top or bottom of range)
- If midrange breaks, close early (potential breakout forming)
- Stop loss: Slightly outside range boundary

**Best Sessions**: Tokyo session (00:00–09:00 UTC) for quiet consolidation

### 4. 📈 Swing Trading Logic

**Goal**: Capture multi-day to multi-session moves

**Entry Logic**:
- Timeframe: 4H–1D
- Identify trend direction: EMA(20) > EMA(50) for uptrend
- Entry on retest of broken resistance (now support) or trendline
- Confirm with volume or OBV rising in direction

**Exit Logic**:
- TP1: 1.5× risk, TP2: Next structure level
- Use trailing stop under swing lows/highs
- Exit when opposite structure breaks or RSI divergence forms

**Risk Logic**:
- Fewer trades (2–5 per week)
- Higher RR ratio (2–3:1)

### 5. 💱 Arbitrage Logic (Placeholder)

**Goal**: Exploit temporary price inefficiencies between exchanges or pairs

**Types**:
- Exchange arbitrage: Buy on cheaper exchange, sell on pricier one
- Triangular arbitrage: Exploit mismatched conversion rates

**Note**: Full implementation requires exchange API integration with real-time order book data.

### 6. 🌍 Session-Based Trading Logic

**Goal**: Use daily volatility cycles for timed entries/exits

**Entry Logic**:
- Identify session ranges (Asia high/low → used for London breakouts)
- Common pattern: Asia consolidates → London breaks out → NY reverses/continues
- Entry: Wait for London breakout with confirmation candle close outside range

**Exit Logic**:
- TP: 1–1.5× Asia range size
- SL: Half of range width
- Exit before NY close (end of high-volatility period)

### 🧩 Combined Daily Trading System

**Strategy Selection by Session**:

| Time (UTC) | Session | Strategy | Notes |
|------------|---------|----------|-------|
| 00:00–07:00 | Asian | Range trading | Quiet session, fade edges |
| 07:00–12:00 | London | Momentum + Scalping | Breakouts begin |
| 12:00–16:00 | London/NY overlap | Momentum + Session breakout | Most profitable |
| 16:00–21:00 | NY session | Range/swing re-entries | Reversals, end of day setups |
| 21:00–00:00 | Post-NY | No trade / review | Reset cycle |

**Balance Settings**:

| Factor | Safe Setting | Aggressive Setting | Balanced Setting |
|--------|--------------|-------------------|------------------|
| Risk per trade | 0.5% | 2–3% | 1–1.5% |
| Max trades/day | 2 | 10+ | 4–6 |
| Stop-loss | Wide | Tight | Structure-based |
| TP target | 1R | 3R+ | 1.5–2R |
| Trade duration | 1h–4h | Minutes | 15m–1h |
| News exposure | Avoid | Trade it | Trade only confirmed continuation |

---

## 📊 Pattern Detection

### Supported Pattern Categories

**Trend Patterns**:
- Golden Cross (MA50 > MA200)
- Death Cross (MA50 < MA200)

**Momentum Patterns**:
- MACD Bullish/Bearish Crossover
- RSI Trend Changes
- RSI Divergence

**Chart Patterns**:
- Bull Flag Breakout
- Triangle Breakout
- Head & Shoulders
- Double Top/Bottom

**Candlestick Patterns**:
- Bullish Engulfing
- Hammer
- Morning Star / Evening Star

**Breakout Patterns**:
- Scalp Breakout
- EMA21 Pullback Bounce
- VWAP Bounce
- EMA Ribbon Hold

**Confluence Engine**:
- Support/Resistance zones with Fibonacci confluence
- Volume and candlestick confirmations
- Multi-timeframe trend context

All patterns are verified with confirmations and confidence scoring. ML can enhance scoring once enough trades are collected.

---

## 🎯 Confidence Scoring System (LAYER 2)

### Weighted Component System

The confidence score is calculated using a 7-component weighted system, each normalized to 0–1:

| Component | Weight | Description |
|-----------|--------|-------------|
| Pattern Structure Quality | 25% | Clarity, depth, symmetry of pattern |
| Trend Alignment | 15% | Matches higher timeframe direction |
| Momentum Confirmation | 15% | RSI/MACD agreement |
| Volume Confirmation | 10% | Above average, consistent |
| Location/Context | 15% | Near S/R, MA cluster, VWAP |
| Risk/Reward Geometry | 10% | Target vs stop ratio quality |
| Session Context | 5% | Session quality adjustment |

**Calculation**:
```python
raw_conf = sum(weight * component for weight, component in components)
confidence_pct = raw_conf * 100
effective_conf = confidence_pct * session_multiplier[session_type]
```

### ML Adjustment (LAYER 4)

Machine Learning is used as a **bounded nudge**, not a full decision maker:
- ML adjustment bound: ±5% initially (configurable via `ML_NUDGE_LIMIT`)
- Can expand to ±10–12% after validation
- Retrain: Weekly rolling window or 500 new trades
- Features: Confidence components + volatility + spread + time-of-day

### Session Multipliers (LAYER 5)

Confidence is adjusted by session quality:
- **High liquidity**: 1.0 (no adjustment)
- **Medium liquidity**: 0.95
- **Low liquidity**: 0.92
- **Weekend**: 0.90

**Confidence Adjustments** (penalties):
- High liquidity: +0
- Medium liquidity: +5
- Low liquidity: +8
- Weekend: +10

---

## 🎯 Trade Decision Logic

### Thresholds (LAYER 1)

**Global Environment Variables**:
- `MIN_CONFIDENCE` (default 55): Minimum confidence to trade
- `MIN_RR` (default 1.3): Minimum required risk-to-reward
- `REQUIRE_MTF_CONFIRMATION` (default 1): Require multi-timeframe confirmation (directional alignment only)
- `APPLY_FILTERS_TO_DAILY` (default 1): Apply filters to daily timeframes
- `MAX_RISK_PCT` (default 0.003): Hard cap for per-trade risk (0.3%)

**Per-Symbol Overrides** (`symbols_config.json`):
- `custom_settings.confidence_threshold` (e.g., 55–70)
- `risk_per_trade_pct` (0.1–0.3% per symbol)
- `max_concurrent_trades`

Decision uses the stricter of per-symbol threshold and `MIN_CONFIDENCE`.

### Multi-Timeframe Confirmation (LAYER 1 - Relaxed)

**Directional Alignment Only**: Instead of demanding full pattern match on higher timeframe, only requires HTF trend direction match:
- For BUY: Higher TF in uptrend (MA50 > MA200)
- For SELL: Higher TF in downtrend (MA50 < MA200)

This avoids over-filtering while maintaining trend alignment.

### Session-Aware Gating (LAYER 5)

Sessions are categorized with different requirements:
- **High liquidity** (US open/overlap): Base thresholds, full position size
- **Medium liquidity** (Asia open): +5 confidence requirement, 0.9× position size
- **Low liquidity**: +8 confidence requirement, 0.7× position size
- **Weekend**: +10 confidence requirement, 0.5× position size

---

## 🛡️ Risk Management (LAYER 3 & 6)

### Dynamic Position Sizing (LAYER 3)

Position size scales with confidence linearly between min_conf and 100:

```python
base_risk = symbol_cfg.risk_per_trade_pct  # e.g., 0.002 (0.2%)
size_pct = base_risk * ((effective_conf - min_conf) / (100 - min_conf))
risk_pct = min(size_pct, MAX_RISK_PCT)  # Cap at 0.3%
```

**Example Risk Scaling**:
- Confidence 55: 0.10% risk
- Confidence 65: 0.18% risk
- Confidence 75: 0.24% risk
- Confidence 90: 0.30% risk (full size)

### Dynamic Reward-to-Risk Targeting (LAYER 3)

RR targets adjust based on confidence:
- Confidence < 65 → Target RR = 1.8
- Confidence < 80 → Target RR = 1.5
- Confidence ≥ 80 → Target RR = 1.2

**Rationale**: High-confidence patterns can afford slightly tighter targets because winrate compensates.

### Account-Level Protections (LAYER 6)

**Max Concurrent Trades**: 6 (configurable)
**Max Total Exposure**: 2% of equity cap
**Daily Loss Limit**: -3% (stops trading for the day)
**Kill Switch**:
- **Soft Alert**: 15% drawdown → Reduces position size by 50%
- **Hard Stop**: 25% drawdown → Stops all trading

### Risk Filters

- Enforce `MIN_RR` for acceptable asymmetry
- Cap risk size by `MAX_RISK_PCT` (0.3%)
- Respect per-symbol `risk_per_trade_pct` and `max_concurrent_trades`
- Session-based position multipliers

---

## ⏰ Session Awareness

### Session Types

**High Liquidity**:
- US Open (13:30–15:00 UTC)
- London/US Overlap (12:00–16:00 UTC)
- Position multiplier: 1.0
- Confidence multiplier: 1.0

**Medium Liquidity**:
- Asia Open (23:00–02:00 UTC)
- Position multiplier: 0.9
- Confidence multiplier: 0.95

**Low Liquidity**:
- Dead hours (late Asia session, Sunday night)
- Position multiplier: 0.7
- Confidence multiplier: 0.92

**Weekend**:
- Minimal position size
- Position multiplier: 0.5
- Confidence multiplier: 0.9
- Optional weekend avoidance flag

### Session Logic

Trades are filtered and sized by market session. Lower-liquidity sessions use smaller size and higher confirmation requirements.

---

## 🤖 Automation

### Automated Trading System

**`auto_trader.py`**:
- Runs all enabled symbols from `symbols_config.json`
- Automatically trains ML model when enough data is available
- Manages trade submission (paper or live)
- Writes analytics and performance metrics
- Monitors account limits and kill switch

**Features**:
- Multi-symbol parallel processing
- Automatic pattern verification
- Real-time performance tracking
- Auto-retraining of ML models

### Performance Monitor

**`examples/performance_monitor.py`**:
- Displays win rate, P&L, drawdown
- Pattern performance analysis
- Session performance breakdown
- Open-trade summary
- Real-time dashboard

---

## 📈 Monitoring & Analytics

### Metrics Tracked

**Performance Metrics**:
- Win rate (overall and by pattern)
- Profit & Loss (total and daily)
- Profit factor
- Sharpe ratio / Sortino ratio
- Maximum drawdown
- Risk-adjusted returns

**Pattern Analytics**:
- Pattern success rate
- Average RR by pattern
- Confidence distribution
- Pattern frequency

**Session Analytics**:
- Performance by session type
- Best/worst trading sessions
- Session win rate
- Volume analysis

**Risk Metrics**:
- Current drawdown
- Daily loss tracking
- Open trades count
- Exposure percentage

### Running Analytics

```bash
python examples/performance_monitor.py
```

---

## 🧾 Enhanced Logging (Testing Metrics)

The bot now records additional testing-friendly metrics to help you quickly determine if, how, and why a configuration is profitable.

### Where logs are written
- `logs/trades.jsonl`: Raw trade events (opens/generic).
- `logs/trades_enhanced.jsonl`: Closed trades with enriched metrics per trade.
- `logs/pair_metrics.json`: Aggregated per-pair (symbol) performance snapshot.

### Per-trade metrics (in `trades_enhanced.jsonl`)
- **symbol**: Trading pair (e.g., `BTCUSDT`).
- **side**: `BUY` or `SELL`.
- **entry**: Executed entry price.
- **exit**: Executed exit price (TP or SL, including simulated slippage/fees where applicable).
- **size**: Position quantity (base asset units).
- **pnl**: Realized profit or loss in quote currency after fees.
- **r_multiple**: Realized R, where 1R equals the initial risk per unit price.
- **pattern**: The pattern/strategy name that triggered the trade.
- **timeframe**: Timeframe of the setup (if available).
- **closed_time**: UTC timestamp when the trade closed.
- **return_pct**: Percentage return relative to trade notional. Calculated as PnL divided by entry price × size, expressed in percent. In words: “how many percent of the capital deployed in this trade was won or lost.”
- **outcome**: `WIN`, `LOSS`, or `BREAKEVEN` based on the sign of PnL.

Notes on formatting: all numeric values in logs are written to avoid scientific notation, with sensible rounding (prices up to 8 decimals; PnL to 2 decimals; percentages to 2–4 decimals).

### Per-pair aggregates (in `pair_metrics.json`)
For each symbol, we maintain cumulative testing stats to date:
- **trades**: Number of closed trades.
- **wins / losses / breakevens**: Counts of outcomes.
- **total_pnl**: Sum of realized PnL across closed trades.
- **avg_pnl**: Average PnL per trade = total_pnl ÷ trades.
- **win_rate_pct**: Wins ÷ trades × 100.
- **gross_profit**: Sum of positive PnL only.
- **gross_loss**: Absolute sum of negative PnL only.
- **profit_factor**: gross_profit ÷ gross_loss (higher is better; undefined set to 0 or 999 if no losses yet).

All metrics are updated automatically whenever a trade closes, both in simulation and in the shared ledger paths, without changing the trading logic.

## 📁 Project Structure

```
project/
├── auto_trader.py              # Automated multi-symbol system
├── main.py                     # Manual entry point
├── bot_manager.py              # Signal ingestion and submission
├── timeframe_bot.py            # Per-timeframe bot
├── simulate_trading.py         # Simulator + state persistence
├── verify_patterns.py          # Verification & MTF/filters
├── pattern_logic.py            # Pattern detection logic
├── session_manager.py          # Session awareness and adjustments
├── confluence_engine.py        # Confluence-based trading engine
├── core/
│   ├── daily_trading_logic.py  # Daily trading strategies (NEW)
│   ├── risk_manager.py         # Risk management & account limits
│   ├── session_logic.py        # Session-based signals
│   ├── hybrid_logic.py         # Confidence blending
│   ├── trade_executor.py       # Trade execution routing
│   └── indicators.py           # Technical indicators
├── utils/
│   ├── ml_pattern_scorer.py    # ML confidence scoring
│   ├── analytics.py             # Performance analytics
│   ├── config_manager.py        # Configuration management
│   ├── logger.py                # Logging system
│   └── strategy_optimizer.py   # Strategy optimization
├── config/
│   ├── trading.json            # Trading mode settings
│   ├── risk.json               # Risk management config
│   ├── sessions.json           # Session multipliers & adjustments
│   ├── indicators.json         # Indicator settings
│   ├── patterns.json           # Pattern parameters
│   └── modes.json              # Trading mode configs
├── data/                       # Local candle cache
├── patterns_unverified/        # Raw detections
├── patterns_verified/          # Verified signals
├── sim_results/                # Status + SQLite databases
│   ├── trades.db               # Trade history
│   └── global_account.db       # Account ledger
└── symbols_config.json         # Symbols and timeframes
```

---

## 🔧 Troubleshooting

### Common Issues

**"No trades executing"**:
- Check thresholds: Lower `MIN_CONFIDENCE` to 50 or `MIN_RR` to 1.2
- Verify timeframes are enabled in `symbols_config.json`
- Check session filters: Ensure trading during high-liquidity sessions
- Review `sim_results/*/status.json` for bot status

**"Poor performance"**:
- Lower `MIN_CONFIDENCE` (e.g., 50) or `MIN_RR` (e.g., 1.2)
- Review analytics per pattern: Disable underperforming patterns
- Adjust risk per trade: Reduce to 0.1–0.2% for testing
- Check session timing: Ensure trading during optimal sessions

**"High CPU usage"**:
- Reduce number of symbols or timeframes
- Increase candle update interval
- Disable unnecessary indicators

**"Kill switch triggered"**:
- Review recent trades for issues
- Check drawdown: Soft alert at 15%, hard stop at 25%
- Reduce risk per trade or increase confidence threshold
- Review daily loss limit settings

---

## 🎓 Best Practices

### Getting Started

1. **Start in Paper Mode**: Run 7–30 days before going live
   ```bash
   TRADE_LIVE=0 python auto_trader.py
   ```

2. **Conservative Settings Initially**:
   - Risk per trade: 0.1–0.2% (lower than default)
   - MIN_CONFIDENCE: 60–65 (higher than default)
   - MIN_RR: 1.5–2.0 (higher than default)
   - Max concurrent trades: 3–4

3. **Gradual Scaling**:
   - After consistent profits, gradually increase risk
   - Monitor drawdown closely
   - Adjust based on performance analytics

### Optimization

1. **Review Analytics Weekly**:
   - Check pattern performance
   - Identify best/worst sessions
   - Adjust confidence thresholds per pattern
   - Review risk management metrics

2. **Pattern Tuning**:
   - Disable underperforming patterns
   - Focus on high-winrate patterns
   - Adjust confidence weights if needed

3. **Session Optimization**:
   - Identify best trading sessions for your setup
   - Adjust session multipliers if needed
   - Consider weekend avoidance for crypto

4. **ML Model Training**:
   - Ensure at least 50 trades before enabling ML
   - Retrain weekly or after 500 new trades
   - Monitor ML adjustment effectiveness

### Risk Management

1. **Never Risk More Than You Can Afford to Lose**
2. **Use Kill Switch**: Set appropriate drawdown limits
3. **Diversify**: Trade multiple symbols, not just one
4. **Monitor Daily**: Check daily loss limits and exposure
5. **Review Regularly**: Weekly performance reviews

---

## ⚠️ Risk Warnings

**IMPORTANT DISCLAIMERS**:

1. **Educational Purposes Only**: This bot is for educational and research purposes. No guarantees of profitability.

2. **High Risk**: Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results.

3. **Never Risk Funds You Can't Afford to Lose**: Only trade with capital you can afford to lose completely.

4. **No Financial Advice**: This software does not constitute financial advice. Always do your own research.

5. **Market Volatility**: Crypto markets are highly volatile. Extreme price movements can occur at any time.

6. **Technical Risks**: Software bugs, API failures, network issues, or exchange problems can cause losses.

7. **Regulatory Risks**: Cryptocurrency regulations vary by jurisdiction and may change.

8. **Start Small**: Always start with paper trading and small position sizes.

9. **Monitor Actively**: Even with automation, monitor your bot regularly.

10. **Use Kill Switch**: Always enable kill switch and daily loss limits.

**USE AT YOUR OWN RISK. THE DEVELOPERS AND CONTRIBUTORS ARE NOT RESPONSIBLE FOR ANY LOSSES.**

---

## 📞 Support & Documentation

### Getting Help

1. **Setup Issues**: Run `python setup_enhancements.py` for guided setup
2. **Configuration**: Review `config/` directory files
3. **Performance**: Check `examples/performance_monitor.py` for analytics
4. **Troubleshooting**: Review logs in `sim_results/` directories

### Key Configuration Files

- **`.env`**: Environment variables and trading parameters
- **`symbols_config.json`**: Symbol and timeframe configuration
- **`config/risk.json`**: Risk management settings
- **`config/sessions.json`**: Session multipliers and adjustments

### Version Information

**Version**: 2.0  
**Last Updated**: December 2024  
**Features**: 
- 7-Layer Trading Logic System
- 6 Daily Trading Strategies
- ML-Enhanced Confidence Scoring
- Advanced Risk Management
- Session-Aware Trading

---

## 🔧 System Architecture & How It Works

This section provides a comprehensive explanation of how the trading bot system works, from data collection to trade execution, without code examples.

### System Overview

The trading bot is a multi-layered system that continuously monitors cryptocurrency markets, detects trading patterns, evaluates trade opportunities, and executes trades based on a sophisticated decision-making framework. The system operates in real-time, processing market data, applying technical analysis, and making trading decisions autonomously.

### Data Collection & Processing

**Market Data Fetching**: The system connects to CoinEx exchange through the CCXT library to fetch real-time and historical price data. Each trading pair (symbol) is monitored across multiple timeframes simultaneously, from 1-minute charts for scalping opportunities to daily charts for swing trading. The data includes open, high, low, close prices, and trading volume for each time period.

**Data Storage**: Market data is stored in rolling JSON files, one per symbol and timeframe combination. This allows the system to maintain a historical context while keeping memory usage manageable. The system continuously updates these files with new candle data as it becomes available.

**Data Normalization**: Before analysis, the system normalizes symbol formats (converting between exchange formats like BTC-USDT and BTC/USDT), ensures data consistency, and handles missing or corrupted data points gracefully.

### Pattern Detection System

**Technical Indicator Calculation**: The system calculates a comprehensive set of technical indicators for each timeframe, including moving averages (MA50, MA200), exponential moving averages (EMA12, EMA26, EMA21), MACD (Moving Average Convergence Divergence), RSI (Relative Strength Index), ATR (Average True Range), ADX (Average Directional Index), and VWAP (Volume Weighted Average Price). These indicators form the foundation for pattern recognition.

**Pattern Recognition**: The bot scans for over 15 different technical patterns, categorized into trend patterns (Golden Cross, Death Cross), momentum patterns (MACD crossovers, RSI divergences), chart patterns (Bull Flags, Triangles, Head & Shoulders), and candlestick patterns (Engulfing, Hammer). Each pattern has specific detection criteria based on price action, indicator values, and volume characteristics.

**Pattern Verification**: Detected patterns are not immediately traded. Instead, they enter a verification queue where additional checks are performed. This includes multi-timeframe confirmation (ensuring higher timeframes align with the trade direction), volume confirmation (ensuring sufficient trading volume), and trend strength validation (using ADX to confirm market is trending).

**Confluence Engine**: A specialized component identifies support and resistance zones using pivot points, Fibonacci levels, and volume clusters. When price approaches these zones with a matching pattern, the system generates a confluence signal, which carries higher confidence than standalone patterns.

### Confidence Scoring System

**Component-Based Scoring**: Every potential trade receives a confidence score from 0 to 100, calculated using seven weighted components. Pattern structure quality (25% weight) evaluates how clear and well-formed the pattern is. Trend alignment (15%) checks if higher timeframes support the trade direction. Momentum confirmation (15%) verifies that momentum indicators agree with the pattern. Volume confirmation (10%) ensures adequate trading volume. Location context (15%) evaluates whether price is near key levels like support/resistance or moving averages. Risk-reward geometry (10%) assesses the quality of the stop-loss and take-profit placement. Session context (5%) adjusts based on current market session quality.

**Confidence Calculation**: Each component is normalized to a 0-1 scale, multiplied by its weight, and summed to produce a raw confidence score. This score is then multiplied by 100 to get a percentage. The system applies session multipliers (high liquidity sessions get full confidence, while low liquidity sessions reduce confidence) to produce the final confidence value.

**Machine Learning Enhancement**: Once the system has collected enough trade data (typically 50+ trades), a machine learning model begins to refine confidence scores. The ML model analyzes historical pattern performance, considering factors like time of day, volatility levels, and pattern characteristics. However, ML adjustments are bounded (initially ±5%) to prevent over-reliance on the model, ensuring the system remains grounded in technical analysis.

### Trade Decision Logic

**Multi-Layer Filtering**: Before a trade is executed, it must pass through multiple layers of filters. Layer 1 includes global thresholds like minimum confidence (typically 55-65%), minimum risk-reward ratio (1.3-1.8:1), and multi-timeframe confirmation requirements. Layer 2 applies the confidence scoring system. Layer 3 determines dynamic risk-reward targets based on confidence levels. Layer 4 applies ML adjustments if available. Layer 5 adjusts for session quality. Layer 6 checks account-level protections.

**Session-Aware Trading**: The system recognizes that cryptocurrency markets, while trading 24/7, still follow traditional finance session patterns for liquidity and volatility. High liquidity sessions (US market open, London-US overlap) allow full position sizes and standard confidence thresholds. Medium liquidity sessions (Asia open) require slightly higher confidence. Low liquidity sessions (late Asia, pre-market) require significantly higher confidence and reduce position sizes. Weekend sessions are treated with extra caution.

**Risk Management Integration**: Every trade decision considers current account equity, open positions, daily profit/loss, and drawdown levels. The system enforces maximum concurrent trades, maximum total exposure, daily loss limits, and kill-switch mechanisms. If account drawdown exceeds soft limits (10-15%), position sizes are reduced. If hard limits are reached (15-25%), trading is halted entirely.

### Position Sizing & Risk Calculation

**Dynamic Position Sizing**: Position sizes are not fixed but dynamically calculated based on multiple factors. The base risk per trade is typically 0.15-0.25% of account equity. This base risk is then scaled based on confidence level - higher confidence trades can use slightly larger positions (up to 0.3%), while lower confidence trades use smaller positions (down to 0.1%). The system also considers session quality, reducing position sizes during low-liquidity periods.

**Risk Per Unit Calculation**: For each trade, the system calculates the risk per unit by determining the distance between entry price and stop-loss price. This risk distance, combined with the desired risk percentage of account equity, determines the position size. The formula ensures that if the stop-loss is hit, the loss will be exactly the predetermined risk percentage, regardless of position size.

**Volatility Adjustment**: Position sizes are adjusted for market volatility using ATR (Average True Range). In highly volatile markets, stop-losses are wider, so position sizes are reduced to maintain the same dollar risk. In calm markets, stop-losses are tighter, allowing slightly larger positions for the same risk.

### Entry, Stop-Loss, and Take-Profit Placement

**Entry Price Determination**: Entry prices are typically set at the current market price (market orders) or slightly better prices (limit orders) depending on exchange capabilities. The system accounts for slippage (the difference between expected and actual execution price) and trading fees in its calculations.

**Stop-Loss Placement**: Stop-losses are placed using multiple methods. ATR-based stops use a multiple of the Average True Range to set stop distance, ensuring stops adapt to volatility. Structure-based stops are placed just beyond recent swing highs or lows. The system ensures stops are not too tight (which would cause premature exits) or too wide (which would risk too much capital).

**Take-Profit Calculation**: Take-profit levels are calculated using dynamic risk-reward ratios that vary with confidence. Lower confidence trades (60-65%) target 1.8-2.0:1 risk-reward ratios. Medium confidence trades (70-75%) target 2.2-2.5:1 ratios. High confidence trades (80-90%+) target 2.8-3.5:1 ratios. The system recalculates the actual risk-reward ratio after all adjustments to ensure accuracy.

**Trailing Stops**: Once a trade moves into profit by a certain amount (typically 1.2R, meaning 1.2 times the initial risk), the system can activate trailing stops. These stops move with the price in the profitable direction, locking in profits while allowing the trade to continue if the trend persists.

### Trade Execution

**Order Placement**: When a trade passes all filters and checks, the system prepares an order. For paper trading, the order is sent to a simulator that tracks performance without using real money. For live trading, the order is sent to CoinEx exchange through their API. The system ensures orders meet exchange minimum requirements (typically $1 USD minimum order size).

**Order Types**: The system primarily uses bracket orders when supported by the exchange, which simultaneously place entry, stop-loss, and take-profit orders. If bracket orders aren't available, the system places separate orders for each component, ensuring risk management is always in place.

**Execution Monitoring**: After order placement, the system continuously monitors open positions, checking if stop-losses or take-profits have been hit. It also tracks running profit/loss, maximum favorable excursion (how much profit was available), and maximum adverse excursion (how much drawdown occurred before exit).

### Performance Tracking & Analytics

**Trade Logging**: Every trade is logged with comprehensive details including entry/exit prices, position size, profit/loss, risk-reward ratio, pattern that triggered it, confidence level, and session information. This data is stored in JSONL format for easy analysis and machine learning training.

**Performance Metrics**: The system calculates numerous performance metrics including win rate, average win/loss, profit factor (gross profit divided by gross loss), expectancy (average R-multiple), maximum drawdown, Sharpe ratio, and Sortino ratio. These metrics are tracked per symbol, per pattern, and overall.

**Pattern Performance Analysis**: The system tracks which patterns are most profitable, which timeframes work best for each pattern, and how pattern performance varies by market conditions. This information is used to adjust pattern weights and confidence calculations over time.

### Multi-Symbol Management

**Symbol Configuration**: The system can trade multiple cryptocurrency pairs simultaneously, each with its own configuration including risk per trade, confidence thresholds, and enabled timeframes. This allows optimization for different market characteristics (e.g., Bitcoin might use different settings than altcoins).

**Resource Management**: To prevent overwhelming the system or exchange API, the system manages resources carefully. It limits the number of concurrent API calls, uses efficient data structures, and implements rate limiting to respect exchange constraints.

**Global Account Management**: All symbols share a single account balance and equity. The system tracks total exposure across all symbols, ensuring that combined risk never exceeds account-level limits. This prevents over-leveraging when trading multiple pairs simultaneously.

### Error Handling & Resilience

**Network Resilience**: The system handles network interruptions, API timeouts, and exchange maintenance gracefully. It implements retry logic with exponential backoff, caches recent data to continue operating during brief outages, and logs all errors for debugging.

**Data Validation**: Before making trading decisions, the system validates all data for completeness, consistency, and reasonableness. It checks for missing candles, suspicious price movements, and data anomalies that might indicate errors.

**Fail-Safe Mechanisms**: Multiple fail-safe mechanisms protect against catastrophic errors. The kill-switch halts trading if drawdown exceeds limits. Daily loss limits prevent excessive losses in a single day. Maximum position size limits prevent oversized trades. All these mechanisms work independently to provide redundant protection.

### Continuous Learning & Adaptation

**Pattern Lifecycle Management**: The system tracks the performance of each pattern type over time. Patterns that consistently underperform have their weights reduced or are disabled. Patterns that perform well receive higher weights and confidence boosts.

**Market Regime Detection**: The system attempts to detect different market regimes (trending, ranging, volatile, calm) and adjusts strategy parameters accordingly. For example, in ranging markets, it might favor mean-reversion patterns, while in trending markets, it favors momentum patterns.

**Parameter Optimization**: Based on historical performance, the system can suggest parameter optimizations. However, these suggestions are conservative to avoid over-optimization (curve-fitting), which can lead to poor performance on new data.

### Integration Points

**Exchange Integration**: The system integrates with CoinEx exchange through the CCXT library, which provides a unified interface to multiple exchanges. This abstraction allows the system to work with different exchanges with minimal code changes.

**Database Storage**: Trade history, account state, and performance metrics are stored in SQLite databases. This provides persistent storage that survives system restarts and allows for historical analysis.

**Logging System**: Comprehensive logging captures all system events, trade decisions, errors, and performance metrics. Logs are structured (JSON format) for easy parsing and analysis, and are rotated to prevent disk space issues.

**Configuration Management**: All system parameters are stored in JSON configuration files, allowing easy adjustment without code changes. Environment variables provide runtime overrides for sensitive settings like API keys and trading modes.

This architecture creates a robust, adaptive trading system that can operate autonomously while maintaining strict risk controls and continuously improving through performance analysis and machine learning enhancements.

---

## 📝 Changelog

### Version 2.0 (December 2024)

**Major Updates**:
- ✅ Implemented 7-Layer Trading Logic System
- ✅ Added 6 Daily Trading Strategies (Scalping, Momentum, Range, Swing, Arbitrage, Session-Based)
- ✅ Redesigned Confidence Scoring with weighted components
- ✅ Dynamic Position Sizing based on confidence
- ✅ Dynamic Reward-to-Risk targeting
- ✅ Account-Level Protections (kill switch, daily limits, max trades)
- ✅ Relaxed MTF Confirmation to directional-only
- ✅ Session Multipliers and Confidence Adjustments
- ✅ ML Bounded Nudge System (±5%)
- ✅ Updated all configuration files with new defaults

**Parameter Changes**:
- MIN_CONFIDENCE: 60 → 55
- MIN_RR: 1.5 → 1.3
- MAX_RISK_PCT: 0.25% → 0.3% (0.003 decimal)
- APPLY_FILTERS_TO_DAILY: 0 → 1 (enabled)
- Added ML_NUDGE_LIMIT: 0.05
- Added RISK_PER_TRADE_PCT: 0.002
- Added account protection parameters

---

**Happy Trading! 🚀**

*Remember: Start with paper trading, monitor closely, and never risk more than you can afford to lose.*
