# Reinforcement Learning & Pattern-Based Trading System

Unified stack for crypto trading that supports:
- Multi-timeframe pattern detection with session-aware risk controls and optional live trading
- A reusable paper/live trade simulator with dynamic stop-loss enforcement
- PPO-based reinforcement learning across multiple symbols and timeframes

The system defaults to **paper trading**. Live trading requires CoinEx API keys and explicit opt-in.

---

## 🚀 Quick Start

1) **Install dependencies**
```bash
pip install -r requirements.txt
```

2) **Run the multi-timeframe pattern bot (paper trading by default)**
```bash
python main.py --symbol BTC-USDT --timeframes 1min 5min 15min 1h 4h --mode spot
```
- Interactive setup runs when no flags are provided (prompts for symbol, timeframes, moderation/trading/operating mode, and live/paper choice) @main.py#77-175
- Live trading only happens when `TRADE_LIVE=1` and CoinEx keys are provided @main.py#131-153

3) **Start RL training**
```bash
python rl_training.py \
  --balance 100000.0 \
  --timesteps 0 \
  --max-trades 100 \
  --model-dir models/rl_models \
  --min-trades 500 \
  --monitor \
  --dashboard
```

4) **Monitor dashboard (training or paper sim)**
```bash
python rl_dashboard.py
```

5) **Fully automated multi-symbol runner**
```bash
python auto_trader.py
```
Uses `symbols_config.json` to launch bots for all enabled symbols and auto-trains the ML scorer when enough trades exist @auto_trader.py#2-218.

---

## 🧩 Components at a Glance

- **Multi-Timeframe Pattern Bot (main.py + BotManager + TimeframeBot)**
  - Spawns a process per timeframe; supports moderation modes (strict/balanced/aggressive), session-aware trading modes (safe/balanced/aggressive), and operating modes (pattern/session/hybrid/auto) @main.py#225-436 @bot_manager.py#24-132.
  - Verified pattern feed submits only vetted setups; RR/confidence filters and session multipliers applied before orders @bot_manager.py#173-238.
  - Paper mode runs the simulator; live mode routes to CoinEx if `TRADE_LIVE=1`.

- **Trade Simulator (simulate_trading.py)**
  - Uses CoinEx public data for price sampling; applies latency and slippage to executions @simulate_trading.py#119-190.
  - Enforces max concurrent trades, kill switch on drawdown, breakeven + trailing stops, and force-close timeout @simulate_trading.py#139-147 @simulate_trading.py#156-337.
  - Dynamic pair-specific minimum stop-loss enforcement on every signal @simulate_trading.py#175-187.
  - Persists state to SQLite (`sim_results/trades.db`) and per-symbol JSON status under `sim_results/<symbol>/` @simulate_trading.py#61-105 @simulate_trading.py#405-456.

- **Session-Aware Risk & Stop Adjustments**
  - Position size multipliers and tighter stops in weaker sessions (Asia/low-liquidity/weekend) @session_manager.py#322-347.
  - Timeframe bot further tightens stops per session when forwarding setups @timeframe_bot.py#451-470.

- **Dynamic Stop-Loss Calculator**
  - Pair configs with ATR, volatility, volume, and price-level scaling; clamps to pair min/max pct and returns enforced SL distance @core/dynamic_sl_calculator.py#22-205.
  - Shared by simulator and RL environment.

- **RL Environment & Trainer**
  - Symbols: BTC/ETH/SOL/BNB/XRP by default; timeframes: 1m, 5m, 15m, 1h, 4h, 6h, 12h, 1d @rl_training.py#66-84.
  - Discrete actions (hold/buy/sell) by default; optional continuous target position sizing with rate limits @rl_trading_env.py#268-303.
  - Dynamic stop-loss selection and enforcement inside the env using primary timeframe data; TP recomputed to preserve RR @rl_trading_env.py#343-447.
  - Pair-specific minimum SL pct calculation with ATR/volatility/volume and price scaling; clamps to per-pair min/max @rl_trading_env.py#873-934.

- **AutoTrader + ML Pattern Scorer**
  - Loads all enabled symbols, starts BotManagers, auto-trains ML scorer when enough trades are logged, and monitors performance @auto_trader.py#2-218.

- **Ledger & Persistence**
  - GlobalAccountLedger keeps shared balances across symbols in paper mode; simulator and bots write SQLite and JSON snapshots @bot_manager.py#97-132 @simulate_trading.py#405-456.

---

## 🛠️ Usage Notes

- **Paper vs Live**
  - Paper is default (`TRADE_LIVE=0`). Live requires `TRADE_LIVE=1` plus `COINEX_API_KEY` and `COINEX_API_SECRET` in `.env` @main.py#131-153.

- **Pattern Bot**
  - CLI flags override prompts. Invalid timeframes are rejected; common set: 1min, 5min, 15min, 30min, 1h, 4h, 6h, 12h, 1d @main.py#248-398.

- **Simulator**
  - Stops and position sizes are enforced per pair; conflicting opposite-direction trades on the same symbol are skipped @simulate_trading.py#156-232.

- **RL Training**
  - Open-ended by default (`--timesteps 0`) until acceptance criteria are met. Model checkpoints and training state are stored under `models/rl_models/`.
  - Dynamic SLs and RR are recomputed per step using current primary timeframe data @rl_trading_env.py#343-447.

---

## 📂 Key Directories

```
data/<base>/<tf>.json        # Rolling candles per timeframe (pattern bots)
patterns_unverified/         # Raw detected patterns
patterns_verified/<symbol>/  # Verified patterns feeding trades
sim_results/<symbol>/        # Status + trade logs (paper)
sim_results/trades.db        # SQLite store for all trades
models/rl_models/            # RL checkpoints, best model, training state
config/                      # Risk/acceptance/session configs
```

---

## 🔐 Safety & Risk Controls

- Kill switch on max drawdown, per-symbol max open trades, breakeven/trailing stops, and 48h force-close timeout @simulate_trading.py#139-339.
- Session-aware sizing and stop tightening; weekend/low-liquidity protection @session_manager.py#322-347 @timeframe_bot.py#451-470.
- Pair-specific minimum stop distances to avoid razor-thin SLs @core/dynamic_sl_calculator.py#22-205.
- RL acceptance criteria and SL clamps ensure the policy cannot pick unrealistically tight stops @rl_trading_env.py#873-934.

---

## ⚙️ Configuration

- `.env`: `TRADE_LIVE`, CoinEx API keys, and optional risk filters (see `MIN_RR`, `MIN_CONFIDENCE`, etc. used in bot filters) @main.py#131-153 @bot_manager.py#215-238.
- `symbols_config.json`: Symbols/timeframes for AutoTrader.
- `config/rl_acceptance_criteria.json`: RL acceptance thresholds, normalization, action space settings.
- `config/` (risk/session): session thresholds, moderation settings, and trading modes.

---

## 📝 Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk. Past performance does not guarantee future results. Use at your own risk.
