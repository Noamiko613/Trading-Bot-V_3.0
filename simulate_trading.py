import json
import os
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from coinEx_getting_data import CoinExDataFetcher
from utils.logger import ComponentLogger
from account_ledger import GlobalAccountLedger
from core.dynamic_sl_calculator import get_sl_calculator


class TradeSimulator:
    def __init__(
        self,
        starting_balance: float = 100000.0,
        mode: str = "spot",
        symbol: str = "BTCUSDT",
        ledger: GlobalAccountLedger = None,
        fee_pct: float = 0.0004,
        slippage_pct: float = 0.0005,
        max_drawdown_pct: float = 20.0,
        kill_switch_enabled: bool = True,
        max_trade_duration_hours: float = None,
        stale_check_interval_sec: int = None,
        is_historical_training: bool = False,
    ):
        self.starting_balance = starting_balance
        # Anchor all persistence to project root to avoid cwd drift
        self.project_root = Path(__file__).resolve().parent
        # Attach shared ledger (optional); mirror local balances for compatibility
        self.ledger = ledger or GlobalAccountLedger(starting_balance=starting_balance)
        self.balance = self.ledger.get_balance()
        self.equity = self.balance
        self.max_equity = starting_balance
        self.min_equity = starting_balance
        self.last_drawdown_pct = 0.0
        self.open_trades: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.mode = mode
        self.symbol = symbol
        self.current_step = 0
        self.is_historical_training = is_historical_training  # Flag for historical training mode
        # Trading costs
        self.fee_pct = fee_pct  # 0.04% taker fee
        self.slippage_pct = slippage_pct  # 0.05% slippage
        # Hard cap on per-trade risk percent (helps PPO avoid oversizing to "win back" losses)
        try:
            self.max_trade_risk_pct = float(os.getenv("MAX_TRADE_RISK_PCT", "0.35"))
        except Exception:
            self.max_trade_risk_pct = 0.35
        # Risk management
        self.max_drawdown_pct = max_drawdown_pct  # Maximum drawdown percentage
        self.kill_switch_enabled = kill_switch_enabled
        self.trading_halted = False  # Kill switch flag
        # Use CoinEx public data for price sampling via ccxt
        from verify_patterns import normalize_symbol_to_ccxt
        symbol_ccxt = normalize_symbol_to_ccxt(symbol)
        self.fetcher = CoinExDataFetcher(symbol=symbol_ccxt, timeframe_internal="1min")
        # Preload some candles (not required for live price, but helpful)
        try:
            self.fetcher.update_initial(limit=50)
        except Exception:
            pass
        os.makedirs(self.project_root / "sim_results", exist_ok=True)
        # SQLite DB for persistent results
        self.db_path = str(self.project_root / "sim_results" / "trades.db")
        self._init_db()
        # Load resume state if present
        self._resume_state()
        # separate status file that updates continuously (pair-specific)
        self.status_path = self.project_root / "sim_results" / self.symbol / "status.json"
        # Ensure per-symbol directory exists
        try:
            os.makedirs(self.status_path.parent, exist_ok=True)
        except Exception:
            pass
        # Write an initial snapshot so files exist immediately
        try:
            self._write_results()
        except Exception:
            pass
        # Health tracking
        self._error_streak = 0
        self._last_error = None
        self._stale_price_cycles = 0  # counts consecutive cycles with missing price
        self._last_price_ts = None
        self._last_price_range: Optional[Dict[str, float]] = None  # cache for TP/SL checks when feed blips
        try:
            env_max_hours = float(os.getenv("MAX_TRADE_DURATION_HOURS", "96"))
        except Exception:
            env_max_hours = 96.0
        self.max_trade_duration_hours = float(max_trade_duration_hours) if max_trade_duration_hours is not None else env_max_hours
        try:
            env_stale_sec = int(os.getenv("STALE_CHECK_INTERVAL_SEC", "90"))
        except Exception:
            env_stale_sec = 90
        self.stale_check_interval_sec = int(stale_check_interval_sec) if stale_check_interval_sec is not None else env_stale_sec
        self._last_stale_check = datetime.utcnow()
        try:
            self._global_open_cap = int(os.getenv('MAX_GLOBAL_OPEN_TRADES','20'))
        except Exception:
            self._global_open_cap = 20

    def _write_results(self):
        """Persist simulator state to SQLite and per-symbol status file."""
        # Persist balances/equity snapshot
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                c = conn.cursor()
                c.execute(
                    "INSERT OR REPLACE INTO state (id, balance, equity, max_equity, min_equity, updated_at) VALUES (1,?,?,?,?,?)",
                    (
                        float(self.balance),
                        float(self.equity),
                        float(self.max_equity),
                        float(self.min_equity),
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            # Avoid breaking the training loop on persistence errors
            pass

        # Write status JSON (used by dashboards/ledger reconciliation)
        try:
            os.makedirs(self.status_path.parent, exist_ok=True)
            status_payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": self.symbol,
                "mode": self.mode,
                "balance": round(float(self.balance), 4),
                "equity": round(float(self.equity), 4),
                "max_equity": round(float(self.max_equity), 4),
                "min_equity": round(float(self.min_equity), 4),
                "last_drawdown_pct": round(float(self.last_drawdown_pct), 4),
                "trading_halted": bool(self.trading_halted),
                "open_trades": self.open_trades,
                # Keep recent closed trades to avoid unbounded growth in status file
                "closed_trades": self.closed_trades[-50:],
                "error": getattr(self, "_last_error", None),
            }
            with open(self.status_path, "w", encoding="utf-8") as f:
                json.dump(status_payload, f, indent=2)
        except Exception:
            # Logging is best-effort; ignore errors to keep loop running
            pass

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK (id=1), balance REAL, equity REAL, max_equity REAL, min_equity REAL, updated_at TEXT)")
            # FIXED: Add symbol column to trades_open and trades_closed tables
            c.execute("CREATE TABLE IF NOT EXISTS trades_open (symbol TEXT, id TEXT PRIMARY KEY, time TEXT, pattern TEXT, side TEXT, entry REAL, stop REAL, tp REAL, rr REAL, risk_pct REAL, size REAL, max_drawdown REAL, min_runup REAL, timeframe TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS trades_closed (symbol TEXT, id TEXT PRIMARY KEY, time TEXT, pattern TEXT, side TEXT, entry REAL, stop REAL, tp REAL, rr REAL, risk_pct REAL, size REAL, exit REAL, pnl REAL, r_multiple REAL, closed_time TEXT, timeframe TEXT)")
            # Backfill schema: add timeframe columns if missing
            try:
                c.execute("ALTER TABLE trades_open ADD COLUMN timeframe TEXT")
            except Exception:
                pass
            try:
                c.execute("ALTER TABLE trades_closed ADD COLUMN timeframe TEXT")
            except Exception:
                pass
            conn.commit()
        finally:
            conn.close()

    def _resume_state(self):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT balance, equity, max_equity, min_equity FROM state WHERE id=1")
            row = c.fetchone()
            if row:
                self.balance, self.equity, self.max_equity, self.min_equity = row
            # Load open trades - FIXED: Only load trades for THIS symbol
            c.execute("SELECT symbol, id, time, pattern, side, entry, stop, tp, rr, risk_pct, size, max_drawdown, min_runup, timeframe FROM trades_open")
            for r in c.fetchall():
                trade_symbol = r[0]
                # FIXED: Only add trades for current symbol
                if trade_symbol != self.symbol:
                    continue
                self.open_trades.append({
                    'symbol': self.symbol,
                    'id': r[1], 'time': r[2], 'pattern': r[3], 'side': r[4], 'entry': r[5], 'stop': r[6], 'tp': r[7],
                    'rr': r[8], 'risk_pct': r[9], 'size': r[10], 'status': 'OPEN', 'max_drawdown': r[11], 'min_runup': r[12], 'timeframe': r[13]
                })
            # Closed trades don't need to be loaded into memory; we append on close
        finally:
            conn.close()

    def _current_price(self) -> Optional[float]:
        """Get current price, preferring close price but with fallback"""
        try:
            latest = self.fetcher.fetch_latest_closed()
        except Exception as e:
            self._last_error = str(e)
            self._error_streak = getattr(self, '_error_streak', 0) + 1
            self._stale_price_cycles = getattr(self, '_stale_price_cycles', 0) + 1
            return None

        if latest:
            try:
                self._stale_price_cycles = 0
                self._last_price_ts = datetime.utcnow()
                return float(latest.get('close', 0.0)) if float(latest.get('close', 0.0)) > 0 else None
            except Exception:
                return None
        self._stale_price_cycles = getattr(self, '_stale_price_cycles', 0) + 1
        return None

    def reset(self, starting_balance: Optional[float] = None, fee_pct: Optional[float] = None, slippage_pct: Optional[float] = None):
        """Reset simulator state for a new episode (used by RL env)."""
        if starting_balance is not None:
            self.starting_balance = starting_balance
            if self.ledger is None:
                self.ledger = GlobalAccountLedger(starting_balance=starting_balance)
            self.ledger.balance = starting_balance
            self.ledger.equity = starting_balance
        # Reset balances/equity
        self.balance = self.starting_balance
        self.equity = self.starting_balance
        self.max_equity = self.starting_balance
        self.min_equity = self.starting_balance
        self.last_drawdown_pct = 0.0
        # Clear trades
        self.open_trades.clear()
        self.closed_trades.clear()
        # Update fees/slippage if provided
        if fee_pct is not None:
            self.fee_pct = fee_pct
        if slippage_pct is not None:
            self.slippage_pct = slippage_pct
        # Reset error tracking
        self._error_streak = 0
        self._last_error = None
        self._stale_price_cycles = 0
        self._last_price_ts = None

    def _refresh_fetcher(self):
        """Recreate the data fetcher when price feed goes stale"""
        try:
            from verify_patterns import normalize_symbol_to_ccxt
            symbol_ccxt = normalize_symbol_to_ccxt(self.symbol)
            # Rebuild fetcher to recover from disconnects/timeouts
            self.fetcher = CoinExDataFetcher(symbol=symbol_ccxt, timeframe_internal="1min")
            try:
                self.fetcher.update_initial(limit=50)
            except Exception:
                pass
            self._error_streak = 0
            self._stale_price_cycles = 0
            self._last_price_ts = datetime.utcnow()
            self._last_price_range = None
            print(f"[SIM] Refreshed price fetcher for {self.symbol} after stale/error state")
        except Exception as e:
            print(f"[SIM] Failed to refresh fetcher for {self.symbol}: {e}")

    def _get_price_range(self) -> Optional[Dict[str, float]]:
        """Get high/low/close of latest candle for TP/SL checking"""
        try:
            latest = self.fetcher.fetch_latest_closed()
        except Exception as e:
            self._last_error = str(e)
            self._error_streak = getattr(self, '_error_streak', 0) + 1
            self._stale_price_cycles = getattr(self, '_stale_price_cycles', 0) + 1
            # Warn but continue; let caller decide
            print(f"[SIM] Price fetch failed ({self._error_streak}): {self._last_error}")
            latest = None
        if latest:
            try:
                self._stale_price_cycles = 0
                self._last_price_ts = datetime.utcnow()
                self._last_price_range = {
                    'high': float(latest.get('high', 0.0)),
                    'low': float(latest.get('low', 0.0)),
                    'close': float(latest.get('close', 0.0))
                }
                return dict(self._last_price_range)
            except Exception:
                pass
        # Fallback: use current close only
        fallback = self._current_price()
        if fallback is not None and fallback > 0:
            self._stale_price_cycles = 0
            self._last_price_ts = datetime.utcnow()
            self._last_price_range = {'high': fallback, 'low': fallback, 'close': fallback}
            return dict(self._last_price_range)
        # Final fallback: reuse the last known price range if it is recent to avoid skipping TP/SL checks
        if self._last_price_range and self._last_price_ts:
            age_sec = (datetime.utcnow() - self._last_price_ts).total_seconds()
            if age_sec <= max(2 * self.stale_check_interval_sec, 180):
                print(f"[SIM] Using cached price range from {age_sec:.0f}s ago for {self.symbol} to continue TP/SL checks")
                return dict(self._last_price_range)
        # Log a single-line warning so we know why trades are not closing
        if getattr(self, "_error_streak", 0) == 0:
            self._error_streak = 1
        self._stale_price_cycles = getattr(self, '_stale_price_cycles', 0) + 1
        print(f"[SIM] Warning: No price data available for {self.symbol}; skipping trade updates")
        return None
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to entry/exit price"""
        if side == 'BUY':
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)
    
    def _simulate_execution_latency(self, signal_price: float, side: str) -> float:
        """Simulate order execution latency by using next candle's price"""
        # In real trading, there's latency between signal generation and execution
        # This simulates that by using a slightly different price
        import random
        
        # Simulate 0.1-0.5% price movement due to latency
        latency_factor = random.uniform(0.001, 0.005)
        
        if side == 'BUY':
            # Price might move up during latency
            return signal_price * (1 + latency_factor)
        else:
            # Price might move down during latency
            return signal_price * (1 - latency_factor)
    
    def _calculate_fees(self, price: float, size: float) -> float:
        """Calculate trading fees for a trade"""
        return price * size * self.fee_pct
    
    def _min_notional_for_price(self, price: float) -> float:
        """
        Dynamic minimum notional risk based on pair price so micro-trades are avoided.
        Higher-priced assets require a higher dollar floor, lower-priced assets use a smaller floor.
        """
        if price >= 20000:
            return 80.0
        if price >= 5000:
            return 40.0
        if price >= 1000:
            return 20.0
        if price >= 200:
            return 10.0
        if price >= 50:
            return 5.0
        return 2.0
    
    def _check_drawdown_breach(self):
        """Check if maximum drawdown has been breached and halt trading if so"""
        if self.max_equity <= 0:
            return
        current_drawdown_pct = (self.max_equity - self.equity) / self.max_equity * 100.0
        self.last_drawdown_pct = current_drawdown_pct
        # Skip hard kill switch if disabled (e.g., RL training mode), but keep tracking drawdown
        if not self.kill_switch_enabled:
            return
        # Prevent log spam once the kill switch is already tripped
        if self.trading_halted:
            return
        if current_drawdown_pct >= self.max_drawdown_pct:
            self.trading_halted = True
            print(f"🚨 KILL SWITCH ACTIVATED: Drawdown {current_drawdown_pct:.2f}% exceeds maximum {self.max_drawdown_pct}%")
            print(f"🛑 Trading halted. Current equity: {self.equity:.2f}, Max equity: {self.max_equity:.2f}")

    def _position_size(self, entry: float, stop: float, risk_pct: float) -> float:
        risk_amount = self.balance * (risk_pct / 100.0)
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return 0.0
        size = risk_amount / risk_per_unit
        return max(size, 0.0)

    def submit_signal(self, setup: Dict):
        # Check kill switch
        if self.trading_halted:
            return
        # Respect max concurrent trades (aligned with config risk_settings: 8)
        if len(self.open_trades) >= 8:
            return
        # Global concurrency cap
        try:
            open_global = int(self.ledger.count_open_trades())
            cap = int(os.getenv('MAX_GLOBAL_OPEN_TRADES','20'))
        except Exception:
            open_global, cap = 0, 20
        if open_global >= cap:
            return

        # CRITICAL FIX: Only allow ONE trade per symbol at a time (same or opposite direction)
        # This prevents the model from opening multiple positions on the same pair
        side = setup.get('side')
        for existing_trade in self.open_trades:
            # Block if there's ANY open trade on this symbol (same or opposite direction)
            if existing_trade.get('symbol') == self.symbol and existing_trade.get('status') == 'OPEN':
                existing_side = existing_trade.get('side', 'UNKNOWN')
                if existing_side == side:
                    print(f"⚠️ {self.symbol}: Skipping {side} signal - already have {existing_side} position open (duplicate blocked)")
                else:
                    print(f"⚠️ {self.symbol}: Skipping {side} signal - already have {existing_side} position open (opposite direction blocked)")
                return
        entry = float(setup.get('entry'))
        stop = float(setup.get('stop'))
        tp = float(setup.get('tp'))
        risk_pct = float(setup.get('risk_pct', 0.25))
        
        # FIXED: Enforce dynamic minimum SL per pair
        try:
            sl_calc = get_sl_calculator()
            stop = sl_calc.enforce_minimum_sl(
                symbol=self.symbol,
                entry_price=entry,
                stop_loss=stop,
                side=side,
                df=None  # Could pass market data if available
            )
        except Exception:
            pass  # Continue with original SL if dynamic calc fails
        
        # Apply slippage and latency to entry price
        entry_with_latency = self._simulate_execution_latency(entry, side)
        entry_with_slippage = self._apply_slippage(entry_with_latency, side)
        
        # Enforce a dynamic minimum notional risk based on pair price
        min_notional = self._min_notional_for_price(entry_with_slippage)
        risk_amount = self.balance * (risk_pct / 100.0)
        if risk_amount < min_notional and self.balance > 0:
            risk_pct = (min_notional / self.balance) * 100.0
            risk_amount = min_notional
        
        size = self._position_size(entry_with_slippage, stop, risk_pct)
        if size <= 0:
            return
        
        # Calculate entry fees
        entry_fees = self._calculate_fees(entry_with_slippage, size)
        
        risk_per_unit = abs(entry_with_slippage - stop)
        
        # Calculate actual RR from final values to ensure accuracy
        actual_risk = abs(entry_with_slippage - stop)
        if side == 'BUY':
            actual_reward = abs(tp - entry_with_slippage)
        else:
            actual_reward = abs(entry_with_slippage - tp)
        actual_rr = (actual_reward / actual_risk) if actual_risk > 0 else float(setup.get('rr', 2.0))
        rr_text = f"{actual_rr:.2f}|1"
        
        # Mark pattern for historical training trades
        pattern = setup.get('name', setup.get('pattern', 'Unknown'))
        if self.is_historical_training:
            pattern = f"HISTORICAL_TRAINING_{pattern}"
        
        trade = {
            'id': f"{datetime.utcnow().timestamp():.0f}-{len(self.open_trades)+len(self.closed_trades)}",
            'time': datetime.utcnow().isoformat(),
            'pattern': pattern,  # Pattern that triggered the trade (marked if historical)
            'side': side,
            'symbol': self.symbol,  # ensure symbol is carried through lifecycle
            'entry': round(entry_with_slippage, 8),  # FIXED: Format prices properly, avoid scientific notation
            'entry_fees': round(entry_fees, 8),
            'stop': round(stop, 8),
            'tp': round(tp, 8),
            'rr': round(actual_rr, 2),  # Use actual calculated RR
            'rr_text': rr_text,
            'risk_pct': risk_pct,
            'size': round(size, 8),
            'status': 'OPEN',
            'max_drawdown': 0.0,
            'min_runup': 0.0,
            'init_stop': round(stop, 8),
            'r_per_unit': round(risk_per_unit, 12),
            'session_allowed': setup.get('session_cause'),  # Session that allowed this trade
            'timeframe': setup.get('timeframe'),
            'is_historical_training': self.is_historical_training  # Flag for filtering
        }
        self.open_trades.append(trade)
        # Log trade open (non-intrusive)
        try:
            ComponentLogger.trading_logger().log_trade({
                'event': 'OPEN',
                'symbol': self.symbol,
                'side': side,
                'entry': trade['entry'],
                'stop': trade['stop'],
                'tp': trade['tp'],
                'size': trade['size'],
                'rr': trade['rr'],
                'risk_pct': trade['risk_pct'],
                'pattern': trade.get('pattern'),
                'timeframe': trade.get('timeframe'),
                'id': trade.get('id')
            })
        except Exception:
            pass
        # Record to global ledger (do not alter sizing/execution)
        try:
            self.ledger.record_trade_open(self.symbol, trade)
        except Exception:
            pass
        # Persist open trade to DB - FIXED: Include symbol
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO trades_open (symbol, id, time, pattern, side, entry, stop, tp, rr, risk_pct, size, max_drawdown, min_runup, timeframe) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (self.symbol, trade['id'], trade['time'], trade['pattern'], trade['side'], trade['entry'], trade['stop'], trade['tp'], trade['rr'], trade['risk_pct'], trade['size'], trade['max_drawdown'], trade['min_runup'], trade.get('timeframe')))
            conn.commit()
        finally:
            conn.close()

    def close_trade(self, trade_id: str, reason: str = "MANUAL") -> bool:
        """
        Close an open trade immediately at the current market price.
        Returns True if the trade was closed, False otherwise.
        """
        price = self._current_price()
        if price is None:
            return False

        for idx, t in enumerate(list(self.open_trades)):
            if t.get('id') != trade_id:
                continue

            side = t.get('side')
            entry = t.get('entry', price)
            stop = t.get('stop', entry)
            tp = t.get('tp', entry)
            size = t.get('size', 0.0)

            # Use slippage-aware exit to mirror execution path
            exit_price = price
            exit_price_with_slippage = self._apply_slippage(exit_price, 'SELL' if side == 'BUY' else 'BUY')
            exit_fees = self._calculate_fees(exit_price_with_slippage, size)

            if side == 'BUY':
                pnl = (exit_price_with_slippage - entry) * size - t.get('entry_fees', 0) - exit_fees
            else:
                pnl = (entry - exit_price_with_slippage) * size - t.get('entry_fees', 0) - exit_fees

            risk_per_unit = float(t.get('r_per_unit') or abs(entry - stop) or 1e-12)
            r_multiple = (exit_price_with_slippage - entry) / risk_per_unit if side == 'BUY' else (entry - exit_price_with_slippage) / risk_per_unit

            # Update balances
            try:
                self.balance = self.ledger.update_balance(pnl)
            except Exception:
                self.balance += pnl
            self.equity = self.balance
            self.max_equity = max(self.max_equity, self.equity)
            self.min_equity = min(self.min_equity, self.equity)
            self._check_drawdown_breach()

            closed_trade = t.copy()
            closed_trade.update({
                'status': 'CLOSED',
                'exit': round(exit_price_with_slippage, 8),
                'pnl': round(pnl, 2),
                'r_multiple': round(r_multiple, 2),
                'closed_time': datetime.utcnow().isoformat(),
                'close_reason': reason,
            })
            self.closed_trades.append(closed_trade)
            try:
                self.ledger.record_trade_close(closed_trade)
            except Exception:
                pass

            # Persist closure to DB and remove from open trades
            conn = sqlite3.connect(self.db_path)
            try:
                c = conn.cursor()
                c.execute("DELETE FROM trades_open WHERE id=? AND symbol=?", (t.get('id'), self.symbol))
                c.execute(
                    "INSERT OR REPLACE INTO trades_closed (symbol, id, time, pattern, side, entry, stop, tp, rr, risk_pct, size, exit, pnl, r_multiple, closed_time, timeframe) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self.symbol, t.get('id'), t.get('time'), t.get('pattern'), t.get('side'), t.get('entry'), t.get('stop'), t.get('tp'),
                     t.get('rr'), t.get('risk_pct'), t.get('size'), closed_trade.get('exit'), closed_trade.get('pnl'), closed_trade.get('r_multiple'),
                     closed_trade.get('closed_time'), t.get('timeframe'))
                )
                conn.commit()
            finally:
                conn.close()
            try:
                self.open_trades.pop(idx)
            except Exception:
                self.open_trades = [ot for ot in self.open_trades if ot.get('id') != trade_id]
            return True

        return False

    def _check_and_close_stale_trades(self):
        """Force-close trades older than the configured maximum duration."""
        if not self.max_trade_duration_hours or self.max_trade_duration_hours <= 0:
            return

        max_age = timedelta(hours=self.max_trade_duration_hours)
        now = datetime.utcnow()
        stale_ids = []

        for t in list(self.open_trades):
            if t.get("symbol") != self.symbol or t.get("status") != "OPEN":
                continue

            try:
                trade_time_str = t.get("time", "")
                if not trade_time_str:
                    continue
                trade_time = datetime.fromisoformat(trade_time_str.replace("Z", "+00:00"))
                if trade_time.tzinfo:
                    trade_time = trade_time.replace(tzinfo=None)
                age = now - trade_time
                if age > max_age:
                    stale_ids.append(t.get("id"))
                    print(f"[SIM] ⏰ Stale trade detected | {self.symbol} | ID: {t.get('id', '')[:8]}... | Age: {age.total_seconds()/3600:.1f}h > {self.max_trade_duration_hours}h")
            except Exception:
                continue

        for trade_id in stale_ids:
            try:
                closed = self.close_trade(trade_id, reason=f"STALE_{self.max_trade_duration_hours}h")
                if closed:
                    print(f"[SIM] 🔴 Force-closed stale trade {trade_id[:8] if trade_id else 'unknown'} for {self.symbol}")
                else:
                    print(f"[SIM] ⚠️ Failed to close stale trade {trade_id[:8] if trade_id else 'unknown'} (possibly already closed)")
            except Exception as e:
                print(f"[SIM] ⚠️ Error closing stale trade {trade_id[:8] if trade_id else 'unknown'}: {e}")

    def _update_open_trades(self):
        import json
        import os
        import time as time_module
        log_path = r"c:\Users\Mini Echo09\Desktop\Trading-Bot-V_2.0-feature-enhanced-logging-metrics\.cursor\debug.log"
        # #region agent log
        try:
            open_count_before = len(self.open_trades)
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_update_entry","timestamp":int(time_module.time()*1000),"location":"simulate_trading.py:632","message":"_update_open_trades() entry","data":{"symbol":self.symbol,"open_trades_count":open_count_before},"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + "\n")
        except: pass
        # #endregion
        # Get price range (high/low/close) for more accurate TP/SL detection
        price_range = self._get_price_range()
        if price_range is None:
            # Nothing to do without prices; keep a small heartbeat
            if self.current_step % 20 == 0:
                print(f"[SIM] No price_range; open={len(self.open_trades)} closed={len(self.closed_trades)}")
            # Fallback to close price only
            price = self._current_price()
            if price is None or price <= 0:
                # Try to get price from fetcher directly if _current_price() fails
                try:
                    latest = self.fetcher.fetch_latest_closed()
                    if latest:
                        price = float(latest.get("close", 0.0))
                    if price is None or price <= 0:
                        # If still no price, skip this update
                        return
                except Exception as e:
                    # If we can't get price at all, skip this update
                    return
            else:
                price_range = {'high': price, 'low': price, 'close': price}
        
        price = price_range['close']
        high = price_range['high']
        low = price_range['low']
        # #region agent log
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_price_data","timestamp":int(time_module.time()*1000),"location":"simulate_trading.py:656","message":"Price data retrieved","data":{"symbol":self.symbol,"price":price,"high":high,"low":low,"open_trades_count":len(self.open_trades)},"sessionId":"debug-session","runId":"run1","hypothesisId":"C,E"}) + "\n")
        except: pass
        # #endregion
        remaining_open = []
        trades_checked = 0
        trades_hit_tp = 0
        trades_hit_sl = 0
        for t in self.open_trades:
            # CRITICAL: Only process trades for THIS symbol
            if t.get('symbol') != self.symbol:
                remaining_open.append(t)  # Keep trades from other symbols
                continue
            
            side = t['side']
            entry = t.get('entry', 0)
            stop = t.get('stop', 0)
            tp = t.get('tp', 0)
            size = t.get('size', 0)
            
            # Validate trade data
            if entry <= 0 or stop <= 0 or tp <= 0 or size <= 0:
                print(f"[WARNING] {self.symbol}: Invalid trade data - Entry: {entry}, Stop: {stop}, TP: {tp}, Size: {size}")
                # Mark as invalid and skip
                t['status'] = 'INVALID'
                continue
            
            # Debug logging for stuck trades and TP/SL proximity
            if t.get('status') == 'OPEN':
                try:
                    from datetime import datetime, timedelta
                    trade_time = datetime.fromisoformat(t.get('time', datetime.utcnow().isoformat()))
                    age_hours = (datetime.utcnow() - trade_time).total_seconds() / 3600
                    
                    # Check proximity to TP/SL
                    if side == 'BUY':
                        dist_to_sl = ((price - stop) / stop * 100) if stop > 0 else 0
                        dist_to_tp = ((tp - price) / price * 100) if price > 0 else 0
                    else:
                        dist_to_sl = ((stop - price) / price * 100) if price > 0 else 0
                        dist_to_tp = ((price - tp) / tp * 100) if tp > 0 else 0
                    
                    # Log if trade has been open for a while or is close to TP/SL
                    if age_hours > 1 or abs(dist_to_sl) < 0.5 or abs(dist_to_tp) < 0.5:
                        print(f"[TRADE_CHECK] {self.symbol}: Trade {t.get('id', 'unknown')[:8]}... | "
                              f"Age: {age_hours:.1f}h | Entry: ${entry:.2f} | Stop: ${stop:.2f} | TP: ${tp:.2f} | "
                              f"Current: ${price:.2f} | Side: {side} | "
                              f"Dist to SL: {dist_to_sl:.2f}% | Dist to TP: {dist_to_tp:.2f}%")
                except Exception as e:
                    pass
            
            # Track drawdown/runup in R
            # Use high/low to detect if TP/SL was hit during the candle period
            if side == 'BUY':
                adverse = (entry - min(low, entry)) / max(entry - stop, 1e-12)
                favorable = (max(high, entry) - entry) / max(entry - stop, 1e-12)
                # For BUY: SL hit if low went below stop, TP hit if high went above TP
                # Use close price as fallback if high/low don't hit
                hit_sl = low <= stop or price <= stop
                hit_tp = high >= tp or price >= tp
                # Also check if very close (within 0.1%) - force close to prevent stuck trades
                if not hit_sl and not hit_tp:
                    sl_distance_pct = abs((price - stop) / stop * 100) if stop > 0 else 100
                    tp_distance_pct = abs((tp - price) / price * 100) if price > 0 else 100
                    if sl_distance_pct < 0.1:
                        hit_sl = True
                        print(f"[FORCE_CLOSE_SL] {self.symbol}: Price ${price:.2f} very close to SL ${stop:.2f} (distance: {sl_distance_pct:.3f}%)")
                    elif tp_distance_pct < 0.1:
                        hit_tp = True
                        print(f"[FORCE_CLOSE_TP] {self.symbol}: Price ${price:.2f} very close to TP ${tp:.2f} (distance: {tp_distance_pct:.3f}%)")
            else:
                adverse = (max(high, entry) - entry) / max(stop - entry, 1e-12)
                favorable = (entry - min(low, entry)) / max(stop - entry, 1e-12)
                # For SELL: SL hit if high went above stop, TP hit if low went below TP
                # Use close price as fallback if high/low don't hit
                hit_sl = high >= stop or price >= stop
                hit_tp = low <= tp or price <= tp
                # Also check if very close (within 0.1%) - force close to prevent stuck trades
                if not hit_sl and not hit_tp:
                    sl_distance_pct = abs((stop - price) / price * 100) if price > 0 else 100
                    tp_distance_pct = abs((price - tp) / tp * 100) if tp > 0 else 100
                    if sl_distance_pct < 0.1:
                        hit_sl = True
                        print(f"[FORCE_CLOSE_SL] {self.symbol}: Price ${price:.2f} very close to SL ${stop:.2f} (distance: {sl_distance_pct:.3f}%)")
                    elif tp_distance_pct < 0.1:
                        hit_tp = True
                        print(f"[FORCE_CLOSE_TP] {self.symbol}: Price ${price:.2f} very close to TP ${tp:.2f} (distance: {tp_distance_pct:.3f}%)")
            
            # Log when TP/SL is hit
            if hit_sl or hit_tp:
                reason = "SL" if hit_sl else "TP"
                trades_hit_sl += 1 if hit_sl else 0
                trades_hit_tp += 1 if hit_tp else 0
                # #region agent log
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_tp_sl_hit","timestamp":int(time_module.time()*1000),"location":"simulate_trading.py:742","message":"TP/SL hit detected","data":{"symbol":self.symbol,"trade_id":t.get('id','unknown')[:8],"reason":reason,"entry":entry,"stop":stop,"tp":tp,"price":price,"side":side,"hit_sl":hit_sl,"hit_tp":hit_tp},"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + "\n")
                except: pass
                # #endregion
                print(f"[TRADE_CLOSE] {self.symbol}: {reason} HIT! Trade {t.get('id', 'unknown')[:8]}... | "
                      f"Entry: ${entry:.2f} | Exit: ${stop if hit_sl else tp:.2f} | Current: ${price:.2f} | Side: {side}")
            trades_checked += 1
            t['max_drawdown'] = round(max(t.get('max_drawdown', 0.0), adverse), 2)  # FIXED: Format properly
            t['min_runup'] = round(max(t.get('min_runup', 0.0), favorable), 2)  # FIXED: Format properly

            # Breakeven and trailing stop logic
            try:
                r_per_unit = float(t.get('r_per_unit', abs(entry - stop)))
                if r_per_unit > 0:
                    if side == 'BUY':
                        runup_r = (max(price, entry) - entry) / r_per_unit
                        # Move stop to breakeven at >= 1R
                        if runup_r >= 1.0 and t['stop'] < entry:
                            t['stop'] = round(entry, 8)
                        # Trail to +0.5R at >= 1.5R
                        if runup_r >= 1.5:
                            target_stop = entry + 0.5 * r_per_unit
                            if t['stop'] < target_stop:
                                t['stop'] = round(target_stop, 8)
                    else:
                        runup_r = (entry - min(price, entry)) / r_per_unit
                        if runup_r >= 1.0 and t['stop'] > entry:
                            t['stop'] = round(entry, 8)
                        if runup_r >= 1.5:
                            target_stop = entry - 0.5 * r_per_unit
                            if t['stop'] > target_stop:
                                t['stop'] = round(target_stop, 8)
            except Exception:
                pass

            # Force close trades that have been open too long (configurable)
            force_close = False
            try:
                trade_time = datetime.fromisoformat(t.get('time', datetime.utcnow().isoformat()))
                age_hours = (datetime.utcnow() - trade_time).total_seconds() / 3600
                if self.max_trade_duration_hours and age_hours > self.max_trade_duration_hours:
                    force_close = True
                    print(f"[AUTO-CLOSE] {self.symbol}: Force closing trade {t.get('id', 'unknown')} "
                          f"after {age_hours:.1f} hours (timeout>{self.max_trade_duration_hours}h)")
            except Exception:
                pass
            
            if hit_sl or hit_tp or force_close:
                pnl = 0.0
                r_multiple = 0.0
                if force_close:
                    exit_price = price  # Close at current price
                    close_reason = "FORCE_CLOSE_TIMEOUT"
                    hit_sl = False
                    hit_tp = False
                else:
                    exit_price = stop if hit_sl else tp
                    close_reason = "SL" if hit_sl else "TP"
                
                # Apply slippage to exit price
                exit_price_with_slippage = self._apply_slippage(exit_price, 'SELL' if side == 'BUY' else 'BUY')
                
                # Calculate exit fees
                exit_fees = self._calculate_fees(exit_price_with_slippage, size)
                
                if side == 'BUY':
                    pnl = (exit_price_with_slippage - entry) * size - t.get('entry_fees', 0) - exit_fees
                    r_multiple = -1.0 if hit_sl else t.get('rr', 0)
                else:
                    pnl = (entry - exit_price_with_slippage) * size - t.get('entry_fees', 0) - exit_fees
                    r_multiple = -1.0 if hit_sl else t.get('rr', 0)

                # Update local and global balances with realized pnl (keep original pnl calc)
                try:
                    self.balance = self.ledger.update_balance(pnl)
                except Exception:
                    self.balance += pnl
                self.equity = self.balance
                self.max_equity = max(self.max_equity, self.equity)
                self.min_equity = min(self.min_equity, self.equity)
                
                # Check for max drawdown breach
                self._check_drawdown_breach()

                t['status'] = 'CLOSED'
                t['exit'] = round(exit_price_with_slippage, 8)  # Use slippage-adjusted exit price
                t['pnl'] = round(pnl, 2)
                t['r_multiple'] = round(r_multiple, 2)
                t['closed_time'] = datetime.utcnow().isoformat()
                t['close_reason'] = close_reason
                
                # Log successful closure
                print(f"[TRADE_CLOSED] {self.symbol}: Trade {t.get('id', 'unknown')[:8]}... closed via {close_reason} | "
                      f"Entry: ${entry:.2f} | Exit: ${t['exit']:.2f} | PnL: ${pnl:.2f} | R: {r_multiple:.2f}")
                
                self.closed_trades.append(t)
                try:
                    # ensure symbol present for downstream logging/ledger
                    t['symbol'] = self.symbol
                    self.ledger.record_trade_close(t)
                except Exception as e:
                    print(f"[ERROR] Failed to record trade close to ledger: {e}")
                
                # Persist close: remove from open and add to closed - FIXED: Include symbol
                conn = sqlite3.connect(self.db_path)
                try:
                    c = conn.cursor()
                    c.execute("DELETE FROM trades_open WHERE id=? AND symbol=?", (t['id'], self.symbol))
                    c.execute("INSERT OR REPLACE INTO trades_closed (symbol, id, time, pattern, side, entry, stop, tp, rr, risk_pct, size, exit, pnl, r_multiple, closed_time, timeframe) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                              (self.symbol, t['id'], t['time'], t['pattern'], t['side'], t['entry'], t['stop'], t['tp'], t['rr'], t['risk_pct'], t['size'], t['exit'], t['pnl'], t['r_multiple'], t['closed_time'], t.get('timeframe')))
                    conn.commit()
                except Exception as e:
                    print(f"[ERROR] Failed to persist trade close to DB: {e}")
                finally:
                    conn.close()
            else:
                remaining_open.append(t)

        self.open_trades = remaining_open
        # #region agent log
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_update_exit","timestamp":int(time_module.time()*1000),"location":"simulate_trading.py:857","message":"_update_open_trades() exit","data":{"symbol":self.symbol,"trades_checked":trades_checked,"trades_hit_tp":trades_hit_tp,"trades_hit_sl":trades_hit_sl,"open_trades_before":open_count_before,"open_trades_after":len(remaining_open),"trades_closed":open_count_before-len(remaining_open)},"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + "\n")
        except: pass
        # #endregion

    def step(self):
        import json
        import os
        import time as time_module
        log_path = r"c:\Users\Mini Echo09\Desktop\Trading-Bot-V_2.0-feature-enhanced-logging-metrics\.cursor\debug.log"
        # #region agent log
        try:
            open_count_before = len(self.open_trades)
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_step_entry","timestamp":int(time_module.time()*1000),"location":"simulate_trading.py:859","message":"step() entry","data":{"symbol":self.symbol,"current_step":self.current_step+1,"open_trades_before":open_count_before},"sessionId":"debug-session","runId":"run1","hypothesisId":"B"}) + "\n")
        except: pass
        # #endregion
        self.current_step += 1
        # Refresh price data before updating trades
        try:
            self.fetcher.update_initial(limit=10)  # Refresh recent candles
            # #region agent log
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_price_refresh_ok","timestamp":int(time_module.time()*1000),"location":"simulate_trading.py:863","message":"Price refresh successful","data":{"symbol":self.symbol,"current_step":self.current_step},"sessionId":"debug-session","runId":"run1","hypothesisId":"E"}) + "\n")
            except: pass
            # #endregion
        except Exception as e:
            self._error_streak = getattr(self, '_error_streak', 0) + 1
            # #region agent log
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_price_refresh_error","timestamp":int(time_module.time()*1000),"location":"simulate_trading.py:864","message":"Price refresh error","data":{"symbol":self.symbol,"error":str(e),"error_streak":self._error_streak,"current_step":self.current_step},"sessionId":"debug-session","runId":"run1","hypothesisId":"E"}) + "\n")
            except: pass
            # #endregion
        # If price feed appears stale or erroring, recreate fetcher
        if getattr(self, '_stale_price_cycles', 0) >= 6 or getattr(self, '_error_streak', 0) >= 5:
            self._refresh_fetcher()
        
        if (datetime.utcnow() - getattr(self, "_last_stale_check", datetime.min)).total_seconds() >= self.stale_check_interval_sec:
            self._check_and_close_stale_trades()
            self._last_stale_check = datetime.utcnow()

        self._update_open_trades()
        # #region agent log
        try:
            open_count_after = len(self.open_trades)
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_step_exit","timestamp":int(time_module.time()*1000),"location":"simulate_trading.py:874","message":"step() exit","data":{"symbol":self.symbol,"current_step":self.current_step,"open_trades_before":open_count_before,"open_trades_after":open_count_after,"trades_closed":open_count_before-open_count_after},"sessionId":"debug-session","runId":"run1","hypothesisId":"B"}) + "\n")
        except: pass
        # #endregion
        # Check drawdown on every step
        self._check_drawdown_breach()
        self._write_results()

    def run_forever(self, poll_seconds: float = 5.0):
        backoff = 0.0
        while True:
            try:
                self.step()
                backoff = 0.0
                self._error_streak = 0
            except Exception as e:
                try:
                    ComponentLogger.simulator_logger().error('simulator_loop_error', error=str(e))
                except Exception:
                    pass
                self._last_error = str(e)
                self._error_streak = getattr(self, '_error_streak', 0) + 1
                backoff = 0.5 if backoff <= 0 else min(backoff * 2.0, 5.0)
                time.sleep(backoff)
                continue
            time.sleep(poll_seconds)


