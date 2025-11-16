import json
import os
import time
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

from coinEx_getting_data import CoinExDataFetcher
from utils.logger import ComponentLogger
from account_ledger import GlobalAccountLedger


class TradeSimulator:
    def __init__(self, starting_balance: float = 100000.0, mode: str = "spot", symbol: str = "BTCUSDT", ledger: GlobalAccountLedger = None, fee_pct: float = 0.0004, slippage_pct: float = 0.0005, max_drawdown_pct: float = 20.0):
        self.starting_balance = starting_balance
        # Attach shared ledger (optional); mirror local balances for compatibility
        self.ledger = ledger or GlobalAccountLedger(starting_balance=starting_balance)
        self.balance = self.ledger.get_balance()
        self.equity = self.balance
        self.max_equity = starting_balance
        self.min_equity = starting_balance
        self.open_trades: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.mode = mode
        self.symbol = symbol
        # Trading costs
        self.fee_pct = fee_pct  # 0.04% taker fee
        self.slippage_pct = slippage_pct  # 0.05% slippage
        # Risk management
        self.max_drawdown_pct = max_drawdown_pct  # Maximum drawdown percentage
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
        os.makedirs("sim_results", exist_ok=True)
        # SQLite DB for persistent results
        self.db_path = os.path.join("sim_results", "trades.db")
        self._init_db()
        # Load resume state if present
        self._resume_state()
        # separate status file that updates continuously (pair-specific)
        self.status_path = f"sim_results/{self.symbol}/status.json"
        # Ensure per-symbol directory exists
        try:
            os.makedirs(os.path.dirname(self.status_path), exist_ok=True)
        except Exception:
            pass
        # Write an initial snapshot so files exist immediately
        try:
            self._write_results()
        except Exception:
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
                    'id': r[1], 'time': r[2], 'pattern': r[3], 'side': r[4], 'entry': r[5], 'stop': r[6], 'tp': r[7],
                    'rr': r[8], 'risk_pct': r[9], 'size': r[10], 'status': 'OPEN', 'max_drawdown': r[11], 'min_runup': r[12], 'timeframe': r[13]
                })
            # Closed trades don't need to be loaded into memory; we append on close
        finally:
            conn.close()

    def _current_price(self) -> Optional[float]:
        latest = self.fetcher.fetch_latest_closed()
        if latest:
            return float(latest.get("close", 0.0))
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
    
    def _check_drawdown_breach(self):
        """Check if maximum drawdown has been breached and halt trading if so"""
        if self.max_equity > 0:
            current_drawdown_pct = (self.max_equity - self.equity) / self.max_equity * 100.0
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
        
        # FIXED: Check for conflicting trades (opposite direction on same symbol)
        side = setup.get('side')
        for existing_trade in self.open_trades:
            if existing_trade.get('side') != side:
                print(f"⚠️ {self.symbol}: Skipping {side} signal - already have {existing_trade.get('side')} position open")
                return
        entry = float(setup.get('entry'))
        stop = float(setup.get('stop'))
        tp = float(setup.get('tp'))
        risk_pct = float(setup.get('risk_pct', 0.25))
        
        # Apply slippage and latency to entry price
        entry_with_latency = self._simulate_execution_latency(entry, side)
        entry_with_slippage = self._apply_slippage(entry_with_latency, side)
        
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
        
        trade = {
            'id': f"{datetime.utcnow().timestamp():.0f}-{len(self.open_trades)+len(self.closed_trades)}",
            'time': datetime.utcnow().isoformat(),
            'pattern': setup.get('name', 'Unknown'),  # Pattern that triggered the trade
            'side': side,
            'entry': round(entry_with_slippage, 8),  # FIXED: Format prices properly, avoid scientific notation
            'entry_fees': round(entry_fees, 8),
            'stop': round(stop, 8),
            'tp': round(tp, 8),
            'rr': round(actual_rr, 2),  # Use actual calculated RR
            'risk_pct': risk_pct,
            'size': round(size, 8),
            'status': 'OPEN',
            'max_drawdown': 0.0,
            'min_runup': 0.0,
            'init_stop': round(stop, 8),
            'r_per_unit': round(risk_per_unit, 12),
            'session_allowed': setup.get('session_cause'),  # Session that allowed this trade
            'timeframe': setup.get('timeframe')
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

    def _update_open_trades(self):
        price = self._current_price()
        if price is None:
            return
        remaining_open = []
        for t in self.open_trades:
            side = t['side']
            entry = t['entry']
            stop = t['stop']
            tp = t['tp']
            size = t['size']
            # Track drawdown/runup in R
            if side == 'BUY':
                adverse = (entry - min(price, entry)) / max(entry - stop, 1e-12)
                favorable = (max(price, entry) - entry) / max(entry - stop, 1e-12)
                hit_sl = price <= stop
                hit_tp = price >= tp
            else:
                adverse = (max(price, entry) - entry) / max(stop - entry, 1e-12)
                favorable = (entry - min(price, entry)) / max(stop - entry, 1e-12)
                hit_sl = price >= stop
                hit_tp = price <= tp
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

            if hit_sl or hit_tp:
                pnl = 0.0
                r_multiple = 0.0
                exit_price = stop if hit_sl else tp
                
                # Apply slippage to exit price
                exit_price_with_slippage = self._apply_slippage(exit_price, 'SELL' if side == 'BUY' else 'BUY')
                
                # Calculate exit fees
                exit_fees = self._calculate_fees(exit_price_with_slippage, size)
                
                if side == 'BUY':
                    pnl = (exit_price_with_slippage - entry) * size - t.get('entry_fees', 0) - exit_fees
                    r_multiple = -1.0 if hit_sl else t['rr']
                else:
                    pnl = (entry - exit_price_with_slippage) * size - t.get('entry_fees', 0) - exit_fees
                    r_multiple = -1.0 if hit_sl else t['rr']

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
                t['exit'] = round(exit_price, 8)  # FIXED: Format exit price properly
                t['pnl'] = round(pnl, 2)  # FIXED: Format PnL to 2 decimals
                t['r_multiple'] = round(r_multiple, 2)
                t['closed_time'] = datetime.utcnow().isoformat()
                self.closed_trades.append(t)
                try:
                    self.ledger.record_trade_close(t)
                except Exception:
                    pass
                # Enhanced logging for closed trade
                try:
                    ComponentLogger.trading_logger().log_trade_close({
                        'event': 'CLOSE',
                        'symbol': self.symbol,
                        'side': t.get('side'),
                        'entry': t.get('entry'),
                        'exit': t.get('exit'),
                        'size': t.get('size'),
                        'pnl': t.get('pnl'),
                        'r_multiple': t.get('r_multiple'),
                        'pattern': t.get('pattern'),
                        'timeframe': t.get('timeframe'),
                        'closed_time': t.get('closed_time'),
                        'id': t.get('id')
                    })
                except Exception:
                    pass
                # Persist close: remove from open and add to closed - FIXED: Include symbol
                conn = sqlite3.connect(self.db_path)
                try:
                    c = conn.cursor()
                    c.execute("DELETE FROM trades_open WHERE id=? AND symbol=?", (t['id'], self.symbol))
                    c.execute("INSERT OR REPLACE INTO trades_closed (symbol, id, time, pattern, side, entry, stop, tp, rr, risk_pct, size, exit, pnl, r_multiple, closed_time, timeframe) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                              (self.symbol, t['id'], t['time'], t['pattern'], t['side'], t['entry'], t['stop'], t['tp'], t['rr'], t['risk_pct'], t['size'], t['exit'], t['pnl'], t['r_multiple'], t['closed_time'], t.get('timeframe')))
                    conn.commit()
                finally:
                    conn.close()
            else:
                remaining_open.append(t)

        self.open_trades = remaining_open

    def _write_results(self):
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'balance': round(self.balance, 2),
            'equity': round(self.equity, 2),
            'max_equity': round(self.max_equity, 2),
            'min_equity': round(self.min_equity, 2),
            'max_drawdown_pct': round(0.0 if self.max_equity == 0 else (self.max_equity - self.equity) / self.max_equity * 100.0, 2),
            'open_trades': self.open_trades,
            'closed_trades': self.closed_trades
        }
        with open(f"sim_results/{self.symbol}/trade_log.json", "w") as f:
            json.dump(report, f, indent=2)
        # Persist state snapshot to DB
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO state (id, balance, equity, max_equity, min_equity, updated_at) VALUES (1,?,?,?,?,?)",
                      (self.balance, self.equity, self.max_equity, self.min_equity, datetime.utcnow().isoformat()))
            # Update open trades snapshots - FIXED: Include symbol
            for t in self.open_trades:
                c.execute("INSERT OR REPLACE INTO trades_open (symbol, id, time, pattern, side, entry, stop, tp, rr, risk_pct, size, max_drawdown, min_runup, timeframe) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (self.symbol, t['id'], t['time'], t['pattern'], t['side'], t['entry'], t['stop'], t['tp'], t['rr'], t['risk_pct'], t['size'], t.get('max_drawdown', 0.0), t.get('min_runup', 0.0), t.get('timeframe')))
            conn.commit()
        finally:
            conn.close()
        # also write a compact status file
        # Load closed trades from DB
        conn_status = sqlite3.connect(self.db_path)
        try:
            c_status = conn_status.cursor()
            c_status.execute("SELECT id, pattern, side, entry, exit, pnl, r_multiple, closed_time FROM trades_closed WHERE symbol=? ORDER BY closed_time DESC LIMIT 100", (self.symbol,))
            closed_rows = c_status.fetchall()
            closed_trades = [
                {'id': row[0], 'pattern': row[1], 'side': row[2], 'entry': row[3], 'exit': row[4], 'pnl': row[5], 'r_multiple': row[6], 'closed_time': row[7]}
                for row in closed_rows
            ]
        finally:
            conn_status.close()
        
        status = {
            'timestamp': report['timestamp'],
            'balance': self.balance,
            'equity': self.equity,
            'open_trades': [
                {'id': t['id'], 'pattern': t['pattern'], 'side': t['side'], 'entry': t['entry'], 'stop': t['stop'], 'tp': t['tp'], 'size': t['size']}
                for t in self.open_trades
            ],
            'closed_trades': closed_trades
        }
        with open(self.status_path, 'w') as sf:
            json.dump(status, sf, indent=2)

    def step(self):
        self._update_open_trades()
        # Check drawdown on every step
        self._check_drawdown_breach()
        self._write_results()

    def run_forever(self, poll_seconds: float = 5.0):
        while True:
            self.step()
            time.sleep(poll_seconds)


