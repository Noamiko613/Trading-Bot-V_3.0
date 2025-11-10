import os
import sqlite3
import json
from datetime import datetime
from typing import Dict
from utils.logger import ComponentLogger

# Singleton instance
_ledger_instance = None

class GlobalAccountLedger:
    """
    SQLite-backed global ledger shared by all simulators across symbols.
    Tracks a single account balance/equity and all trades for reporting.
    Uses singleton pattern to ensure all symbols share the same balance.
    """

    def __new__(cls, db_path: str = None, starting_balance: float = 100000.0):
        global _ledger_instance
        if _ledger_instance is None:
            _ledger_instance = super(GlobalAccountLedger, cls).__new__(cls)
            _ledger_instance._initialized = False
        return _ledger_instance

    def __init__(self, db_path: str = None, starting_balance: float = 100000.0):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.db_path = db_path or os.path.join('sim_results', 'global_account.db')
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db(starting_balance)
        self._initialized = True

    def _init_db(self, starting_balance: float):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY CHECK (id=1), balance REAL, equity REAL, created_at TEXT, updated_at TEXT)")
            c.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    time TEXT,
                    pattern TEXT,
                    side TEXT,
                    entry REAL,
                    stop REAL,
                    tp REAL,
                    rr REAL,
                    risk_pct REAL,
                    size REAL,
                    status TEXT,
                    exit REAL,
                    pnl REAL,
                    r_multiple REAL,
                    closed_time TEXT,
                    timeframe TEXT
                )
            """)
            # Backfill schema: add timeframe if missing
            try:
                c.execute("ALTER TABLE trades ADD COLUMN timeframe TEXT")
            except Exception:
                pass
            # initialize account row if missing
            c.execute("SELECT balance, equity FROM account WHERE id=1")
            row = c.fetchone()
            if not row:
                now = datetime.utcnow().isoformat()
                c.execute("INSERT INTO account (id, balance, equity, created_at, updated_at) VALUES (1, ?, ?, ?, ?)", (starting_balance, starting_balance, now, now))
            conn.commit()
        finally:
            conn.close()

    def get_balance(self) -> float:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT balance FROM account WHERE id=1")
            row = c.fetchone()
            return float(row[0]) if row else 0.0
        finally:
            conn.close()

    def update_balance(self, delta: float) -> float:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT balance FROM account WHERE id=1")
            row = c.fetchone()
            bal = float(row[0]) if row else 0.0
            bal += float(delta)
            now = datetime.utcnow().isoformat()
            c.execute("UPDATE account SET balance=?, equity=?, updated_at=? WHERE id=1", (bal, bal, now))
            conn.commit()
            return bal
        finally:
            conn.close()

    def record_trade_open(self, symbol: str, trade: Dict):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            payload = (
                trade.get('id'), symbol, trade.get('time'), trade.get('pattern'), trade.get('side'),
                trade.get('entry'), trade.get('stop'), trade.get('tp'), trade.get('rr'), trade.get('risk_pct'),
                trade.get('size'), 'OPEN', None, None, None, None, trade.get('timeframe')
            )
            c.execute("INSERT OR REPLACE INTO trades (id, symbol, time, pattern, side, entry, stop, tp, rr, risk_pct, size, status, exit, pnl, r_multiple, closed_time, timeframe) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", payload)
            conn.commit()
        finally:
            conn.close()

    def record_trade_close(self, trade: Dict):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute(
                "UPDATE trades SET status=?, exit=?, pnl=?, r_multiple=?, closed_time=?, timeframe=? WHERE id=?",
                (trade.get('status'), trade.get('exit'), trade.get('pnl'), trade.get('r_multiple'), trade.get('closed_time'), trade.get('timeframe'), trade.get('id'))
            )
            conn.commit()
        finally:
            conn.close()
        # Non-intrusive enhanced logging
        try:
            ComponentLogger.trading_logger().log_trade_close({
                'event': 'CLOSE',
                'symbol': trade.get('symbol'),
                'side': trade.get('side'),
                'entry': trade.get('entry'),
                'exit': trade.get('exit'),
                'size': trade.get('size'),
                'pnl': trade.get('pnl'),
                'r_multiple': trade.get('r_multiple'),
                'pattern': trade.get('pattern'),
                'timeframe': trade.get('timeframe'),
                'closed_time': trade.get('closed_time'),
                'id': trade.get('id')
            })
        except Exception:
            pass

    def count_open_trades(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'")
            row = c.fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def write_status(self):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT balance, equity FROM account WHERE id=1")
            row = c.fetchone()
            balance = float(row[0]) if row else 0.0
            equity = float(row[1]) if row else balance
            # Reconcile open trades counts with per-symbol status files to avoid stale entries
            open_ids_by_symbol = {}
            try:
                base_dir = os.path.join('sim_results')
                for sym in os.listdir(base_dir):
                    sym_dir = os.path.join(base_dir, sym)
                    if not os.path.isdir(sym_dir):
                        continue
                    
                    # Try to find a status file with open_trades data
                    status = None
                    for status_file in ['status.json', 'live_status.json']:
                        status_fp = os.path.join(sym_dir, status_file)
                        if os.path.exists(status_fp):
                            try:
                                with open(status_fp, 'r') as sf:
                                    temp_status = json.load(sf)
                                    # Only use this file if it contains open_trades field
                                    if 'open_trades' in temp_status:
                                        status = temp_status
                                        break
                            except Exception:
                                continue
                    
                    # Process status if found
                    if status:
                        open_list = status.get('open_trades', []) or []
                        open_ids_by_symbol[sym] = {t.get('id') for t in open_list if t.get('id')}
                    else:
                        open_ids_by_symbol[sym] = set()
            except Exception:
                pass
            # Count open trades considering current simulator reports
            c.execute("SELECT id, symbol FROM trades WHERE status='OPEN'")
            rows = c.fetchall() or []
            open_cnt = 0
            for tid, sym in rows:
                sym_ids = open_ids_by_symbol.get(sym)
                if sym_ids is None:
                    open_cnt += 1
                else:
                    if tid in sym_ids:
                        open_cnt += 1
            c.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'")
            closed_cnt = int(c.fetchone()[0])
            # Compute cumulative PnL and win rate
            c.execute("SELECT COALESCE(SUM(pnl),0) FROM trades WHERE status='CLOSED'")
            cum_pnl = float(c.fetchone()[0] or 0.0)
            c.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl > 0")
            wins = int(c.fetchone()[0] or 0)
            win_rate = (wins / closed_cnt * 100.0) if closed_cnt > 0 else 0.0
            
            # Get per-pair statistics - include ALL symbols with trades (open or closed)
            c.execute("SELECT DISTINCT symbol FROM trades")
            all_symbols = [row[0] for row in c.fetchall()]
            pair_stats = {}
            for symbol in all_symbols:
                # Get closed trades stats
                c.execute("SELECT COUNT(*), COALESCE(SUM(pnl),0) FROM trades WHERE status='CLOSED' AND symbol=?", (symbol,))
                closed_row = c.fetchone()
                closed_trades = int(closed_row[0] or 0)
                pnl = float(closed_row[1] or 0.0)
                
                # Get open trades count - prefer current status file data if available
                if symbol in open_ids_by_symbol:
                    open_trades = len(open_ids_by_symbol[symbol])
                else:
                    c.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN' AND symbol=?", (symbol,))
                    open_trades = int(c.fetchone()[0] or 0)
                
                # Calculate win rate
                c.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND symbol=? AND pnl > 0", (symbol,))
                wins = int(c.fetchone()[0] or 0)
                win_rate_pair = (wins / closed_trades * 100.0) if closed_trades > 0 else 0.0
                
                pair_stats[symbol] = {
                    'open_trades': open_trades,
                    'closed_trades': closed_trades,
                    'pnl': round(pnl, 2),
                    'win_rate_pct': round(win_rate_pair, 2)
                }
            
            status = {
                'timestamp': datetime.utcnow().isoformat(),
                'balance': round(balance, 2),
                'equity': round(equity, 2),
                'open_trades': open_cnt,
                'closed_trades': closed_cnt,
                'cum_pnl': round(cum_pnl, 2),
                'win_rate_pct': round(win_rate, 2),
                'pair_stats': pair_stats
            }
            os.makedirs('sim_results', exist_ok=True)
            with open(os.path.join('sim_results', 'global_status.json'), 'w') as f:
                json.dump(status, f, indent=2)
        finally:
            conn.close()
