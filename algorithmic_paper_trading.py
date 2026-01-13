"""
Isolated Paper Trading Instance for Algorithmic Trading System
==============================================================

This module provides a completely isolated paper trading simulator
specifically for the algorithmic trading system. It maintains its own
balance, trades, and state separate from the main trading system and RL training.

ISOLATION GUARANTEES:
- Uses separate database: algorithmic_trading/paper_trading_{SYMBOL}.db
- Does not share balance with GlobalAccountLedger or RL training
- All files stored in algorithmic_trading/ directory
- No shared state with other trading systems
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import deque

from coinEx_getting_data import CoinExDataFetcher
from utils.logger import ComponentLogger
from verify_patterns import normalize_symbol_to_ccxt


class AlgorithmicPaperTrading:
    """
    Isolated paper trading simulator for algorithmic trading system.
    Each pair gets its own instance with separate balance tracking.
    """
    
    def __init__(
        self,
        symbol: str,
        starting_balance: float = 10000.0,
        mode: str = "spot",
        fee_pct: float = 0.0004,
        slippage_pct: float = 0.0005,
    ):
        self.symbol = symbol
        self.starting_balance = starting_balance
        self.mode = mode
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        
        # Anchor to project root
        self.project_root = Path(__file__).resolve().parent
        
        # Isolated database for this system
        algo_dir = self.project_root / "algorithmic_trading"
        os.makedirs(algo_dir, exist_ok=True)
        
        # Per-pair database - normalize symbol to avoid collisions
        # Use original symbol as-is for uniqueness, just make filesystem-safe
        safe_symbol = symbol.replace('/', '_').replace('-', '_').upper()
        self.db_path = algo_dir / f"paper_trading_{safe_symbol}.db"
        self._init_db()
        
        # Load or initialize balance
        self.balance = self._load_balance()
        needs_save = (self.balance is None)
        if self.balance is None:
            self.balance = starting_balance
        
        # Initialize equity before saving (needed by _save_balance)
        self.equity = self.balance
        self.max_equity = self.balance
        self.min_equity = self.balance
        
        # Save balance if it was just initialized
        if needs_save:
            self._save_balance()
        
        # Trade tracking
        self.open_trades: List[Dict] = []
        self.closed_trades: List[Dict] = []
        
        # Price fetcher
        symbol_ccxt = normalize_symbol_to_ccxt(symbol)
        self.fetcher = CoinExDataFetcher(symbol=symbol_ccxt, timeframe_internal="1min", mode=mode)
        try:
            self.fetcher.update_initial(limit=50)
        except Exception:
            pass
        
        # Status file - use same normalization as database
        safe_symbol = symbol.replace('/', '_').replace('-', '_').upper()
        self.status_path = algo_dir / safe_symbol / "status.json"
        os.makedirs(self.status_path.parent, exist_ok=True)
        
        # Logger
        self.logger = ComponentLogger.algorithmic_logger()
    
    def _init_db(self):
        """Initialize isolated database"""
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    balance REAL,
                    equity REAL,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    algorithm_id TEXT,
                    symbol TEXT,
                    time TEXT,
                    side TEXT,
                    entry REAL,
                    stop REAL,
                    tp REAL,
                    size REAL,
                    status TEXT,
                    exit REAL,
                    pnl REAL,
                    r_multiple REAL,
                    closed_time TEXT,
                    algorithm_name TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()
    
    def _load_balance(self) -> Optional[float]:
        """Load balance from database"""
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT balance FROM account WHERE id=1")
            row = c.fetchone()
            return float(row[0]) if row else None
        finally:
            conn.close()
    
    def _save_balance(self):
        """Save balance to database"""
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            now = datetime.utcnow().isoformat()
            c.execute("SELECT id FROM account WHERE id=1")
            if c.fetchone():
                c.execute(
                    "UPDATE account SET balance=?, equity=?, updated_at=? WHERE id=1",
                    (self.balance, self.equity, now)
                )
            else:
                c.execute(
                    "INSERT INTO account (id, balance, equity, created_at, updated_at) VALUES (1, ?, ?, ?, ?)",
                    (self.balance, self.equity, now, now)
                )
            conn.commit()
        finally:
            conn.close()
    
    def get_current_price(self) -> Optional[float]:
        """Get current market price"""
        try:
            candles = self.fetcher.get_kline_data(limit=1)
            if candles and len(candles) > 0:
                return float(candles[-1]['close'])
        except Exception as e:
            self.logger.warning("price_fetch_error", error=str(e))
        return None
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to price"""
        if side.upper() == 'BUY':
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)
    
    def _calculate_fees(self, price: float, size: float) -> float:
        """Calculate trading fees"""
        return price * size * self.fee_pct
    
    def open_trade(
        self,
        algorithm_id: str,
        algorithm_name: str,
        side: str,
        entry: float,
        stop: float,
        tp: float,
        risk_pct: float = 0.01,
    ) -> Optional[str]:
        """Open a new trade"""
        try:
            current_price = self.get_current_price()
            if current_price is None:
                return None
            
            # Calculate position size based on risk
            risk_per_unit = abs(entry - stop) if stop else entry * 0.02
            if risk_per_unit == 0:
                return None
            
            risk_amount = self.balance * risk_pct
            size = risk_amount / risk_per_unit
            
            # Apply slippage to entry
            entry_with_slippage = self._apply_slippage(entry, side)
            entry_fees = self._calculate_fees(entry_with_slippage, size)
            
            # Check if we have enough balance
            total_cost = entry_with_slippage * size + entry_fees
            if total_cost > self.balance:
                return None
            
            # Create trade
            trade_id = f"ALGO_{algorithm_id}_{int(time.time() * 1000)}"
            trade = {
                'id': trade_id,
                'algorithm_id': algorithm_id,
                'algorithm_name': algorithm_name,
                'symbol': self.symbol,
                'time': datetime.utcnow().isoformat(),
                'side': side.upper(),
                'entry': entry_with_slippage,
                'stop': stop,
                'tp': tp,
                'size': size,
                'status': 'OPEN',
                'entry_fees': entry_fees,
            }
            
            # Deduct from balance
            self.balance -= total_cost
            self._save_balance()
            
            # Store trade
            self.open_trades.append(trade)
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            try:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO trades (id, algorithm_id, symbol, time, side, entry, stop, tp, size, status, algorithm_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id, algorithm_id, self.symbol, trade['time'], side.upper(),
                    entry_with_slippage, stop, tp, size, 'OPEN', algorithm_name
                ))
                conn.commit()
            finally:
                conn.close()
            
            self.logger.info("trade_opened", trade_id=trade_id, algorithm=algorithm_name, side=side, size=size)
            return trade_id
            
        except Exception as e:
            self.logger.error("trade_open_error", error=str(e))
            return None
    
    def close_trade(self, trade_id: str, reason: str = "MANUAL") -> bool:
        """Close an open trade"""
        try:
            # Find trade
            trade = None
            for idx, t in enumerate(self.open_trades):
                if t.get('id') == trade_id:
                    trade = self.open_trades.pop(idx)
                    break
            
            if not trade:
                return False
            
            # Get exit price
            current_price = self.get_current_price()
            if current_price is None:
                # Re-add trade if we can't get price
                self.open_trades.append(trade)
                return False
            
            side = trade.get('side')
            entry = trade.get('entry')
            stop = trade.get('stop')
            tp = trade.get('tp')
            size = trade.get('size')
            
            # Determine exit price (check if TP or SL hit)
            exit_price = current_price
            if side == 'BUY':
                if current_price >= tp:
                    exit_price = tp
                elif current_price <= stop:
                    exit_price = stop
            else:  # SELL
                if current_price <= tp:
                    exit_price = tp
                elif current_price >= stop:
                    exit_price = stop
            
            # Apply slippage
            exit_price_with_slippage = self._apply_slippage(exit_price, 'SELL' if side == 'BUY' else 'BUY')
            exit_fees = self._calculate_fees(exit_price_with_slippage, size)
            
            # Calculate PnL
            if side == 'BUY':
                pnl = (exit_price_with_slippage - entry) * size - trade.get('entry_fees', 0) - exit_fees
            else:
                pnl = (entry - exit_price_with_slippage) * size - trade.get('entry_fees', 0) - exit_fees
            
            # Calculate R multiple
            risk_per_unit = abs(entry - stop) if stop else 1e-12
            if side == 'BUY':
                r_multiple = (exit_price_with_slippage - entry) / risk_per_unit
            else:
                r_multiple = (entry - exit_price_with_slippage) / risk_per_unit
            
            # Update balance
            self.balance += entry * size + pnl
            self.equity = self.balance
            self.max_equity = max(self.max_equity, self.equity)
            self.min_equity = min(self.min_equity, self.equity)
            self._save_balance()
            
            # Update trade
            closed_trade = trade.copy()
            closed_trade.update({
                'status': 'CLOSED',
                'exit': exit_price_with_slippage,
                'pnl': round(pnl, 2),
                'r_multiple': round(r_multiple, 2),
                'closed_time': datetime.utcnow().isoformat(),
                'close_reason': reason,
            })
            self.closed_trades.append(closed_trade)
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            try:
                c = conn.cursor()
                c.execute("""
                    UPDATE trades SET status=?, exit=?, pnl=?, r_multiple=?, closed_time=?
                    WHERE id=?
                """, ('CLOSED', exit_price_with_slippage, pnl, r_multiple, closed_trade['closed_time'], trade_id))
                conn.commit()
            finally:
                conn.close()
            
            self.logger.info("trade_closed", trade_id=trade_id, pnl=pnl, r_multiple=r_multiple)
            return True
            
        except Exception as e:
            self.logger.error("trade_close_error", error=str(e))
            return False
    
    def step(self):
        """Update open trades (check for TP/SL hits)"""
        for trade in list(self.open_trades):
            current_price = self.get_current_price()
            if current_price is None:
                continue
            
            side = trade.get('side')
            entry = trade.get('entry')
            stop = trade.get('stop')
            tp = trade.get('tp')
            
            # Check if TP or SL hit
            should_close = False
            reason = "TP_HIT"
            
            if side == 'BUY':
                if current_price >= tp:
                    should_close = True
                    reason = "TP_HIT"
                elif current_price <= stop:
                    should_close = True
                    reason = "SL_HIT"
            else:  # SELL
                if current_price <= tp:
                    should_close = True
                    reason = "TP_HIT"
                elif current_price >= stop:
                    should_close = True
                    reason = "SL_HIT"
            
            if should_close:
                self.close_trade(trade.get('id'), reason=reason)
    
    def get_statistics(self, algorithm_id: Optional[str] = None) -> Dict:
        """Get trading statistics"""
        # Filter trades by algorithm if specified
        if algorithm_id:
            closed = [t for t in self.closed_trades if t.get('algorithm_id') == algorithm_id]
        else:
            closed = self.closed_trades
        
        if not closed:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_r_multiple': 0.0,
                'total_pnl': 0.0,
                'profit_factor': 0.0,
            }
        
        wins = [t for t in closed if t.get('pnl', 0) > 0]
        losses = [t for t in closed if t.get('pnl', 0) <= 0]
        
        total_pnl = sum(t.get('pnl', 0) for t in closed)
        win_rate = (len(wins) / len(closed) * 100) if closed else 0.0
        avg_r_multiple = sum(t.get('r_multiple', 0) for t in closed) / len(closed) if closed else 0.0
        
        gross_profit = sum(t.get('pnl', 0) for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.get('pnl', 0) for t in losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        
        return {
            'total_trades': len(closed),
            'win_rate': round(win_rate, 2),
            'avg_r_multiple': round(avg_r_multiple, 2),
            'total_pnl': round(total_pnl, 2),
            'profit_factor': round(profit_factor, 2),
            'wins': len(wins),
            'losses': len(losses),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
        }
    
    def write_status(self):
        """Write status to JSON file"""
        try:
            status = {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': self.symbol,
                'balance': round(self.balance, 2),
                'equity': round(self.equity, 2),
                'max_equity': round(self.max_equity, 2),
                'min_equity': round(self.min_equity, 2),
                'open_trades': len(self.open_trades),
                'closed_trades': len(self.closed_trades),
                'statistics': self.get_statistics(),
            }
            with open(self.status_path, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            self.logger.error("status_write_error", error=str(e))

