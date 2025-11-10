"""
Cooldown Manager for Trading Bot

Implements cooldown logic to prevent revenge trading and over-trading:
- After 3 consecutive losses: pause for 3 hours
- After 10 trades per day: halt until next session
- After 5 losses in a day: pause until UTC midnight
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional


class CooldownManager:
    """Manages trading cooldowns based on consecutive losses and daily trade limits"""
    
    def __init__(self, state_file: str = "logs/cooldown_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load cooldown configuration from config/risk.json"""
        config_path = Path('config') / 'risk.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    risk_config = json.load(f)
                    return risk_config.get('cooldown', {
                        "enabled": True,
                        "consecutive_losses_threshold": 3,
                        "cooldown_hours": 3,
                        "max_trades_per_day": 10
                    })
            except Exception:
                pass
        # Defaults
        return {
            "enabled": True,
            "consecutive_losses_threshold": 3,
            "cooldown_hours": 3,
            "max_trades_per_day": 10
        }
    
    def _load_state(self) -> Dict:
        """Load cooldown state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        # Initialize default state
        return {
            "consecutive_losses": 0,
            "last_loss_time": None,
            "cooldown_until": None,
            "daily_trades": {},
            "daily_losses": {},
            "last_reset_date": None
        }
    
    def _save_state(self):
        """Save cooldown state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass
    
    def _reset_daily_if_needed(self):
        """Reset daily counters if it's a new day"""
        now = datetime.now(timezone.utc)
        current_date = now.date().isoformat()
        last_reset = self.state.get('last_reset_date')
        
        if last_reset != current_date:
            # New day - reset daily counters
            self.state['daily_trades'] = {}
            self.state['daily_losses'] = {}
            self.state['last_reset_date'] = current_date
            self._save_state()
    
    def record_trade(self, symbol: str, is_win: bool, pnl: float = 0.0):
        """
        Record a trade result and update cooldown state.
        
        Args:
            symbol: Trading symbol
            is_win: True if trade was profitable, False if loss
            pnl: Profit/loss amount (negative for losses)
        """
        if not self.config.get('enabled', True):
            return
        
        self._reset_daily_if_needed()
        now = datetime.now(timezone.utc)
        current_date = now.date().isoformat()
        
        # Update daily trade count
        if current_date not in self.state['daily_trades']:
            self.state['daily_trades'][current_date] = {}
        if symbol not in self.state['daily_trades'][current_date]:
            self.state['daily_trades'][current_date][symbol] = 0
        self.state['daily_trades'][current_date][symbol] += 1
        
        # Update consecutive losses
        if not is_win:
            self.state['consecutive_losses'] += 1
            self.state['last_loss_time'] = now.isoformat()
            
            # Check if we hit consecutive loss threshold
            threshold = self.config.get('consecutive_losses_threshold', 3)
            if self.state['consecutive_losses'] >= threshold:
                cooldown_hours = self.config.get('cooldown_hours', 3)
                cooldown_until = now + timedelta(hours=cooldown_hours)
                self.state['cooldown_until'] = cooldown_until.isoformat()
        else:
            # Reset consecutive losses on win
            self.state['consecutive_losses'] = 0
            self.state['cooldown_until'] = None
        
        # Track daily losses
        if not is_win:
            if current_date not in self.state['daily_losses']:
                self.state['daily_losses'][current_date] = {}
            if symbol not in self.state['daily_losses'][current_date]:
                self.state['daily_losses'][current_date][symbol] = 0
            self.state['daily_losses'][current_date][symbol] += 1
        
        self._save_state()
    
    def can_trade(self, symbol: str) -> tuple[bool, Optional[str]]:
        """
        Check if trading is allowed for the given symbol.
        
        Returns:
            (can_trade: bool, reason: Optional[str])
        """
        if not self.config.get('enabled', True):
            return True, None
        
        self._reset_daily_if_needed()
        now = datetime.now(timezone.utc)
        current_date = now.date().isoformat()
        
        # Check cooldown from consecutive losses
        cooldown_until = self.state.get('cooldown_until')
        if cooldown_until:
            try:
                cooldown_time = datetime.fromisoformat(cooldown_until.replace('Z', '+00:00'))
                if now < cooldown_time:
                    remaining = (cooldown_time - now).total_seconds() / 3600
                    return False, f"Cooldown active: {remaining:.1f}h remaining (3 consecutive losses)"
            except Exception:
                pass
        
        # Check daily trade limit
        max_trades = self.config.get('max_trades_per_day', 10)
        daily_trades = self.state.get('daily_trades', {}).get(current_date, {})
        total_trades_today = sum(daily_trades.values())
        
        if total_trades_today >= max_trades:
            return False, f"Daily trade limit reached ({total_trades_today}/{max_trades})"
        
        # Check daily loss limit (5 losses in a day = pause until midnight)
        daily_losses = self.state.get('daily_losses', {}).get(current_date, {})
        total_losses_today = sum(daily_losses.values())
        
        if total_losses_today >= 5:
            # Pause until next UTC midnight
            next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            if now < next_midnight:
                remaining = (next_midnight - now).total_seconds() / 3600
                return False, f"Daily loss limit reached (5 losses). Paused until UTC midnight ({remaining:.1f}h)"
        
        return True, None
    
    def get_status(self) -> Dict:
        """Get current cooldown status for monitoring"""
        self._reset_daily_if_needed()
        now = datetime.now(timezone.utc)
        current_date = now.date().isoformat()
        
        daily_trades = self.state.get('daily_trades', {}).get(current_date, {})
        total_trades_today = sum(daily_trades.values())
        
        cooldown_until = self.state.get('cooldown_until')
        cooldown_active = False
        cooldown_remaining = None
        if cooldown_until:
            try:
                cooldown_time = datetime.fromisoformat(cooldown_until.replace('Z', '+00:00'))
                if now < cooldown_time:
                    cooldown_active = True
                    cooldown_remaining = (cooldown_time - now).total_seconds() / 3600
            except Exception:
                pass
        
        return {
            "consecutive_losses": self.state.get('consecutive_losses', 0),
            "daily_trades": total_trades_today,
            "cooldown_active": cooldown_active,
            "cooldown_remaining_hours": cooldown_remaining,
            "can_trade": cooldown_active is False and total_trades_today < self.config.get('max_trades_per_day', 10)
        }

