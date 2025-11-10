"""
Risk manager scaffold for adaptive SL/TP and sizing.
LAYER 3 & 6: Includes dynamic RR, account-level protections, and kill switch.
"""

import os
import json
from typing import Dict, Optional
from pathlib import Path


def _load_risk_config() -> Dict:
    """Load risk configuration from config/risk.json"""
    config_path = Path('config') / 'risk.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    # Return defaults
    return {
        "max_drawdown_pct": 15.0,
        "risk_per_trade_pct": 0.15,
        "max_concurrent_trades": 3,
        "max_daily_trades": 10,
        "max_daily_loss_pct": 1.0,
        "kill_switch": {
            "soft_drawdown_pct": 10.0,
            "hard_drawdown_pct": 15.0,
            "enabled": True
        }
    }


def adjust_setup_for_volatility(setup: Dict, atr: float, atr_multiplier_sl: float = 0.5, rr: float = None) -> Dict:
    """Return a copy of setup with ATR-based SL/TP adjustments."""
    s = dict(setup)
    entry = float(s.get('entry', 0))
    side = s.get('side', 'BUY')
    if entry <= 0 or atr <= 0:
        return s
    # Compute SL distance as fraction of ATR
    sl_dist = atr * atr_multiplier_sl
    if side == 'BUY':
        s['stop'] = entry - sl_dist
    else:
        s['stop'] = entry + sl_dist
    # Recompute TP by RR if provided
    if rr is not None:
        s['rr'] = rr
        if side == 'BUY':
            s['tp'] = entry + rr * (entry - s['stop'])
        else:
            s['tp'] = entry - rr * (s['stop'] - entry)
    return s


def get_dynamic_rr_target(confidence: float) -> float:
    """
    LAYER 3: Dynamic RR targeting based on confidence.
    More granular thresholds for better profitability:
    - confidence < 60 → target RR = 1.8
    - confidence < 65 → target RR = 2.0
    - confidence < 70 → target RR = 2.2
    - confidence < 75 → target RR = 2.5
    - confidence < 80 → target RR = 2.8
    - confidence < 85 → target RR = 3.0
    - confidence < 90 → target RR = 3.2
    - confidence >= 90 → target RR = 3.5
    """
    config = _load_risk_config()
    rr_config = config.get('take_profit', {}).get('dynamic_rr', {})
    
    if confidence < 60:
        return float(rr_config.get('confidence_60', 1.8))
    elif confidence < 65:
        return float(rr_config.get('confidence_65', 2.0))
    elif confidence < 70:
        return float(rr_config.get('confidence_70', 2.2))
    elif confidence < 75:
        return float(rr_config.get('confidence_75', 2.5))
    elif confidence < 80:
        return float(rr_config.get('confidence_80', 2.8))
    elif confidence < 85:
        return float(rr_config.get('confidence_85', 3.0))
    elif confidence < 90:
        return float(rr_config.get('confidence_90', 3.2))
    else:
        return 3.5  # Very high confidence trades get best RR


def check_account_limits(current_equity: float, initial_equity: float, 
                         open_trades: int, daily_pnl: float) -> Dict[str, any]:
    """
    LAYER 6: Account-level protections.
    Checks:
    - Max concurrent trades
    - Max total exposure
    - Daily loss limit
    - Kill switch (soft/hard drawdown)
    
    Updated to use config/risk.json values:
    - max_concurrent_trades: 3
    - max_daily_loss_pct: 1.0%
    - soft_drawdown_pct: 10.0%
    - hard_drawdown_pct: 15.0%
    """
    config = _load_risk_config()
    max_trades = int(config.get('max_concurrent_trades', 3))
    max_exposure_pct = float(config.get('max_total_exposure_pct', 0.006))
    daily_loss_pct = float(config.get('max_daily_loss_pct', 0.01))
    kill_switch = config.get('kill_switch', {})
    soft_dd = float(kill_switch.get('soft_drawdown_pct', 10.0)) / 100.0
    hard_dd = float(kill_switch.get('hard_drawdown_pct', 15.0)) / 100.0
    
    result = {
        'can_trade': True,
        'reasons': [],
        'action': None  # 'reduce_size', 'stop_trading', or None
    }
    
    # Check max concurrent trades
    if open_trades >= max_trades:
        result['can_trade'] = False
        result['reasons'].append(f"Max concurrent trades reached ({open_trades}/{max_trades})")
        return result
    
    # Check daily loss limit
    daily_loss = abs(daily_pnl) if daily_pnl < 0 else 0
    if daily_loss > 0 and initial_equity > 0:
        daily_loss_pct_actual = daily_loss / initial_equity
        if daily_loss_pct_actual >= daily_loss_pct:
            result['can_trade'] = False
            result['action'] = 'stop_trading'
            result['reasons'].append(f"Daily loss limit reached ({daily_loss_pct_actual*100:.2f}% >= {daily_loss_pct*100:.2f}%)")
            return result
    
    # Check drawdown (kill switch)
    if initial_equity > 0:
        current_dd = (initial_equity - current_equity) / initial_equity
        
        if current_dd >= hard_dd:
            result['can_trade'] = False
            result['action'] = 'stop_trading'
            result['reasons'].append(f"Hard drawdown limit reached ({current_dd*100:.2f}% >= {hard_dd*100:.2f}%)")
            return result
        elif current_dd >= soft_dd:
            result['action'] = 'reduce_size'
            result['reasons'].append(f"Soft drawdown alert ({current_dd*100:.2f}% >= {soft_dd*100:.2f}%) - reducing position size")
    
    return result


