"""
Session-based intraday trading logic.

Uses the comprehensive daily trading logic system from core.daily_trading_logic.
This module provides a wrapper that maintains compatibility with the existing system
while using the new comprehensive daily trading strategies.

The new system includes:
  - Scalping Logic (1m-5m timeframes)
  - Momentum Trading Logic (5m-1h)
  - Range Trading Logic (15m-1h)
  - Swing Trading Logic (4h-1d)
  - Session-Based Trading Logic (15m-1h)
  - Combined system that selects strategies based on session and timeframe

Returns setups compatible with the rulebook in `pattern_logic.decide_trade_signals`.
"""

from typing import Dict, Any, List
import os


def evaluate_session_signals(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build session-driven setups using the comprehensive daily trading logic system.
    
    This function replaces the old simple momentum/VWAP logic with the new
    comprehensive daily trading strategies that adapt to market sessions.

    Expected context keys:
      - df: DataFrame with columns open, high, low, close, volume and indicators
      - timeframe: str (e.g., '1min', '5min', '15min', '30min', '1h', '4h', '1d')
      - session_info: dict from SessionManager.get_session_info() (optional)
    
    Returns:
      List of setup dictionaries with keys: name, side, entry, stop, tp, rr, risk_pct, strategy
    """
    try:
        # Import the new comprehensive daily trading logic
        from core.daily_trading_logic import evaluate_daily_trading_signals
        
        # Get enabled strategies from environment or use defaults
        enabled_strategies_str = os.getenv('DAILY_TRADING_STRATEGIES', 'scalping,momentum,range,swing,session_based')
        enabled_strategies = [s.strip() for s in enabled_strategies_str.split(',') if s.strip()]
        
        # Remove 'arbitrage' from defaults if present (requires exchange API)
        if 'arbitrage' in enabled_strategies:
            enabled_strategies.remove('arbitrage')
        
        # Ensure we have at least some strategies enabled
        if not enabled_strategies:
            enabled_strategies = ['scalping', 'momentum', 'range', 'swing', 'session_based']
        
        # Call the new comprehensive daily trading logic
        setups = evaluate_daily_trading_signals(context, enabled_strategies=enabled_strategies)
        
        # Ensure all setups have required fields for compatibility
        for setup in setups:
            # Ensure risk_pct is in percentage format (0.5 = 0.5%, not 0.5% = 50%)
            # The new system uses decimal format (0.005 = 0.5%), but old system expects percentage
            if 'risk_pct' in setup:
                # Convert from decimal to percentage if needed (if > 1, assume it's already percentage)
                if setup['risk_pct'] < 1.0:
                    # Already in decimal format, keep as is (will be handled by bot_manager)
                    pass
                # Otherwise assume it's already in percentage format
        
        return setups
        
    except ImportError as e:
        # Fallback: if new module not available, return empty list
        print(f"[WARNING] Could not import daily trading logic: {e}")
        return []
    except Exception as e:
        # Error handling: log and return empty list
        print(f"[ERROR] Error in evaluate_session_signals: {e}")
        return []


def run_session_mode() -> None:
    """Standalone session mode hook (reserved)."""
    return None
    # Will integrate with BotManager/timeframe bots in future steps
    pass


