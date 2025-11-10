"""
Daily Trading Logic - Comprehensive Trading System
====================================================

Implements multiple trading styles for daily trading:
1. Scalping Logic
2. Momentum Trading Logic
3. Range Trading Logic
4. Swing Trading Logic
5. Arbitrage Logic
6. Session-Based Trading Logic

All strategies follow the baseline daily trading structure with time segmentation
and market context classification.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Import filter functions from pattern_logic
try:
    from pattern_logic import (
        check_volume_weighted_confirmation,
        check_atr_volatility_filter,
        check_adx_trend_filter,
        calculate_indicators
    )
except ImportError:
    # Fallback if pattern_logic not available
    def check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
        return True
    def check_atr_volatility_filter(df, max_multiplier=2.0, min_multiplier=0.5, median_period=14):
        return True
    def check_adx_trend_filter(df, min_adx=25):
        return True
    def calculate_indicators(df):
        return df


# ============================================================================
# BASELINE DAILY TRADING STRUCTURE (Universal Rules)
# ============================================================================

class MarketSession:
    """Market session time definitions (UTC) - Updated per new config"""
    LONDON_START = 7   # 07:00 UTC
    LONDON_END = 11    # 11:00 UTC (reduced from 16:00)
    NY_START = 12      # 12:00 UTC
    NY_END = 17        # 17:00 UTC (reduced from 21:00)
    OVERLAP_START = 12 # London/NY overlap
    OVERLAP_END = 16   # 16:00 UTC
    # Tokyo/Asia session disabled per new config


def get_current_session(utc_time: Optional[datetime] = None) -> str:
    """
    Identify current trading session.
    Returns: 'london', 'ny', 'overlap', 'closed' (Asia/Tokyo disabled)
    """
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    
    hour = utc_time.hour
    
    # Overlap takes priority
    if MarketSession.OVERLAP_START <= hour < MarketSession.OVERLAP_END:
        return 'overlap'
    elif MarketSession.LONDON_START <= hour < MarketSession.LONDON_END:
        return 'london'
    elif MarketSession.NY_START <= hour < MarketSession.NY_END:
        return 'ny'
    else:
        return 'closed'  # All other times (Asia/Tokyo disabled)


def classify_market_context(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Market Context Classification:
    - Identify market structure (trend or range)
    - Identify session volatility (quiet or expanding)
    - Identify key levels
    """
    if len(df) < 50:
        return {'structure': 'unknown', 'volatility': 'unknown', 'levels': {}}
    
    last = df.iloc[-1]
    lookback = min(50, len(df))
    recent = df.tail(lookback)
    
    # Structure: Trend or Range
    ma20 = recent['close'].rolling(20).mean()
    ma50 = recent['close'].rolling(50).mean() if len(recent) >= 50 else ma20
    
    price_range = recent['high'].max() - recent['low'].min()
    price_center = recent['close'].mean()
    range_pct = (price_range / price_center) * 100 if price_center > 0 else 0
    
    # Trend if price consistently above/below MA and range > threshold
    is_uptrend = (last['close'] > ma20.iloc[-1] and ma20.iloc[-1] > ma50.iloc[-1]) if len(ma50) > 0 else False
    is_downtrend = (last['close'] < ma20.iloc[-1] and ma20.iloc[-1] < ma50.iloc[-1]) if len(ma50) > 0 else False
    
    if is_uptrend or is_downtrend:
        structure = 'trend'
    elif range_pct < 2.0:  # Tight range
        structure = 'range'
    else:
        structure = 'transition'
    
    # Volatility: Quiet or Expanding
    atr = recent['high'].sub(recent['low']).abs().rolling(14).mean().iloc[-1]
    atr_pct = (atr / last['close']) * 100 if last['close'] > 0 else 0
    
    if atr_pct < 1.0:
        volatility = 'quiet'
    elif atr_pct > 3.0:
        volatility = 'expanding'
    else:
        volatility = 'normal'
    
    # Key Levels
    levels = {
        'session_high': float(recent['high'].max()),
        'session_low': float(recent['low'].min()),
        'prev_day_high': float(df['high'].iloc[-24:].max()) if len(df) >= 24 else float(recent['high'].max()),
        'prev_day_low': float(df['low'].iloc[-24:].min()) if len(df) >= 24 else float(recent['low'].min()),
        'support': float(recent['low'].min()),
        'resistance': float(recent['high'].max()),
    }
    
    return {
        'structure': structure,
        'trend_direction': 'up' if is_uptrend else ('down' if is_downtrend else 'neutral'),
        'volatility': volatility,
        'levels': levels,
        'range_pct': range_pct
    }


# ============================================================================
# 1. SCALPING LOGIC
# ============================================================================

def evaluate_scalping_signals(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Scalping: Small, frequent profits during high liquidity (5m timeframes).
    Focus: NY session only (12:00-17:00 UTC) per new config.
    Updated with enhanced filters: volume confirmation, ATR filter, minimum 2:1 R:R.
    """
    df = context.get('df')
    timeframe = context.get('timeframe', '').lower()
    session = get_current_session()
    
    if df is None or len(df) < 50:
        return []
    
    # Ensure indicators are calculated
    if 'atr' not in df.columns or 'adx' not in df.columns:
        df = calculate_indicators(df.copy())
    
    # Only on 5m timeframe (1min removed per new config)
    if timeframe != '5min':
        return []
    
    # Only NY session for scalping per new config
    if session != 'ny':
        return []
    
    # ATR volatility filter
    if not check_atr_volatility_filter(df, max_multiplier=2.0, min_multiplier=0.5):
        return []
    
    setups = []
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    
    # Calculate EMAs
    if 'ema9' not in df.columns:
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    if 'ema20' not in df.columns:
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    ema9 = float(df['ema9'].iloc[-1])
    ema20 = float(df['ema20'].iloc[-1])
    price = float(last['close'])
    
    # RSI
    if 'rsi' not in df.columns:
        from pattern_logic import rsi
        df['rsi'] = rsi(df['close'], period=14)
    rsi_val = float(df['rsi'].iloc[-1])
    
    # Volume-weighted confirmation (20% above average) - REQUIRED
    if not check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
        return []
    
    # ATR-based stop (1.5x ATR for scalps)
    atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else price * 0.01
    stop_distance = 1.5 * atr
    
    # Entry Logic: Micro-trend with confirmations
    if price > ema9 > ema20 and rsi_val > 50 and rsi_val < 70:  # Bullish micro-trend
        stop = price - stop_distance
        risk = price - stop
        tp = price + 2.0 * risk  # Minimum 2:1 R:R
        setups.append({
            'name': 'Scalp Long',
            'side': 'BUY',
            'entry': round(price, 8),
            'stop': round(stop, 8),
            'tp': round(tp, 8),
            'rr': 2.0,
            'risk_pct': 0.15,  # Updated to 0.15%
            'confidence': 70,  # Base confidence
            'strategy': 'scalping'
        })
    
    if price < ema9 < ema20 and rsi_val < 50 and rsi_val > 30:  # Bearish micro-trend
        stop = price + stop_distance
        risk = stop - price
        tp = price - 2.0 * risk  # Minimum 2:1 R:R
        setups.append({
            'name': 'Scalp Short',
            'side': 'SELL',
            'entry': round(price, 8),
            'stop': round(stop, 8),
            'tp': round(tp, 8),
            'rr': 2.0,
            'risk_pct': 0.15,  # Updated to 0.15%
            'confidence': 70,  # Base confidence
            'strategy': 'scalping'
        })
    
    return setups


# ============================================================================
# 2. MOMENTUM TRADING LOGIC
# ============================================================================

def evaluate_momentum_signals(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Momentum: Capture strong directional moves (best during London→NY overlap).
    Updated with: ADX filter (>25), volume confirmation, ATR filter, minimum 2:1 R:R.
    """
    df = context.get('df')
    timeframe = context.get('timeframe', '').lower()
    session = get_current_session()
    
    if df is None or len(df) < 50:
        return []
    
    # Ensure indicators are calculated
    if 'atr' not in df.columns or 'adx' not in df.columns:
        df = calculate_indicators(df.copy())
    
    # 5m-1h timeframes
    if timeframe not in ('5min', '15min', '1h'):
        return []
    
    # London and NY sessions only (Asia disabled)
    if session not in ('london', 'ny', 'overlap'):
        return []
    
    # ATR volatility filter
    if not check_atr_volatility_filter(df, max_multiplier=2.0, min_multiplier=0.5):
        return []
    
    # ADX trend filter - REQUIRED for momentum (must be >25)
    if not check_adx_trend_filter(df, min_adx=25):
        return []
    
    setups = []
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    
    # Volume-weighted confirmation (20% above average) - REQUIRED
    if not check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
        return []
    
    # MACD
    if 'macd_hist' in df.columns:
        macd_hist = float(df['macd_hist'].iloc[-1])
        macd_expanding = abs(macd_hist) > abs(float(df['macd_hist'].iloc[-2])) if len(df) >= 2 else False
    else:
        macd_expanding = False
    
    # RSI
    if 'rsi' not in df.columns:
        from pattern_logic import rsi
        df['rsi'] = rsi(df['close'], period=14)
    rsi_val = float(df['rsi'].iloc[-1])
    
    # Momentum ignition: large candle breaking consolidation
    candle_size = abs(float(last['close']) - float(last['open']))
    body_pct = (candle_size / float(last['close'])) * 100 if last['close'] > 0 else 0
    
    price = float(last['close'])
    atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else price * 0.01
    
    # Bullish momentum
    if (body_pct > 0.5 and  # Large body
        price > float(prev['high']) and  # Breakout
        rsi_val > 50 and rsi_val < 70 and
        macd_expanding):
        
        # ATR-based stop (1.5x for momentum on shorter TFs, 2.5x for longer)
        stop_multiplier = 1.5 if timeframe in ('5min', '15min') else 2.5
        stop = price - stop_multiplier * atr
        risk = price - stop
        tp = price + 2.5 * risk  # Minimum 2.5:1 R:R for momentum
        setups.append({
            'name': 'Momentum Long',
            'side': 'BUY',
            'entry': round(price, 8),
            'stop': round(stop, 8),
            'tp': round(tp, 8),
            'rr': 2.5,
            'risk_pct': 0.15,  # Updated to 0.15%
            'confidence': 75,  # Higher confidence for momentum with ADX
            'strategy': 'momentum'
        })
    
    # Bearish momentum
    if (body_pct > 0.5 and
        price < float(prev['low']) and
        rsi_val < 50 and rsi_val > 30 and
        macd_expanding):
        
        stop_multiplier = 1.5 if timeframe in ('5min', '15min') else 2.5
        stop = price + stop_multiplier * atr
        risk = stop - price
        tp = price - 2.5 * risk
        setups.append({
            'name': 'Momentum Short',
            'side': 'SELL',
            'entry': round(price, 8),
            'stop': round(stop, 8),
            'tp': round(tp, 8),
            'rr': 2.5,
            'risk_pct': 0.15,  # Updated to 0.15%
            'confidence': 75,  # Higher confidence for momentum with ADX
            'strategy': 'momentum'
        })
    
    return setups


# ============================================================================
# 3. RANGE TRADING LOGIC
# ============================================================================

def evaluate_range_signals(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Range Trading: DISABLED per new config (only trade during London/NY sessions).
    Range trading was best during Tokyo/Asia, which is now disabled.
    """
    # Range trading disabled - only London and NY sessions allowed
    return []


# ============================================================================
# 4. SWING TRADING LOGIC
# ============================================================================

def evaluate_swing_signals(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Swing Trading: Capture multi-day moves (4H-1D timeframes).
    Updated with: ADX filter, volume confirmation, ATR-based stops (2.5x), minimum 2.5:1 R:R.
    """
    df = context.get('df')
    timeframe = context.get('timeframe', '').lower()
    market_ctx = classify_market_context(df)
    
    if df is None or len(df) < 100:
        return []
    
    # Ensure indicators are calculated
    if 'atr' not in df.columns or 'adx' not in df.columns:
        df = calculate_indicators(df.copy())
    
    # 4H-1D timeframes
    if timeframe not in ('4h', '1d'):
        return []
    
    # Trend structure required
    if market_ctx['structure'] != 'trend':
        return []
    
    # ATR volatility filter
    if not check_atr_volatility_filter(df, max_multiplier=2.0, min_multiplier=0.5):
        return []
    
    # ADX trend filter - REQUIRED for swing trades
    if not check_adx_trend_filter(df, min_adx=25):
        return []
    
    # Volume-weighted confirmation (20% above average) - REQUIRED
    if not check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
        return []
    
    setups = []
    last = df.iloc[-1]
    price = float(last['close'])
    
    # EMAs
    if 'ema20' not in df.columns:
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    if 'ema50' not in df.columns:
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    ema20 = float(df['ema20'].iloc[-1])
    ema50 = float(df['ema50'].iloc[-1])
    
    # ATR-based stop (2.5x ATR for swings)
    atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else price * 0.02
    
    # Uptrend: EMA20 > EMA50, price breaking structure
    if ema20 > ema50 and market_ctx['trend_direction'] == 'up':
        # Entry on retest of broken resistance (now support)
        support = market_ctx['levels']['support']
        if abs(price - support) / price < 0.02:  # Near support
            stop = price - 2.5 * atr  # ATR-based stop
            risk = price - stop
            tp = price + 3.0 * risk  # Minimum 3:1 R:R for swings
            setups.append({
                'name': 'Swing Long',
                'side': 'BUY',
                'entry': round(price, 8),
                'stop': round(stop, 8),
                'tp': round(tp, 8),
                'rr': 3.0,
                'risk_pct': 0.15,  # Updated to 0.15%
                'confidence': 75,  # Higher confidence for swing with ADX
                'strategy': 'swing'
            })
    
    # Downtrend
    if ema20 < ema50 and market_ctx['trend_direction'] == 'down':
        resistance = market_ctx['levels']['resistance']
        if abs(price - resistance) / price < 0.02:
            stop = price + 2.5 * atr  # ATR-based stop
            risk = stop - price
            tp = price - 3.0 * risk
            setups.append({
                'name': 'Swing Short',
                'side': 'SELL',
                'entry': round(price, 8),
                'stop': round(stop, 8),
                'tp': round(tp, 8),
                'rr': 3.0,
                'risk_pct': 0.15,  # Updated to 0.15%
                'confidence': 75,  # Higher confidence for swing with ADX
                'strategy': 'swing'
            })
    
    return setups


# ============================================================================
# 5. ARBITRAGE LOGIC (Placeholder - requires exchange API)
# ============================================================================

def evaluate_arbitrage_signals(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Arbitrage: Exploit price inefficiencies (requires exchange API integration).
    Note: This is a placeholder - full implementation requires real-time order book data.
    """
    # Placeholder - would need exchange API integration
    return []


# ============================================================================
# 6. SESSION-BASED TRADING LOGIC
# ============================================================================

def evaluate_session_based_signals(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Session-Based: Use daily volatility cycles for timed entries.
    Updated for London session (07:00-11:00 UTC) - momentum breakouts only.
    Enhanced with: volume confirmation, ATR filter, ADX filter, minimum 2.5:1 R:R.
    """
    df = context.get('df')
    timeframe = context.get('timeframe', '').lower()
    session = get_current_session()
    
    if df is None or len(df) < 100:
        return []
    
    # Ensure indicators are calculated
    if 'atr' not in df.columns or 'adx' not in df.columns:
        df = calculate_indicators(df.copy())
    
    # 15m-1h timeframes
    if timeframe not in ('15min', '1h'):
        return []
    
    # Only London session for momentum breakouts per new config
    if session != 'london':
        return []
    
    # ATR volatility filter
    if not check_atr_volatility_filter(df, max_multiplier=2.0, min_multiplier=0.5):
        return []
    
    # ADX trend filter - REQUIRED for momentum breakouts
    if not check_adx_trend_filter(df, min_adx=25):
        return []
    
    # Volume-weighted confirmation (20% above average) - REQUIRED
    if not check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
        return []
    
    setups = []
    last = df.iloc[-1]
    price = float(last['close'])
    
    # Get previous session range (lookback for consolidation)
    lookback = min(100, len(df))
    recent = df.tail(lookback)
    prev_high = float(recent['high'].max())
    prev_low = float(recent['low'].min())
    prev_range = prev_high - prev_low
    
    # ATR-based stop
    atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else price * 0.01
    stop_multiplier = 2.5  # Swing stop for session breakouts
    
    # London breakout above previous high
    if price > prev_high * 1.001:  # 0.1% breakout
        stop = price - stop_multiplier * atr
        risk = price - stop
        tp = price + 2.5 * risk  # Minimum 2.5:1 R:R
        setups.append({
            'name': 'Session Breakout Long',
            'side': 'BUY',
            'entry': round(price, 8),
            'stop': round(stop, 8),
            'tp': round(tp, 8),
            'rr': 2.5,
            'risk_pct': 0.15,  # Updated to 0.15%
            'confidence': 75,  # Higher confidence with ADX
            'strategy': 'session_based'
        })
    
    # London breakout below previous low
    if price < prev_low * 0.999:
        stop = price + stop_multiplier * atr
        risk = stop - price
        tp = price - 2.5 * risk
        setups.append({
            'name': 'Session Breakout Short',
            'side': 'SELL',
            'entry': round(price, 8),
            'stop': round(stop, 8),
            'tp': round(tp, 8),
            'rr': 2.5,
            'risk_pct': 0.15,  # Updated to 0.15%
            'confidence': 75,  # Higher confidence with ADX
            'strategy': 'session_based'
        })
    
    return setups


# ============================================================================
# 7. COMBINED DAILY TRADING SYSTEM
# ============================================================================

def evaluate_daily_trading_signals(context: Dict[str, Any], 
                                   enabled_strategies: List[str] = None) -> List[Dict[str, Any]]:
    """
    Combined system that evaluates all enabled strategies based on session and timeframe.
    
    Strategy selection by session:
    - 00:00-07:00 UTC (Tokyo): Range trading
    - 07:00-12:00 UTC (London): Momentum + Scalping
    - 12:00-16:00 UTC (Overlap): Momentum + Session breakout (most profitable)
    - 16:00-21:00 UTC (NY): Range/swing re-entries
    - 21:00-00:00 UTC (Post-NY): No trade / review
    
    Args:
        context: Dictionary with keys:
            - df: DataFrame with OHLCV data and indicators
            - timeframe: str (e.g., '1min', '5min', '15min', '30min', '1h', '4h', '1d')
            - session_info: Optional dict from SessionManager.get_session_info()
        enabled_strategies: List of strategy names to enable. Defaults to all except arbitrage.
    
    Returns:
        List of setup dictionaries compatible with pattern_logic format.
    """
    if enabled_strategies is None:
        enabled_strategies = ['scalping', 'momentum', 'range', 'swing', 'session_based']
    
    # Validate context
    df = context.get('df')
    if df is None or len(df) < 20:
        return []
    
    session = get_current_session()
    timeframe = context.get('timeframe', '').lower()
    
    all_setups = []
    
    # Strategy selection by session (updated per new config)
    if session == 'closed':
        # All other times (Asia/Tokyo disabled) - no trading
        return []
    
    elif session == 'london':
        # London (07:00-11:00 UTC): Momentum breakouts only
        if 'momentum' in enabled_strategies:
            all_setups.extend(evaluate_momentum_signals(context))
        if 'session_based' in enabled_strategies:
            all_setups.extend(evaluate_session_based_signals(context))
    
    elif session == 'overlap':
        # Overlap (12:00-16:00 UTC): Momentum + Scalping
        if 'scalping' in enabled_strategies and timeframe == '5min':
            all_setups.extend(evaluate_scalping_signals(context))
        if 'momentum' in enabled_strategies:
            all_setups.extend(evaluate_momentum_signals(context))
    
    elif session == 'ny':
        # NY (12:00-17:00 UTC): Scalping + Trend continuations
        if 'scalping' in enabled_strategies and timeframe == '5min':
            all_setups.extend(evaluate_scalping_signals(context))
        if 'momentum' in enabled_strategies:
            all_setups.extend(evaluate_momentum_signals(context))
        if 'swing' in enabled_strategies and timeframe in ('4h', '1d'):
            all_setups.extend(evaluate_swing_signals(context))
    
    # Always check swing on higher timeframes (regardless of session)
    if 'swing' in enabled_strategies and timeframe in ('4h', '1d'):
        swing_setups = evaluate_swing_signals(context)
        # Avoid duplicates
        existing_setups = {(s['name'], s['side']) for s in all_setups}
        for setup in swing_setups:
            key = (setup['name'], setup['side'])
            if key not in existing_setups:
                all_setups.append(setup)
    
    return all_setups

