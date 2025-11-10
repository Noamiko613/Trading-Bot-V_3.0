import requests
import os
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import inspect
import ccxt

from pattern_logic import (
    detect_golden_cross,
    detect_death_cross,
    detect_rsi_trend,
    detect_macd_cross,
    detect_bull_flag,
    detect_triangle_breakout,
    detect_head_shoulders,
    detect_double_top_bottom,
    detect_bullish_engulfing,
    detect_hammer,
    detect_rsi_divergence,
    detect_scalp_breakout,
    detect_pullback_to_ema,
    detect_vwap_bounce,
    detect_ema_ribbon_hold,
)

# =====================
# CONFIG
# =====================
PATTERN_UNVERIFIED = Path("patterns_unverified")
PATTERN_VERIFIED = Path("patterns_verified")
assume_tz_offset_hours = 0

# Interval mappings for CoinEx
interval_alias = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "4h": "4h"
}

interval_ms_map = {
    "1min": 60*1000,
    "5min": 5*60*1000,
    "15min": 15*60*1000,
    "30min": 30*60*1000,
    "1h": 60*60*1000,
    "4h": 4*60*60*1000
}

# =====================
# Helpers
# =====================
import requests
from datetime import datetime, timezone, timedelta
import pandas as pd

assume_tz_offset_hours = 0

interval_ms_map = {
    "1min": 60*1000, "5min": 5*60*1000, "15min": 15*60*1000,
    "30min": 30*60*1000, "1h": 60*60*1000, "4h": 4*60*60*1000
}

def normalize_symbol_to_ccxt(symbol: str) -> str:
    """Convert various symbol formats to ccxt format (e.g., 'ETH-USDT' -> 'ETH/USDT')."""
    if not symbol:
        return 'BTC/USDT'
    
    s = symbol.upper().replace(' ', '')
    
    # Handle different separators
    s = s.replace('-', '/').replace('_', '/').replace(':', '/')
    
    # If no separator and ends with USDT, add separator
    if '/' not in s and s.endswith('USDT'):
        s = s[:-4] + '/USDT'
    elif '/' not in s and len(s) >= 6:  # Assume format like BTCUSDT
        # Try to split at common points
        for i in range(3, len(s)-2):
            if s[i:].startswith(('USDT', 'USDC', 'BTC', 'ETH')):
                s = s[:i] + '/' + s[i:]
                break
    
    return s

def normalize_symbol_to_filename(symbol: str) -> str:
    """Convert symbol to filename-safe format (e.g., 'ETH/USDT' -> 'ETH-USDT')."""
    if not symbol:
        return 'BTC-USDT'
    
    s = symbol.upper().replace(' ', '')
    s = s.replace('/', '-').replace('_', '-').replace(':', '-')
    return s

# ---------------------
# Time conversion
# ---------------------
def iso_to_ms(iso_ts):
    dt = pd.to_datetime(iso_ts, utc=True)
    return int(dt.timestamp() * 1000)

def fetch_window_local(symbol_ccxt: str, tf_internal: str, ts_ms: int, mode: str = 'spot') -> pd.DataFrame:
    """Fetch a window of candles from local data files.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        # Convert symbol to local file format
        symbol_normalized = symbol_ccxt.replace('-', '').replace('/', '').lower()
        base_asset = symbol_normalized.replace('usdt', '').replace('btc', '').replace('eth', '').replace('bnb', '').replace('sol', '').replace('ada', '').replace('xrp', '')
        
        # Map to local data folder
        symbol_map = {
            'btcusdt': 'btc',
            'ethusdt': 'eth', 
            'bnbusdt': 'bnb',
            'solusdt': 'sol',
            'adausdt': 'ada',
            'xrpusdt': 'xrp'
        }
        
        local_symbol = symbol_map.get(symbol_normalized, base_asset)
        data_file = f"data/{local_symbol}/{tf_internal}.json"
        
        if not os.path.exists(data_file):
            print(f"[WARNING] Local data file not found: {data_file}")
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        
        # Load local data (JSONL format)
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        if not data:
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter data around the timestamp (get last 300 candles)
        target_time = pd.to_datetime(ts_ms, unit='ms')
        df = df.tail(300)  # Get last 300 candles
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch local data for {symbol_ccxt}: {e}")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

def fetch_window_ccxt(symbol_ccxt: str, tf_internal: str, ts_ms: int, mode: str = 'spot') -> pd.DataFrame:
    """Fetch a window of candles around ts_ms using ccxt/CoinEx.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    ccxt_tf = interval_alias.get(tf_internal)
    if not ccxt_tf:
        raise ValueError(f"Unsupported timeframe: {tf_internal}")
    exchange_params = {'enableRateLimit': True}
    if (mode or 'spot').lower() == 'futures':
        exchange_params['options'] = {'defaultType': 'swap'}
    exchange = ccxt.coinex(exchange_params)
    # Compute since and limit
    win_before = 300
    win_after = 20
    ms_per_bar = interval_ms_map[tf_internal]
    since = max(0, ts_ms - win_before * ms_per_bar)
    limit = min(1000, win_before + win_after + 10)
    ohlcv = exchange.fetch_ohlcv(symbol_ccxt, timeframe=ccxt_tf, since=since, limit=limit)
    if not ohlcv:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]) 
    ohlcv = sorted(ohlcv, key=lambda r: r[0])
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df


# =====================
# Pattern mapping
# =====================
pattern_checks = {
    "Golden Cross": detect_golden_cross,
    "Death Cross": detect_death_cross,
    "RSI Trend": detect_rsi_trend,
    "MACD Bullish Crossover": detect_macd_cross,
    "MACD Bearish Crossover": detect_macd_cross,
    "Bull Flag Breakout": detect_bull_flag,
    "Triangle Breakout": detect_triangle_breakout,
    "Head and Shoulders": detect_head_shoulders,
    "Double Top/Bottom": detect_double_top_bottom,
    "Bullish Engulfing Candle": detect_bullish_engulfing,
    "Hammer": detect_hammer,
    "RSI Bullish Divergence": detect_rsi_divergence,
    "RSI Bearish Divergence": detect_rsi_divergence,
    "5min Breakout": detect_scalp_breakout,
    "Pullback to EMA": detect_pullback_to_ema,
    "VWAP Bounce": detect_vwap_bounce,
    "EMA Ribbon Hold": detect_ema_ribbon_hold,
}

# Aliases
pattern_checks["RSI Cross Below 50"] = detect_rsi_trend
pattern_checks["RSI Cross Above 50"] = detect_rsi_trend
pattern_checks["Double Top Bottom"] = detect_double_top_bottom
pattern_checks["Head & Shoulders Detected"] = detect_head_shoulders

# =====================
# Indicator calculations
# =====================
def add_indicators(df):
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    # RSI14
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']  # MACD Histogram
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

def _check_multi_timeframe_confirmation_directional(symbol_ccxt: str, tf: str, ts_ms: int, mode: str, df: pd.DataFrame, side: str) -> bool:
    """
    LAYER 1: Relaxed MTF confirmation - only requires directional alignment.
    Instead of demanding full pattern match on higher timeframe, only require HTF trend direction match.
    """
    # Determine higher timeframe
    tf_map = {
        '1min': '5min', '5min': '15min', '15min': '30min',
        '30min': '1h', '1h': '4h', '4h': '1d', '1d': '1w'
    }
    higher_tf = tf_map.get(tf.lower())
    if not higher_tf:
        return True  # No higher TF available, allow trade
    
    # Fetch higher timeframe data
    try:
        from coinEx_getting_data import fetch_window_local, add_indicators
        higher_df = fetch_window_local(symbol_ccxt, higher_tf, ts_ms, mode=mode)
        if higher_df is None or len(higher_df) < 50:
            return True  # Insufficient data, allow trade
        
        add_indicators(higher_df)
        
        # Check directional alignment only (not full pattern match)
        last = higher_df.iloc[-1]
        price = float(last.get('close', 0))
        
        # Use MA50 vs MA200 for trend direction
        if 'ma50' in higher_df.columns and 'ma200' in higher_df.columns:
            ma50 = float(last.get('ma50', price))
            ma200 = float(last.get('ma200', price))
            
            # For BUY: require higher TF in uptrend (ma50 > ma200)
            if side == 'BUY':
                return ma50 > ma200
            # For SELL: require higher TF in downtrend (ma50 < ma200)
            elif side == 'SELL':
                return ma50 < ma200
            else:
                return True  # Unknown side, allow
        
        # Fallback: use EMA21 trend
        if 'ema21' in higher_df.columns:
            ema21 = float(last.get('ema21', price))
            if side == 'BUY':
                return price > ema21
            elif side == 'SELL':
                return price < ema21
        
        return True  # No indicators available, allow trade
    except Exception:
        return True  # Error, allow trade to proceed

def _check_multi_timeframe_confirmation(symbol_ccxt: str, tf: str, ts_ms: int, mode: str, df: pd.DataFrame) -> bool:
    """
    Check higher timeframe trend alignment for signal confirmation.
    
    Parameters:
    symbol_ccxt: Symbol in ccxt format
    tf: Current timeframe
    ts_ms: Timestamp in milliseconds
    mode: Trading mode (spot/futures)
    df: Current timeframe DataFrame
    
    Returns:
    bool: True if higher timeframe confirms the signal
    """
    # Map to next higher timeframe
    higher_tf_map = {
        "1min": "5min",
        "5min": "15min", 
        "15min": "1h",
        "30min": "1h",
        "1h": "4h",
        "4h": "1d"
    }
    
    higher_tf = higher_tf_map.get(tf)
    if not higher_tf:
        return True  # No higher TF available, allow signal
    
    try:
        # Fetch higher timeframe data from local files
        htf_df = fetch_window_local(symbol_ccxt, higher_tf, ts_ms, mode=mode)
        if len(htf_df) < 200:
            return True  # Not enough data, allow signal
        
        # Add indicators to higher timeframe
        add_indicators(htf_df)
        
        # OPTIMIZED: Much more lenient multi-timeframe confirmation for daily trading
        # Check trend alignment - be very permissive to allow more trades
        current_ma50 = htf_df['ma50'].iloc[-1]
        current_ma200 = htf_df['ma200'].iloc[-1]
        
        # OPTIMIZED: Allow patterns even in sideways/weak trend markets
        # Only reject if there's a strong opposite trend (MA50 significantly below MA200)
        ma_ratio = current_ma50 / current_ma200
        return ma_ratio > 0.95  # OPTIMIZED: Allow patterns if MA50 is within 5% of MA200
        
    except Exception as e:
        # If we can't fetch higher TF data, allow the signal
        return True

# =====================
# Verification loop
# =====================
def _write_verified(record, symbol: str):
    try:
        symbol_verified = PATTERN_VERIFIED / symbol
        symbol_verified.mkdir(parents=True, exist_ok=True)
        key = f"{record.get('timeframe')}|{record.get('timestamp')}|{record.get('pattern')}"
        all_file = symbol_verified / "all_patterns.jsonl"
        if all_file.exists():
            try:
                with open(all_file, 'r') as _af:
                    for line in _af:
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        k2 = f"{d.get('timeframe')}|{d.get('timestamp')}|{d.get('pattern')}"
                        if k2 == key:
                            return
            except Exception:
                pass

        tf_file = symbol_verified / f"{record.get('timeframe')}_patterns.jsonl"
        with open(tf_file, 'a') as tf_out:
            tf_out.write(json.dumps(record) + "\n")
        with open(all_file, 'a') as all_out:
            all_out.write(json.dumps(record) + "\n")
    except Exception as werr:
        print(f"⚠️ Failed writing verified record: {werr}")

import os

def verify_record(symbol: str, record: dict, debug: bool = False) -> bool:
    tf = record.get('timeframe')
    raw_pattern = record.get('pattern')
    ts = record.get('timestamp')
    pattern = raw_pattern.split(" — ")[0] if raw_pattern else None
    tf = tf.strip()

    if not tf or not pattern or not ts:
        return False
    if pattern not in pattern_checks:
        return False

    mode = (record.get('mode') or 'spot').lower()
    interval = interval_alias.get(tf)
    if not interval:
        return False

    # Determine symbol: prefer record's symbol, fallback to passed in
    rec_sym = (record.get('symbol') or symbol or 'BTC-USDT')
    symbol_ccxt = normalize_symbol_to_ccxt(rec_sym)

    try:
        ts_ms = iso_to_ms(ts)
        # Fetch data from local files
        df = fetch_window_local(symbol_ccxt, tf, ts_ms, mode=mode)

        add_indicators(df)
        
        # Determine if daily timeframe to protect daily trading flow
        tf_norm = (tf or '').strip().lower()
        is_daily = tf_norm in ('1d', 'd', '1day', 'daily')
        apply_to_daily = os.getenv('APPLY_FILTERS_TO_DAILY', '0') == '1'

        # Optional: require multi-timeframe confirmation (env flag) - skip for daily unless explicitly enabled
        if os.getenv('REQUIRE_MTF_CONFIRMATION', '0') == '1' and (apply_to_daily or not is_daily):
            if not _check_multi_timeframe_confirmation(symbol_ccxt, tf, ts_ms, mode, df):
                if debug:
                    print(f"[WARNING] Multi-timeframe confirmation failed for {pattern}")
                return False

    except Exception as e:
        if debug:
            print(f"[ERROR] Error fetching data: {e}")
        return False

    check_fn = pattern_checks[pattern]
    sig = inspect.signature(check_fn)
    try:
        if len(sig.parameters) == 1:
            result = check_fn(df)
        else:
            result = check_fn(df, ts)
    except Exception as e:
        if debug:
            print(f"[ERROR] Error in pattern function {pattern}: {e}")
        return False

    # Optional: minimum confidence gate (skip for daily unless explicitly enabled)
    min_conf = None
    try:
        min_conf = int(os.getenv('MIN_CONFIDENCE', '0'))
    except Exception:
        min_conf = 0
    if min_conf and (apply_to_daily or not is_daily) and int(record.get('confidence') or 0) < min_conf:
        if debug:
            print(f"[INFO] Confidence {record.get('confidence')} < {min_conf}, rejecting")
        return False

    ok = False
    if isinstance(result, tuple) and len(result) == 2:
        ok = result[0]
    else:
        ok = result

    if isinstance(ok, pd.Series):
        ok = ok.any()
    elif isinstance(ok, pd.DataFrame):
        ok = not ok.empty
    elif isinstance(ok, list):
        ok = len(ok) > 0
    else:
        try:
            ok = bool(ok)
        except:
            ok = False

    if debug:
        print("\n====================================")
        print(f"Pattern: {pattern}")
        print(f"Timeframe: {tf}, Timestamp: {ts}, Mode: {mode}")
        print(f"Fetching symbol: {symbol_ccxt}, interval: {interval}")
        print("Last 10 candles:")
        cols = [c for c in ["timestamp","open","high","low","close","volume","rsi","macd","signal","ema21"] if c in df.columns]
        print(df[cols].tail(10))
        print("Detection function returned:", result)
        print("Final decision:", "TRUE [OK]" if ok else "FALSE [X]")
        print("====================================\n")

    if ok:
        _write_verified(record, rec_sym)
    return ok

def verify_file(file_path):
    print(f"\n--- Checking {file_path} ---")
    with open(file_path) as f:
        for line_no, line in enumerate(f, 1):
            d = json.loads(line)
            tf, raw_pattern, ts = d.get('timeframe'), d.get('pattern'), d.get('timestamp')
            pattern = raw_pattern.split(" — ")[0]

            ok = verify_record(d.get('symbol') or 'BTC-USDT', d, debug=True)
            status = "TRUE ✅" if ok else "FALSE ❌"
            print(f"[line {line_no}] {tf} {raw_pattern} at {ts} -> {status}")

def watch_unverified(symbol: str):
    PATTERN_UNVERIFIED.mkdir(parents=True, exist_ok=True)
    PATTERN_VERIFIED.mkdir(parents=True, exist_ok=True)
    # Create symbol-specific directories
    symbol_unverified = PATTERN_UNVERIFIED / symbol
    symbol_verified = PATTERN_VERIFIED / symbol
    symbol_unverified.mkdir(parents=True, exist_ok=True)
    symbol_verified.mkdir(parents=True, exist_ok=True)
    
    positions = {}
    while True:
        for file in sorted(symbol_unverified.glob("*.jsonl")):
            if file.name == "all_patterns.jsonl":
                continue
            processed = positions.get(file, 0)
            try:
                with open(file) as f:
                    for line_no, line in enumerate(f, 1):
                        if line_no <= processed:
                            continue
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        verify_record(symbol, d)
                    positions[file] = line_no
            except Exception:
                continue
        import time as _t
        _t.sleep(1)

def main():
    PATTERN_UNVERIFIED.mkdir(parents=True, exist_ok=True)
    PATTERN_VERIFIED.mkdir(parents=True, exist_ok=True)
    for file in sorted(PATTERN_UNVERIFIED.glob("*.jsonl")):
        if file.name == "all_patterns.jsonl":
            continue
        verify_file(file)

if __name__ == "__main__":
    main()
