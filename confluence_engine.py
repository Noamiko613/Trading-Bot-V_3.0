"""confluence_engine.py

Rule-based confluence engine combining:
- Support/Resistance zones (swing-based + merge into zones)
- Fibonacci retracements on major swing
- Volume and Candlestick confirmations
- Multi-timeframe trend context

Returns structured signals (BUY/SELL/HOLD) plus SL/TP/position_size and a
features dict for logging or later ML re-ranking.

Usage example:
    import pandas as pd
    from confluence_engine import ConfluenceEngine

    engine = ConfluenceEngine()
    result = engine.generate_signal(entry_tf_df=df_15m, higher_tfs={"1h": df_1h, "4h": df_4h})
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import math
import numpy as np
import pandas as pd


# ---------------------------
# CONFIG: exact params & rules
# ---------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Swing detection - OPTIMIZED for daily trading
    "SWING_LEFT": 10,                # OPTIMIZED: Reduced from 15 to 10 for more responsive zones
    "SWING_RIGHT": 10,               # OPTIMIZED: Reduced from 15 to 10 for more responsive zones
    "SWING_MERGE_TOL_PCT": 1.5,      # OPTIMIZED: Increased to 1.5% for better zone merging
    "ZONE_MIN_TESTS": 1,             # OPTIMIZED: Reduced from 2 to 1 for more zones
    "ZONE_AGE_WEIGHT": 0.8,          # OPTIMIZED: Increased to prefer recent tests more

    # Fibonacci - OPTIMIZED for daily trading
    "FIB_LEVELS": [0.236, 0.382, 0.5, 0.618, 0.786],
    "FIB_CONFLUENCE_TOL_PCT": 1.5,   # OPTIMIZED: Increased to 1.5% for more fib confluence
    "FIB_PRIMARY": [0.382, 0.618],   # primary preferred fib levels

    # Volume - OPTIMIZED for daily trading
    "VOL_MA_LEN": 15,                # OPTIMIZED: Reduced from 20 to 15 for more responsive volume
    "VOL_MULTIPLIER_WEAK": 0.8,      # OPTIMIZED: Reduced to 0.8 for quieter markets
    "VOL_MULTIPLIER_STRONG": 1.1,    # OPTIMIZED: Reduced to 1.1 for quieter markets

    # Candlestick confirmation
    "CANDLE_CONFIRMATION": [
        "hammer",
        "bullish_engulfing",
        "morning_star",
        "big_bull_body",
        "shooting_star",
        "bearish_engulfing",
        "evening_star",
        "big_bear_body",
    ],
    "CANDLE_WICK_PCT": 0.4,          # OPTIMIZED: Reduced from 0.5 to 0.4 for more candle patterns

    # Trend context
    "EMA_SHORT": 20,
    "EMA_MID": 50,
    "EMA_LONG": 200,

    # Stops & sizing - OPTIMIZED for daily trading
    "ATR_LEN": 14,
    "SL_BUFFER_ATR_MULT": 0.3,       # OPTIMIZED: Reduced from 0.5 to 0.3 for tighter stops
    "SL_BUFFER_PCT": 0.002,          # OPTIMIZED: Reduced from 0.003 to 0.002 for tighter stops
    "RISK_PER_TRADE": 0.008,         # OPTIMIZED: Increased from 0.005 to 0.008 for more aggressive sizing
    "RISK_REWARD": 1.5,              # OPTIMIZED: Reduced from 2.0 to 1.5 for more achievable targets

    # Execution & tolerances - OPTIMIZED for daily trading
    "ZONE_TOUCH_TOL_PCT": 1.5,       # OPTIMIZED: Increased to 1.5% for crypto volatility
    "FIB_WEAK_TOL_PCT": 2.5,         # OPTIMIZED: Increased to 2.5% for crypto volatility
    "FIRST_TEST_VETO": False,        # OPTIMIZED: Disabled first test veto for more patterns
    "FIRST_TEST_REQUIRED_VOLUME_MULT": 1.0,  # OPTIMIZED: Reduced to 1.0 for more patterns

    # Safety caps
    "MAX_CONCURRENT_EXPOSURE_PCT": 0.05,  # OPTIMIZED: Increased from 0.03 to 0.05 for more exposure
    "MAX_DAILY_LOSS_PCT": 0.08,      # OPTIMIZED: Increased from 0.05 to 0.08 for more risk tolerance
}


# ---------------------------
# Helper numeric utilities
# ---------------------------
def pct(a: float, b: float) -> float:
    """Return absolute percent difference between a and b as percentage (e.g., 0.6 = 0.6%)."""
    if b == 0:
        return float("inf")
    return abs(a - b) / b * 100.0


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


# ---------------------------
# Swing pivots -> S/R zones
# ---------------------------
def detect_swings(
    df: pd.DataFrame, left: int, right: int
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Detect swing highs and lows using local windows.

    Returns:
      swing_lows: list of (index, price)
      swing_highs: list of (index, price)
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []
    for i in range(left, n - right):
        window_high = highs[i - left : i + right + 1]
        window_low = lows[i - left : i + right + 1]
        if highs[i] == window_high.max():
            swing_highs.append((i, float(highs[i])))
        if lows[i] == window_low.min():
            swing_lows.append((i, float(lows[i])))
    return swing_lows, swing_highs


def merge_pivots_to_zones(pivots: List[Tuple[int, float]], merge_tol_pct: float) -> List[Dict[str, Any]]:
    """Merge pivot levels within merge_tol_pct into compact zones."""
    if not pivots:
        return []
    pivots_sorted = sorted([p for _, p in pivots])
    zones: List[Dict[str, Any]] = []
    current: List[float] = [pivots_sorted[0]]
    for p in pivots_sorted[1:]:
        center = float(np.mean(current))
        if pct(p, center) <= merge_tol_pct:
            current.append(p)
        else:
            low, high = min(current), max(current)
            center = (low + high) / 2.0
            zones.append({"price_min": low, "price_max": high, "center": center, "tests": len(current)})
            current = [p]
    low, high = min(current), max(current)
    center = (low + high) / 2.0
    zones.append({"price_min": low, "price_max": high, "center": center, "tests": len(current)})
    return zones


# ---------------------------
# Fibonacci calculation
# ---------------------------
def fibonacci_levels_for_range(high_price: float, low_price: float, fib_levels: List[float]) -> Dict[float, float]:
    diff = high_price - low_price
    levels: Dict[float, float] = {}
    for f in fib_levels:
        levels[f] = high_price - diff * f
    return levels


def choose_major_swing_range(df: pd.DataFrame, lookback: int = 200) -> Tuple[float, float]:
    recent = df[-lookback:]
    high = float(recent["high"].max())
    low = float(recent["low"].min())
    return high, low


# ---------------------------
# Candlestick pattern heuristics
# ---------------------------
def candlestick_type(
    last_candle: pd.Series, prev_candle: Optional[pd.Series] = None, prev2: Optional[pd.Series] = None
) -> str:
    o, h, l, c, v = last_candle[["open", "high", "low", "close", "volume"]]
    body = abs(c - o)
    full = h - l if (h - l) > 0 else 1e-9
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l

    if body / full > 0.6:
        return "big_bull_body" if c > o else "big_bear_body"

    if body / full <= 0.4 and lower_wick / full >= 0.5 and c > o:
        return "hammer"
    if body / full <= 0.4 and upper_wick / full >= 0.5 and c < o:
        return "shooting_star"

    if prev_candle is not None:
        o2, h2, l2, c2, v2 = prev_candle[["open", "high", "low", "close", "volume"]]
        if c2 < o2 and c > o and (c - o) > (o2 - c2):
            return "bullish_engulfing"
        if c2 > o2 and c < o and (o - c) > (c2 - o2):
            return "bearish_engulfing"

    if prev_candle is not None and prev2 is not None:
        o1, c1 = prev2["open"], prev2["close"]
        o2, c2 = prev_candle["open"], prev_candle["close"]
        if c1 < o1 and abs(c2 - o2) / (prev_candle["high"] - prev_candle["low"] + 1e-9) < 0.25 and c > o and (c - o) > (o1 - c1):
            return "morning_star"
        if c1 > o1 and abs(c2 - o2) / (prev_candle["high"] - prev_candle["low"] + 1e-9) < 0.25 and c < o and (o - c) > (c1 - o1):
            return "evening_star"

    return "neutral"


# ---------------------------
# Volume confirmation
# ---------------------------
def volume_confirmation(df: pd.DataFrame, idx: int, vol_ma_len: int, multiplier: float) -> bool:
    if idx < 0 or idx >= len(df):
        return False
    vol_ma = df["volume"].rolling(vol_ma_len, min_periods=1).mean().iloc[idx]
    vol = float(df["volume"].iloc[idx])
    return vol >= vol_ma * multiplier


# ---------------------------
# Zone evaluation & confluence
# ---------------------------
def find_zones_and_confluences(entry_df: pd.DataFrame, higher_df: pd.DataFrame, config: Dict) -> List[Dict[str, Any]]:
    left = config["SWING_LEFT"]
    right = config["SWING_RIGHT"]
    merge_tol = config["SWING_MERGE_TOL_PCT"]
    lookback = min(len(higher_df), 200)

    swing_lows, swing_highs = detect_swings(higher_df, left, right)
    low_zones = merge_pivots_to_zones(swing_lows, merge_tol)
    high_zones = merge_pivots_to_zones(swing_highs, merge_tol)
    zones = [{"type": "support", **z} for z in low_zones] + [{"type": "resistance", **z} for z in high_zones]

    swing_high, swing_low = choose_major_swing_range(higher_df, lookback=lookback)
    fibs = fibonacci_levels_for_range(swing_high, swing_low, config["FIB_LEVELS"])

    candidates: List[Dict[str, Any]] = []
    for z in zones:
        z_center = z["center"]
        z_tests = z["tests"]
        confluences: List[Dict[str, Any]] = []
        for f_lvl, f_price in fibs.items():
            if pct(z_center, f_price) <= config["FIB_CONFLUENCE_TOL_PCT"]:
                confluences.append({"fib": f_lvl, "price": f_price, "strength": "strong"})
            elif pct(z_center, f_price) <= config["FIB_WEAK_TOL_PCT"]:
                confluences.append({"fib": f_lvl, "price": f_price, "strength": "weak"})
        candidate = {
            "zone": z,
            "confluences": confluences,
            "swing_high": float(swing_high),
            "swing_low": float(swing_low),
            "fibs": fibs,
            "zone_tests": z_tests,
        }
        candidates.append(candidate)
    return candidates


# ---------------------------
# Main Confluence Engine class
# ---------------------------
@dataclass
class Signal:
    action: str  # "BUY", "SELL", "HOLD"
    entry_price: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    position_size: Optional[float]  # as fraction of account (e.g., 0.005 = 0.5%)
    reason: str
    features: Dict[str, Any]


class ConfluenceEngine:
    def __init__(self, config: Dict | None = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df["ema_short"] = ema(df["close"], self.config["EMA_SHORT"])
        df["ema_mid"] = ema(df["close"], self.config["EMA_MID"])
        df["ema_long"] = ema(df["close"], self.config["EMA_LONG"])
        df["atr"] = atr(df, self.config["ATR_LEN"])
        df["vol_ma"] = df["volume"].rolling(self.config["VOL_MA_LEN"], min_periods=1).mean()
        return df

    def is_price_touching_zone(self, price: float, zone: Dict[str, Any], tol_pct: float) -> bool:
        center = zone["center"]
        return pct(price, center) <= tol_pct

    def compute_sl_tp_and_size(
        self, price: float, zone: Dict[str, Any], side: str, df_entry: pd.DataFrame
    ) -> Tuple[float, float, float]:
        atr_val = float(df_entry["atr"].iloc[-1])
        sl_buffer_amt = max(
            self.config["SL_BUFFER_ATR_MULT"] * atr_val,
            self.config["SL_BUFFER_PCT"] * price,
        )
        if side == "LONG":
            sl = zone["price_min"] - sl_buffer_amt
            sl = min(sl, price * (1 - 1e-6))
            risk_per_unit = price - sl
        else:
            sl = zone["price_max"] + sl_buffer_amt
            sl = max(sl, price * (1 + 1e-6))
            risk_per_unit = sl - price

        if risk_per_unit <= 0:
            position_size = 0.0
        else:
            position_size = self.config["RISK_PER_TRADE"] / (risk_per_unit / price)
            position_size = min(position_size, self.config["MAX_CONCURRENT_EXPOSURE_PCT"])
        tp = (
            price + (self.config["RISK_REWARD"] * (price - sl))
            if side == "LONG"
            else price - (self.config["RISK_REWARD"] * (sl - price))
        )
        return sl, tp, position_size

    def evaluate_candidate(
        self, entry_df: pd.DataFrame, higher_df: pd.DataFrame, candidate: Dict[str, Any]
    ) -> Optional[Signal]:
        config = self.config
        last_close = float(entry_df["close"].iloc[-1])
        last_idx = len(entry_df) - 1
        prev_idx = last_idx - 1
        prev2_idx = last_idx - 2 if last_idx - 2 >= 0 else None

        zone = candidate["zone"]
        side = "LONG" if zone["type"] == "support" else "SHORT"

        touching = self.is_price_touching_zone(last_close, zone, config["ZONE_TOUCH_TOL_PCT"])
        if not touching:
            return None

        fibs = candidate["confluences"]
        if not fibs:
            fib_ok = False
        else:
            fib_ok = any(
                cf["strength"] == "strong" and (cf["fib"] in config["FIB_PRIMARY"])
                for cf in fibs
            ) or any(cf["strength"] == "strong" for cf in fibs)

        last_candle = entry_df.iloc[-1]
        prev_candle = entry_df.iloc[-2] if prev_idx >= 0 else None
        prev2_candle = entry_df.iloc[-3] if prev2_idx is not None else None
        candle_type_name = candlestick_type(last_candle, prev_candle, prev2_candle)

        first_test = zone["tests"] <= 1
        if first_test and config["FIRST_TEST_VETO"]:
            vol_required = config["FIRST_TEST_REQUIRED_VOLUME_MULT"]
        else:
            vol_required = config["VOL_MULTIPLIER_WEAK"]
        vol_ok = volume_confirmation(entry_df, last_idx, config["VOL_MA_LEN"], vol_required)

        bullish_ok = False
        bearish_ok = False
        if side == "LONG":
            bullish_ok = candle_type_name in [
                "hammer",
                "bullish_engulfing",
                "morning_star",
                "big_bull_body",
            ]
            if candle_type_name == "big_bull_body" and vol_ok and (not fibs):
                bullish_ok = True
        else:
            bearish_ok = candle_type_name in [
                "shooting_star",
                "bearish_engulfing",
                "evening_star",
                "big_bear_body",
            ]
            if candle_type_name == "big_bear_body" and vol_ok and (not fibs):
                bearish_ok = True

        trend_ok = True
        try:
            hf_close = higher_df["close"].iloc[-1]
            hf_ema_long = higher_df["close"].ewm(span=config["EMA_LONG"], adjust=False).mean().iloc[-1]
            trend_ok = hf_close > hf_ema_long if side == "LONG" else hf_close < hf_ema_long
        except Exception:
            trend_ok = True

        if not vol_ok or not trend_ok:
            return None

        candle_ok = bullish_ok if side == "LONG" else bearish_ok
        special_override = False
        if not fib_ok and candle_ok and vol_ok:
            special_override = True
        if not fib_ok and not candle_ok and not special_override:
            return None

        sl, tp, size = self.compute_sl_tp_and_size(price=last_close, zone=zone, side=side, df_entry=entry_df)
        if size <= 0:
            return None

        reason = (
            f"{('LONG' if side=='LONG' else 'SHORT')} entry: zone_type={zone['type']}, "
            f"zone_center={zone['center']:.2f}, candle={candle_type_name}, vol_ok={vol_ok}, "
            f"fib_present={bool(fibs)}, first_test={first_test}"
        )
        features = {
            "zone_center": zone["center"],
            "zone_tests": zone["tests"],
            "last_close": last_close,
            "candle": candle_type_name,
            "vol_ratio": float(entry_df["volume"].iloc[-1] / (entry_df["vol_ma"].iloc[-1] + 1e-9)),
            "fib_confluences": candidate["confluences"],
            "hf_trend_ok": trend_ok,
        }
        action = "BUY" if side == "LONG" else "SELL"
        return Signal(
            action=action,
            entry_price=last_close,
            sl=sl,
            tp=tp,
            position_size=size,
            reason=reason,
            features=features,
        )

    def generate_signal(self, entry_tf_df: pd.DataFrame, higher_tfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        config = self.config
        if "4h" in higher_tfs:
            hf_df = higher_tfs["4h"]
        elif "1h" in higher_tfs:
            hf_df = higher_tfs["1h"]
        else:
            hf_df = entry_tf_df

        entry_df = self.compute_indicators(entry_tf_df)
        higher_df = self.compute_indicators(hf_df)

        candidates = find_zones_and_confluences(entry_df, higher_df, config)
        candidates_sorted = sorted(candidates, key=lambda c: c["zone"]["tests"], reverse=True)
        for cand in candidates_sorted:
            sig = self.evaluate_candidate(entry_df, higher_df, cand)
            if sig:
                return {
                    "action": sig.action,
                    "entry_price": sig.entry_price,
                    "sl": sig.sl,
                    "tp": sig.tp,
                    "position_size": sig.position_size,
                    "reason": sig.reason,
                    "features": sig.features,
                }
        return {"action": "HOLD", "reason": "No confluence found", "features": {}}


def example_run():  # pragma: no cover - illustrative only
    idx = pd.date_range("2025-01-01", periods=300, freq="15T")
    price = np.cumsum(np.random.randn(len(idx)) * 0.5) + 100.0
    high = price + np.random.rand(len(idx)) * 0.5
    low = price - np.random.rand(len(idx)) * 0.5
    openp = price + np.random.randn(len(idx)) * 0.1
    close = price + np.random.randn(len(idx)) * 0.1
    volume = (np.abs(np.random.randn(len(idx))) + 1.0) * 1000.0

    df_15 = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    df_1h = df_15.resample("1H").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    df_4h = df_1h.resample("4H").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

    engine = ConfluenceEngine()
    res = engine.generate_signal(entry_tf_df=df_15, higher_tfs={"1h": df_1h, "4h": df_4h})
    import json as _json
    print(_json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover
    example_run()


