"""
Indicator helpers used by session and hybrid logic.

Note: Core pattern indicators live in pattern_logic.calculate_indicators.
These helpers complement that set for session-style strategies.
"""

from typing import Tuple
import pandas as pd
import numpy as np


def add_atr(df: pd.DataFrame, period: int = 14, col_name: str = "atr") -> pd.DataFrame:
    """Add ATR(period) to df if not present; returns the same df for chaining."""
    if col_name in df.columns:
        return df
    if not {"high", "low", "close"}.issubset(df.columns):
        return df
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].shift(1).astype(float)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    df[col_name] = tr.rolling(period).mean()
    return df


def add_vwap(df: pd.DataFrame, col_name: str = "vwap") -> pd.DataFrame:
    """Add VWAP to df if not present; returns the same df for chaining."""
    if col_name in df.columns:
        return df
    if not {"close", "volume"}.issubset(df.columns):
        return df
    price = df["close"].astype(float)
    vol = df["volume"].astype(float)
    cum_vol = vol.cumsum().replace(0, np.nan)
    cum_pv = (price * vol).cumsum()
    df[col_name] = (cum_pv / cum_vol).bfill().ffill()
    return df


def rolling_high_low(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
    """Return (high, low) over the most recent window bars."""
    if len(df) < max(1, window) or not {"high", "low"}.issubset(df.columns):
        return (np.nan, np.nan)
    recent = df.iloc[-window:]
    return float(recent["high"].max()), float(recent["low"].min())


def volume_change_rate(df: pd.DataFrame, lookback: int = 20) -> float:
    """Return current_volume / average_volume over lookback window (ratio)."""
    if len(df) < lookback + 1 or "volume" not in df.columns:
        return float("nan")
    avg = df["volume"].rolling(lookback).mean().iloc[-1]
    cur = float(df["volume"].iloc[-1])
    if avg is None or avg == 0:
        return float("nan")
    return float(cur / avg)



