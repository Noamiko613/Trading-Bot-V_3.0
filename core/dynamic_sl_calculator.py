"""
Dynamic Stop Loss Calculator
=============================

Calculates pair-specific minimum stop loss distances based on:
- ATR (Average True Range) for volatility
- Price level (higher prices need larger absolute SL, but smaller %)
- Volume analysis (liquidity)
- Historical volatility patterns per pair

This ensures SL is appropriate for each pair's characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class DynamicSLCalculator:
    """Calculate dynamic, pair-specific minimum stop loss distances."""
    
    # Pair-specific base parameters (can be tuned)
    PAIR_CONFIG = {
        # Even wider mins so model learns to avoid razor-thin stops
        'BTC': {'atr_multiplier': 1.8, 'min_pct': 1.8, 'max_pct': 4.0, 'price_scale': 1.0},
        'ETH': {'atr_multiplier': 2.3, 'min_pct': 1.8, 'max_pct': 4.2, 'price_scale': 1.0},
        'SOL': {'atr_multiplier': 2.5, 'min_pct': 2.0, 'max_pct': 4.8, 'price_scale': 0.8},
        'XRP': {'atr_multiplier': 2.7, 'min_pct': 2.4, 'max_pct': 5.5, 'price_scale': 0.6},
        'ADA': {'atr_multiplier': 2.6, 'min_pct': 2.1, 'max_pct': 5.0, 'price_scale': 0.7},
        'BNB': {'atr_multiplier': 2.2, 'min_pct': 1.6, 'max_pct': 4.2, 'price_scale': 0.9},
        'DOGE': {'atr_multiplier': 2.8, 'min_pct': 2.4, 'max_pct': 5.5, 'price_scale': 0.5},
        'AVAX': {'atr_multiplier': 2.3, 'min_pct': 1.8, 'max_pct': 4.6, 'price_scale': 0.8},
        'MATIC': {'atr_multiplier': 2.5, 'min_pct': 2.0, 'max_pct': 5.0, 'price_scale': 0.6},
        'LINK': {'atr_multiplier': 2.2, 'min_pct': 1.6, 'max_pct': 4.2, 'price_scale': 0.9},
    }
    
    def __init__(self):
        """Initialize calculator."""
        pass
    
    @staticmethod
    def _extract_base_asset(symbol: str) -> str:
        """Extract base asset from symbol (e.g., 'BTCUSDT' -> 'BTC')."""
        symbol = (symbol or '').upper().replace('-', '').replace('/', '')
        
        # Remove common quote currencies
        for quote in ['USDT', 'USDC', 'USD', 'BTC', 'ETH', 'BUSD']:
            if symbol.endswith(quote):
                return symbol[:-len(quote)]
        
        return symbol[:3]  # Fallback: first 3 chars
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from OHLC data."""
        if df is None or len(df) < period:
            return 0.0
        
        try:
            df = df.copy()
            
            # Ensure numeric columns
            for col in ['high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if len(df) < 2:
                return 0.0
            
            # Calculate True Range
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return float(atr) if pd.notna(atr) and atr > 0 else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def _calculate_volatility(df: pd.DataFrame, period: int = 20) -> float:
        """Calculate recent volatility (std dev of returns)."""
        if df is None or len(df) < period:
            return 0.0
        
        try:
            closes = pd.to_numeric(df['close'], errors='coerce')
            if len(closes) < 2:
                return 0.0
            
            returns = closes.pct_change().dropna()
            if len(returns) < period:
                return float(returns.std())
            
            vol = returns.tail(period).std()
            return float(vol) if pd.notna(vol) else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def _calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
        """Calculate current volume vs average (liquidity indicator)."""
        if df is None or len(df) < period:
            return 1.0
        
        try:
            volumes = pd.to_numeric(df['volume'], errors='coerce')
            if len(volumes) < period:
                return 1.0
            
            current_vol = volumes.iloc[-1]
            avg_vol = volumes.tail(period).mean()
            
            if avg_vol > 0:
                return float(current_vol / avg_vol)
            return 1.0
        except Exception:
            return 1.0
    
    def calculate_min_sl_pct(
        self,
        symbol: str,
        current_price: float,
        df: Optional[pd.DataFrame] = None,
        atr_override: Optional[float] = None
    ) -> float:
        """
        Calculate minimum stop loss percentage for a pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'SOL-USDT')
            current_price: Current market price
            df: OHLC DataFrame (optional, for ATR calculation)
            atr_override: Override ATR value (for testing)
        
        Returns:
            Minimum SL distance as percentage (e.g., 0.015 = 1.5%)
        """
        if current_price <= 0:
            return 0.02  # Default 2%
        
        base_asset = self._extract_base_asset(symbol)
        config = self.PAIR_CONFIG.get(base_asset, {
            'atr_multiplier': 2.0,
            'min_pct': 1.2,
            'max_pct': 4.0,
            'price_scale': 0.8
        })
        
        # Calculate ATR-based SL
        atr_val = atr_override
        if atr_val is None and df is not None:
            atr_val = self._calculate_atr(df)
        
        if atr_val is not None and atr_val > 0:
            # ATR-based: atr_multiplier * ATR / price
            atr_sl_pct = (config['atr_multiplier'] * atr_val) / current_price
        else:
            atr_sl_pct = 0.0
        
        # Calculate volatility adjustment
        vol_adjustment = 1.0
        if df is not None:
            vol = self._calculate_volatility(df)
            # Higher volatility -> larger SL
            vol_adjustment = 1.0 + (vol * 7.0)  # Stronger scaling
            vol_adjustment = np.clip(vol_adjustment, 1.0, 2.5)
        
        # Calculate volume adjustment (lower volume -> larger SL for safety)
        vol_ratio_adjustment = 1.0
        if df is not None:
            vol_ratio = self._calculate_volume_ratio(df)
            # Low volume -> higher SL
            if vol_ratio < 0.7:
                vol_ratio_adjustment = 1.5
            elif vol_ratio < 1.0:
                vol_ratio_adjustment = 1.2
            else:
                vol_ratio_adjustment = 1.0
        
        # Combine ATR-based with adjustments
        if atr_sl_pct > 0:
            min_sl_pct = atr_sl_pct * vol_adjustment * vol_ratio_adjustment
        else:
            # Fallback: use base min_pct with adjustments
            min_sl_pct = (config['min_pct'] / 100.0) * vol_adjustment * vol_ratio_adjustment
        
        # Apply price scaling (lower price pairs need larger % SL)
        price_scale = config['price_scale']
        if current_price < 1.0:
            min_sl_pct *= 2.0  # Very low price: increase SL more
        elif current_price < 10.0:
            min_sl_pct *= 1.4  # Low price: increase SL
        elif current_price > 10000:
            min_sl_pct *= 0.9  # Very high price: slightly tighter
        
        # Clamp to configured range
        min_pct = config['min_pct'] / 100.0
        max_pct = config['max_pct'] / 100.0
        min_sl_pct = np.clip(min_sl_pct, min_pct, max_pct)
        
        return float(min_sl_pct)
    
    def calculate_min_sl_distance(
        self,
        symbol: str,
        current_price: float,
        df: Optional[pd.DataFrame] = None,
        atr_override: Optional[float] = None
    ) -> float:
        """
        Calculate minimum stop loss distance in absolute price units.
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            df: OHLC DataFrame (optional)
            atr_override: Override ATR value
        
        Returns:
            Minimum SL distance in price units (e.g., 2.5 for $2.50)
        """
        min_sl_pct = self.calculate_min_sl_pct(symbol, current_price, df, atr_override)
        return current_price * min_sl_pct
    
    def enforce_minimum_sl(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        side: str,
        df: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Enforce minimum SL distance, adjusting if too close to entry.
        
        Args:
            symbol: Trading pair
            entry_price: Entry price
            stop_loss: Proposed stop loss price
            side: 'BUY' or 'SELL'
            df: OHLC DataFrame (optional)
        
        Returns:
            Adjusted stop loss price (may be same as input if already valid)
        """
        if entry_price <= 0:
            return stop_loss
        
        min_sl_distance = self.calculate_min_sl_distance(symbol, entry_price, df)
        
        if side == 'BUY':
            # For BUY: SL should be below entry
            min_sl = entry_price - min_sl_distance
            if stop_loss > min_sl:
                # SL is too close to entry, move it down
                return min_sl
        else:  # SELL
            # For SELL: SL should be above entry
            max_sl = entry_price + min_sl_distance
            if stop_loss < max_sl:
                # SL is too close to entry, move it up
                return max_sl
        
        return stop_loss
    
    def get_pair_config(self, symbol: str) -> Dict:
        """Get configuration for a specific pair."""
        base_asset = self._extract_base_asset(symbol)
        return self.PAIR_CONFIG.get(base_asset, {
            'atr_multiplier': 2.0,
            'min_pct': 1.2,
            'max_pct': 4.0,
            'price_scale': 0.8
        })


# Global instance
_calculator = None

def get_sl_calculator() -> DynamicSLCalculator:
    """Get global SL calculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = DynamicSLCalculator()
    return _calculator
