"""
Historical Data Replay Module
==============================

Provides functionality to replay historical market data chronologically
for offline pre-training of RL models.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Deque
from collections import deque
from pathlib import Path
import json
from coinEx_getting_data import CoinExDataFetcher
from verify_patterns import normalize_symbol_to_ccxt

# Cache directory (same as download_historical_data.py)
CACHE_DIR = Path("data/historical_cache")


class HistoricalDataReplay:
    """
    Replays historical market data chronologically for training.
    
    Loads historical data once and then serves it step-by-step,
    simulating real-time market conditions for offline training.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframes: List[str],
        start_date: str,
        end_date: str,
        mode: str = 'spot',
        lookback_window: int = 100,
    ):
        """
        Initialize historical data replay.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframes: List of timeframes to replay (e.g., ['1h', '4h', '1d'])
            start_date: Start date in ISO format (e.g., '2019-01-01')
            end_date: End date in ISO format (e.g., '2024-12-31')
            mode: Trading mode ('spot' or 'futures')
            lookback_window: Number of historical candles to keep in buffer
        """
        self.symbol = symbol
        self.symbol_ccxt = normalize_symbol_to_ccxt(symbol)
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        self.lookback_window = lookback_window
        
        # Validate symbol normalization
        print(f"[HistoricalReplay] Symbol: {symbol} -> {self.symbol_ccxt}")
        
        # Historical data storage: {timeframe: DataFrame}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        # Current position in replay (index into dataframes)
        self.current_indices: Dict[str, int] = {}
        
        # Buffers for current lookback window
        self.data_buffers: Dict[str, Deque[Dict]] = {}
        
        # Load all historical data
        self._load_historical_data()
    
    def _load_from_cache(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Try to load data from cache first."""
        cache_path = CACHE_DIR / f"{self.symbol.replace('/', '_')}_{timeframe}.csv"
        
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_csv(cache_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            
            if df.empty:
                return None
            
            # Filter to requested date range, but be flexible
            start_dt = pd.to_datetime(self.start_date, utc=True)
            end_dt = pd.to_datetime(self.end_date, utc=True)
            
            # Get actual date range of cached data
            cache_start = df['timestamp'].min()
            cache_end = df['timestamp'].max()
            
            # Use cached data if it overlaps with requested range
            # Allow some flexibility - use data if it's within 30 days of requested range
            overlap_start = max(start_dt, cache_start)
            overlap_end = min(end_dt, cache_end)
            
            if overlap_start <= overlap_end:
                # There's overlap - filter to the overlapping range
                mask = (df['timestamp'] >= overlap_start) & (df['timestamp'] <= overlap_end)
                df_filtered = df[mask].copy()
                
                if not df_filtered.empty:
                    print(f"[HistoricalReplay] ✅ Loaded {len(df_filtered)} candles from cache for {timeframe}")
                    print(f"[HistoricalReplay]   Cache range: {cache_start} to {cache_end}")
                    print(f"[HistoricalReplay]   Using range: {overlap_start} to {overlap_end}")
                    return df_filtered
            
            # If no overlap but cache data is close (within 30 days), use it anyway
            days_before = (start_dt - cache_end).days if cache_end < start_dt else 0
            days_after = (cache_start - end_dt).days if cache_start > end_dt else 0
            
            if days_before <= 30 or days_after <= 30:
                # Close enough - use all cached data
                print(f"[HistoricalReplay] ✅ Loaded {len(df)} candles from cache for {timeframe} (close to requested range)")
                print(f"[HistoricalReplay]   Cache range: {cache_start} to {cache_end}")
                print(f"[HistoricalReplay]   Requested: {start_dt} to {end_dt}")
                return df
            
            # No overlap and not close enough
            print(f"[HistoricalReplay] ⚠️ Cache exists but no data in requested range for {timeframe}")
            print(f"[HistoricalReplay]   Cache range: {cache_start} to {cache_end}")
            print(f"[HistoricalReplay]   Requested: {start_dt} to {end_dt}")
            return None
        except Exception as e:
            print(f"[HistoricalReplay] ⚠️ Error loading from cache: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_historical_data(self):
        """Load all historical data for all timeframes."""
        print(f"[HistoricalReplay] Loading historical data for {self.symbol} from {self.start_date} to {self.end_date}")
        
        for tf in self.timeframes:
            print(f"[HistoricalReplay] Loading {tf} timeframe...")
            try:
                # Try cache first
                cached_df = self._load_from_cache(tf)
                if cached_df is not None and not cached_df.empty:
                    self.historical_data[tf] = cached_df
                    self.current_indices[tf] = 0
                    self.data_buffers[tf] = deque(maxlen=self.lookback_window)
                    
                    # Show date range
                    first_date = cached_df['timestamp'].iloc[0]
                    last_date = cached_df['timestamp'].iloc[-1]
                    print(f"[HistoricalReplay] Loaded {len(cached_df)} candles for {tf} timeframe")
                    print(f"[HistoricalReplay]   Date range: {first_date} to {last_date}")
                    continue
                
                # If not in cache, fetch from CoinEx
                print(f"[HistoricalReplay] Cache miss - fetching {tf} data from CoinEx...")
                fetcher = CoinExDataFetcher(
                    symbol=self.symbol_ccxt,
                    timeframe_internal=tf,
                    max_candles=100000,  # Large enough for historical data
                    mode=self.mode,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )
                
                # Fetch historical data
                candles = fetcher.fetch_historical_range(self.start_date, self.end_date)
                
                if not candles:
                    print(f"[HistoricalReplay] Warning: No data found for {tf}")
                    self.historical_data[tf] = pd.DataFrame()
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                
                # Validate data
                if df.empty:
                    print(f"[HistoricalReplay] Warning: No valid data found for {tf}")
                    self.historical_data[tf] = pd.DataFrame()
                    continue
                
                # Check for reasonable data range
                if len(df) < 10:
                    print(f"[HistoricalReplay] Warning: Very few candles ({len(df)}) for {tf} - CoinEx may not have data for this range")
                
                # Validate symbol matches (check prices are reasonable)
                if 'close' in df.columns:
                    sample_price = float(df['close'].iloc[-1])
                    # Basic sanity check - BTC should be in reasonable range
                    if 'BTC' in self.symbol.upper() and (sample_price < 1000 or sample_price > 200000):
                        print(f"[HistoricalReplay] Warning: Suspicious price {sample_price} for {self.symbol} - data may be incorrect")
                
                if not df.empty:
                    self.historical_data[tf] = df
                    self.current_indices[tf] = 0
                    self.data_buffers[tf] = deque(maxlen=self.lookback_window)
                    
                    # Show date range
                    first_date = df['timestamp'].iloc[0]
                    last_date = df['timestamp'].iloc[-1]
                    print(f"[HistoricalReplay] Loaded {len(df)} candles for {tf} timeframe")
                    print(f"[HistoricalReplay]   Date range: {first_date} to {last_date}")
                    
                    # Calculate expected vs actual
                    expected_days = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days
                    actual_days = (last_date - first_date).days if isinstance(last_date, pd.Timestamp) else 0
                    coverage = (actual_days / expected_days * 100) if expected_days > 0 else 0
                    print(f"[HistoricalReplay]   Coverage: {actual_days}/{expected_days} days ({coverage:.1f}%)")
                else:
                    # No data for this timeframe, skip it
                    print(f"[HistoricalReplay] Warning: No data found for {tf}")
                    self.historical_data[tf] = pd.DataFrame()
                    # Don't create buffer for timeframes with no data
                
            except Exception as e:
                print(f"[HistoricalReplay] Error loading {tf}: {e}")
                self.historical_data[tf] = pd.DataFrame()
                # Don't create buffer for timeframes with errors
        
        # Find primary timeframe (use first available)
        self.primary_timeframe = None
        for tf in self.timeframes:
            if not self.historical_data.get(tf, pd.DataFrame()).empty:
                self.primary_timeframe = tf
                break
        
        if self.primary_timeframe is None:
            raise ValueError(f"No historical data loaded for {self.symbol}")
        
        print(f"[HistoricalReplay] Primary timeframe: {self.primary_timeframe}")
        print(f"[HistoricalReplay] Total candles loaded: {sum(len(df) for df in self.historical_data.values())}")
    
    def reset(self):
        """Reset replay to beginning."""
        # Only reset timeframes that have data loaded
        for tf in list(self.data_buffers.keys()):
            self.current_indices[tf] = 0
            self.data_buffers[tf].clear()
        
        # Initialize buffers with initial data
        for tf in list(self.historical_data.keys()):
            df = self.historical_data.get(tf, pd.DataFrame())
            if not df.empty and tf in self.data_buffers:
                # Fill buffer with initial candles
                for idx in range(min(self.lookback_window, len(df))):
                    candle = df.iloc[idx].to_dict()
                    self.data_buffers[tf].append(candle)
    
    def step(self) -> bool:
        """
        Advance to next candle in primary timeframe.
        
        Returns:
            True if more data available, False if reached end
        """
        primary_df = self.historical_data.get(self.primary_timeframe, pd.DataFrame())
        if primary_df.empty:
            return False
        
        current_idx = self.current_indices.get(self.primary_timeframe, 0)
        
        if current_idx >= len(primary_df) - 1:
            return False  # Reached end
        
        # Advance primary timeframe
        current_idx += 1
        self.current_indices[self.primary_timeframe] = current_idx
        
        # Update primary buffer
        if current_idx < len(primary_df):
            new_candle = primary_df.iloc[current_idx].to_dict()
            self.data_buffers[self.primary_timeframe].append(new_candle)
        
        # Advance secondary timeframes to match primary timestamp
        primary_timestamp = primary_df.iloc[current_idx]['timestamp']
        
        # Only process timeframes that have data loaded
        for tf in list(self.historical_data.keys()):
            if tf == self.primary_timeframe:
                continue
            
            df = self.historical_data.get(tf, pd.DataFrame())
            if df.empty or tf not in self.data_buffers:
                continue
            
            # Find candles up to primary timestamp
            tf_idx = self.current_indices.get(tf, 0)
            while tf_idx < len(df) - 1:
                next_timestamp = df.iloc[tf_idx + 1]['timestamp']
                if next_timestamp <= primary_timestamp:
                    tf_idx += 1
                    new_candle = df.iloc[tf_idx].to_dict()
                    self.data_buffers[tf].append(new_candle)
                else:
                    break
            
            self.current_indices[tf] = tf_idx
        
        return True
    
    def get_current_data(self, timeframe: Optional[str] = None) -> pd.DataFrame:
        """
        Get current data buffer for a timeframe.
        
        Args:
            timeframe: Timeframe to get data for (default: primary)
        
        Returns:
            DataFrame with current lookback window
        """
        tf = timeframe or self.primary_timeframe
        buffer = self.data_buffers.get(tf, deque())
        
        if not buffer:
            return pd.DataFrame()
        
        return pd.DataFrame(list(buffer))
    
    def get_all_current_data(self) -> Dict[str, pd.DataFrame]:
        """Get current data for all timeframes that have data loaded."""
        return {
            tf: self.get_current_data(tf)
            for tf in list(self.data_buffers.keys())
        }
    
    def get_current_price(self) -> Optional[float]:
        """Get current price from primary timeframe."""
        primary_df = self.get_current_data()
        if primary_df.empty:
            return None
        
        return float(primary_df['close'].iloc[-1])
    
    def is_done(self) -> bool:
        """Check if replay has reached the end."""
        primary_df = self.historical_data.get(self.primary_timeframe, pd.DataFrame())
        if primary_df.empty:
            return True
        
        current_idx = self.current_indices.get(self.primary_timeframe, 0)
        return current_idx >= len(primary_df) - 1
    
    def get_progress(self) -> float:
        """Get replay progress (0.0 to 1.0)."""
        primary_df = self.historical_data.get(self.primary_timeframe, pd.DataFrame())
        if primary_df.empty:
            return 1.0
        
        current_idx = self.current_indices.get(self.primary_timeframe, 0)
        total = len(primary_df)
        return min(1.0, current_idx / max(1, total - 1))
