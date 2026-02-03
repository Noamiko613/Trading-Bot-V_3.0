"""
Historical Data Downloader
==========================

Pre-downloads and caches historical market data for RL training using CoinEx V2 API.
This script should be run before rl_training.py to ensure all data is available.

Uses CoinEx V2 API which properly supports historical data back to 2019-12.

Usage:
    python download_historical_data.py [--symbols BTCUSDT,ETHUSDT] [--start-date 2019-12-01] [--end-date 2025-12-31]
"""

import os
import json
import argparse
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

# Default symbols/timeframes for RL training (must match rl_training.RL_TRAINING_*).
# Defined here to avoid importing rl_training (which pulls in torch) when only downloading.
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "6h", "12h", "1d"]

DEFAULT_START_DATE = "2019-12-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# CoinEx V2 API endpoint
COINEX_V2_API = "https://api.coinex.com/v2/spot/kline"

# Map internal timeframes to CoinEx V2 API timeframes
TIMEFRAME_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "2h": "2hour",
    "4h": "4hour",
    "6h": "6hour",
    "12h": "12hour",
    "1d": "1day",
    "3d": "3day",
    "1w": "1week",
}

# Cache directory
CACHE_DIR = Path("data/historical_cache")
CACHE_METADATA_FILE = CACHE_DIR / "metadata.json"


class HistoricalDataDownloader:
    """Downloads and caches historical market data."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Downloader] Warning: Could not load metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"[Downloader] Warning: Could not save metadata: {e}")
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for symbol and timeframe."""
        symbol_clean = symbol.replace("/", "_")
        return self.cache_dir / f"{symbol_clean}_{timeframe}.csv"
    
    def _is_cached(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> bool:
        """Check if data is already cached. Once cached (from 2019 / requested start), use it—do not re-download."""
        cache_path = self._get_cache_path(symbol, timeframe)
        
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            return False
        
        parsed_ok = False
        start_ok = False
        
        # Verify cache has valid data and starts from 2019 (or requested start)
        try:
            df = pd.read_csv(cache_path)
            if df.empty or 'timestamp' not in df.columns:
                pass
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                df = df.dropna(subset=['timestamp'])
                if not df.empty:
                    parsed_ok = True
                    cached_start_dt = df['timestamp'].min()
                    requested_start_dt = pd.to_datetime(start_date, utc=True)
                    # Cached must start at or before requested start (e.g. 2019-12-01). Allow 30-day flexibility.
                    start_ok = cached_start_dt <= requested_start_dt or (requested_start_dt - cached_start_dt).days <= 30
                    if start_ok:
                        return True
        except Exception:
            pass
        
        # If we parsed but cache doesn't start from 2019, re-download
        if parsed_ok and not start_ok:
            return False
        
        # Fallback: metadata says we have cached data from requested start
        key = f"{symbol}_{timeframe}"
        if key in self.metadata:
            cached_start = self.metadata[key].get('start_date') or self.metadata[key].get('actual_start_date')
            if cached_start:
                try:
                    cached_start_dt = datetime.fromisoformat(str(cached_start).replace('Z', '+00:00'))
                    if cached_start_dt.tzinfo:
                        cached_start_dt = cached_start_dt.replace(tzinfo=None)
                    requested_start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    if requested_start_dt.tzinfo:
                        requested_start_dt = requested_start_dt.replace(tzinfo=None)
                    start_ok = cached_start_dt <= requested_start_dt or (requested_start_dt - cached_start_dt).days <= 30
                    if start_ok:
                        return True
                except Exception as e:
                    print(f"[Downloader] Warning: Error checking cache metadata: {e}")
        
        # Last resort: non-empty cache exists but we couldn't validate (e.g. parse error). Use it; use --force-download to re-download.
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return True
        return False
    
    def _save_to_cache(self, symbol: str, timeframe: str, candles: List[Dict], 
                       start_date: str, end_date: str):
        """Save downloaded data to cache."""
        if not candles:
            return
        
        cache_path = self._get_cache_path(symbol, timeframe)
        
        try:
            df = pd.DataFrame(candles)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    print(f"[Downloader] ⚠️ Warning: Missing column {col} in data")
            
            # Save to CSV
            df.to_csv(cache_path, index=False)
            
            # Update metadata
            key = f"{symbol}_{timeframe}"
            if key not in self.metadata:
                self.metadata[key] = {}
            
            # Get actual date range from data
            if 'timestamp' in df.columns and not df.empty:
                actual_start = df['timestamp'].min()
                actual_end = df['timestamp'].max()
                actual_start_str = actual_start.isoformat() if hasattr(actual_start, 'isoformat') else str(actual_start)
                actual_end_str = actual_end.isoformat() if hasattr(actual_end, 'isoformat') else str(actual_end)
            else:
                actual_start_str = start_date
                actual_end_str = end_date
            
            self.metadata[key].update({
                'start_date': start_date,
                'end_date': end_date,
                'actual_start_date': actual_start_str,
                'actual_end_date': actual_end_str,
                'download_date': datetime.now(timezone.utc).isoformat(),
                'candle_count': len(df),
                'cache_file': str(cache_path),
            })
            
            self._save_metadata()
            
            print(f"[Downloader] ✅ Cached {len(df)} candles to {cache_path}")
            print(f"[Downloader]   Date range: {actual_start_str} to {actual_end_str}")
        except Exception as e:
            print(f"[Downloader] ❌ Error saving cache: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_from_cache(self, symbol: str, timeframe: str) -> Optional[List[Dict]]:
        """Load data from cache."""
        cache_path = self._get_cache_path(symbol, timeframe)
        
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_csv(cache_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert back to list of dicts
            candles = df.to_dict('records')
            print(f"[Downloader] ✅ Loaded {len(candles)} candles from cache")
            return candles
        except Exception as e:
            print(f"[Downloader] ❌ Error loading cache: {e}")
            return None
    
    def _fetch_klines_v2(self, market: str, interval: str, end_time: int) -> Optional[List[Dict]]:
        """Fetch candles from CoinEx V2 API (backward-fetching method)."""
        params = {
            "market": market,
            "period": interval,
            "limit": 1000,
            "end_time": end_time
        }
        try:
            response = requests.get(COINEX_V2_API, params=params, timeout=30)
            data = response.json()
            if data.get("code") == 0:
                return data.get("data")
            else:
                print(f"[Downloader] API Error: {data.get('message', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"[Downloader] Connection Error: {e}")
            return None
    
    def download_symbol_timeframe(self, symbol: str, timeframe: str, 
                                 start_date: str, end_date: str, 
                                 use_cache: bool = True) -> List[Dict]:
        """Download or load from cache data for a symbol and timeframe using CoinEx V2 API."""
        # Check cache first
        if use_cache:
            if self._is_cached(symbol, timeframe, start_date, end_date):
                print(f"[Downloader] Using cached data for {symbol} {timeframe}")
                cached_data = self._load_from_cache(symbol, timeframe)
                if cached_data:
                    return cached_data
        
        # Convert timeframe to CoinEx V2 format
        coinex_tf = TIMEFRAME_MAP.get(timeframe, timeframe)
        
        # Download from CoinEx V2 API using backward-fetching
        print(f"[Downloader] Downloading {symbol} {timeframe} from {start_date} to {end_date}...")
        print(f"[Downloader] Using CoinEx V2 API (backward-fetching method)...")
        
        try:
            # Parse dates
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            start_ts = int(start_dt.timestamp() * 1000)  # milliseconds
            end_ts = int(end_dt.timestamp() * 1000)
            
            all_data = []
            current_end_cursor = end_ts  # Start from end date and go backward
            seen_ts = set()
            iteration = 0
            max_iterations = 10000  # Safety limit
            
            while iteration < max_iterations:
                batch = self._fetch_klines_v2(symbol, coinex_tf, current_end_cursor)
                
                if not batch or len(batch) == 0:
                    print(f"[Downloader] End of data reached for {symbol} {timeframe}")
                    break
                
                # Check for repetition (we've seen this data before)
                batch_oldest_ts = batch[0]['created_at']
                batch_latest_ts = batch[-1]['created_at']
                
                if batch_oldest_ts in seen_ts:
                    print(f"[Downloader] Repeated data detected. History ends here.")
                    break
                
                seen_ts.add(batch_oldest_ts)
                all_data.extend(batch)
                
                # Move cursor back: New end_time is the oldest timestamp from the batch minus 1ms
                current_end_cursor = batch_oldest_ts - 1
                
                readable_date = datetime.fromtimestamp(batch_oldest_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d')
                print(f"[Downloader] Progress: Back to {readable_date} | Rows: {len(all_data)}", end="\r")
                
                # Stop if we've reached the start date
                if batch_oldest_ts <= start_ts:
                    print(f"\n[Downloader] ✅ Successfully reached target date ({start_date})!")
                    break
                
                # Rate limiting (CoinEx V2 allows ~1-2 requests per second)
                time.sleep(0.5)
                iteration += 1
            
            if all_data:
                # Convert V2 format to standard format expected by historical_data_replay
                formatted_candles = []
                seen_timestamps = set()
                
                for candle in all_data:
                    # V2 format: {created_at, open, close, high, low, volume, value}
                    # Standard format: {timestamp, open, high, low, close, volume}
                    ts_ms = candle['created_at']
                    
                    # Skip duplicates
                    if ts_ms in seen_timestamps:
                        continue
                    seen_timestamps.add(ts_ms)
                    
                    # Convert timestamp to ISO format string
                    ts_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    
                    formatted_candles.append({
                        'timestamp': ts_dt.isoformat(),
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': float(candle['volume']),
                        'base_volume': float(candle['volume']),
                        'quote_volume': float(candle.get('value', 0)),
                        'interval': timeframe,
                        'confirm': '1'
                    })
                
                # Sort by timestamp
                formatted_candles.sort(key=lambda x: x['timestamp'])
                
                # Save to cache
                self._save_to_cache(symbol, timeframe, formatted_candles, start_date, end_date)
                print(f"\n[Downloader] ✅ Downloaded {len(formatted_candles)} candles for {symbol} {timeframe}")
                
                # Show date range
                if formatted_candles:
                    first_ts = formatted_candles[0]['timestamp']
                    last_ts = formatted_candles[-1]['timestamp']
                    print(f"[Downloader]   Date range: {first_ts} to {last_ts}")
                
                return formatted_candles
            else:
                print(f"[Downloader] ⚠️ No data returned for {symbol} {timeframe}")
                return []
        except Exception as e:
            print(f"[Downloader] ❌ Error downloading {symbol} {timeframe}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def download_all(self, symbols: List[str], timeframes: List[str],
                     start_date: str, end_date: str, use_cache: bool = True):
        """Download data for all symbols and timeframes."""
        print(f"\n{'='*80}")
        print(f"Historical Data Downloader")
        print(f"{'='*80}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Timeframes: {', '.join(timeframes)}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"{'='*80}\n")
        
        total_downloads = len(symbols) * len(timeframes)
        completed = 0
        failed = 0
        
        for symbol in symbols:
            print(f"\n[Downloader] Processing {symbol}...")
            for timeframe in timeframes:
                try:
                    candles = self.download_symbol_timeframe(
                        symbol, timeframe, start_date, end_date, use_cache
                    )
                    if candles:
                        completed += 1
                        print(f"[Downloader] ✅ {symbol} {timeframe}: {len(candles)} candles")
                    else:
                        failed += 1
                        print(f"[Downloader] ⚠️ {symbol} {timeframe}: No data")
                except Exception as e:
                    failed += 1
                    print(f"[Downloader] ❌ {symbol} {timeframe}: Error - {e}")
        
        print(f"\n{'='*80}")
        print(f"Download Summary")
        print(f"{'='*80}")
        print(f"Total: {total_downloads}")
        print(f"Completed: {completed}")
        print(f"Failed/No data: {failed}")
        print(f"Cache location: {self.cache_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Download and cache historical market data')
    parser.add_argument('--symbols', type=str, default=','.join(DEFAULT_SYMBOLS),
                       help=f'Comma-separated list of symbols (default: {",".join(DEFAULT_SYMBOLS)})')
    parser.add_argument('--timeframes', type=str, default=','.join(DEFAULT_TIMEFRAMES),
                       help=f'Comma-separated list of timeframes (default: {",".join(DEFAULT_TIMEFRAMES)})')
    parser.add_argument('--start-date', type=str, default=DEFAULT_START_DATE,
                       help=f'Start date in YYYY-MM-DD format (default: {DEFAULT_START_DATE})')
    parser.add_argument('--end-date', type=str, default=DEFAULT_END_DATE,
                       help=f'End date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Force re-download even if cached')
    parser.add_argument('--cache-dir', type=str, default=str(CACHE_DIR),
                       help=f'Cache directory (default: {CACHE_DIR})')
    
    args = parser.parse_args()
    
    # Parse symbols and timeframes
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    # Create downloader
    downloader = HistoricalDataDownloader(cache_dir=Path(args.cache_dir))
    
    # Download all data
    downloader.download_all(
        symbols=symbols,
        timeframes=timeframes,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=not args.no_cache
    )
    
    print("[Downloader] ✅ Download complete! You can now run rl_training.py")


if __name__ == "__main__":
    main()
