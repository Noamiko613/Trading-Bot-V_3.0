"""
Test script to verify historical data fetching works correctly.
Run this to check if CoinEx has data for your date range.
"""

import sys
from coinEx_getting_data import CoinExDataFetcher
from verify_patterns import normalize_symbol_to_ccxt

def test_historical_fetch(symbol: str, timeframe: str, start_date: str, end_date: str):
    """Test fetching historical data for a symbol."""
    print(f"\n{'='*80}")
    print(f"Testing Historical Data Fetch")
    print(f"{'='*80}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"{'='*80}\n")
    
    symbol_ccxt = normalize_symbol_to_ccxt(symbol)
    print(f"Normalized symbol: {symbol} -> {symbol_ccxt}\n")
    
    try:
        fetcher = CoinExDataFetcher(
            symbol=symbol_ccxt,
            timeframe_internal=timeframe,
            max_candles=100000,
            mode='spot',
            start_date=start_date,
            end_date=end_date,
        )
        
        print(f"Fetching historical data...")
        candles = fetcher.fetch_historical_range(start_date, end_date)
        
        print(f"\n{'='*80}")
        print(f"Results:")
        print(f"{'='*80}")
        print(f"Total candles fetched: {len(candles)}")
        
        if candles:
            import pandas as pd
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.dropna(subset=['timestamp']).sort_values('timestamp')
            
            print(f"First candle: {df['timestamp'].iloc[0]}")
            print(f"Last candle: {df['timestamp'].iloc[-1]}")
            print(f"First price: ${df['close'].iloc[0]:.2f}")
            print(f"Last price: ${df['close'].iloc[-1]:.2f}")
            
            # Calculate expected vs actual
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            expected_days = (end_dt - start_dt).days
            
            first_ts = df['timestamp'].iloc[0]
            last_ts = df['timestamp'].iloc[-1]
            actual_days = (last_ts - first_ts).days if hasattr(last_ts, 'days') else 0
            
            print(f"\nExpected date range: {expected_days} days")
            print(f"Actual date range: {actual_days} days")
            print(f"Coverage: {actual_days/expected_days*100:.1f}%")
            
            # Check for gaps
            if len(df) > 1:
                df['time_diff'] = df['timestamp'].diff()
                avg_diff = df['time_diff'].mean()
                print(f"\nAverage time between candles: {avg_diff}")
        else:
            print("No candles fetched!")
            print("\nPossible reasons:")
            print("1. CoinEx doesn't have data for this date range")
            print("2. Symbol format is incorrect")
            print("3. Network/API error")
            print("\nTry:")
            print("- Using a more recent date range (e.g., 2020-2024)")
            print("- Checking if the symbol exists on CoinEx")
            print("- Verifying your internet connection")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with BTCUSDT
    test_historical_fetch(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date="2020-01-01",
        end_date="2024-12-31"
    )
    
    # Test with recent data to see if API works
    print("\n\n" + "="*80)
    print("Testing with recent data (last 30 days)...")
    print("="*80)
    
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    test_historical_fetch(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date=start_date,
        end_date=end_date
    )
