import requests
import os
import time
import pandas as pd
from datetime import datetime, timezone

# --- CONFIGURATION ---
# Add any pairs you want here
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]

# All 13 requested timeframes
TIMEFRAMES = [
    "1min", "3min", "5min", "15min", "30min", 
    "1hour", "2hour", "4hour", "6hour", "12hour", 
    "1day", "3day", "1week"
]

# Target: Dec 1, 2019 (CoinEx usually starts around late 2017 for BTC)
START_DATE = datetime(2019, 12, 1, tzinfo=timezone.utc)
START_TS = int(START_DATE.timestamp() * 1000) # V2 uses milliseconds

SAVE_DIR = "coinex_master_data"
BASE_URL = "https://api.coinex.com/v2/spot/kline"

# ---------------------

def fetch_klines_v2(market, interval, end_time):
    """Fetches 1000 candles ending AT or BEFORE end_time."""
    params = {
        "market": market,
        "period": interval,
        "limit": 1000,
        "end_time": end_time 
    }
    try:
        # Note: CoinEx V2 public endpoints are rate-limited. 
        # If you get 429 errors, increase the time.sleep() in main.
        response = requests.get(BASE_URL, params=params, timeout=15)
        data = response.json()
        if data.get("code") == 0:
            return data.get("data")
        else:
            print(f"\n[!] API Error: {data.get('message')}")
            return None
    except Exception as e:
        print(f"\n[!] Connection Error: {e}")
        return None

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for pair in PAIRS:
        for tf in TIMEFRAMES:
            file_path = os.path.join(SAVE_DIR, f"{pair}_{tf}.csv")
            
            # Skip if we already have the file (Optional: remove this if you want to overwrite)
            if os.path.exists(file_path):
                print(f"[-] Skipping {pair} {tf} (File already exists).")
                continue

            all_data = []
            current_end_cursor = int(time.time() * 1000) # Start from now in ms
            seen_ts = set()
            
            print(f"[*] BACKFILLING: {pair} | {tf}")
            
            while True:
                batch = fetch_klines_v2(pair, tf, current_end_cursor)
                
                if not batch or len(batch) == 0:
                    print(f"\n[!] End of data reached for {pair} {tf}.")
                    break
                
                # Check for repetition
                batch_oldest_ts = batch[0]['created_at']
                batch_latest_ts = batch[-1]['created_at']
                
                if batch_oldest_ts in seen_ts:
                    print(f"\n[!] Repeated data detected. History ends here.")
                    break
                
                seen_ts.add(batch_oldest_ts)
                all_data.extend(batch)
                
                # Move cursor back: New end_time is the oldest timestamp from the batch minus 1ms
                current_end_cursor = batch_oldest_ts - 1
                
                readable_date = datetime.fromtimestamp(batch_oldest_ts/1000).strftime('%Y-%m-%d')
                print(f"    > Progress: Back to {readable_date} | Rows: {len(all_data)}", end="\r")
                
                if batch_oldest_ts <= START_TS:
                    print(f"\n[+] Successfully reached target date!")
                    break
                
                # CoinEx V2 Rate Limit is roughly 1-2 requests per second for public K-lines
                time.sleep(0.5)

            if all_data:
                # Convert list of dicts to DataFrame
                df = pd.DataFrame(all_data)
                
                # V2 fields: created_at, open, close, high, low, volume, value
                df = df.drop_duplicates(subset=['created_at']).sort_values('created_at')
                df['readable_date'] = pd.to_datetime(df['created_at'], unit='ms')
                
                # Reorder for clean CSV
                cols = ['readable_date', 'created_at', 'open', 'high', 'low', 'close', 'volume', 'value']
                df = df[cols]
                
                df.to_csv(file_path, index=False)
                print(f"\n[SAVE] {len(df)} rows saved to {file_path}\n")

if __name__ == "__main__":
    main()