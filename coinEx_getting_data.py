import os
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Dict

import ccxt


INTERNAL_TO_CCXT_TF = {
    '1min': '1m', '3min': '3m', '5min': '5m', '15min': '15m', '30min': '30m',
    '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '12h': '12h', '1d': '1d'
}
# '8h': '8h' is not supported by CoinEx

def _format_candle(ohlcv, timeframe_internal: str, mode: str = 'spot') -> Dict:
    ts_ms, o, h, l, c, v = ohlcv[:6]
    return {
        'timestamp': datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
        'open': float(o),
        'high': float(h),
        'low': float(l),
        'close': float(c),
        'base_volume': float(v),
        'volume': float(v),
        'quote_volume': None,
        'interval': INTERNAL_TO_CCXT_TF.get(timeframe_internal, timeframe_internal),
        'mode': mode,
        'confirm': '1'
    }


class CoinExDataFetcher:
    def __init__(self, symbol: str, timeframe_internal: str, max_candles: int = 250, mode: str = 'spot', start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.symbol = symbol  # ccxt format e.g. BTC/USDT
        self.timeframe_internal = timeframe_internal
        self.timeframe = INTERNAL_TO_CCXT_TF.get(timeframe_internal, '1m')
        self.max_candles = max_candles
        self.mode = mode
        self.start_date = start_date
        self.end_date = end_date
        options = {}
        if (mode or 'spot').lower() == 'futures':
            options = {'options': {'defaultType': 'swap'}}
        # Increase timeout and enable rate limit for stability
        self.exchange = ccxt.coinex({**options, 'enableRateLimit': True, 'timeout': 20000})
        self.candles = deque(maxlen=max_candles)

    def _safe_fetch_ohlcv(self, limit: int) -> list:
        """Fetch OHLCV with retries and exponential backoff to handle ReadTimeouts."""
        import math, time
        max_retries = 5
        backoff_base = 0.5
        last_exc = None
        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
                return ohlcv or []
            except Exception as e:
                last_exc = e
                # Backoff and retry on network/timeout errors
                sleep_s = backoff_base * (2 ** attempt)
                time.sleep(min(8.0, sleep_s))
                # On last attempt, re-raise
        raise last_exc

    def update_initial(self, limit: int) -> bool:
        ohlcv = self._safe_fetch_ohlcv(limit=min(limit, self.max_candles))
        # Ensure ascending order
        ohlcv = sorted(ohlcv, key=lambda r: r[0])
        self.candles.clear()
        for r in ohlcv:
            self.candles.append(_format_candle(r, self.timeframe_internal, self.mode))
        return len(self.candles) > 0

    def get_kline_data(self, limit: int = 3):
        ohlcv = self._safe_fetch_ohlcv(limit=limit)
        ohlcv = sorted(ohlcv, key=lambda r: r[0])
        return [_format_candle(r, self.timeframe_internal, self.mode) for r in ohlcv]

    def fetch_latest_closed(self):
        candles = self.get_kline_data(limit=2)
        return candles[-2] if len(candles) >= 2 else (candles[-1] if candles else None)


class CoinExTrader:
    def __init__(self, mode: str = 'spot', api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.mode = (mode or 'spot').lower()
        api_key = api_key or os.getenv('COINEX_API_KEY', '')
        api_secret = api_secret or os.getenv('COINEX_API_SECRET', '')
        params = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        }
        if self.mode == 'futures':
            params['options'] = {'defaultType': 'swap'}
        self.exchange = ccxt.coinex(params)
        try:
            self.exchange.load_markets()
        except Exception:
            pass

    def fetch_balance(self) -> Dict:
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            return {'error': str(e)}

    def fetch_usdt_equity(self) -> float:
        bal = self.fetch_balance()
        try:
            total = float(bal.get('total', {}).get('USDT', 0) or 0)
            if total == 0:
                total = float(bal.get('free', {}).get('USDT', 0) or 0)
            return total
        except Exception:
            return 0.0

    def place_bracket(self, symbol: str, side: str, entry: float, stop: float, tp: float, size_in_quote: float):
        try:
            # Normalize symbol to ccxt format robustly
            raw = (symbol or '').strip().upper().replace(' ', '')
            # Generate candidate forms
            candidates = set()
            # Detect base/quote from common quotes
            common_quotes = ['USDT', 'USD', 'USDC', 'BTC', 'ETH']
            base = raw
            quote = None
            for q in common_quotes:
                if raw.endswith(q) and len(raw) > len(q):
                    base = raw[:-len(q)]
                    quote = q
                    break
            if quote:
                candidates.add(f"{base}/{quote}")
                candidates.add(f"{base}{quote}")
                if self.mode == 'futures' and quote == 'USDT':
                    candidates.add(f"{base}/{quote}:USDT")
            # Hyphen/underscore forms
            candidates.add(raw.replace('-', '/').replace('_', '/'))
            candidates.add(raw.replace('-', '').replace('_', ''))

            # Load and resolve against markets
            try:
                self.exchange.load_markets()
            except Exception:
                pass
            markets = getattr(self.exchange, 'markets', {}) or {}
            resolved = None
            for c in list(candidates):
                if c in markets:
                    resolved = c
                    break
            if not resolved:
                # Fallback by matching base/quote from market metadata
                for mk, md in markets.items():
                    try:
                        if base and md.get('base', '').upper() == base and (not quote or md.get('quote', '').upper() == quote):
                            if self.mode == 'futures':
                                if md.get('type') in ('swap', 'future') or md.get('swap'):
                                    resolved = mk
                                    break
                            else:
                                if md.get('spot'):
                                    resolved = mk
                                    break
                    except Exception:
                        continue
            if not resolved:
                return {'error': f'coinex does not have market symbol {symbol}'}

            m = self.exchange.market(resolved)
            # Always use resolved market symbol for API calls
            m_symbol = m['symbol']
            ticker = self.exchange.fetch_ticker(m_symbol)
            last = float(ticker.get('last') or entry)

            # Enforce CoinEx min notional rule (>= 1 USD) and amount limits/precision
            min_notional_usd = 1.0
            # Some markets expose explicit cost limits
            try:
                lim_cost = (m.get('limits') or {}).get('cost') or {}
                if lim_cost.get('min'):
                    min_notional_usd = max(min_notional_usd, float(lim_cost.get('min')))
            except Exception:
                pass

            # Ensure the quote size we use is at least the min notional (with a small buffer)
            size_quote = float(size_in_quote or 0.0)
            if size_quote < min_notional_usd:
                size_quote = min_notional_usd * 1.05  # 5% buffer to avoid rounding below

            # Convert to base amount and respect precision
            raw_amount = size_quote / max(last, 1e-12)
            try:
                amount = float(self.exchange.amount_to_precision(m_symbol, raw_amount))
            except Exception:
                amount = max(raw_amount, 0.0)

            # Respect min amount if provided by market
            try:
                lim_amt = (m.get('limits') or {}).get('amount') or {}
                min_amount = float(lim_amt.get('min')) if lim_amt.get('min') else 0.0
            except Exception:
                min_amount = 0.0
            if min_amount > 0 and amount < min_amount:
                amount = min_amount
                try:
                    amount = float(self.exchange.amount_to_precision(m_symbol, amount))
                except Exception:
                    pass

            # Final guard: if notional still below min, bump to exact min-notional rounded to precision
            if amount * last < min_notional_usd:
                min_amount_for_notional = min_notional_usd / max(last, 1e-12)
                try:
                    amount = float(self.exchange.amount_to_precision(m_symbol, min_amount_for_notional * 1.02))
                except Exception:
                    amount = max(min_amount_for_notional * 1.02, amount)

            # Prevent zero/negative
            amount = max(amount, 0.0)

            order = self.exchange.create_order(symbol=m_symbol, type='market', side=side.lower(), amount=amount)
            # SL/TP emulation not guaranteed; return entry only
            return {
                'entry_order': order,
                'amount': amount,
                'entry_price_ref': last,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def resync_time(self):
        """Resync exchange time to ensure accurate timestamps"""
        try:
            self.exchange.fetch_time()
        except Exception as e:
            print(f"Warning: Failed to resync time: {e}")

import requests
import json
import os
from time import sleep
from datetime import datetime, timezone

COINEX_API_URL = "https://api.coinex.com/v1/market/kline"

# Allowed intervals by CoinEx API
VALID_INTERVALS = {
    "1min", "5min", "15min", "30min",
    "1hour", "2hour", "4hour", "6hour", "12hour",
    "1day", "1week"
}

def format_candle(raw_candle, interval, mode="spot"):
    timestamp, o, h, l, c, volume, turnover = raw_candle
    return {
        "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "base_volume": volume,
        "volume": volume,
        "quote_volume": turnover,
        "interval": interval,
        "mode": mode
    }

def fetch_candles(pair, timeframe, limit=250):
    params = {
        "market": pair,
        "type": timeframe,
        "limit": limit
    }
    response = requests.get(COINEX_API_URL, params=params)
    data = response.json()
    if data['code'] != 0:
        raise Exception(f"API Error: {data}")
    return data['data']

def save_candles_to_file(pair, timeframe, candles):
    filename = f"{pair.replace(':', '_')}_{timeframe}.json"
    with open(filename, "w") as f:
        json.dump(candles, f, indent=4)
    print(f"Saved {len(candles)} candles to {filename}")

def load_candles_from_file(pair, timeframe):
    filename = f"{pair.replace(':', '_')}_{timeframe}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

def update_candles(pair, timeframe, mode):
    candles = load_candles_from_file(pair, timeframe)
    latest_raw = fetch_candles(pair, timeframe, limit=1)
    latest_candle = format_candle(latest_raw[0], timeframe, mode)

    if candles and latest_candle['timestamp'] == candles[-1]['timestamp']:
        return  # No new candle

    candles.append(latest_candle)
    if len(candles) > 250:
        candles = candles[-250:]
    save_candles_to_file(pair, timeframe, candles)

def main():
    pair = input("Enter trading pair (e.g., BTCUSDT): ").strip()
    timeframes_input = input(
        "Enter timeframes separated by comma (allowed: 1min,5min,15min,30min,1hour,2hour,4hour,6hour,12hour,1day,1week): "
    )
    mode = input("Enter mode (spot/futures): ").strip().lower()
    if mode not in {"spot", "futures"}:
        print("Invalid mode, defaulting to 'spot'.")
        mode = "spot"

    timeframes = [tf.strip() for tf in timeframes_input.split(",")]

    # Validate intervals
    for tf in timeframes:
        if tf not in VALID_INTERVALS:
            print(f"Invalid interval: {tf}. Skipping.")
    timeframes = [tf for tf in timeframes if tf in VALID_INTERVALS]

    if not timeframes:
        print("No valid intervals provided. Exiting.")
        return

    # Initial fetch
    for tf in timeframes:
        try:
            raw_candles = fetch_candles(pair, tf)
            candles = [format_candle(c, tf, mode) for c in raw_candles]
            save_candles_to_file(pair, tf, candles)
        except Exception as e:
            print(f"Error fetching {tf}: {e}")

    print("Initial fetch complete. Now updating in real-time...")

    while True:
        for tf in timeframes:
            try:
                update_candles(pair, tf, mode)
            except Exception as e:
                print(f"Error updating {tf}: {e}")
        sleep(60)  # Adjust based on smallest timeframe

if __name__ == "__main__":
    main()
