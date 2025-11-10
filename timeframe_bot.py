import asyncio
import json
import os
import time
from collections import deque
from datetime import datetime, timedelta
from multiprocessing import Queue
from typing import Dict, Optional
import pandas as pd
from pandas.tseries.frequencies import to_offset

from pattern_logic import calculate_indicators, decide_trade_signals
from core.session_logic import evaluate_session_signals
from confluence_engine import ConfluenceEngine
from coinEx_getting_data import CoinExDataFetcher
from session_manager import SessionManager


class CandleStore:
    """In-process dictionary of deques keyed by symbol/timeframe."""
    def __init__(self):
        self._buffers: Dict[str, deque] = {}

    def _key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}:{timeframe}"

    def get_buffer(self, symbol: str, timeframe: str, maxlen: int) -> deque:
        key = self._key(symbol, timeframe)
        buf = self._buffers.get(key)
        # Create or reset buffer if maxlen changed
        if buf is None or buf.maxlen != maxlen:
            buf = deque(maxlen=maxlen)
            self._buffers[key] = buf
        return buf


# Module-level store (per process)
CANDLE_STORE = CandleStore()

class TimeframeBot:
    """
    Individual bot that monitors a specific timeframe.
    Runs as a separate process and writes candle data to a rolling JSON file.
    """
    
    # Using Binance REST for historical backfill and live candle-close updates ???????
    
    def __init__(self, symbol: str, timeframe: str, data_file: str, signal_queue: Queue, status_dict=None, mode: str = "spot", moderation_mode: str = "balanced", trading_mode: str = "balanced", operating_mode: str = "hybrid"):  # ADJUSTED: Added trading_mode parameter; NEW: operating_mode
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_file = data_file
        self.signal_queue = signal_queue
        self.status_dict = status_dict
        self.mode = (mode or "spot").lower()
        self.moderation_mode = moderation_mode or "balanced"  # ADJUSTED: Store moderation mode
        self.trading_mode = trading_mode or "balanced"  # ADJUSTED: Store trading mode for session awareness
        self.operating_mode = (operating_mode or "hybrid").lower()  # NEW: Strategy operating mode
        
        # Data management - Increased to 500 to allow proper pattern detection with MA200
        self.max_candles = 500  # Increased to allow pattern detection (needs 200+ candles for MA200)
        # Use deque-based buffer from shared in-process store
        self.candles = CANDLE_STORE.get_buffer(self.symbol, self.timeframe, self.max_candles)
        self.last_candle_time = None
        
        # Integrity checking
        self.integrity_check_counter = 0
        self.integrity_check_interval = 10  # Check every 10 candles
         
        # Signal tracking
        self.previous_signals = set()
        self.signal_cooldown = {}
        self.cooldown_period = 120  # 2-minute cooldown
        
        # Connection management
        self.max_retries = 5
        self.retry_count = 0
        # CoinEx fetcher via ccxt; use symbols like BTC/USDT
        from verify_patterns import normalize_symbol_to_ccxt
        symbol_ccxt = normalize_symbol_to_ccxt(self.symbol)
        self._coinex_fetcher = CoinExDataFetcher(
            symbol=symbol_ccxt,
            timeframe_internal=self.timeframe,
            max_candles=self.max_candles,
            mode=self.mode
        )
        # Map internal timeframe to minute/hour durations for polling
        self._interval_seconds_map = {
            '1min': 60, '3min': 180, '5min': 300, '15min': 900, '30min': 1800,
            '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '12h': 43200, '1d': 86400
        }
        # '8h': 28800 is not supported by CoinEx
        # Confluence engine instance with moderation mode configuration
        self._confluence_engine = ConfluenceEngine(self._get_moderation_config())  # ADJUSTED: Pass moderation config
        
        # Session manager for session-aware trading
        self._session_manager = SessionManager(self.trading_mode)

    def _get_moderation_config(self) -> dict:
        """ADJUSTED: Get configuration based on moderation mode to adjust pattern detection sensitivity"""
        if self.moderation_mode == "strict":
            # Original strict settings - very conservative
            return {
                "SWING_MERGE_TOL_PCT": 0.6,      # Keep original strict tolerance
                "FIB_CONFLUENCE_TOL_PCT": 0.6,   # Keep original strict tolerance
                "VOL_MULTIPLIER_WEAK": 1.15,     # Keep original strict volume requirement
                "VOL_MULTIPLIER_STRONG": 1.5,    # Keep original strict volume requirement
                "ZONE_TOUCH_TOL_PCT": 0.6,       # Keep original strict zone touch tolerance
                "FIB_WEAK_TOL_PCT": 1.2,         # Keep original strict fib tolerance
                "FIRST_TEST_VETO": True,         # Keep original strict first test veto
                "FIRST_TEST_REQUIRED_VOLUME_MULT": 2.0,  # Keep original strict first test volume
            }
        elif self.moderation_mode == "aggressive":
            # More lenient settings - more patterns but higher false positive risk
            return {
                "SWING_MERGE_TOL_PCT": 1.5,      # ADJUSTED: Increased from 0.6% to 1.5% for more zone merging
                "FIB_CONFLUENCE_TOL_PCT": 1.5,   # ADJUSTED: Increased from 0.6% to 1.5% for more fib confluence
                "VOL_MULTIPLIER_WEAK": 0.8,      # ADJUSTED: Decreased from 1.15 to 0.8 for quieter markets
                "VOL_MULTIPLIER_STRONG": 1.2,    # ADJUSTED: Decreased from 1.5 to 1.2 for quieter markets
                "ZONE_TOUCH_TOL_PCT": 1.5,       # ADJUSTED: Increased from 0.6% to 1.5% for crypto volatility
                "FIB_WEAK_TOL_PCT": 2.5,         # ADJUSTED: Increased from 1.2% to 2.5% for crypto volatility
                "FIRST_TEST_VETO": False,        # ADJUSTED: Disabled first test veto for more patterns
                "FIRST_TEST_REQUIRED_VOLUME_MULT": 1.0,  # ADJUSTED: Reduced from 2.0 to 1.0 for more patterns
            }
        else:  # balanced (default)
            # Moderate settings - good balance between patterns and quality
            return {
                "SWING_MERGE_TOL_PCT": 1.0,      # ADJUSTED: Increased from 0.6% to 1.0% for better zone merging
                "FIB_CONFLUENCE_TOL_PCT": 1.0,   # ADJUSTED: Increased from 0.6% to 1.0% for better fib confluence
                "VOL_MULTIPLIER_WEAK": 1.0,      # ADJUSTED: Decreased from 1.15 to 1.0 for quieter markets
                "VOL_MULTIPLIER_STRONG": 1.3,    # ADJUSTED: Decreased from 1.5 to 1.3 for quieter markets
                "ZONE_TOUCH_TOL_PCT": 1.0,       # ADJUSTED: Increased from 0.6% to 1.0% for crypto volatility
                "FIB_WEAK_TOL_PCT": 1.8,         # ADJUSTED: Increased from 1.2% to 1.8% for crypto volatility
                "FIRST_TEST_VETO": True,         # Keep first test veto for quality
                "FIRST_TEST_REQUIRED_VOLUME_MULT": 1.5,  # ADJUSTED: Reduced from 2.0 to 1.5 for more patterns
            }

    def _derive_base_asset_folder_for_path(self) -> str:
        """Return lowercase base-asset folder name based on symbol (BTC-USDT -> btc)."""
        s = (self.symbol or "").strip()
        if not s:
            return "unknown"
        for sep in ["-", "/", "_"]:
            if sep in s:
                parts = s.split(sep)
                if parts:
                    return parts[0].lower()
        u = s.upper()
        common_quotes = [
            "USDT", "USD", "USDC", "BTC", "ETH", "BUSD", "TUSD", "EUR", "GBP", "JPY"
        ]
        for q in common_quotes:
            if u.endswith(q) and len(u) > len(q):
                base = u[: -len(q)]
                return base.lower()
        return s.lower()
        
    def remove_duplicates_in_file(self):
        """Remove duplicate rows in the data file based on timestamp, keeping only the latest entry for each timestamp."""
        if not os.path.exists(self.data_file):
            return
        try:
            seen = {}
            with open(self.data_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            candle = json.loads(line.strip())
                            ts = candle.get('timestamp')
                            if ts:
                                seen[ts] = candle
                        except Exception:
                            continue
            # Write back only unique candles, sorted by timestamp
            unique_candles = list(seen.values())
            unique_candles.sort(key=lambda c: c['timestamp'])
            with open(self.data_file, 'w') as f:
                for candle in unique_candles:
                    f.write(json.dumps(candle) + '\n')
            print(f"[CLEAN] {self.timeframe} bot: Removed duplicates, {len(unique_candles)} unique candles remain.")
        except Exception as e:
            print(f"[ERROR] Error removing duplicates in {self.data_file}: {e}")

    def load_existing_data(self):
        """Load existing candle data from file, create file if missing, and remove duplicates."""
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                pass  # Create empty file
        # Remove duplicates before loading
        self.remove_duplicates_in_file()
        if os.path.exists(self.data_file):
            try:
                # Repopulate deque without reassigning
                self.candles.clear()
                with open(self.data_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            candle = json.loads(line.strip())
                            self.candles.append(candle)
                if self.candles:
                    self.last_candle_time = pd.to_datetime(self.candles[-1]['timestamp'])
                print(f"[DATA] Loaded {len(self.candles)} existing candles for {self.timeframe}")
                # Check integrity of loaded data
                if len(self.candles) >= self.max_candles:
                    print(f"[CHECK] {self.timeframe} bot: Checking integrity of loaded data...")
                    self.check_data_integrity()
            except Exception as e:
                print(f"⚠️ Error loading data for {self.timeframe}: {e}")

    def save_candle(self, candle: Dict):
        """Save candle to memory buffer and file"""
        # Add timestamp if not present
        if 'timestamp' not in candle:
            candle['timestamp'] = datetime.now().isoformat()
            
        # Add to memory buffer
        self.candles.append(candle)
            
        # Write to file
        self._write_candle_to_file(candle)
        
    def _write_candle_to_file(self, candle: Dict):
        """Write a single candle to file and always keep only last max_candles entries. Also remove duplicates and check for gaps after writing."""
        try:
            with open(self.data_file, 'a') as f:
                f.write(json.dumps(candle) + '\n')
            # Always trim file to last max_candles lines
            with open(self.data_file, 'r') as f:
                lines = f.readlines()
            recent_lines = lines[-self.max_candles:]
            with open(self.data_file, 'w') as f:
                f.writelines(recent_lines)
            # Remove duplicates after writing
            self.remove_duplicates_in_file()
            # Reload candles from file for gap check
            self.candles.clear()
            with open(self.data_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.candles.append(json.loads(line.strip()))
            # Check for gaps after writing
            if len(self.candles) >= 2:
                self.check_data_integrity()
            # Debug: Show when candle is written and file trimmed
            timestamp = candle.get('timestamp', 'Unknown')
            print(f"📝 {self.timeframe} bot: Wrote completed candle to file - {timestamp} (file trimmed to last {self.max_candles}, deduped, gap-checked)")
        except Exception as e:
            print(f"[ERROR] Error writing candle to file for {self.timeframe}: {e}")
            
    def _truncate_file(self):
        """Truncate file to keep only recent data"""
        try:
            # Read last N lines and rewrite
            with open(self.data_file, 'r') as f:
                lines = f.readlines()
                
            # Keep only the last max_candles lines
            recent_lines = lines[-self.max_candles:]
            
            with open(self.data_file, 'w') as f:
                f.writelines(recent_lines)
                
        except Exception as e:
            print(f"⚠️ Error truncating file for {self.timeframe}: {e}")
            
    def log_pattern(self, pattern: str, candle: Dict, confidence: int, setup: Dict = None):
        """Log detected pattern to unverified patterns folder with detailed information"""
        log_entry = {
            'timeframe': self.timeframe,
            'symbol': self.symbol,
            'mode': self.mode,
            'pattern': pattern,
            'timestamp': candle.get('timestamp'),
            'price': candle.get('close'),
            'open': candle.get('open'),
            'high': candle.get('high'),
            'low': candle.get('low'),
            'volume': candle.get('volume'),
            'confidence': confidence,
            'detection_time': datetime.now().isoformat(),
            'setup': setup
        }
        try:
            # Create unverified patterns directory if it doesn't exist
            os.makedirs(f'patterns_unverified/{self.symbol}', exist_ok=True)
            
            # Write to timeframe-specific unverified pattern file
            pattern_file = f'patterns_unverified/{self.symbol}/{self.timeframe}_patterns.jsonl'
            with open(pattern_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Also write to general unverified pattern log
            with open(f'patterns_unverified/{self.symbol}/all_patterns.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
            print(f"📝 {self.timeframe} bot: Unverified pattern logged to {pattern_file}")
        except Exception as e:
            print(f"[ERROR] Error writing to unverified pattern log: {e}")
            
    def check_patterns(self):
        """Check for patterns in recent candles"""
        # Reduced minimum data requirement from 50 to 20 for earlier pattern detection
        if len(self.candles) < 20:  # Need minimum data for basic patterns
            return
            
        # Check data integrity before pattern detection
        if not self.check_data_integrity():
            print(f"⚠️ {self.timeframe} bot: Skipping pattern detection due to data integrity issues")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Get structured setups from existing rulebook
        setups = decide_trade_signals(df)

        # Augment with confluence engine signals: treat HOLD as no-op, BUY/SELL as a setup
        try:
            # Build higher timeframe inputs for confluence engine using local files if available
            higher_tfs: Dict[str, pd.DataFrame] = {}
            # Attempt to load sibling timeframe files for context (1h, 4h preferred)
            # Files are maintained in data/{base_asset}/{tf}.json with ISO timestamps per line
            desired_context = ["1h", "4h"]
            for tf_ctx in desired_context:
                ctx_path = f"data/{self._derive_base_asset_folder_for_path()}/{tf_ctx}.json"
                if os.path.exists(ctx_path):
                    ctx_rows = []
                    with open(ctx_path, 'r') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                ctx_rows.append(json.loads(line))
                            except Exception:
                                continue
                    if ctx_rows:
                        ctx_df = pd.DataFrame(ctx_rows)
                        if 'timestamp' in ctx_df.columns:
                            ctx_df['timestamp'] = pd.to_datetime(ctx_df['timestamp'])
                            ctx_df = ctx_df.sort_values('timestamp')
                            ctx_df.set_index('timestamp', inplace=True)
                        higher_tfs[tf_ctx] = ctx_df.tail(300)

            # Prepare entry timeframe dataframe for engine
            entry_df = df.copy()
            entry_df = entry_df.reset_index()
            if 'timestamp' in entry_df.columns:
                entry_df.set_index('timestamp', inplace=True)

            res = self._confluence_engine.generate_signal(entry_tf_df=entry_df[[c for c in ["open","high","low","close","volume"] if c in entry_df.columns]],
                                                          higher_tfs=higher_tfs)
            action = (res or {}).get('action')
            if action in ("BUY", "SELL"):
                last_close = float(entry_df['close'].iloc[-1])
                sl = float(res.get('sl') or last_close)
                tp = float(res.get('tp') or last_close)
                # Calculate actual RR from final values
                actual_risk = abs(last_close - sl)
                if action == 'BUY':
                    actual_reward = abs(tp - last_close)
                else:
                    actual_reward = abs(last_close - tp)
                rr = (actual_reward / actual_risk) if actual_risk > 0 else 2.0
                setups.append({
                    'name': 'Confluence Engine',
                    'side': action,
                    'entry': round(last_close, 8),
                    'stop': round(sl, 8),
                    'tp': round(tp, 8),
                    'rr': round(float(max(0.1, rr)), 2),  # Use actual calculated RR
                    'risk_pct': 0.25
                })
        except Exception as _:
            # Fail-closed: if cannot compute confluence, continue with existing setups
            pass

        # Add session-based daily trading setups (e.g., overlap momentum, VWAP bounce)
        try:
            session_info = self._session_manager.get_session_info()
            session_setups = evaluate_session_signals({
                'df': df,
                'timeframe': self.timeframe,
                'session_info': session_info
            })
            if session_setups:
                setups.extend(session_setups)
        except Exception:
            pass

        # Apply cooldown filtering per setup name
        current_time = time.time()
        filtered_setups = []
        for setup in setups:
            name = setup.get('name', 'Unknown')
            if name in self.signal_cooldown:
                if current_time - self.signal_cooldown[name] < self.cooldown_period:
                    continue
            filtered_setups.append(setup)
            self.signal_cooldown[name] = current_time

        # Check for new setups by name+side to avoid duplicates
        new_keys = {f"{s['name']}|{s['side']}" for s in filtered_setups}
        prev_keys = set(self.previous_signals)
        new_setups = [s for s in filtered_setups if f"{s['name']}|{s['side']}" not in prev_keys]

        # Emit setups with pre-filters and session-aware filtering
        for setup in new_setups:
            if self.candles:
                completed_candle = self.candles[-2] if len(self.candles) > 1 else self.candles[-1]
                pattern_str = f"{setup['name']} — {setup['side']}"
                confidence = self._calculate_confidence(pattern_str, df)
                
                # Check for confluence and volume confirmation
                has_confluence = setup['name'] == 'Confluence Engine'
                has_volume_confirmation = self._check_volume_confirmation(df)
                
                # Pre-filters: trend alignment, reasonable volatility, minimum RR
                try:
                    last_close = float(df['close'].iloc[-1])
                    ma50 = float(df.get('ma50', pd.Series([last_close])).iloc[-1])
                    ma200 = float(df.get('ma200', pd.Series([last_close])).iloc[-1])
                    ema21_val = float(df.get('ema21', pd.Series([last_close])).iloc[-1])
                    atr_val = float(df.get('atr', pd.Series([(df['high']-df['low']).rolling(14).mean().iloc[-1]])).iloc[-1])
                    atr_pct = (atr_val / max(last_close, 1e-9)) * 100.0
                    rr_ratio = float(setup.get('rr', 2.0))
                    side = setup.get('side', 'BUY')
                    if not has_confluence:
                        # Trend alignment for non-confluence rulebook setups
                        if side == 'BUY' and not ((ma50 > ma200) or (last_close > ema21_val)):
                            continue
                        if side == 'SELL' and not ((ma50 < ma200) or (last_close < ema21_val)):
                            continue
                        # Reasonable volatility window
                        if not (0.3 <= atr_pct <= 5.0):
                            continue
                        # Enforce a slightly higher minimum RR for discretionary setups
                        if rr_ratio < 1.5:
                            continue
                except Exception:
                    pass

                # Apply session-aware trading decision
                should_trade, session_reason = self._session_manager.should_trade(
                    confidence, pattern_str, has_confluence, has_volume_confirmation
                )
                
                if should_trade:
                    # Apply session-based position sizing
                    original_risk_pct = setup.get('risk_pct', 0.25)
                    adjusted_risk_pct = self._session_manager.get_position_size_multiplier(original_risk_pct)
                    setup['risk_pct'] = adjusted_risk_pct
                    
                    # Apply session-based stop loss adjustment
                    stop_loss_multiplier = self._session_manager.get_stop_loss_multiplier()
                    if stop_loss_multiplier != 1.0:
                        entry_price = setup['entry']
                        stop_price = setup['stop']
                        tp_price = setup['tp']
                        
                        # Adjust stop loss (tighter stops in low liquidity)
                        if setup['side'] == 'BUY':
                            new_stop = stop_price + (entry_price - stop_price) * (1 - stop_loss_multiplier)
                            setup['stop'] = new_stop
                        else:
                            new_stop = stop_price - (stop_price - entry_price) * (1 - stop_loss_multiplier)
                            setup['stop'] = new_stop
                        
                        # Recalculate TP to maintain RR ratio
                        rr_ratio = setup.get('rr', 2.0)
                        if setup['side'] == 'BUY':
                            setup['tp'] = entry_price + (entry_price - new_stop) * rr_ratio
                        else:
                            setup['tp'] = entry_price - (new_stop - entry_price) * rr_ratio
                        
                        # Recalculate actual RR from final values
                        actual_risk = abs(entry_price - new_stop)
                        actual_reward = abs(setup['tp'] - entry_price) if setup['side'] == 'BUY' else abs(entry_price - setup['tp'])
                        setup['rr'] = round((actual_reward / actual_risk) if actual_risk > 0 else rr_ratio, 2)
                    
                    # Add session_cause to setup for proper logging
                    session_info = self._session_manager.get_session_info()
                    setup['session_cause'] = f"{session_info['session_type']} session"
                    setup['session_info'] = session_info
                    
                    payload = {
                        'pattern': pattern_str,
                        'price': completed_candle.get('close', 0),
                        'timestamp': completed_candle.get('timestamp'),
                        'confidence': confidence,
                        'setup': setup,
                        'session_info': session_info
                    }
                    self.signal_queue.put(payload)
                    self.log_pattern(pattern_str, completed_candle, confidence, setup)
                    
                    # Enhanced logging with session information
                    session_summary = self._session_manager.get_session_summary()
                    print(f"[SIGNAL] {self.timeframe} bot: {pattern_str} @ {setup['entry']} SL {setup['stop']} TP {setup['tp']} RR {setup['rr']} (Confidence: {confidence}%, Session: {session_info['session_type']}, Risk: {adjusted_risk_pct:.3f})")
                    print(f"[SESSION] {session_summary}")
                else:
                    # Log session-based filtering decision
                    log_msg = self._session_manager.log_session_decision(pattern_str, confidence, has_confluence, has_volume_confirmation)
                    print(f"⚠️ {self.timeframe} bot: {log_msg}")

        self.previous_signals = new_keys
        
    def _calculate_confidence(self, signal: str, df: pd.DataFrame) -> int:
        """
        Calculate confidence level using LAYER 2 weighted component system.
        Each component normalized 0-1, then weighted sum.
        """
        if len(df) < 50:
            return 40
            
        last = df.iloc[-1]
        price = float(last.get('close', 1))
        
        # LAYER 2: Component weights
        weights = {
            'pattern_structure': 0.25,
            'trend_alignment': 0.15,
            'momentum_confirmation': 0.15,
            'volume_confirmation': 0.10,
            'location_context': 0.15,
            'risk_reward_geometry': 0.10,
            'session_context': 0.05
        }
        
        components = {}
        
        # 1. Pattern structure quality (0-1)
        pattern_score = 0.0
        if "Golden Cross" in signal or "Death Cross" in signal:
            pattern_score = 0.9
        elif "MACD" in signal or "RSI" in signal:
            pattern_score = 0.7
        elif "Breakout" in signal:
            pattern_score = 0.75
        elif "Engulfing" in signal or "Hammer" in signal:
            pattern_score = 0.6
        else:
            pattern_score = 0.5
        components['pattern_structure'] = pattern_score
        
        # 2. Trend alignment (0-1) - Higher timeframe direction
        trend_score = 0.5  # Neutral default
        if 'ma50' in df.columns and 'ma200' in df.columns:
            ma50 = float(last.get('ma50', price))
            ma200 = float(last.get('ma200', price))
            if "Bullish" in signal and ma50 > ma200:
                trend_score = 1.0
            elif "Bearish" in signal and ma50 < ma200:
                trend_score = 1.0
            elif ("Bullish" in signal and ma50 < ma200) or ("Bearish" in signal and ma50 > ma200):
                trend_score = 0.3  # Counter-trend
            else:
                trend_score = 0.7  # General alignment
        components['trend_alignment'] = trend_score
        
        # 3. Momentum confirmation (0-1) - RSI/MACD agreement
        momentum_score = 0.5
        macd_ok = False
        rsi_ok = False
        
        if 'macd_hist' in df.columns:
            macd_hist = float(last.get('macd_hist', 0))
            if "Bullish" in signal and macd_hist > 0:
                macd_ok = True
                momentum_score += 0.25
            elif "Bearish" in signal and macd_hist < 0:
                macd_ok = True
                momentum_score += 0.25
            elif "MACD" in signal and abs(macd_hist) > 0.001:
                momentum_score += 0.15
        
        if 'rsi' in df.columns:
            rsi_val = float(last.get('rsi', 50))
            if "Bullish" in signal and 30 < rsi_val < 70:
                rsi_ok = True
                momentum_score += 0.25
            elif "Bearish" in signal and 30 < rsi_val < 70:
                rsi_ok = True
                momentum_score += 0.25
        
        if macd_ok and rsi_ok:
            momentum_score = 1.0
        elif macd_ok or rsi_ok:
            momentum_score = min(1.0, momentum_score)
        else:
            momentum_score = 0.5
        components['momentum_confirmation'] = momentum_score
        
        # 4. Volume confirmation (0-1)
        volume_score = 0.5
        if 'volume' in df.columns:
            recent_volume = float(df['volume'].rolling(20).mean().iloc[-1])
            current_volume = float(last.get('volume', 0))
            if recent_volume > 0:
                vol_ratio = current_volume / recent_volume
                if vol_ratio >= 1.2:
                    volume_score = 1.0
                elif vol_ratio >= 0.8:
                    volume_score = 0.8
                elif vol_ratio >= 0.6:
                    volume_score = 0.6
                else:
                    volume_score = 0.3
        components['volume_confirmation'] = volume_score
        
        # 5. Location/Context (0-1) - Near S/R, MA cluster, VWAP
        location_score = 0.5
        if 'ema21' in df.columns:
            ema21_val = float(last.get('ema21', price))
            if abs(price - ema21_val) / price < 0.01:  # Within 1% of EMA21
                location_score += 0.2
        if 'vwap' in df.columns:
            vwap_val = float(last.get('vwap', price))
            if abs(price - vwap_val) / price < 0.01:  # Within 1% of VWAP
                location_score += 0.2
        if 'ma50' in df.columns:
            ma50_val = float(last.get('ma50', price))
            if abs(price - ma50_val) / price < 0.015:  # Near MA50
                location_score += 0.1
        location_score = min(1.0, location_score)
        components['location_context'] = location_score
        
        # 6. Risk/Reward geometry (0-1) - This will be updated by setup later
        # For now, assume reasonable RR
        rr_score = 0.7  # Default reasonable
        components['risk_reward_geometry'] = rr_score
        
        # 7. Session context (0-1) - Will be adjusted by session logic
        session_score = 0.5  # Neutral, adjusted externally
        components['session_context'] = session_score
        
        # Compute weighted sum
        raw_conf = sum(weights[k] * components[k] for k in weights.keys())
        
        # Convert to percentage (0-100)
        confidence_pct = raw_conf * 100
        
        # Clamp to 0-100
        return max(0, min(100, int(confidence_pct)))
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if current volume confirms the signal - OPTIMIZED for daily trading"""
        if len(df) < 20 or 'volume' not in df.columns:
            return False
        
        last_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        # OPTIMIZED: Volume confirmation if current volume is at least 60% of average (was 80%)
        return last_volume >= avg_volume * 0.6
            
    def _backfill_from_coinex(self):
        """Fetch up to 250 recent candles from CoinEx and persist them."""
        try:
            print(f"[{self.timeframe}] 📡 Backfilling candles from CoinEx...")
            if self._coinex_fetcher.update_initial(limit=self.max_candles):
                # Replace current deque and file with fetched candles (keep only last max_candles)
                candles = list(self._coinex_fetcher.candles)[-self.max_candles:]
                self.candles.clear()
                for c in candles:
                    self.candles.append(c)
                # Rewrite file
                with open(self.data_file, 'w') as f:
                    for candle in self.candles:
                        f.write(json.dumps(candle) + '\n')
                if self.candles:
                    self.last_candle_time = pd.to_datetime(self.candles[-1]['timestamp'])
                print(f"[{self.timeframe}] [OK] Backfill complete: {len(self.candles)} candles")
                return True
            else:
                print(f"[{self.timeframe}] [ERROR] Backfill failed: could not fetch from CoinEx")
        except Exception as e:
            print(f"[{self.timeframe}] [ERROR] Backfill error: {e}")
        return False

    async def _poll_coinex_on_close(self):
        """Poll CoinEx at each candle close to append the latest closed candle."""
        interval_seconds = self._interval_seconds_map.get(self.timeframe)
        if not interval_seconds:
            if self.status_dict is not None:
                self.status_dict[self.timeframe] = {'status': 'error', 'msg': 'Unknown interval'}
            print(f"[ERROR] Unknown interval for {self.timeframe}")
            return
        while True:
            now = datetime.utcnow()
            # Compute precise boundary using epoch mod to avoid drift
            interval_seconds = self._interval_seconds_map[self.timeframe]
            now_ts = int(now.timestamp())
            remainder = now_ts % interval_seconds
            sleep_s = (interval_seconds - remainder) + 1.5  # fetch ~1.5s after close
            seconds_until_fetch = max(0.0, sleep_s)
            if self.status_dict is not None:
                self.status_dict[self.timeframe] = {
                    'status': 'waiting',
                    'msg': f'Fetching at close in {int(seconds_until_fetch)}s',
                    'last_candle': str(self.last_candle_time) if self.last_candle_time else '-',
                }
            await asyncio.sleep(seconds_until_fetch)

            try:
                # Fetch a small batch and append any missed candles (ccxt are confirmed)
                candles = self._coinex_fetcher.get_kline_data(limit=10)
                if not candles:
                    print(f"[{self.timeframe}] ⚠️ No candles returned from CoinEx at close time")
                    continue
                appended = 0
                for c in candles:
                    # Treat as confirmed
                    c_ts = pd.to_datetime(c['timestamp'])
                    if self.last_candle_time is None or c_ts > self.last_candle_time:
                        self.last_candle_time = c_ts
                        self.candles.append(c)
                        self._write_candle_to_file(c)
                        appended += 1
                if appended:
                    print(f"[{self.timeframe}] [OK] Appended {appended} closed candle(s), last {self.last_candle_time}")
                    if self.status_dict is not None:
                        self.status_dict[self.timeframe] = {
                            'status': 'connected',
                            'msg': f'Fetched {appended} candle(s)',
                            'last_candle': str(self.last_candle_time)
                        }
                    if len(self.candles) >= 20:
                        self.check_patterns()
                else:
                    print(f"[{self.timeframe}] ℹ️ No new closed candle (last {self.last_candle_time})")
            except Exception as e:
                print(f"[{self.timeframe}] [ERROR] Error during close polling: {e}")

    def run(self):
        """Main run method for the bot process"""
        print(f"[START] Starting {self.timeframe} bot for {self.symbol}")
        
        # Load existing data
        self.load_existing_data()
        
        # If insufficient data, backfill from CoinEx to 250 candles
        if len(self.candles) < self.max_candles:
            self._backfill_from_coinex()
        
        # If we have data, do an initial pattern check
        if len(self.candles) >= 20:
            print(f"[CHECK] {self.timeframe} bot: Performing initial pattern check on loaded data...")
            self.check_patterns()
        
        # Start live close polling using CoinEx via ccxt
        try:
            asyncio.run(self._poll_coinex_on_close())
        except KeyboardInterrupt:
            print(f"⚠️ {self.timeframe} bot: Received interrupt signal")
        except Exception as e:
            print(f"[ERROR] {self.timeframe} bot: Unexpected error: {e}")
        finally:
            print(f"[STOP] {self.timeframe} bot: Stopped")

    def force_pattern_check(self):
        """Force a pattern check on current data - useful for debugging"""
        print(f"[CHECK] {self.timeframe} bot: Forcing pattern check...")
        if len(self.candles) >= 20:
            self.check_patterns()
        else:
            print(f"⚠️ {self.timeframe} bot: Not enough data for pattern check (need 20+, have {len(self.candles)})")
            
    def check_data_integrity(self):
        """Check for missing candles in this bot's data file and auto-fix if needed"""
        if len(self.candles) < 2:
            return True  # Not enough data to check
            
        # Determine expected interval in minutes
        interval_map = {
            '1min': 1, '3min': 3, '5min': 5, '15min': 15, '30min': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
        }
        # '8h': 480 is not supported by CoinEx
        expected_minutes = interval_map.get(self.timeframe)
        if not expected_minutes:
            print(f"[ERROR] Unknown interval for {self.timeframe}")
            return False
            
        # Convert candles to DataFrame for easier analysis
        df = pd.DataFrame(self.candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Check for gaps
        timestamps = df['timestamp'].tolist()
        gaps_found = []
        
        for i in range(len(timestamps) - 1):
            prev = timestamps[i]
            curr = timestamps[i + 1]
            diff_minutes = int((curr - prev).total_seconds() // 60)
            
            # Allow some tolerance for timing variations
            tolerance = max(1, expected_minutes * 0.2)  # 20% tolerance
            if diff_minutes > (expected_minutes + tolerance):
                gaps_found.append((prev, curr, diff_minutes))
                
        if not gaps_found:
            print(f"[OK] {self.timeframe} bot: Data integrity check passed - no gaps found")
            return True  # No gaps found
            
        # Report gaps found
        print(f"⚠️ {self.timeframe} bot: Found {len(gaps_found)} gaps in data:")
        for prev, curr, diff in gaps_found:
            print(f"   Gap: {prev} -> {curr} ({diff} minutes)")
            
        # Try to fix gaps
        if self._fix_data_gaps(gaps_found):
            print(f"[OK] {self.timeframe} bot: Data gaps fixed successfully")
            return True
        else:
            print(f"[ERROR] {self.timeframe} bot: Failed to fix data gaps, skipping pattern detection")
            return False
            
    def _fix_data_gaps(self, gaps):
        """Try to fix data gaps by fetching missing data or truncating after gaps"""
        try:
            # For now, we'll truncate after the first gap
            # In the future, this could fetch missing data from API
            first_gap_end = gaps[0][1]  # The timestamp after the gap

            # Keep only candles from the first candle after the gap
            valid_candles = []
            for candle in list(self.candles):
                candle_time = pd.to_datetime(candle['timestamp'])
                if candle_time >= first_gap_end:
                    valid_candles.append(candle)

            # Update memory buffer without replacing deque
            self.candles.clear()
            for c in valid_candles:
                self.candles.append(c)

            # Rewrite the file with only valid data
            with open(self.data_file, 'w') as f:
                for candle in valid_candles:
                    f.write(json.dumps(candle) + '\n')

            print(f"🗑️ {self.timeframe} bot: Truncated data before gap at {first_gap_end}")
            return True

        except Exception as e:
            print(f"[ERROR] {self.timeframe} bot: Error fixing data gaps: {e}")
            return False


if __name__ == "__main__":
    # Test individual bot
    from multiprocessing import Queue
    
    queue = Queue()
    bot = TimeframeBot(
        symbol="BTCUSDT",
        timeframe="1min",
        data_file="test_1min.json",
        signal_queue=queue
    )
    bot.run() 