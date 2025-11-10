import os
import sys
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from multiprocessing import Process, Queue
from typing import Dict, List, Optional
import pandas as pd
import multiprocessing
import sys
import threading

# Load .env file with override to ensure paper trading mode is respected
load_dotenv(override=True)

from timeframe_bot import TimeframeBot
from simulate_trading import TradeSimulator
from verify_patterns import watch_unverified
from coinEx_getting_data import CoinExTrader
from account_ledger import GlobalAccountLedger

class BotManager:
    """
    Manages multiple timeframe bots running as separate processes.
    Each bot monitors a specific timeframe and writes to its own data file.
    """
    
    def __init__(self, symbol: str = "BTCUSDT", timeframes: List[str] = None, mode: str = "spot", 
                 leverage: int = 20, margin_mode: str = "isolated", moderation_mode: str = "balanced", trading_mode: str = "balanced", operating_mode: str = "hybrid"):  # ADJUSTED: Added trading_mode parameter; NEW: operating_mode
        self.symbol = symbol
        self.timeframes = timeframes or ["1min", "5min", "15min", "30min", "1h", "4h", "1d", "3d", "1w", "1mon"]
        self.mode = (mode or "spot").lower()
        self.leverage = leverage if self.mode == "futures" else None
        self.margin_mode = margin_mode if self.mode == "futures" else None
        self.moderation_mode = moderation_mode or "balanced"  # ADJUSTED: Store moderation mode
        self.trading_mode = trading_mode or "balanced"  # ADJUSTED: Store trading mode for session awareness
        self.operating_mode = (operating_mode or "hybrid").lower()  # NEW: Strategy operating mode
        self.bots: Dict[str, Process] = {}
        self.queues: Dict[str, Queue] = {}
        self.running = False
        # Shared status dictionary for all bots
        self.manager = multiprocessing.Manager()
        self.status_dict = self.manager.dict()
        # Create data and pattern directories if they don't exist
        os.makedirs("data", exist_ok=True)
        # Derive base asset folder from symbol and ensure subdirectory exists, e.g., data/btc
        self.base_asset_folder = self._derive_base_asset_folder(self.symbol)
        os.makedirs(os.path.join("data", self.base_asset_folder), exist_ok=True)
        os.makedirs("patterns_unverified", exist_ok=True)
        os.makedirs("patterns_verified", exist_ok=True)
        
    def start_all_bots(self):
        """Start all timeframe bots as separate processes with rate limiting"""
        print(f"[START] Starting {len(self.timeframes)} timeframe bots for {self.symbol}")
        print("=" * 60)
        
        # Rate limiting: Start bots with delays to avoid API rate limits
        # Bitunix has ~10 requests per second per IP limit
        delay_between_bots = 5.0  # 5 seconds between each bot start
        
        for i, timeframe in enumerate(self.timeframes):
            queue = Queue()
            # Pass status_dict to each bot
            bot = TimeframeBot(
                symbol=self.symbol,
                timeframe=timeframe,
                data_file=f"data/{self.base_asset_folder}/{timeframe}.json",
                signal_queue=queue,
                status_dict=self.status_dict,
                mode=self.mode,
                moderation_mode=self.moderation_mode,  # ADJUSTED: Pass moderation mode to timeframe bot
                trading_mode=self.trading_mode,  # ADJUSTED: Pass trading mode for session awareness
                operating_mode=self.operating_mode  # NEW: Pass operating mode
            )
            
            process = Process(target=bot.run)
            process.start()
            
            self.bots[timeframe] = process
            self.queues[timeframe] = queue
            
            print(f"[OK] Started {timeframe} bot (PID: {process.pid})")
            
            # Add delay between bot starts to avoid rate limiting
            if i < len(self.timeframes) - 1:  # Don't delay after the last bot
                print(f"[WAIT] Waiting {delay_between_bots}s before starting next bot...")
                time.sleep(delay_between_bots)
            
        self.running = True
        print("=" * 60)
        print("[RUNNING] All bots are now running and monitoring for patterns...")
        print("[INFO] Check individual data files in the 'data/' directory")
        print("[ALERT] Pattern alerts will appear below:")
        print("=" * 60)
        use_live = os.getenv('TRADE_LIVE', '0') == '1'
        print(f"ENV TRADE_LIVE={os.getenv('TRADE_LIVE', '0')} MODE={self.mode} SYMBOL={self.symbol}")
        if self.mode == "futures":
            print(f"[FUTURES] Parameters: Leverage={self.leverage}x, Margin Mode={self.margin_mode}")
        
        if use_live:
            self.trader = CoinExTrader(mode=self.mode)
            print("[LIVE] Live trading ENABLED (env TRADE_LIVE=1). Orders will be sent to CoinEx.")
        else:
            # Start simulator - use singleton GlobalAccountLedger so all symbols share same balance
            self.global_ledger = GlobalAccountLedger(starting_balance=100000.0)
            self.simulator = TradeSimulator(starting_balance=100000.0, mode=self.mode, symbol=self.symbol, ledger=self.global_ledger)
            # Background simulator thread
            t = threading.Thread(target=self.simulator.run_forever, kwargs={'poll_seconds': 5.0}, daemon=True)
            t.start()
            # Background global status writer
            def _ledger_status_loop():
                import time as _t
                while True:
                    try:
                        self.global_ledger.write_status()
                    except Exception:
                        pass
                    _t.sleep(2.0)
            ls = threading.Thread(target=_ledger_status_loop, daemon=True)
            ls.start()
        # Start status writer (writes real balances if live, else writes note)
        s = threading.Thread(target=self._write_live_status_loop, daemon=True)
        s.start()
        # Start background verifier to populate patterns_verified
        v = threading.Thread(target=watch_unverified, args=(self.symbol,), daemon=True)
        v.start()
        # Start verified trade feeder
        vf = threading.Thread(target=self._feed_verified_trades_loop, daemon=True)
        vf.start()
        
    def _derive_base_asset_folder(self, symbol: str) -> str:
        """Return a lowercase base-asset name for foldering given symbol formats.
        Accepts formats like BTCUSDT, BTC-USDT, BTC/USDT, btc_usdt, etc.
        """
        s = (symbol or "").strip()
        if not s:
            return "unknown"
        # Normalize separators to a single delimiter
        for sep in ["-", "/", "_"]:
            if sep in s:
                parts = s.split(sep)
                if parts:
                    return parts[0].lower()
        # No separator: try to split base from quote for common quotes
        u = s.upper()
        common_quotes = [
            "USDT", "USD", "USDC", "BTC", "ETH", "BUSD", "TUSD", "EUR", "GBP", "JPY"
        ]
        for q in common_quotes:
            if u.endswith(q) and len(u) > len(q):
                base = u[: -len(q)]
                return base.lower()
        # Fallback: whole symbol lowercased
        return s.lower()
        
    def monitor_signals(self):
        """Monitor signal queues from all bots and display alerts"""
        while self.running:
            for timeframe, queue in self.queues.items():
                try:
                    # Non-blocking check for signals
                    if not queue.empty():
                        signal = queue.get_nowait()
                        self.display_signal(timeframe, signal)
                        # Do not submit raw signals; trades are fed from verified watcher
                except:
                    pass
                    
            time.sleep(0.1)  # Small delay to prevent CPU overuse

    def _feed_verified_trades_loop(self):
        """Tail patterns_verified/all_patterns.jsonl and submit setups to simulator."""
        import time as _t
        import json as _j
        import os as _os
        verified_path = _os.path.join('patterns_verified', self.symbol, 'all_patterns.jsonl')
        processed_offsets = 0
        seen_keys = set()
        while True:
            try:
                if _os.path.exists(verified_path):
                    with open(verified_path, 'r') as f:
                        # Fast-forward to last processed position
                        for _ in range(processed_offsets):
                            next(f, None)
                        line_no = processed_offsets
                        for line in f:
                            line_no += 1
                            try:
                                rec = _j.loads(line)
                            except Exception:
                                continue
                            # Dedup by tf|ts|pattern
                            key = f"{rec.get('timeframe')}|{rec.get('timestamp')}|{rec.get('pattern')}"
                            if key in seen_keys:
                                continue
                            seen_keys.add(key)
                            # Symbol/mode gate
                            if (rec.get('symbol') or '').upper() != (self.symbol or '').upper():
                                continue
                            if (rec.get('mode') or 'spot').lower() != (self.mode or 'spot').lower():
                                continue
                            setup = rec.get('setup')
                            if not setup:
                                continue
                            # FIXED: Check that this is a verified pattern before submitting
                            pattern_name = setup.get('name', '')
                            # Skip if setup doesn't have proper pattern identification
                            if not pattern_name or pattern_name == 'Unknown':
                                continue
                            # Enforce quality gates: RR and risk cap
                            try:
                                min_rr = float(os.getenv('MIN_RR', '1.3'))
                            except Exception:
                                min_rr = 1.3
                            rr_val = float(setup.get('rr', 0.0))
                            # Skip RR gate for daily timeframe unless explicitly enabled
                            tf_norm = (rec.get('timeframe') or '').strip().lower()
                            is_daily = tf_norm in ('1d', 'd', '1day', 'daily')
                            apply_to_daily = os.getenv('APPLY_FILTERS_TO_DAILY', '1') == '1'
                            if rr_val < min_rr and (apply_to_daily or not is_daily):
                                continue
                            # Use risk percent from setup to allocate quote size (cap by env)
                            # LAYER 3: Dynamic position sizing based on confidence
                            confidence = float(setup.get('confidence', 50))
                            min_conf = float(os.getenv('MIN_CONFIDENCE', '55'))
                            base_risk = float(os.getenv('RISK_PER_TRADE_PCT', '0.002'))
                            risk_cap = float(os.getenv('MAX_RISK_PCT', '0.003'))
                            
                            # Scale risk by confidence: size_pct = base_risk * ((conf - min_conf) / (100 - min_conf))
                            if confidence >= min_conf:
                                size_pct = base_risk * ((confidence - min_conf) / (100 - min_conf))
                                risk_pct = min(size_pct, risk_cap)
                            else:
                                risk_pct = base_risk * 0.5  # Half size for below-min confidence
                            
                            # Apply session multiplier if available
                            session_mult = float(setup.get('session_multiplier', 1.0))
                            risk_pct = risk_pct * session_mult
                            
                            # Final cap
                            if risk_pct > risk_cap:
                                risk_pct = risk_cap
                            if os.getenv('TRADE_LIVE', '0') == '1':
                                # Live: compute size by risk percent of account balance
                                # Note: risk_pct is already a decimal (0.003 = 0.3%), not a percentage
                                try:
                                    from core.risk_manager import check_account_limits
                                    equity = self.trader.fetch_usdt_equity()
                                    
                                    # LAYER 6: Check account-level protections
                                    # Note: This is a simplified check - in production, track initial equity and daily PnL
                                    account_check = check_account_limits(
                                        current_equity=equity,
                                        initial_equity=equity,  # Simplified - should track from start
                                        open_trades=0,  # Simplified - should track actual open trades
                                        daily_pnl=0.0  # Simplified - should track actual daily PnL
                                    )
                                    
                                    if not account_check['can_trade']:
                                        print(f"⚠️ Account limit reached: {', '.join(account_check['reasons'])}")
                                        continue
                                    
                                    # Apply size reduction if soft DD triggered
                                    if account_check.get('action') == 'reduce_size':
                                        risk_pct = risk_pct * 0.5  # Reduce to half size
                                    
                                    size_in_quote = equity * risk_pct  # risk_pct is already decimal (0.003 = 0.3%)
                                    side = setup.get('side')
                                    print(f"[LIVE ORDER] {side} {self.symbol} size_quote={size_in_quote:.2f} entry={setup['entry']} sl={setup['stop']} tp={setup['tp']}")
                                    res = self.trader.place_bracket(self.symbol, side, setup['entry'], setup['stop'], setup['tp'], size_in_quote)
                                    
                                    try:
                                        with open(os.path.join('sim_results','orders.log'), 'a') as ol:
                                            ol.write(json.dumps({'timestamp': datetime.utcnow().isoformat(), 'event': 'live_order', 'symbol': self.symbol, 'side': side, 'size_quote': size_in_quote, 'result': res})+"\n")
                                    except Exception:
                                        pass
                                except Exception as e:
                                    print(f"❌ Live trade submit failed: {e}")
                                    try:
                                        with open(os.path.join('sim_results','orders.log'), 'a') as ol:
                                            ol.write(json.dumps({'timestamp': datetime.utcnow().isoformat(), 'event': 'error', 'symbol': self.symbol, 'error': str(e)})+"\n")
                                    except Exception:
                                        pass
                            else:
                                # Paper: submit to simulator
                                try:
                                    # Propagate timeframe from record into setup for DB persistence/display
                                    tf = rec.get('timeframe')
                                    if tf and isinstance(setup, dict):
                                        setup['timeframe'] = tf
                                except Exception:
                                    pass
                                self.simulator.submit_signal(setup)
                        processed_offsets = line_no
            except Exception:
                pass
            _t.sleep(1.0)

    def _write_live_status_loop(self):
        import json as _j
        path = os.path.join('sim_results', self.symbol, 'live_status.json')
        os.makedirs(os.path.join('sim_results', self.symbol), exist_ok=True)
        last_resync = 0
        # Write an initial placeholder immediately
        try:
            with open(path, 'w') as f:
                _j.dump({'timestamp': datetime.utcnow().isoformat(), 'mode': self.mode, 'symbol': self.symbol, 'balance_total_USDT': None, 'balance_free_USDT': None, 'error': 'initializing' }, f, indent=2)
        except Exception:
            pass
        while True:
            total_usdt = None
            free_usdt = None
            err = None
            try:
                if os.getenv('TRADE_LIVE', '0') == '1':
                    # Resync time every 30 minutes (1800 seconds)
                    current_time = time.time()
                    if current_time - last_resync > 1800:
                        try:
                            self.trader.resync_time()
                            last_resync = current_time
                        except Exception as _:
                            pass
                    try:
                        bal = self.trader.fetch_balance()
                        if isinstance(bal, dict) and 'error' in bal:
                            total_usdt = 0
                            free_usdt = 0
                            err = bal['error']
                        else:
                            total_usdt = (bal.get('total', {}) or {}).get('USDT') or 0
                            free_usdt = (bal.get('free', {}) or {}).get('USDT') or 0
                            err = None
                    except Exception as e:
                        err = f"balance_error: {e}"
                else:
                    err = 'paper_mode'
            except Exception as e:
                err = f"status_loop_error: {e}"
            finally:
                status = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'mode': self.mode,
                    'symbol': self.symbol,
                    'balance_total_USDT': total_usdt,
                    'balance_free_USDT': free_usdt,
                    'error': err
                }
                try:
                    with open(path, 'w') as f:
                        _j.dump(status, f, indent=2)
                except Exception:
                    pass
            time.sleep(3)
            
    def display_signal(self, timeframe: str, signal: dict):
        """Display a formatted signal alert"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Color coding based on signal type
        if "BUY" in signal.get('pattern', ''):
            icon = "[BUY]"
            action = "BUY"
        elif "SELL" in signal.get('pattern', ''):
            icon = "[SELL]"
            action = "SELL"
        else:
            icon = "[NEUTRAL]"
            action = "NEUTRAL"
            
        print(f"\n{icon} [{timestamp}] {timeframe.upper()} SIGNAL")
        print(f"   Pattern: {signal.get('pattern', 'Unknown')}")
        print(f"   Price: ${signal.get('price', 0):,.2f}")
        print(f"   Action: {action}")
        if signal.get('confidence'):
            print(f"   Confidence: {signal.get('confidence')}%")
        setup = signal.get('setup')
        if setup:
            print(f"   Entry: {setup.get('entry')}, SL: {setup.get('stop')}, TP: {setup.get('tp')} (RR {setup.get('rr')})")
        print("-" * 50)
        
    def stop_all_bots(self):
        """Stop all running bots"""
        print("\n[STOP] Stopping all bots...")
        
        for timeframe, process in self.bots.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                print(f"[OK] Stopped {timeframe} bot")
                
        self.running = False
        print("[DONE] All bots stopped")
        
    def get_bot_status(self) -> Dict[str, bool]:
        """Get status of all bots"""
        return {
            timeframe: process.is_alive() 
            for timeframe, process in self.bots.items()
        }
        
    def run_dashboard(self):
        try:
            while self.running:
                # Clear screen and move cursor to top
                sys.stdout.write('\033[2J\033[H')
                sys.stdout.flush()
                # Market time (UTC) and UAE time
                now_utc = datetime.utcnow()
                now_uae = now_utc + timedelta(hours=4)
                print(f"\033[1mMarket Time (UTC):\033[0m {now_utc.strftime('%Y-%m-%d %H:%M:%S')}   ", end='')
                print(f"\033[1mUAE Time (+4):\033[0m {now_uae.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Display session information
                self._display_session_info()
                
                print("\n\033[1mInterval   Status        Message                        Last Candle\033[0m")
                print("---------------------------------------------------------------")
                for tf in self.timeframes:
                    status = self.status_dict.get(tf, {})
                    st = status.get('status', 'unknown')
                    msg = status.get('msg', '-')
                    last_candle = status.get('last_candle', '-')
                    # Color for status
                    if st == 'connected':
                        color = '\033[32m'  # Green
                    elif st == 'connecting' or st == 'waiting':
                        color = '\033[33m'  # Yellow
                    elif st == 'disconnected':
                        color = '\033[31m'  # Red
                    elif st == 'error':
                        color = '\033[41m'  # Red background
                    else:
                        color = '\033[0m'   # Default
                    print(f"{tf:<10} {color}{st:<13}\033[0m {msg:<30} {last_candle}")
                print("\n(Press Ctrl+C to exit)")
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    def _display_session_info(self):
        """Display current session information"""
        try:
            from session_manager import SessionManager
            session_mgr = SessionManager(self.trading_mode)
            session_info = session_mgr.get_session_info()
            
            print(f"\n\033[1mSession Status:\033[0m {session_info['session_type'].upper()} | "
                  f"Mode: {session_info['trading_mode'].upper()} | "
                  f"Confidence Threshold: {session_info['confidence_threshold']}% | "
                  f"Position Multiplier: {session_info['position_size_multiplier']:.1f}x")
            
            # Display active sessions
            active_sessions = []
            inactive_sessions = []
            
            for session_name, session_data in session_info['session_status'].items():
                if session_data['is_active']:
                    active_sessions.append(f"\033[32m{session_data['name']}\033[0m ({session_data['status_text']})")
                else:
                    inactive_sessions.append(f"{session_data['name']} ({session_data['status_text']})")
            
            if active_sessions:
                print(f"\033[1mActive Sessions:\033[0m {' | '.join(active_sessions)}")
            
            # Show next upcoming session
            if inactive_sessions:
                print(f"\033[1mNext Session:\033[0m {inactive_sessions[0]}")
                
        except Exception as e:
            print(f"\n\033[1mSession Status:\033[0m Error loading session info: {e}")

    def run(self):
        """Main run method - starts bots and shows dashboard"""
        try:
            self.start_all_bots()
            self.running = True
            self.run_dashboard()
        except KeyboardInterrupt:
            print("\n⚠️ Received interrupt signal...")
        finally:
            self.stop_all_bots()
