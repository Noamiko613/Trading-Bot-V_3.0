#!/usr/bin/env python3
"""
Multi-Timeframe Pattern Detection Bot System
============================================

This is the new main entry point for the redesigned trading bot system.
It launches multiple timeframe bots as separate processes, each monitoring
a specific timeframe and detecting patterns independently.

Features:
- Multi-process architecture for better performance
- Rolling data storage per timeframe
- Centralized signal monitoring
- Automatic reconnection handling
- Clean shutdown on interrupt
"""

from bot_manager import BotManager
import argparse
import sys
import threading
import os
from dotenv import load_dotenv

# Load .env file with override to ensure it takes precedence over shell environment
load_dotenv(override=True)

def _load_project_env(env_path: str = ".env"):
    """Legacy function - now handled by load_dotenv above"""
    pass

def _load_project_env_old(env_path: str = ".env"):
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass

def _persist_env_vars(updates: dict, env_path: str = ".env"):
    # Read existing
    existing = {}
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        existing[k.strip()] = v.strip()
        except Exception:
            pass
    # Apply updates
    for k, v in updates.items():
        existing[k] = str(v)
        os.environ[k] = str(v)
    # Write back
    try:
        with open(env_path, 'w') as f:
            for k, v in existing.items():
                f.write(f"{k}={v}\n")
    except Exception:
        pass

def _interactive_setup() -> dict:
    """Prompt user for symbol, timeframes, live/paper, moderation mode, trading mode; ensure API keys if live.
    Returns a dict with keys: symbol, timeframes, mode, trade_live ('1' or '0'), moderation_mode, trading_mode.
    """
    _load_project_env()
    # Symbol (pair) - normalize to standard format
    from verify_patterns import normalize_symbol_to_filename
    default_symbol = os.getenv('DEFAULT_SYMBOL', 'BTC-USDT')
    sym_raw = input(f"Enter instrument (e.g., BTC-USDT, ETH-USDT, BTC/USDT) [{default_symbol}]: ").strip()
    symbol = normalize_symbol_to_filename(sym_raw or default_symbol)
    # Timeframes
    print("Enter timeframes to start (space-separated), e.g. 1min 5min 15min 30min 1h 2h 4h")
    tf_raw = input("Timeframes: ").strip()
    timeframes = [p for p in tf_raw.split() if p] or ['1min','5min', '15min', '30min', '1h', '2h', '4h']
    
    # Moderation mode selection - ADJUSTED: Added moderation mode selection
    existing_moderation = os.getenv('DEFAULT_MODERATION_MODE', 'balanced')
    print("\nPattern Detection Moderation Mode:")
    print("  strict    - Very conservative, fewer patterns but higher quality (original settings)")
    print("  balanced  - Moderate settings, good balance of patterns and quality (recommended)")
    print("  aggressive - More patterns detected, higher risk of false signals")
    moderation_resp = input(f"Select moderation mode (strict/balanced/aggressive) [{existing_moderation}]: ").strip().lower()
    if moderation_resp in ['strict', 'balanced', 'aggressive']:
        moderation_mode = moderation_resp
    else:
        moderation_mode = existing_moderation
    _persist_env_vars({'DEFAULT_MODERATION_MODE': moderation_mode})
    
    # Session-aware trading mode selection - ADJUSTED: Added trading mode selection
    existing_trading_mode = os.getenv('DEFAULT_TRADING_MODE', 'balanced')
    print("\nSession-Aware Trading Mode:")
    print("  safe      - Trade only US/London sessions, avoid first 30m of US open, skip quiet hours")
    print("  balanced  - Trade US/London, allow Asia trades with multiple confirmations (recommended)")
    print("  aggressive - Trade everything, but scale down size in weak sessions")
    trading_resp = input(f"Select trading mode (safe/balanced/aggressive) [{existing_trading_mode}]: ").strip().lower()
    if trading_resp in ['safe', 'balanced', 'aggressive']:
        trading_mode = trading_resp
    else:
        trading_mode = existing_trading_mode
    _persist_env_vars({'DEFAULT_TRADING_MODE': trading_mode})
    
    # Trading mode (spot/swap/futures)
    existing_mode = os.getenv('DEFAULT_MODE', 'spot')
    print("\nAvailable trading modes:")
    print("  spot    - Spot trading (buy/sell actual cryptocurrency)")
    print("  swap    - Perpetual swap trading (futures with leverage)")
    print("  futures - Traditional futures trading")
    mode_resp = input(f"Select trading mode (spot/swap/futures) [{existing_mode}]: ").strip().lower()
    if mode_resp in ['spot', 'swap', 'futures']:
        mode = mode_resp
    else:
        mode = existing_mode
    _persist_env_vars({'DEFAULT_MODE': mode})
    
    # Live or paper
    existing_trade_live = os.getenv('TRADE_LIVE')
    if existing_trade_live in ('0', '1'):
        use_live_default = 'yes' if existing_trade_live == '1' else 'no'
        resp = input(f"Enable LIVE trading? (yes/no) [{use_live_default}]: ").strip().lower()
        if not resp:
            trade_live = existing_trade_live
        else:
            trade_live = '1' if resp in ('y','yes') else '0'
    else:
        resp = input("Enable LIVE trading? (yes/no) [no]: ").strip().lower()
        trade_live = '1' if resp in ('y','yes') else '0'
    _persist_env_vars({'TRADE_LIVE': trade_live})
    # If live, ensure API keys
    if trade_live == '1':
        need_updates = {}
        if not os.getenv('COINEX_API_KEY'):
            need_updates['COINEX_API_KEY'] = input('Enter COINEX_API_KEY: ').strip()
        if not os.getenv('COINEX_API_SECRET'):
            need_updates['COINEX_API_SECRET'] = input('Enter COINEX_API_SECRET: ').strip()
        if need_updates:
            _persist_env_vars(need_updates)
    # Strategy operating mode selection (pattern/session/hybrid/auto)
    existing_operating_mode = os.getenv('DEFAULT_OPERATING_MODE', 'hybrid')
    print("\nOperating Mode (strategy):")
    print("  pattern - High-confidence pattern trading only")
    print("  session - Market session intraday trading")
    print("  hybrid  - Combine session + pattern signals (recommended)")
    print("  auto    - Auto-switch between modes based on conditions")
    operating_resp = input(f"Select operating mode (pattern/session/hybrid/auto) [{existing_operating_mode}]: ").strip().lower()
    if operating_resp in ['pattern', 'session', 'hybrid', 'auto']:
        operating_mode = operating_resp
    else:
        operating_mode = existing_operating_mode
    _persist_env_vars({'DEFAULT_OPERATING_MODE': operating_mode})

    return {
        'symbol': symbol,
        'timeframes': timeframes,
        'mode': mode,
        'trade_live': trade_live,
        'moderation_mode': moderation_mode,
        'trading_mode': trading_mode,
        'operating_mode': operating_mode,
    }

from verify_patterns import watch_unverified


def _organize_file_structure(symbol: str):
    """Organize files by trading pair into subfolders"""
    import shutil
    import glob
    
    # Create pair-specific directories
    pair_dirs = {
        'sim_results': f'sim_results/{symbol}',
        'patterns_unverified': f'patterns_unverified/{symbol}',
        'patterns_verified': f'patterns_verified/{symbol}'
    }
    
    for base_dir, pair_dir in pair_dirs.items():
        os.makedirs(pair_dir, exist_ok=True)
    
    # Move existing files to pair-specific directories
    # Move status files (keep global_status.json in root)
    status_files = ['status.json', 'live_status.json', 'trade_log.json']
    for file in status_files:
        src = f'sim_results/{file}'
        if os.path.exists(src):
            dst = f'sim_results/{symbol}/{file}'
            shutil.move(src, dst)
            print(f"Moved {file} to {symbol}/")
    
    # Ensure global_status.json stays in root (move it back if it was moved)
    global_status_src = f'sim_results/{symbol}/global_status.json'
    global_status_dst = 'sim_results/global_status.json'
    if os.path.exists(global_status_src):
        shutil.move(global_status_src, global_status_dst)
        print(f"Moved global_status.json back to root")
    
    # Move pattern files
    pattern_files = ['all_patterns.jsonl', '2h_patterns.jsonl']
    for file in pattern_files:
        src = f'patterns_unverified/{file}'
        if os.path.exists(src):
            dst = f'patterns_unverified/{symbol}/{file}'
            shutil.move(src, dst)
            print(f"Moved {file} to {symbol}/")
    
    
    print(f"File structure organized for {symbol}")


def main():
    """Main entry point for the multi-timeframe bot system"""
    
    # Parse command line arguments first to check if we should skip interactive setup
    parser = argparse.ArgumentParser(
        description="Multi-Timeframe Pattern Detection Bot System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                        # Run with default settings
  python main.py --symbol ETH-USDT                      # Monitor ETH instead of BTC
  python main.py --timeframes 1min 5min 15min           # Custom timeframes
  python main.py --mode swap --leverage 10              # Perpetual swaps with 10x leverage
  python main.py --mode futures --leverage 5            # Traditional futures with 5x leverage
  python main.py --symbol BTC-USDT --timeframes 1min 5min 15min 30min 1h 4h --mode spot
        """
    )
    
    parser.add_argument(
        # Had in the help Blofin instead of the CoinEx
        '--symbol', 
        default='BTC-USDT',
        help='Trading instrument to monitor (CoinEx format, default: BTC-USDT)'
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=None,
        help='Timeframes to monitor (e.g., 1min 5min 15min). If omitted, you will be prompted.'
    )
    parser.add_argument(
        '--mode',
        choices=['spot','swap','futures'],
        default='spot',
        help='Trading mode: spot (spot trading), swap (perpetual swaps), or futures (traditional futures) (default: spot)'
    )
    parser.add_argument(
        '--operating-mode',
        choices=['pattern', 'session', 'hybrid', 'auto'],
        default=None,
        help='Strategy operating mode at startup (pattern/session/hybrid/auto). If omitted, you will be prompted.'
    )
    
    parser.add_argument(
        '--leverage',
        type=int,
        default=20,
        help='Leverage for futures trading (default: 20, only used in futures mode)'
    )
    
    parser.add_argument(
        '--margin-mode',
        choices=['isolated', 'cross'],
        default='isolated',
        help='Margin mode for futures trading: isolated or cross (default: isolated)'
    )
    
    parser.add_argument(
        '--paper-trading',
        action='store_true',
        help='Enable paper trading mode (simulated trading without real money)'
    )
    
    parser.add_argument(
        '--moderation-mode',
        choices=['strict', 'balanced', 'aggressive'],
        default=None,
        help='Pattern detection moderation mode: strict (conservative), balanced (recommended), or aggressive (more patterns) (default: balanced)'
    )
    
    parser.add_argument(
        '--trading-mode',
        choices=['safe', 'balanced', 'aggressive'],
        default=None,
        help='Session-aware trading mode: safe (US/London only), balanced (recommended), or aggressive (all sessions) (default: balanced)'
    )
    
    args = parser.parse_args()
    
    # Check if we should run interactive setup or partial setup
    provided_args = []
    if args.symbol != 'BTC-USDT':
        provided_args.append('symbol')
    if args.timeframes is not None:
        provided_args.append('timeframes')
    if args.mode != 'spot':
        provided_args.append('mode')
    if args.operating_mode is not None:
        provided_args.append('operating_mode')
    if args.paper_trading:
        provided_args.append('paper_trading')
    if args.moderation_mode is not None:
        provided_args.append('moderation_mode')
    if args.trading_mode is not None:
        provided_args.append('trading_mode')
    
    if len(provided_args) == 0:
        # No command line arguments provided, run full interactive setup
        print("No command line arguments provided, starting interactive setup...")
        setup = _interactive_setup()
        
        # Override with interactive selections
        args.symbol = setup['symbol']
        args.timeframes = setup['timeframes']
        args.mode = setup.get('mode', 'spot')
        args.moderation_mode = setup.get('moderation_mode', 'balanced')
        args.trading_mode = setup.get('trading_mode', 'balanced')
        args.operating_mode = setup.get('operating_mode', 'hybrid') if args.operating_mode is None else args.operating_mode
    else:
        # Some command line arguments provided, use defaults for missing values and prompt for missing ones
        print(f"Starting bot with provided arguments: {', '.join(provided_args)}")
        
        # Set defaults for missing arguments
        if args.timeframes is None:
            args.timeframes = ['1min', '5min', '15min', '30min', '1h', '4h']  # Default timeframes
        if args.operating_mode is None:
            args.operating_mode = 'hybrid'
        
        # Set default values for missing arguments
        if args.moderation_mode is None:
            args.moderation_mode = 'balanced'
        if args.trading_mode is None:
            args.trading_mode = 'balanced'
        
        # Prompt for missing critical settings
        _load_project_env()
        
        # Check if we need to prompt for live trading
        if not args.paper_trading:
            # Always prompt for live trading when using command line arguments
            resp = input("Enable LIVE trading? (yes/no) [no]: ").strip().lower()
            trade_live = '1' if resp in ('y','yes') else '0'
            _persist_env_vars({'TRADE_LIVE': trade_live})
        else:
            trade_live = '0'  # Paper trading mode
            _persist_env_vars({'TRADE_LIVE': '0'})
        
        # Store the trade_live value for later use
        os.environ['TRADE_LIVE'] = trade_live
    
    # Organize file structure for the selected trading pair
    _organize_file_structure(args.symbol)
    
    # Handle paper trading mode
    if args.paper_trading:
        print("PAPER TRADING MODE ENABLED - No real money will be used")
        # Set paper trading mode in environment or configuration
        os.environ['PAPER_TRADING'] = 'true'
    else:
        # Check if live trading is actually enabled
        trade_live = os.getenv('TRADE_LIVE', '0')
        if trade_live == '1':
            print("LIVE TRADING MODE ENABLED - Real money will be used")
        else:
            print("PAPER TRADING MODE ENABLED - No real money will be used")
        os.environ['PAPER_TRADING'] = 'false' if trade_live == '1' else 'true'
    
    # Validate timeframes
    valid_timeframes = [
        '1min', '3min', '5min', '15min', '30min', 
        '1h', '2h', '4h', '6h', '12h', '1d',
        '3d', '1w', '1mon'
    ]
    # '8h' is not supported by CoinEx
    
    # Skip fallback prompt since interactive setup already collected timeframes

    invalid_timeframes = [tf for tf in args.timeframes if tf not in valid_timeframes]
    if invalid_timeframes:
        print(f"Invalid timeframes: {invalid_timeframes}")
        print(f"Valid timeframes: {', '.join(valid_timeframes)}")
        sys.exit(1)
    
    # Display startup information
    print("Multi-Timeframe Pattern Detection Bot System")
    print("=" * 60)
    print(f"Instrument: {args.symbol}")
    print(f"Timeframes: {', '.join(args.timeframes)}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Moderation: {args.moderation_mode.upper()}")  # ADJUSTED: Added moderation mode display
    print(f"Session Mode: {args.trading_mode.upper()}")  # ADJUSTED: Added trading mode display
    print(f"Operating Mode: {args.operating_mode.upper()}")
    if args.mode in ["futures", "swap"]:
        print(f"{args.mode.upper()} Settings: {args.leverage}x leverage, {args.margin_mode} margin")
    print(f"Architecture: Multi-process ({len(args.timeframes)} bots)")
    print("=" * 60)
    
    # Create and run bot manager
    try:
        manager = BotManager(
            symbol=args.symbol,
            timeframes=args.timeframes,
            mode=args.mode,
            leverage=args.leverage,
            margin_mode=args.margin_mode,
            moderation_mode=args.moderation_mode,  # ADJUSTED: Pass moderation mode to bot manager
            trading_mode=args.trading_mode,  # ADJUSTED: Pass trading mode for session awareness
            operating_mode=args.operating_mode  # NEW: Pass strategy operating mode
        )
        # Start background verifier
        verifier_thread = threading.Thread(target=watch_unverified, args=(args.symbol,), daemon=True)
        verifier_thread.start()
        manager.run()
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down gracefully...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
    finally:
        print("\nSystem shutdown complete.")


if __name__ == "__main__":
    main() 