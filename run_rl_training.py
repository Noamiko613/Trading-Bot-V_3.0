"""
RL Training Runner
==================

Wrapper script that:
1. First downloads/caches all historical data
2. Then runs RL training on the cached data

Usage:
    python run_rl_training.py [same arguments as rl_training.py]
"""

import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from download_historical_data import HistoricalDataDownloader, DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES

# Import rl_training defaults
try:
    from rl_training import RL_TRAINING_SYMBOLS, RL_TRAINING_TIMEFRAMES
    TRAINING_SYMBOLS = RL_TRAINING_SYMBOLS
    TRAINING_TIMEFRAMES = RL_TRAINING_TIMEFRAMES
except ImportError:
    TRAINING_SYMBOLS = DEFAULT_SYMBOLS
    TRAINING_TIMEFRAMES = DEFAULT_TIMEFRAMES


def main():
    parser = argparse.ArgumentParser(
        description='Download historical data and run RL training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script:
1. Downloads and caches historical data for all symbols/timeframes
2. Runs RL training using the cached data

All arguments are passed to rl_training.py after data download completes.
        """
    )
    
    # Data download arguments
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download and go straight to training')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download even if cached')
    parser.add_argument('--download-only', action='store_true',
                       help='Only download data, do not run training')
    
    # Pass through arguments for rl_training.py
    parser.add_argument('--symbol', type=str, help='Single symbol to train on')
    parser.add_argument('--timesteps', type=int, help='Number of training timesteps')
    parser.add_argument('--historical-start-date', type=str, help='Historical data start date')
    parser.add_argument('--historical-end-date', type=str, help='Historical data end date')
    parser.add_argument('--historical-pretrain-timesteps', type=int, help='Historical pre-training timesteps')
    
    # Parse known args (we'll pass unknown args to rl_training.py)
    args, unknown_args = parser.parse_known_args()
    
    # Determine symbols and timeframes from RL training config
    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        # Use RL training symbols (automatically imported)
        symbols = TRAINING_SYMBOLS
    
    # Use RL training timeframes (automatically imported)
    timeframes = TRAINING_TIMEFRAMES
    
    print(f"[Runner] Will download data for:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    
    # Determine date range
    start_date = args.historical_start_date or "2019-12-01"
    end_date = args.historical_end_date or datetime.now().strftime("%Y-%m-%d")
    
    if not args.skip_download and not args.download_only:
        print("="*80)
        print("STEP 1: Downloading Historical Data")
        print("="*80)
        
        # Create downloader
        downloader = HistoricalDataDownloader()
        
        # Check what's already cached
        print("\n[Runner] Checking cache status...")
        all_cached = True
        for symbol in symbols:
            for tf in timeframes:
                if not downloader._is_cached(symbol, tf, start_date, end_date):
                    all_cached = False
                    print(f"  ⚠️  {symbol}/{tf}: Not cached - will download")
                else:
                    print(f"  ✅ {symbol}/{tf}: Cached - will use cache")
        
        if all_cached and not args.force_download:
            print("\n[Runner] ✅ All data is cached! Skipping download.")
            print("[Runner] Using cached data. (Use --force-download to re-download)")
        else:
            # Download all data
            downloader.download_all(
                symbols=symbols,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                use_cache=not args.force_download
            )
        
        print("\n" + "="*80)
        print("STEP 2: Starting RL Training")
        print("="*80 + "\n")
    
    if args.download_only:
        print("[Runner] Download complete. Exiting (--download-only specified).")
        return
    
    # Build command for rl_training.py
    cmd = [sys.executable, "rl_training.py"]
    
    # Add known arguments
    if args.symbol:
        cmd.extend(["--symbol", args.symbol])
    if args.timesteps:
        cmd.extend(["--timesteps", str(args.timesteps)])
    if args.historical_start_date:
        cmd.extend(["--historical-start-date", args.historical_start_date])
    if args.historical_end_date:
        cmd.extend(["--historical-end-date", args.historical_end_date])
    if args.historical_pretrain_timesteps:
        cmd.extend(["--historical-pretrain-timesteps", str(args.historical_pretrain_timesteps)])
    
    # Add unknown arguments (pass through)
    cmd.extend(unknown_args)
    
    # Run rl_training.py
    print(f"[Runner] Running: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n[Runner] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Runner] Error running training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
