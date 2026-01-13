#!/usr/bin/env python3
"""
Algorithmic Trading System Entry Point
======================================

Starts the algorithmic trading system that discovers and trades
profitable algorithms independently from RL training.
"""

import os
import sys
import argparse
import signal
from pathlib import Path

from algorithmic_trader import AlgorithmicTradingSystem
from algorithmic_dashboard import AlgorithmicDashboard
from utils.logger import ComponentLogger


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Algorithmic Trading System - Discover and trade profitable algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_algorithmic_trading.py --pairs BTCUSDT ETHUSDT
  python start_algorithmic_trading.py --pairs BTCUSDT --mode spot --balance 10000
  python start_algorithmic_trading.py --pairs BTCUSDT ETHUSDT SOLUSDT --dashboard
        """
    )
    
    parser.add_argument(
        '--pairs',
        nargs='+',
        required=True,
        help='Trading pairs to monitor (e.g., BTCUSDT ETHUSDT)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['spot', 'futures'],
        default='spot',
        help='Trading mode (default: spot)'
    )
    
    parser.add_argument(
        '--balance',
        type=float,
        default=10000.0,
        help='Starting balance per pair (default: 10000)'
    )
    
    parser.add_argument(
        '--min-trades',
        type=int,
        default=100,
        help='Minimum trades for algorithm validation (default: 100)'
    )
    
    parser.add_argument(
        '--target-win-rate',
        type=float,
        default=66.0,
        help='Target win rate percentage (default: 66.0)'
    )
    
    parser.add_argument(
        '--target-rr',
        type=float,
        default=2.0,
        help='Target risk/reward ratio (default: 2.0)'
    )
    
    parser.add_argument(
        '--discovery-interval',
        type=int,
        default=3600,
        help='Algorithm discovery interval in seconds (default: 3600 = 1 hour)'
    )
    
    parser.add_argument(
        '--signal-interval',
        type=int,
        default=60,
        help='Signal check interval in seconds (default: 60 = 1 minute)'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Show real-time dashboard'
    )
    
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Run without dashboard (background mode)'
    )
    
    args = parser.parse_args()
    
    # Ensure paper trading mode
    if os.getenv('TRADE_LIVE', '0') == '1':
        print("⚠️  WARNING: Algorithmic trading system only supports paper trading mode.")
        print("⚠️  Setting TRADE_LIVE=0")
        os.environ['TRADE_LIVE'] = '0'
    
    # Create system
    system = AlgorithmicTradingSystem(
        pairs=args.pairs,
        mode=args.mode,
        starting_balance_per_pair=args.balance,
        discovery_interval=args.discovery_interval,
        signal_check_interval=args.signal_interval,
        min_trades_for_validation=args.min_trades,
        target_win_rate=args.target_win_rate,
        target_rr=args.target_rr,
    )
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        print("\n\n⚠️  Shutting down algorithmic trading system...")
        system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start system
    print("=" * 80)
    print("ALGORITHMIC TRADING SYSTEM")
    print("=" * 80)
    print(f"Pairs: {', '.join(args.pairs)}")
    print(f"Mode: {args.mode}")
    print(f"Starting Balance per Pair: ${args.balance:,.2f}")
    print(f"Target Win Rate: {args.target_win_rate}%")
    print(f"Target R:R: {args.target_rr}:1")
    print(f"Min Trades for Validation: {args.min_trades}")
    print("=" * 80)
    print("\n🚀 Starting system...\n")
    
    system.start()
    
    # Start dashboard if requested
    if args.dashboard and not args.no_dashboard:
        dashboard = AlgorithmicDashboard(pairs=args.pairs)
        try:
            dashboard.start()
        except KeyboardInterrupt:
            dashboard.stop()
            system.stop()
    else:
        # Run without dashboard
        print("System running in background mode. Press Ctrl+C to stop.")
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            system.stop()


if __name__ == "__main__":
    main()

