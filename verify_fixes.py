#!/usr/bin/env python3
"""
Verification script for multi-pair trading and dynamic SL fixes.
Run this to verify the fixes are working correctly.
"""

import json
import os
from pathlib import Path
from core.dynamic_sl_calculator import DynamicSLCalculator

def verify_dynamic_sl():
    """Verify dynamic SL calculator works for all pairs."""
    print("\n" + "="*70)
    print("VERIFYING DYNAMIC SL CALCULATOR")
    print("="*70)
    
    calc = DynamicSLCalculator()
    
    # Test pairs with their typical price ranges
    test_cases = [
        ('BTCUSDT', 45000.0),
        ('ETHUSDT', 2500.0),
        ('SOLUSDT', 133.0),
        ('XRPUSDT', 2.50),
        ('ADAUSDT', 1.10),
        ('BNBUSDT', 620.0),
        ('DOGEUSDT', 0.30),
        ('AVAXUSDT', 45.0),
        ('MATICUSDT', 0.85),
        ('LINKUSDT', 28.0),
    ]
    
    print("\nPair-Specific SL Calculations:")
    print("-" * 70)
    print(f"{'Pair':<12} {'Price':<12} {'Min SL %':<12} {'Min SL $':<12} {'Config':<20}")
    print("-" * 70)
    
    for symbol, price in test_cases:
        min_sl_pct = calc.calculate_min_sl_pct(symbol, price)
        min_sl_distance = calc.calculate_min_sl_distance(symbol, price)
        config = calc.get_pair_config(symbol)
        
        print(f"{symbol:<12} ${price:<11.2f} {min_sl_pct*100:<11.2f}% ${min_sl_distance:<11.2f} "
              f"ATR:{config['atr_multiplier']:.1f}x")
    
    print("-" * 70)
    print("\n✅ Dynamic SL Calculator verified successfully!")
    return True

def verify_symbols_config():
    """Verify symbols_config.json has all pairs enabled."""
    print("\n" + "="*70)
    print("VERIFYING SYMBOLS CONFIGURATION")
    print("="*70)
    
    config_path = Path('symbols_config.json')
    if not config_path.exists():
        print(f"❌ {config_path} not found!")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    symbols = config.get('symbols', [])
    enabled = [s for s in symbols if s.get('enabled', True)]
    
    print(f"\nTotal symbols configured: {len(symbols)}")
    print(f"Enabled symbols: {len(enabled)}")
    print("\nEnabled symbols:")
    print("-" * 70)
    
    for sym_config in enabled:
        symbol = sym_config.get('symbol', 'UNKNOWN')
        timeframes = sym_config.get('timeframes', [])
        print(f"  • {symbol:<12} - Timeframes: {', '.join(timeframes)}")
    
    print("-" * 70)
    
    if len(enabled) < 5:
        print(f"⚠️  Warning: Only {len(enabled)} symbols enabled. Expected at least 5.")
        return False
    
    print(f"\n✅ Symbols configuration verified! {len(enabled)} pairs ready to trade.")
    return True

def verify_imports():
    """Verify all necessary imports work."""
    print("\n" + "="*70)
    print("VERIFYING IMPORTS")
    print("="*70)
    
    try:
        from core.dynamic_sl_calculator import get_sl_calculator, DynamicSLCalculator
        print("✅ core.dynamic_sl_calculator imports OK")
    except Exception as e:
        print(f"❌ Failed to import dynamic_sl_calculator: {e}")
        return False
    
    try:
        from rl_trading_env import TradingEnv
        print("✅ rl_trading_env imports OK")
    except Exception as e:
        print(f"❌ Failed to import rl_trading_env: {e}")
        return False
    
    try:
        from simulate_trading import TradeSimulator
        print("✅ simulate_trading imports OK")
    except Exception as e:
        print(f"❌ Failed to import simulate_trading: {e}")
        return False
    
    print("\n✅ All imports verified successfully!")
    return True

def verify_trade_logging():
    """Check if trade logs exist and have proper symbol field."""
    print("\n" + "="*70)
    print("VERIFYING TRADE LOGGING")
    print("="*70)
    
    log_dir = Path('logs')
    if not log_dir.exists():
        print(f"⚠️  Log directory not found: {log_dir}")
        print("   (This is expected if trading hasn't started yet)")
        return True
    
    trades_log = log_dir / 'trades_enhanced.jsonl'
    if not trades_log.exists():
        print(f"⚠️  Trade log not found: {trades_log}")
        print("   (This is expected if trading hasn't started yet)")
        return True
    
    print(f"\nAnalyzing: {trades_log}")
    
    symbols_seen = set()
    none_symbols = 0
    total_trades = 0
    
    try:
        with open(trades_log, 'r') as f:
            for line in f:
                try:
                    trade = json.loads(line)
                    total_trades += 1
                    symbol = trade.get('symbol')
                    if symbol is None:
                        none_symbols += 1
                    else:
                        symbols_seen.add(symbol)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"❌ Error reading trade log: {e}")
        return False
    
    print(f"\nTotal trades logged: {total_trades}")
    print(f"Unique symbols: {len(symbols_seen)}")
    print(f"Trades with symbol=None: {none_symbols}")
    
    if symbols_seen:
        print(f"\nSymbols traded:")
        for sym in sorted(symbols_seen):
            print(f"  • {sym}")
    
    if none_symbols > 0:
        print(f"\n⚠️  Warning: {none_symbols} trades have symbol=None (should be 0 after fix)")
        return False
    
    if len(symbols_seen) > 1:
        print(f"\n✅ Multi-pair trading verified! {len(symbols_seen)} different pairs traded.")
        return True
    else:
        print(f"\n⚠️  Only {len(symbols_seen)} unique symbol(s) traded (expected multiple)")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("TRADING BOT FIXES VERIFICATION")
    print("="*70)
    
    checks = [
        ("Imports", verify_imports),
        ("Symbols Config", verify_symbols_config),
        ("Dynamic SL Calculator", verify_dynamic_sl),
        ("Trade Logging", verify_trade_logging),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("✅ ALL CHECKS PASSED!")
        print("="*70)
        print("\nThe fixes have been successfully applied:")
        print("  1. Multi-pair trading is enabled")
        print("  2. Dynamic SL per pair is configured")
        print("  3. All imports are working")
        print("\nYou can now start trading with:")
        print("  python auto_trader_optimized.py")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  SOME CHECKS FAILED")
        print("="*70)
        print("\nPlease review the errors above and fix them.")
        return 1

if __name__ == '__main__':
    exit(main())
