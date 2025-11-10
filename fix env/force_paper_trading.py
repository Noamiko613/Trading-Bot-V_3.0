"""
Force Paper Trading Mode
=========================

Run this before starting the bot to ensure paper trading is active.
"""

import os
from pathlib import Path

# Update .env file
env_file = Path('.env')
if env_file.exists():
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    with open(env_file, 'w') as f:
        trade_live_found = False
        for line in lines:
            if line.startswith('TRADE_LIVE='):
                f.write('TRADE_LIVE=0\n')
                trade_live_found = True
            else:
                f.write(line)
        
        if not trade_live_found:
            f.write('\nTRADE_LIVE=0\n')
    
    print("✅ Paper trading mode forced in .env file")
    print("✅ TRADE_LIVE=0")
else:
    print("⚠️  .env file not found")

# Check if shell environment variable is set (takes precedence over .env)
if 'TRADE_LIVE' in os.environ:
    print("\n" + "=" * 60)
    print("⚠️  WARNING: Shell environment variable TRADE_LIVE is set!")
    print(f"   Current value: TRADE_LIVE={os.environ['TRADE_LIVE']}")
    print("\n   This will override the .env file!")
    print("\n   To fix, run ONE of these commands:")
    print("   1. unset TRADE_LIVE")
    print("   2. Close this terminal and open a new one")
    print("=" * 60)
else:
    print("\n✅ No conflicting shell environment variables found")
    print("\nNow start your bot with: python auto_trader.py")
