"""
Check current trading mode and environment settings
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 60)
print("TRADING MODE CHECK")
print("=" * 60)

# Check .env file
env_path = '.env'
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('TRADE_LIVE'):
                print(f"\n.env file: {line.strip()}")
                break
else:
    print("\n⚠️  .env file not found!")

# Check environment variable
trade_live = os.getenv('TRADE_LIVE', 'NOT SET')
print(f"Environment variable TRADE_LIVE: {trade_live}")

# Interpret
print("\n" + "=" * 60)
if trade_live == '0':
    print("✅ PAPER TRADING MODE (Simulated)")
    print("   - No real money will be used")
    print("   - Trades are simulated")
    print("   - Safe for testing")
elif trade_live == '1':
    print("⚠️  LIVE TRADING MODE (Real Money)")
    print("   - Real money WILL be used")
    print("   - Orders sent to exchange")
    print("   - Use with caution!")
else:
    print("❌ TRADE_LIVE not properly set")
    print("   Please run: python force_paper_trading.py")

print("=" * 60)
print("\nNOTE: If you changed .env file while bot is running,")
print("you must STOP and RESTART the bot for changes to take effect!")
print("=" * 60)
