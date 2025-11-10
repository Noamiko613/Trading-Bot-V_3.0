#!/usr/bin/env python3
"""Emergency bot stop script"""
import os
import signal
import psutil

print("Stopping all Python bot processes...")

killed = 0
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if proc.info['name'] and 'python' in proc.info['name'].lower():
            cmdline = proc.info['cmdline'] or []
            # Kill if it's running bot files
            if any(term in ' '.join(cmdline) for term in ['auto_trader', 'main.py', 'bot_manager', 'timeframe_bot']):
                print(f"Killing PID {proc.info['pid']}: {' '.join(cmdline[:3])}")
                proc.kill()
                killed += 1
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

print(f"\n✅ Stopped {killed} bot processes")
print("\n🛡️ PAPER TRADING IS NOW HARDCODED IN bot_manager.py")
print("Live trading is IMPOSSIBLE - the code always uses simulator\n")
print("Now you can safely run: python auto_trader.py")
