#!/usr/bin/env python3
"""
Start verification processes for all symbols that have unverified patterns
"""

import os
import threading
from pathlib import Path
from verify_patterns import watch_unverified

def start_verification_for_all_symbols():
    """Start verification processes for all symbols with unverified patterns"""
    
    patterns_unverified_dir = Path("patterns_unverified")
    if not patterns_unverified_dir.exists():
        print("No patterns_unverified directory found")
        return
    
    # Find all symbols with unverified patterns
    symbols = []
    for symbol_dir in patterns_unverified_dir.iterdir():
        if symbol_dir.is_dir():
            # Check if there are any pattern files
            pattern_files = list(symbol_dir.glob("*.jsonl"))
            if pattern_files:
                symbols.append(symbol_dir.name)
                print(f"Found unverified patterns for: {symbol_dir.name}")
    
    if not symbols:
        print("No symbols with unverified patterns found")
        return
    
    print(f"Starting verification processes for {len(symbols)} symbols...")
    
    # Start verification process for each symbol
    verification_threads = []
    for symbol in symbols:
        print(f"Starting verification for {symbol}...")
        thread = threading.Thread(
            target=watch_unverified, 
            args=(symbol,), 
            daemon=True,
            name=f"verify_{symbol}"
        )
        thread.start()
        verification_threads.append(thread)
    
    print(f"Started {len(verification_threads)} verification processes")
    print("Verification processes are running in the background...")
    
    # Keep the script running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping verification processes...")

if __name__ == "__main__":
    start_verification_for_all_symbols()
