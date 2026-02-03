#!/usr/bin/env python3
"""
RL Training Status Checker
==========================

Diagnostic script to check if RL training is running properly and identify issues.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def check_state_file(state_file_path: Path):
    """Check training state file"""
    print("\n" + "="*70)
    print("TRAINING STATE FILE CHECK")
    print("="*70)
    
    if not state_file_path.exists():
        print(f"❌ State file does not exist: {state_file_path}")
        print("   This suggests training has never been started or state file was deleted.")
        return None
    
    try:
        with open(state_file_path, 'r') as f:
            state = json.load(f)
        
        print(f"✅ State file exists: {state_file_path}")
        print(f"\nState Contents:")
        print(f"  Episode: {state.get('episode', 'N/A')}")
        print(f"  Timesteps: {state.get('total_timesteps', 0):,}")
        print(f"  Total Trades: {state.get('total_trades', 0)}")
        print(f"  Mode: {state.get('mode', 'N/A')}")
        print(f"  Best Reward: {state.get('best_reward', 'N/A')}")
        
        # Check timestamp
        import json
        import time as time_module
        log_path = r"c:\Users\Mini Echo09\Desktop\Trading-Bot-V_2.0-feature-enhanced-logging-metrics\.cursor\debug.log"
        timestamp_str = state.get('timestamp')
        if timestamp_str:
            try:
                # #region agent log
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_status_parse_start","timestamp":int(time_module.time()*1000),"location":"check_rl_training_status.py:45","message":"Status script parsing timestamp","data":{"timestamp_str":timestamp_str},"sessionId":"debug-session","runId":"run1","hypothesisId":"I"}) + "\n")
                except: pass
                # #endregion
                state_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if state_time.tzinfo:
                    state_time = state_time.replace(tzinfo=None)
                # Use UTC time consistently for comparison
                now = datetime.now(timezone.utc)
                if now.tzinfo:
                    now = now.replace(tzinfo=None)
                age_minutes = (now - state_time).total_seconds() / 60
                # #region agent log
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_status_parse_result","timestamp":int(time_module.time()*1000),"location":"check_rl_training_status.py:51","message":"Status script timestamp calculation","data":{"state_time":state_time.isoformat(),"now":now.isoformat(),"age_minutes":age_minutes,"state_time_tz":str(state_time.tzinfo),"now_tz":str(now.tzinfo)},"sessionId":"debug-session","runId":"run1","hypothesisId":"I,J"}) + "\n")
                except: pass
                # #endregion
                print(f"  Last Update: {state_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Age: {age_minutes:.1f} minutes ago")
                
                if age_minutes > 60:
                    print(f"\n⚠️  WARNING: State file is {age_minutes:.0f} minutes old!")
                    print(f"   Training loop may have crashed or stopped.")
                    if age_minutes > 1440:  # 24 hours
                        print(f"   State file is over 24 hours old - training is likely stalled.")
                elif age_minutes < 5:
                    print(f"\n✅ State file is recent - training appears to be running")
                else:
                    print(f"\n⚠️  State file is {age_minutes:.0f} minutes old - check if training is running")
                
            except Exception as e:
                print(f"  Timestamp: {timestamp_str} (parse error: {e})")
        else:
            print(f"  ⚠️  No timestamp in state file")
        
        # Check checkpoint
        checkpoint = state.get('last_checkpoint')
        if checkpoint:
            if os.path.exists(checkpoint):
                print(f"\n✅ Checkpoint exists: {checkpoint}")
            else:
                print(f"\n⚠️  Checkpoint path in state file doesn't exist: {checkpoint}")
        else:
            print(f"\n⚠️  No checkpoint path in state file")
        
        return state
        
    except Exception as e:
        print(f"❌ Error reading state file: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_model_checkpoints(model_dir: Path):
    """Check for model checkpoints"""
    print("\n" + "="*70)
    print("MODEL CHECKPOINTS CHECK")
    print("="*70)
    
    checkpoints_dir = model_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"⚠️  Checkpoints directory doesn't exist: {checkpoints_dir}")
        print("   Training may not have created any checkpoints yet.")
        return []
    
    checkpoints = sorted(checkpoints_dir.glob("rl_model_*.zip"))
    if not checkpoints:
        print(f"⚠️  No checkpoints found in: {checkpoints_dir}")
        print("   Training may not have started or checkpoints are being saved elsewhere.")
        return []
    
    print(f"✅ Found {len(checkpoints)} checkpoints")
    print(f"\nLatest checkpoints:")
    for cp in checkpoints[-5:]:  # Show last 5
        size_mb = cp.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(cp.stat().st_mtime)
        age_minutes = (datetime.now() - mtime).total_seconds() / 60
        print(f"  {cp.name}")
        print(f"    Size: {size_mb:.1f} MB")
        print(f"    Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({age_minutes:.1f}m ago)")
    
    return checkpoints

def check_database(db_path: str):
    """Check trades database"""
    print("\n" + "="*70)
    print("TRADES DATABASE CHECK")
    print("="*70)
    
    if not os.path.exists(db_path):
        print(f"⚠️  Database doesn't exist: {db_path}")
        print("   Trades may not have been executed yet.")
        return
    
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check closed trades
        cursor.execute("SELECT COUNT(*) FROM trades_closed")
        closed_count = cursor.fetchone()[0]
        
        # Check open trades
        cursor.execute("SELECT COUNT(*) FROM trades_open")
        open_count = cursor.fetchone()[0]
        
        print(f"✅ Database exists: {db_path}")
        print(f"  Closed Trades: {closed_count}")
        print(f"  Open Trades: {open_count}")
        
        if closed_count == 0 and open_count == 0:
            print(f"\n⚠️  No trades in database - training may not have started")
        elif closed_count > 0:
            # Get latest closed trade
            cursor.execute("SELECT closed_time FROM trades_closed ORDER BY closed_time DESC LIMIT 1")
            latest = cursor.fetchone()
            if latest:
                try:
                    latest_time = datetime.fromisoformat(latest[0].replace('Z', '+00:00'))
                    if latest_time.tzinfo:
                        latest_time = latest_time.replace(tzinfo=None)
                    # Use UTC time consistently
                    now_utc = datetime.now(timezone.utc)
                    if now_utc.tzinfo:
                        now_utc = now_utc.replace(tzinfo=None)
                    age_minutes = (now_utc - latest_time).total_seconds() / 60
                    print(f"\n  Latest Closed Trade: {latest_time.strftime('%Y-%m-%d %H:%M:%S')} ({age_minutes:.1f}m ago)")
                    if age_minutes > 60:
                        print(f"  ⚠️  Last trade was over an hour ago - training may be stalled")
                except Exception:
                    pass
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main diagnostic function"""
    print("="*70)
    print("RL TRAINING STATUS DIAGNOSTIC")
    print("="*70)
    
    # Default paths
    model_dir = Path("models/rl_models")
    state_file = model_dir / "rl_training_state.json"
    db_path = "sim_results/trades.db"
    
    # Check state file
    state = check_state_file(state_file)
    
    # Check checkpoints
    checkpoints = check_model_checkpoints(model_dir)
    
    # Check database
    check_database(db_path)
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    if state:
        timesteps = state.get('total_timesteps', 0)
        episode = state.get('episode', 0)
        timestamp_str = state.get('timestamp')
        
        if timesteps == 0 and episode == 0:
            print("❌ Training appears to have never started or crashed immediately")
            print("   - Episode: 0, Timesteps: 0")
            print("   - Suggestion: Start training with: python rl_training.py")
        elif timestamp_str:
            try:
                state_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if state_time.tzinfo:
                    state_time = state_time.replace(tzinfo=None)
                # Use UTC time consistently for comparison
                now = datetime.now(timezone.utc)
                if now.tzinfo:
                    now = now.replace(tzinfo=None)
                age_minutes = (now - state_time).total_seconds() / 60
                
                if age_minutes > 60:
                    print(f"⚠️  Training loop appears STALLED")
                    print(f"   - State file hasn't been updated in {age_minutes:.0f} minutes")
                    print(f"   - Training loop may have crashed")
                    print(f"   - Suggestion: Check training logs for errors")
                    print(f"   - Suggestion: Restart training if needed")
                else:
                    print(f"✅ Training appears to be running")
                    print(f"   - State file updated {age_minutes:.1f} minutes ago")
            except:
                pass
        
        if not checkpoints:
            print("\n⚠️  No checkpoints found - training may not be saving progress")
    else:
        print("❌ Cannot diagnose - state file missing or unreadable")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
