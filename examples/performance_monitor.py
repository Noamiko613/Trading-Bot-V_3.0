"""Real-Time Performance Monitor
==============================

Standalone script to monitor trading bot performance in real-time.
Displays live metrics and sends alerts when thresholds are breached.
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.analytics import PerformanceAnalytics, RealTimeMonitor
from utils.logger import get_logger
from utils.config_manager import get_config

# Initialize
logger = get_logger('performance_monitor')
analytics = PerformanceAnalytics()
monitor = RealTimeMonitor(analytics)


def load_global_status(path: str = os.path.join('sim_results', 'global_status.json')) -> Dict:
    """Load global status for open trades summary"""
    try:
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def load_symbol_statuses(base_dir: str = 'sim_results') -> Dict[str, Dict]:
    """Load symbol statuses for open trades summary"""
    statuses: Dict[str, Dict] = {}
    if not os.path.exists(base_dir):
        return statuses
    for name in os.listdir(base_dir):
        sym_dir = os.path.join(base_dir, name)
        if not os.path.isdir(sym_dir):
            continue
        # Try live_status.json first, then fall back to status.json
        status_fp = os.path.join(sym_dir, 'live_status.json')
        if not os.path.exists(status_fp):
            status_fp = os.path.join(sym_dir, 'status.json')
        if os.path.exists(status_fp):
            try:
                with open(status_fp, 'r') as f:
                    statuses[name] = json.load(f)
            except Exception:
                statuses[name] = {}
    return statuses


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def display_dashboard():
    """Display real-time performance dashboard"""
    
    clear_screen()
    
    print("=" * 100)
    print(f"{'TRADING BOT PERFORMANCE DASHBOARD':^100}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^100}")
    print("=" * 100)
    
    # Last 24 hours metrics
    print(f"\n{'📊 LAST 24 HOURS':^100}")
    print("-" * 100)
    
    trades_24h = analytics.get_closed_trades(days=1)
    if trades_24h:
        metrics_24h = analytics.calculate_metrics(trades_24h)
        
        print(f"{'Metric':<30} {'Value':>20} {'Status':>30}")
        print("-" * 100)
        print(f"{'Total Trades':<30} {metrics_24h['total_trades']:>20} {'':<30}")
        print(f"{'Winning Trades':<30} {metrics_24h['winning_trades']:>20} {'':<30}")
        print(f"{'Losing Trades':<30} {metrics_24h['losing_trades']:>20} {'':<30}")
        
        # Win rate with status
        win_rate = metrics_24h['win_rate']
        wr_status = "✅ Good" if win_rate >= 50 else "⚠️ Low" if win_rate >= 40 else "❌ Poor"
        print(f"{'Win Rate':<30} {f'{win_rate:.1f}%':>20} {wr_status:>30}")
        
        # PnL with status
        total_pnl = metrics_24h['total_pnl']
        pnl_status = "✅ Profitable" if total_pnl > 0 else "❌ Loss"
        print(f"{'Total P&L':<30} {f'${total_pnl:,.2f}':>20} {pnl_status:>30}")
        
        print(f"{'Average P&L':<30} {f'${metrics_24h["avg_pnl"]:,.2f}':>20} {'':<30}")
        print(f"{'Average Win':<30} {f'${metrics_24h["avg_win"]:,.2f}':>20} {'':<30}")
        print(f"{'Average Loss':<30} {f'${metrics_24h["avg_loss"]:,.2f}':>20} {'':<30}")
        
        # Profit factor with status
        pf = metrics_24h['profit_factor']
        pf_status = "✅ Excellent" if pf >= 2.0 else "✅ Good" if pf >= 1.5 else "⚠️ Marginal" if pf >= 1.0 else "❌ Poor"
        print(f"{'Profit Factor':<30} {f'{pf:.2f}':>20} {pf_status:>30}")
        
        # R-multiple
        r_mult = metrics_24h['avg_r_multiple']
        r_status = "✅ Excellent" if r_mult >= 1.0 else "⚠️ Marginal" if r_mult >= 0.5 else "❌ Poor"
        print(f"{'Avg R-Multiple':<30} {f'{r_mult:.2f}R':>20} {r_status:>30}")
        
    else:
        print("\nNo trades in the last 24 hours")
    
    # Last 7 days metrics
    print(f"\n{'📅 LAST 7 DAYS':^100}")
    print("-" * 100)
    
    trades_7d = analytics.get_closed_trades(days=7)
    if trades_7d:
        metrics_7d = analytics.calculate_metrics(trades_7d)
        
        print(f"{'Metric':<30} {'Value':>20} {'Status':>30}")
        print("-" * 100)
        print(f"{'Total Trades':<30} {metrics_7d['total_trades']:>20} {'':<30}")
        print(f"{'Win Rate':<30} {f'{metrics_7d["win_rate"]:.1f}%':>20} {'':<30}")
        print(f"{'Total P&L':<30} {f'${metrics_7d["total_pnl"]:,.2f}':>20} {'':<30}")
        
        # Sharpe ratio
        sharpe = metrics_7d['sharpe_ratio']
        sharpe_status = "✅ Excellent" if sharpe >= 2.0 else "✅ Good" if sharpe >= 1.0 else "⚠️ Fair"
        print(f"{'Sharpe Ratio':<30} {f'{sharpe:.2f}':>20} {sharpe_status:>30}")
        
        # Sortino ratio
        sortino = metrics_7d['sortino_ratio']
        sortino_status = "✅ Excellent" if sortino >= 2.0 else "✅ Good" if sortino >= 1.0 else "⚠️ Fair"
        print(f"{'Sortino Ratio':<30} {f'{sortino:.2f}':>20} {sortino_status:>30}")
        
        # Max drawdown
        max_dd = metrics_7d['max_drawdown_pct']
        dd_threshold = get_config('max_drawdown_pct', default=20.0, config_name='risk')
        dd_status = "✅ Safe" if max_dd < dd_threshold * 0.5 else "⚠️ Caution" if max_dd < dd_threshold else "❌ ALERT"
        print(f"{'Max Drawdown':<30} {f'{max_dd:.1f}%':>20} {dd_status:>30}")
        
        print(f"{'Max Consecutive Wins':<30} {metrics_7d['max_consecutive_wins']:>20} {'':<30}")
        print(f"{'Max Consecutive Losses':<30} {metrics_7d['max_consecutive_losses']:>20} {'':<30}")
    else:
        print("\nNo trades in the last 7 days")
    
    # Open trades summary (PRESERVED FROM ORIGINAL)
    print(f"\n{'📍 OPEN TRADES SUMMARY':^100}")
    print("-" * 100)
    print()
    
    global_status = load_global_status()
    statuses = load_symbol_statuses()
    
    if statuses:
        total_open = 0
        for sym, status in sorted(statuses.items()):
            open_trades = status.get('open_trades', []) or []
            if open_trades:
                total_open += len(open_trades)
                print(f"  {sym}:")
                for trade in open_trades:
                    print(f"    {trade.get('id', 'N/A'):20} | {trade.get('pattern', 'N/A'):15} | {trade.get('side', 'N/A'):4} | Entry: ${trade.get('entry', 0):.6f} | TP: ${trade.get('tp', 0):.6f} | SL: ${trade.get('stop', 0):.6f}")
        
        if total_open == 0:
            print("  No open trades")
            
        # Global stats if available
        if global_status:
            print()
            print(f"  Global Balance: ${global_status.get('balance', 0):,.2f} | Equity: ${global_status.get('equity', 0):,.2f}")
            print(f"  Total Open: {global_status.get('open_trades', 0)} | Total Closed: {global_status.get('closed_trades', 0)} | Cumulative PnL: ${global_status.get('cum_pnl', 0):,.2f}")
    else:
        print("(Could not load open trades summary)")

    # Pattern performance
    if trades_7d:
        print(f"\n{'🎯 PATTERN PERFORMANCE (7 DAYS)':^100}")
        print("-" * 100)
        
        pattern_perf = analytics.pattern_performance(trades_7d)
        if pattern_perf:
            print(f"{'Pattern':<30} {'Trades':>10} {'Win Rate':>15} {'Total P&L':>20} {'Avg R':>15}")
            print("-" * 100)
            
            # Sort by total PnL
            sorted_patterns = sorted(pattern_perf.items(), 
                                    key=lambda x: x[1]['total_pnl'], 
                                    reverse=True)
            
            for pattern, stats in sorted_patterns[:10]:  # Top 10 patterns
                print(f"{pattern[:29]:<30} "
                      f"{stats['total_trades']:>10} "
                      f"{stats['win_rate']:>14.1f}% "
                      f"${stats['total_pnl']:>18.2f} "
                      f"{stats['avg_r_multiple']:>14.2f}R")
    
    # Active alerts
    print(f"\n{'🚨 ACTIVE ALERTS':^100}")
    print("-" * 100)
    
    alerts = []
    
    # Check drawdown
    max_allowed_dd = get_config('max_drawdown_pct', default=20.0, config_name='risk')
    dd_alert = monitor.check_drawdown_alert(max_allowed_dd)
    if dd_alert:
        alerts.append(dd_alert)
    
    # Check win rate
    wr_alert = monitor.check_win_rate_alert(min_trades=10, min_win_rate=40.0)
    if wr_alert:
        alerts.append(wr_alert)
    
    # Check consecutive losses
    loss_alert = monitor.check_consecutive_losses_alert(5)
    if loss_alert:
        alerts.append(loss_alert)
    
    if alerts:
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("  ✅ No active alerts - System operating normally")
    
    # Recent alerts
    recent_alerts = monitor.get_recent_alerts(hours=24)
    if recent_alerts and len(recent_alerts) > len(alerts):
        print(f"\n  📋 Recent alerts (24h): {len(recent_alerts)}")
    
    print("\n" + "=" * 100)
    print(f"{'Last updated: ' + datetime.now().strftime('%H:%M:%S'):<50} {'Press Ctrl+C to exit':>50}")
    print("=" * 100)


def main():
    """Main monitoring loop"""
    
    logger.info("Starting performance monitor...")
    
    update_interval = 3  # seconds
    
    try:
        while True:
            try:
                display_dashboard()
                time.sleep(update_interval)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                time.sleep(update_interval)
    
    except KeyboardInterrupt:
        clear_screen()
        print("\n" + "=" * 100)
        print(f"{'PERFORMANCE MONITOR STOPPED':^100}")
        print("=" * 100)
        
        # Final summary
        trades = analytics.get_closed_trades(days=30)
        if trades:
            print(f"\n{'📊 FINAL SUMMARY (30 DAYS)':^100}")
            print("-" * 100)
            
            metrics = analytics.calculate_metrics(trades)
            print(f"  Total Trades:     {metrics['total_trades']}")
            print(f"  Win Rate:         {metrics['win_rate']:.1f}%")
            print(f"  Total P&L:        ${metrics['total_pnl']:,.2f}")
            print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
            print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:.1f}%")
            
            print("\n" + "=" * 100)
        
        logger.info("Performance monitor stopped")


if __name__ == "__main__":
    main()
