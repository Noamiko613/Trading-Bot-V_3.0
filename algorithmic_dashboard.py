"""
Algorithmic Trading Dashboard
==============================

Real-time dashboard for monitoring the algorithmic trading system.
Shows statistics, algorithm performance, and trading activity.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Colors:
    """ANSI color codes for terminal"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    GRAY = '\033[90m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_RED = '\033[91m'
    
    @staticmethod
    def disable():
        """Return empty strings for all colors (for terminals that don't support them)"""
        class NoColors:
            RESET = ''
            BOLD = ''
            RED = ''
            GREEN = ''
            YELLOW = ''
            BLUE = ''
            MAGENTA = ''
            CYAN = ''
            WHITE = ''
            GRAY = ''
            BRIGHT_GREEN = ''
            BRIGHT_YELLOW = ''
            BRIGHT_RED = ''
        return NoColors()


class AlgorithmicDashboard:
    """Dashboard for algorithmic trading system"""
    
    def __init__(
        self,
        pairs: List[str],
        update_interval: float = 2.0,
    ):
        self.pairs = pairs
        self.update_interval = update_interval
        self.running = False
        
        # Detect if colors are supported (Windows PowerShell often doesn't)
        self.use_colors = self._check_color_support()
        
        # Get terminal width
        try:
            self.terminal_width = os.get_terminal_size().columns
        except:
            self.terminal_width = 120
    
    def _check_color_support(self) -> bool:
        """Check if terminal supports ANSI colors"""
        # On Windows, check if we're in a modern terminal
        if os.name == 'nt':
            # PowerShell 7+ and Windows Terminal support colors, but older PowerShell doesn't
            # Check environment variable or try to detect
            term = os.getenv('TERM', '')
            if 'xterm' in term.lower() or 'ansi' in term.lower():
                return True
            # Disable by default on Windows to avoid escape code issues
            return False
        # On Unix-like systems, assume colors work
        return True
    
    def start(self):
        """Start dashboard"""
        self.running = True
        try:
            while self.running:
                self._render()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            self.running = False
    
    def stop(self):
        """Stop dashboard"""
        self.running = False
    
    def _load_status(self, pair: str) -> Optional[Dict]:
        """Load status for a pair"""
        try:
            pair_normalized = pair.replace('/', '_').replace('-', '_')
            status_path = Path("algorithmic_trading") / pair_normalized / "algorithm_status.json"
            
            if status_path.exists():
                with open(status_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return None
    
    def _render(self):
        """Render dashboard"""
        # Use colors or not based on support
        c = Colors() if self.use_colors else Colors.disable()
        
        # Clear screen (only if colors supported)
        if self.use_colors:
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()
        
        # Header
        print(f"{c.BOLD}{c.CYAN}{'='*self.terminal_width}{c.RESET}")
        print(f"{c.BOLD}{c.CYAN}ALGORITHMIC TRADING SYSTEM DASHBOARD{c.RESET}")
        print(f"{c.BOLD}{c.CYAN}{'='*self.terminal_width}{c.RESET}")
        print(f"{c.GRAY}Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{c.RESET}\n")
        
        # Render each pair
        for pair in self.pairs:
            self._render_pair(pair)
            print()
        
        # Footer
        c = Colors() if self.use_colors else Colors.disable()
        print(f"{c.GRAY}{'='*self.terminal_width}{c.RESET}")
        print(f"{c.GRAY}Press Ctrl+C to exit{c.RESET}")
    
    def _render_pair(self, pair: str):
        """Render statistics for a single pair"""
        c = Colors() if self.use_colors else Colors.disable()
        
        status = self._load_status(pair)
        
        if not status:
            print(f"{c.YELLOW}[{pair}] No data available{c.RESET}")
            return
        
        stats = status.get('statistics', {})
        overall = stats.get('overall_stats', {})
        algo_stats = stats.get('algorithm_stats', {})
        system_stats = stats.get('system_stats', {})
        
        # Get balance - check multiple locations in status file
        balance = status.get('balance') or stats.get('balance') or overall.get('balance', 0)
        equity = status.get('equity') or stats.get('equity') or overall.get('equity', 0)
        
        # If balance is still 0, it might be a new instance - show starting balance hint
        if balance == 0 and equity == 0:
            balance = 10000.0  # Default starting balance
            equity = 10000.0
        
        # Pair header
        print(f"{c.BOLD}{c.WHITE}[{pair}]{c.RESET}")
        print(f"{c.GRAY}{'-'*80}{c.RESET}")
        
        # Account info
        balance_color = c.GREEN if equity >= balance else c.RED
        
        print(f"  {c.BOLD}Balance:{c.RESET} {balance_color}${balance:,.2f}{c.RESET} | "
              f"{c.BOLD}Equity:{c.RESET} {balance_color}${equity:,.2f}{c.RESET}")
        
        # Trading stats
        open_trades = overall.get('open_trades', 0)
        closed_trades = overall.get('closed_trades', 0)
        win_rate = overall.get('win_rate', 0)
        total_pnl = overall.get('total_pnl', 0)
        profit_factor = overall.get('profit_factor', 0)
        
        win_rate_color = c.GREEN if win_rate >= 66 else c.YELLOW if win_rate >= 50 else c.RED
        pnl_color = c.GREEN if total_pnl > 0 else c.RED
        
        print(f"  {c.BOLD}Open Trades:{c.RESET} {open_trades} | "
              f"{c.BOLD}Closed Trades:{c.RESET} {closed_trades}")
        print(f"  {c.BOLD}Win Rate:{c.RESET} {win_rate_color}{win_rate:.2f}%{c.RESET} | "
              f"{c.BOLD}PnL:{c.RESET} {pnl_color}${total_pnl:,.2f}{c.RESET} | "
              f"{c.BOLD}Profit Factor:{c.RESET} {profit_factor:.2f}")
        
        # Algorithm stats
        algorithms_active = system_stats.get('algorithms_active', 0)
        algorithms_tested = system_stats.get('algorithms_tested', 0)
        algorithms_validated = system_stats.get('algorithms_validated', 0)
        
        print(f"  {c.BOLD}Algorithms:{c.RESET} {c.GREEN}{algorithms_active} active{c.RESET} | "
              f"{algorithms_validated} validated | {algorithms_tested} tested")
        
        # Active algorithms detail
        if algo_stats:
            print(f"\n  {c.BOLD}Active Algorithms:{c.RESET}")
            for algo_id, algo_data in list(algo_stats.items())[:5]:  # Show top 5
                name = algo_data.get('name', 'Unknown')
                trades = algo_data.get('total_trades', 0)
                wr = algo_data.get('win_rate', 0)
                pnl = algo_data.get('total_pnl', 0)
                
                wr_color = c.GREEN if wr >= 66 else c.YELLOW if wr >= 50 else c.RED
                pnl_color = c.GREEN if pnl > 0 else c.RED
                
                print(f"    • {name}: {trades} trades | "
                      f"WR: {wr_color}{wr:.1f}%{c.RESET} | "
                      f"PnL: {pnl_color}${pnl:,.2f}{c.RESET}")
        
        # Probationary algorithms
        probation_stats = stats.get('probationary_stats', {})
        if probation_stats:
            print(f"\n  {c.BOLD}{c.YELLOW}Probationary Algorithms (Being Monitored):{c.RESET}")
            for algo_id, algo_data in list(probation_stats.items())[:3]:  # Show top 3
                name = algo_data.get('name', 'Unknown')
                trades = algo_data.get('total_trades', 0)
                wr = algo_data.get('win_rate', 0)
                pnl = algo_data.get('total_pnl', 0)
                needed = max(0, 50 - trades)  # Show how many more trades needed
                
                wr_color = c.GREEN if wr >= 60 else c.YELLOW if wr >= 50 else c.RED
                pnl_color = c.GREEN if pnl > 0 else c.RED
                
                print(f"    • {c.YELLOW}{name}{c.RESET}: {trades}/50 trades | "
                      f"WR: {wr_color}{wr:.1f}%{c.RESET} | "
                      f"PnL: {pnl_color}${pnl:,.2f}{c.RESET} | "
                      f"{needed} more needed")

