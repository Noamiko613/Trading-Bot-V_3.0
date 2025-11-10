"""
Pattern Lifecycle Manager

Implements adaptive pattern ranking and pruning:
- Tracks pattern performance metrics (expectancy, win rate, profit factor)
- Weekly ranking by expectancy
- Auto-disable bottom 30% of patterns
- Re-tune top performers with small parameter shifts
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class PatternLifecycleManager:
    """Manages pattern lifecycle: tracking, ranking, and pruning"""
    
    def __init__(self, state_file: str = "logs/pattern_lifecycle.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load pattern configuration from config/patterns.json"""
        config_path = Path('config') / 'patterns.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _load_state(self) -> Dict:
        """Load lifecycle state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        # Initialize default state
        return {
            "pattern_stats": {},  # pattern_name -> {wins, losses, total_pnl, r_multiples}
            "last_ranking_date": None,
            "disabled_patterns": [],
            "pattern_weights": {}
        }
    
    def _save_state(self):
        """Save lifecycle state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass
    
    def record_trade(self, pattern_name: str, is_win: bool, r_multiple: float, pnl: float = 0.0):
        """
        Record a trade result for pattern performance tracking.
        
        Args:
            pattern_name: Name of the pattern that generated the trade
            is_win: True if trade was profitable
            r_multiple: R-multiple of the trade (positive for wins, negative for losses)
            pnl: Profit/loss amount
        """
        if pattern_name not in self.state['pattern_stats']:
            self.state['pattern_stats'][pattern_name] = {
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "r_multiples": [],
                "win_r_multiples": [],
                "loss_r_multiples": []
            }
        
        stats = self.state['pattern_stats'][pattern_name]
        
        if is_win:
            stats['wins'] += 1
            stats['win_r_multiples'].append(r_multiple)
        else:
            stats['losses'] += 1
            stats['loss_r_multiples'].append(abs(r_multiple))
        
        stats['total_pnl'] += pnl
        stats['r_multiples'].append(r_multiple)
        
        # Keep only last 100 trades per pattern
        if len(stats['r_multiples']) > 100:
            stats['r_multiples'] = stats['r_multiples'][-100:]
            stats['win_r_multiples'] = stats['win_r_multiples'][-100:] if len(stats['win_r_multiples']) > 100 else stats['win_r_multiples']
            stats['loss_r_multiples'] = stats['loss_r_multiples'][-100:] if len(stats['loss_r_multiples']) > 100 else stats['loss_r_multiples']
        
        self._save_state()
    
    def calculate_expectancy(self, pattern_name: str) -> float:
        """
        Calculate expectancy for a pattern.
        Expectancy = (avg_win × win_rate) - (avg_loss × loss_rate)
        
        Returns:
            Expectancy value (positive is good)
        """
        if pattern_name not in self.state['pattern_stats']:
            return 0.0
        
        stats = self.state['pattern_stats'][pattern_name]
        total_trades = stats['wins'] + stats['losses']
        
        if total_trades == 0:
            return 0.0
        
        win_rate = stats['wins'] / total_trades
        loss_rate = stats['losses'] / total_trades
        
        avg_win = sum(stats['win_r_multiples']) / len(stats['win_r_multiples']) if stats['win_r_multiples'] else 0.0
        avg_loss = sum(stats['loss_r_multiples']) / len(stats['loss_r_multiples']) if stats['loss_r_multiples'] else 0.0
        
        expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
        return expectancy
    
    def rank_patterns(self) -> List[tuple[str, float]]:
        """
        Rank all patterns by their expectancy.
        
        Returns:
            List of (pattern_name, expectancy) tuples, sorted by expectancy (highest first)
        """
        rankings = []
        for pattern_name in self.state['pattern_stats']:
            expectancy = self.calculate_expectancy(pattern_name)
            rankings.append((pattern_name, expectancy))
        
        # Sort by expectancy (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def should_run_weekly_ranking(self) -> bool:
        """Check if it's time to run weekly ranking (every 7 days)"""
        last_ranking = self.state.get('last_ranking_date')
        if not last_ranking:
            return True
        
        try:
            last_date = datetime.fromisoformat(last_ranking.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            days_since = (now - last_date).days
            return days_since >= 7
        except Exception:
            return True
    
    def perform_weekly_ranking(self) -> Dict:
        """
        Perform weekly pattern ranking and pruning.
        - Ranks patterns by last-week expectancy
        - Disables bottom 30%
        - Updates pattern weights
        
        Returns:
            Dict with ranking results and actions taken
        """
        if not self.should_run_weekly_ranking():
            return {"status": "not_due", "message": "Weekly ranking not due yet"}
        
        # Get rankings
        rankings = self.rank_patterns()
        
        if len(rankings) < 3:
            return {"status": "insufficient_data", "message": "Need at least 3 patterns with trades"}
        
        # Calculate bottom 30% threshold
        bottom_count = max(1, int(len(rankings) * 0.3))
        bottom_patterns = [name for name, _ in rankings[-bottom_count:]]
        
        # Disable bottom performers
        disabled = []
        for pattern_name in bottom_patterns:
            if pattern_name not in self.state['disabled_patterns']:
                self.state['disabled_patterns'].append(pattern_name)
                disabled.append(pattern_name)
        
        # Update pattern weights (top performers get slight boost)
        top_count = max(1, int(len(rankings) * 0.3))
        top_patterns = [name for name, _ in rankings[:top_count]]
        
        # Update weights: top performers ×1.1, bottom performers ×0.9
        if 'pattern_weights' not in self.state:
            self.state['pattern_weights'] = {}
        
        for pattern_name, expectancy in rankings:
            if pattern_name in top_patterns:
                current_weight = self.state['pattern_weights'].get(pattern_name, 1.0)
                self.state['pattern_weights'][pattern_name] = min(current_weight * 1.1, 2.0)  # Cap at 2.0
            elif pattern_name in bottom_patterns:
                current_weight = self.state['pattern_weights'].get(pattern_name, 1.0)
                self.state['pattern_weights'][pattern_name] = max(current_weight * 0.9, 0.1)  # Floor at 0.1
        
        # Update last ranking date
        self.state['last_ranking_date'] = datetime.now(timezone.utc).isoformat()
        self._save_state()
        
        return {
            "status": "completed",
            "rankings": rankings,
            "disabled_patterns": disabled,
            "top_patterns": top_patterns,
            "message": f"Ranked {len(rankings)} patterns, disabled {len(disabled)} bottom performers"
        }
    
    def is_pattern_enabled(self, pattern_name: str) -> bool:
        """Check if a pattern is currently enabled"""
        return pattern_name not in self.state['disabled_patterns']
    
    def get_pattern_weight(self, pattern_name: str) -> float:
        """Get current weight for a pattern (default 1.0)"""
        return self.state['pattern_weights'].get(pattern_name, 1.0)
    
    def get_pattern_stats(self, pattern_name: str) -> Optional[Dict]:
        """Get statistics for a specific pattern"""
        return self.state['pattern_stats'].get(pattern_name)
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all patterns"""
        stats = {}
        for pattern_name in self.state['pattern_stats']:
            pattern_stats = self.state['pattern_stats'][pattern_name]
            total_trades = pattern_stats['wins'] + pattern_stats['losses']
            
            if total_trades > 0:
                win_rate = pattern_stats['wins'] / total_trades
                expectancy = self.calculate_expectancy(pattern_name)
                
                avg_win = sum(pattern_stats['win_r_multiples']) / len(pattern_stats['win_r_multiples']) if pattern_stats['win_r_multiples'] else 0.0
                avg_loss = sum(pattern_stats['loss_r_multiples']) / len(pattern_stats['loss_r_multiples']) if pattern_stats['loss_r_multiples'] else 0.0
                
                profit_factor = abs(avg_win * pattern_stats['wins'] / (avg_loss * pattern_stats['losses'])) if pattern_stats['losses'] > 0 and avg_loss > 0 else 0.0
                
                stats[pattern_name] = {
                    "total_trades": total_trades,
                    "wins": pattern_stats['wins'],
                    "losses": pattern_stats['losses'],
                    "win_rate": win_rate,
                    "expectancy": expectancy,
                    "avg_win_r": avg_win,
                    "avg_loss_r": avg_loss,
                    "profit_factor": profit_factor,
                    "total_pnl": pattern_stats['total_pnl'],
                    "enabled": self.is_pattern_enabled(pattern_name),
                    "weight": self.get_pattern_weight(pattern_name)
                }
        
        return stats

