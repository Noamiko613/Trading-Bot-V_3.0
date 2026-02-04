"""
Multi-Metric Evaluation System for RL Training
==============================================

Implements comprehensive evaluation gates with stability windows to prevent
false positives and ensure robust model performance before paper trading.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
import json
from pathlib import Path

from utils.analytics import PerformanceAnalytics


class StabilityWindow:
    """Represents a single evaluation window"""
    
    def __init__(self, window_id: int, trades: List[Dict], metrics: Dict):
        self.window_id = window_id
        self.trades = trades
        self.metrics = metrics
        self.timestamp = datetime.now().isoformat()
        self.passed = False


class MultiMetricEvaluator:
    """
    Evaluates model performance using multiple metrics with stability windows.
    
    Requires:
    - Win rate ≥ threshold
    - Profit factor ≥ threshold
    - Sharpe ratio ≥ threshold
    - Max drawdown ≤ threshold
    - Multiple consecutive successful windows
    - No recent safety trips (kill-switches)
    """
    
    def __init__(
        self,
        win_rate_threshold: float = 68.0,
        profit_factor_threshold: float = 1.3,
        sharpe_threshold: float = 1.0,
        max_drawdown_threshold: float = 12.0,
        min_trades_per_window: int = 500,
        stability_windows_required: int = 3,
        window_size_trades: int = 500,
        db_path: str = "sim_results/trades.db",
    ):
        self.win_rate_threshold = win_rate_threshold
        self.profit_factor_threshold = profit_factor_threshold
        self.sharpe_threshold = sharpe_threshold
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_trades_per_window = min_trades_per_window
        self.stability_windows_required = stability_windows_required
        self.window_size_trades = window_size_trades
        
        self.analytics = PerformanceAnalytics(db_path=db_path)
        
        # Track evaluation windows
        self.evaluation_windows: deque = deque(maxlen=stability_windows_required * 2)
        self.window_counter = 0
        
        # Track safety trips (kill-switch activations)
        self.safety_trips: deque = deque(maxlen=100)
        
        # State persistence
        self.state_file = Path("models/rl_models/evaluation_state.json")
        self._load_state()
    
    def _load_state(self):
        """Load evaluation state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.window_counter = state.get('window_counter', 0)
                    # Load recent windows
                    for w in state.get('recent_windows', []):
                        self.evaluation_windows.append(StabilityWindow(
                            window_id=w['window_id'],
                            trades=[],
                            metrics=w['metrics']
                        ))
            except Exception:
                pass
    
    def _save_state(self):
        """Save evaluation state to disk"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                'window_counter': self.window_counter,
                'recent_windows': [
                    {
                        'window_id': w.window_id,
                        'metrics': w.metrics,
                        'timestamp': w.timestamp,
                        'passed': w.passed
                    }
                    for w in list(self.evaluation_windows)[-self.stability_windows_required:]
                ]
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
    
    def record_safety_trip(self, reason: str):
        """Record a safety trip (kill-switch activation)"""
        self.safety_trips.append({
            'timestamp': datetime.now().isoformat(),
            'reason': reason
        })
    
    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """
        Calculate Sharpe ratio from equity curve.
        Uses rolling equity changes as returns.
        """
        if len(trades) < 2:
            return 0.0
        
        # Build equity curve from trades
        equity_curve = []
        cumulative_pnl = 0.0
        starting_equity = 100000.0  # Default starting balance
        
        # Sort trades by closed_time
        sorted_trades = sorted(trades, key=lambda t: t.get('closed_time', ''))
        
        for trade in sorted_trades:
            cumulative_pnl += trade.get('pnl', 0.0)
            equity_curve.append(starting_equity + cumulative_pnl)
        
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate returns
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Annualized Sharpe (assuming ~252 trading days for crypto)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        return float(sharpe)
    
    def evaluate_window(self, trades: List[Dict]) -> Tuple[bool, Dict]:
        """
        Evaluate a single window of trades against all metrics.
        
        Returns:
            (passed: bool, metrics: Dict)
        """
        if len(trades) < self.min_trades_per_window:
            return False, {
                'total_trades': len(trades),
                'required': self.min_trades_per_window,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0
            }
        
        # Filter out invalid trades
        valid_trades = [t for t in trades if t.get('pnl') is not None]
        if len(valid_trades) < self.min_trades_per_window:
            return False, {
                'total_trades': len(valid_trades),
                'required': self.min_trades_per_window,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0
            }
        
        # Calculate all metrics
        metrics = self.analytics.calculate_metrics(valid_trades)
        
        # Calculate Sharpe ratio (fix the calculation)
        sharpe = self._calculate_sharpe_ratio(valid_trades)
        metrics['sharpe_ratio'] = sharpe
        
        # Fix max_drawdown_pct calculation - if it's 0 but we have trades, recalculate
        if metrics.get('max_drawdown_pct', 0.0) == 0.0 and len(valid_trades) > 0:
            # Recalculate drawdown from equity curve (cap equity at 0 so drawdown % at most 100%)
            starting_equity = 100000.0  # Default starting balance
            cumulative_pnl = 0.0
            equity_curve = [starting_equity]
            
            sorted_trades = sorted(valid_trades, key=lambda t: t.get('closed_time', ''))
            for trade in sorted_trades:
                cumulative_pnl += trade.get('pnl', 0.0)
                equity_curve.append(max(0.0, starting_equity + cumulative_pnl))
            
            if len(equity_curve) > 1:
                equity_array = np.array(equity_curve)
                running_max = np.maximum.accumulate(equity_array)
                drawdowns = (running_max - equity_array) / np.maximum(running_max, 1.0) * 100.0
                max_dd = min(100.0, float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0)
                metrics['max_drawdown_pct'] = max_dd
        
        # Check all thresholds
        win_rate_ok = metrics.get('win_rate', 0.0) >= self.win_rate_threshold
        profit_factor_ok = metrics.get('profit_factor', 0.0) >= self.profit_factor_threshold
        sharpe_ok = sharpe >= self.sharpe_threshold
        drawdown_ok = metrics.get('max_drawdown_pct', 100.0) <= self.max_drawdown_threshold
        
        # Check for recent safety trips (within last window)
        recent_trips = [
            trip for trip in self.safety_trips
            if datetime.fromisoformat(trip['timestamp']) > datetime.now() - timedelta(days=1)
        ]
        no_safety_trips = len(recent_trips) == 0
        
        # All metrics must pass
        passed = win_rate_ok and profit_factor_ok and sharpe_ok and drawdown_ok and no_safety_trips
        
        metrics['evaluation_passed'] = passed
        metrics['win_rate_ok'] = win_rate_ok
        metrics['profit_factor_ok'] = profit_factor_ok
        metrics['sharpe_ok'] = sharpe_ok
        metrics['drawdown_ok'] = drawdown_ok
        metrics['no_safety_trips'] = no_safety_trips
        
        return passed, metrics
    
    def check_stability(self, all_trades: List[Dict]) -> Tuple[bool, Dict]:
        """
        Check if model has passed stability requirements.
        
        Evaluates the most recent N windows and requires all to pass.
        
        Returns:
            (stable: bool, summary: Dict)
        """
        # Filter out trades without required fields
        valid_trades = [t for t in all_trades if t.get('closed_time') and t.get('pnl') is not None]
        
        if len(valid_trades) < self.min_trades_per_window * self.stability_windows_required:
            return False, {
                'reason': 'insufficient_trades',
                'total_trades': len(valid_trades),
                'required': self.min_trades_per_window * self.stability_windows_required
            }
        
        # Get recent trades for evaluation (sort by closed_time, handle missing timestamps)
        def get_closed_time(trade):
            ct = trade.get('closed_time', '')
            if isinstance(ct, str):
                try:
                    return datetime.fromisoformat(ct.replace('Z', '+00:00'))
                except:
                    return datetime.min
            return datetime.min
        
        recent_trades = sorted(valid_trades, key=get_closed_time)[-self.window_size_trades * self.stability_windows_required:]
        
        # Evaluate current window
        current_window_trades = recent_trades[-self.window_size_trades:]
        passed, metrics = self.evaluate_window(current_window_trades)
        
        # Create window record
        self.window_counter += 1
        window = StabilityWindow(
            window_id=self.window_counter,
            trades=current_window_trades,
            metrics=metrics
        )
        window.passed = passed
        self.evaluation_windows.append(window)
        
        # Check if we have enough consecutive passing windows
        recent_windows = list(self.evaluation_windows)[-self.stability_windows_required:]
        
        if len(recent_windows) < self.stability_windows_required:
            self._save_state()
            return False, {
                'reason': 'insufficient_windows',
                'windows_evaluated': len(recent_windows),
                'windows_required': self.stability_windows_required,
                'current_window': metrics
            }
        
        # All recent windows must have passed
        all_passed = all(w.passed for w in recent_windows)
        
        summary = {
            'stable': all_passed,
            'windows_evaluated': len(recent_windows),
            'windows_passed': sum(1 for w in recent_windows if w.passed),
            'current_window': metrics,
            'recent_windows': [
                {
                    'window_id': w.window_id,
                    'passed': w.passed,
                    'metrics': w.metrics
                }
                for w in recent_windows
            ]
        }
        
        self._save_state()
        return all_passed, summary

