"""
Strategy Optimizer and Backtesting Framework
=============================================

Advanced optimization system for:
- Parameter optimization using grid search and genetic algorithms
- Walk-forward analysis
- Monte Carlo simulation
- Strategy comparison
- Performance attribution
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed


class StrategyOptimizer:
    """Optimize trading strategy parameters"""
    
    def __init__(self, results_dir: str = "optimization_results"):
        """
        Initialize strategy optimizer
        
        Args:
            results_dir: Directory to save optimization results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def grid_search(self, parameter_grid: Dict[str, List], 
                   eval_function, metric: str = 'sharpe_ratio',
                   n_jobs: int = -1) -> Dict:
        """
        Perform grid search optimization
        
        Args:
            parameter_grid: Dictionary of parameters and their ranges
            eval_function: Function that evaluates a parameter set
            metric: Metric to optimize
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        
        Returns:
            Best parameters and results
        """
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(product(*param_values))
        
        print(f"[OPTIMIZER] Testing {len(combinations)} parameter combinations...")
        
        results = []
        
        if n_jobs == 1:
            # Sequential execution
            for i, combo in enumerate(combinations):
                params = dict(zip(param_names, combo))
                result = eval_function(params)
                result['params'] = params
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"[OPTIMIZER] Completed {i + 1}/{len(combinations)} combinations")
        else:
            # Parallel execution
            max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for combo in combinations:
                    params = dict(zip(param_names, combo))
                    future = executor.submit(eval_function, params)
                    futures[future] = params
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        result['params'] = futures[future]
                        results.append(result)
                        
                        if (i + 1) % 10 == 0:
                            print(f"[OPTIMIZER] Completed {i + 1}/{len(combinations)} combinations")
                    except Exception as e:
                        print(f"[OPTIMIZER] Error evaluating {futures[future]}: {e}")
        
        # Find best parameters
        results_df = pd.DataFrame(results)
        best_idx = results_df[metric].idxmax()
        best_result = results_df.loc[best_idx].to_dict()
        
        # Save results
        self._save_optimization_results(results_df, 'grid_search')
        
        return best_result
    
    def walk_forward_analysis(self, data: pd.DataFrame, 
                             train_period_days: int = 30,
                             test_period_days: int = 7,
                             parameter_grid: Dict = None,
                             strategy_function = None) -> Dict:
        """
        Perform walk-forward analysis
        
        Args:
            data: Historical data
            train_period_days: Training period in days
            test_period_days: Testing period in days
            parameter_grid: Parameters to optimize
            strategy_function: Strategy function to evaluate
        
        Returns:
            Walk-forward results
        """
        results = []
        
        # Split data into train/test windows
        total_days = (data.index[-1] - data.index[0]).days
        n_windows = (total_days - train_period_days) // test_period_days
        
        print(f"[WFA] Running {n_windows} walk-forward windows...")
        
        for i in range(n_windows):
            # Define train and test periods
            train_start = data.index[0] + timedelta(days=i * test_period_days)
            train_end = train_start + timedelta(days=train_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_period_days)
            
            # Get data for this window
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Optimize on training data
            best_params = self._optimize_on_window(train_data, parameter_grid, strategy_function)
            
            # Test on out-of-sample data
            test_result = strategy_function(test_data, best_params)
            
            results.append({
                'window': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'best_params': best_params,
                'test_metrics': test_result
            })
            
            print(f"[WFA] Window {i + 1}/{n_windows} complete")
        
        # Save results
        self._save_wfa_results(results)
        
        return results
    
    def _optimize_on_window(self, data: pd.DataFrame, parameter_grid: Dict, 
                          strategy_function) -> Dict:
        """Optimize parameters on a single window"""
        # Simplified optimization for walk-forward
        # In production, use grid_search or other method
        
        best_score = -np.inf
        best_params = {}
        
        # Test a subset of parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        # Sample combinations instead of testing all
        n_samples = min(50, len(list(product(*param_values))))
        combinations = np.random.choice(len(list(product(*param_values))), 
                                       size=n_samples, replace=False)
        
        for combo_idx in combinations:
            combo = list(product(*param_values))[combo_idx]
            params = dict(zip(param_names, combo))
            
            result = strategy_function(data, params)
            score = result.get('sharpe_ratio', 0)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params
    
    def monte_carlo_simulation(self, trades: List[Dict], n_simulations: int = 1000) -> Dict:
        """
        Perform Monte Carlo simulation on trade results
        
        Args:
            trades: List of historical trades
            n_simulations: Number of simulations to run
        
        Returns:
            Simulation results with confidence intervals
        """
        if not trades:
            return {}
        
        trade_returns = [t.get('pnl', 0) for t in trades]
        n_trades = len(trade_returns)
        
        simulation_results = []
        
        print(f"[MC] Running {n_simulations} Monte Carlo simulations...")
        
        for i in range(n_simulations):
            # Randomly resample trades with replacement
            resampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)
            
            # Calculate cumulative return
            cumulative_return = np.sum(resampled_returns)
            
            # Calculate drawdown
            cumulative = np.cumsum(resampled_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown = np.max(drawdown)
            
            simulation_results.append({
                'cumulative_return': cumulative_return,
                'max_drawdown': max_drawdown
            })
        
        # Calculate statistics
        returns = [r['cumulative_return'] for r in simulation_results]
        drawdowns = [r['max_drawdown'] for r in simulation_results]
        
        results = {
            'n_simulations': n_simulations,
            'return': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_25': np.percentile(returns, 25),
                'percentile_75': np.percentile(returns, 75),
                'percentile_95': np.percentile(returns, 95)
            },
            'drawdown': {
                'mean': np.mean(drawdowns),
                'median': np.median(drawdowns),
                'std': np.std(drawdowns),
                'min': np.min(drawdowns),
                'max': np.max(drawdowns),
                'percentile_5': np.percentile(drawdowns, 5),
                'percentile_25': np.percentile(drawdowns, 25),
                'percentile_75': np.percentile(drawdowns, 75),
                'percentile_95': np.percentile(drawdowns, 95)
            }
        }
        
        # Save results
        self._save_monte_carlo_results(results)
        
        return results
    
    def compare_strategies(self, strategy_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Args:
            strategy_results: Dictionary of strategy name -> results
        
        Returns:
            Comparison DataFrame
        """
        comparison = []
        
        for name, results in strategy_results.items():
            comparison.append({
                'strategy': name,
                'total_pnl': results.get('total_pnl', 0),
                'win_rate': results.get('win_rate', 0),
                'profit_factor': results.get('profit_factor', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'avg_r_multiple': results.get('avg_r_multiple', 0)
            })
        
        df = pd.DataFrame(comparison)
        
        # Save comparison
        csv_path = os.path.join(self.results_dir, 
                               f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(csv_path, index=False)
        
        return df
    
    def _save_optimization_results(self, results_df: pd.DataFrame, method: str):
        """Save optimization results"""
        csv_path = os.path.join(self.results_dir, 
                               f"{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"[OPTIMIZER] Results saved to {csv_path}")
    
    def _save_wfa_results(self, results: List[Dict]):
        """Save walk-forward analysis results"""
        json_path = os.path.join(self.results_dir, 
                                f"wfa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Convert datetime objects to strings
        serializable_results = []
        for r in results:
            serializable_r = r.copy()
            for key in ['train_start', 'train_end', 'test_start', 'test_end']:
                if key in serializable_r and isinstance(serializable_r[key], datetime):
                    serializable_r[key] = serializable_r[key].isoformat()
            serializable_results.append(serializable_r)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"[WFA] Results saved to {json_path}")
    
    def _save_monte_carlo_results(self, results: Dict):
        """Save Monte Carlo simulation results"""
        json_path = os.path.join(self.results_dir, 
                                f"monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[MC] Results saved to {json_path}")


class BacktestEngine:
    """Backtesting engine for strategy evaluation"""
    
    def __init__(self):
        """Initialize backtest engine"""
        self.trades = []
        self.balance = 100000.0
        self.initial_balance = 100000.0
    
    def run_backtest(self, data: pd.DataFrame, strategy, params: Dict) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: Historical OHLCV data
            strategy: Strategy function
            params: Strategy parameters
        
        Returns:
            Backtest results
        """
        self.trades = []
        self.balance = self.initial_balance
        equity_curve = []
        
        # Run strategy on historical data
        signals = strategy(data, params)
        
        # Execute trades based on signals
        for i, signal in enumerate(signals):
            if signal['action'] == 'BUY':
                trade = self._execute_buy(signal, data.iloc[i])
                if trade:
                    self.trades.append(trade)
            elif signal['action'] == 'SELL' and self.trades:
                self._execute_sell(signal, data.iloc[i])
            
            equity_curve.append(self.balance)
        
        # Calculate metrics
        return self._calculate_backtest_metrics(equity_curve)
    
    def _execute_buy(self, signal: Dict, candle: pd.Series) -> Optional[Dict]:
        """Execute buy order"""
        # Simplified trade execution
        entry_price = candle['close']
        position_size = signal.get('size', 0.1)
        
        trade = {
            'entry_price': entry_price,
            'entry_time': candle.name,
            'size': position_size,
            'stop_loss': signal.get('stop_loss', entry_price * 0.98),
            'take_profit': signal.get('take_profit', entry_price * 1.04)
        }
        
        return trade
    
    def _execute_sell(self, signal: Dict, candle: pd.Series):
        """Execute sell order"""
        if not self.trades:
            return
        
        trade = self.trades[-1]
        exit_price = candle['close']
        
        pnl = (exit_price - trade['entry_price']) * trade['size']
        self.balance += pnl
        
        trade['exit_price'] = exit_price
        trade['exit_time'] = candle.name
        trade['pnl'] = pnl
    
    def _calculate_backtest_metrics(self, equity_curve: List[float]) -> Dict:
        """Calculate backtest performance metrics"""
        equity = np.array(equity_curve)
        
        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0] * 100
        
        # Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100
        max_drawdown = np.max(drawdown)
        
        # Sharpe ratio
        returns = np.diff(equity) / equity[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 0 else 0
        
        return {
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': equity[-1],
            'n_trades': len([t for t in self.trades if 'exit_price' in t])
        }


if __name__ == "__main__":
    # Test strategy optimizer
    optimizer = StrategyOptimizer()
    
    # Example parameter grid
    param_grid = {
        'rsi_period': [10, 14, 20],
        'rsi_overbought': [65, 70, 75],
        'rsi_oversold': [25, 30, 35],
        'ma_period': [20, 50, 100]
    }
    
    # Example evaluation function
    def eval_params(params):
        # Simulate strategy evaluation
        import random
        return {
            'sharpe_ratio': random.uniform(0.5, 2.0),
            'total_return': random.uniform(-10, 50),
            'max_drawdown': random.uniform(5, 25)
        }
    
    # Run grid search (sequential for testing)
    # best = optimizer.grid_search(param_grid, eval_params, n_jobs=1)
    # print(f"\nBest parameters: {best}")
    
    print("Strategy Optimizer initialized. Ready for optimization tasks.")
