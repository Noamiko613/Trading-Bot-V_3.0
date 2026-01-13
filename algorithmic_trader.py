"""
Algorithmic Trading System
==========================

Main system that monitors pairs, discovers profitable algorithms,
and trades using validated strategies. Runs independently from RL training.

ISOLATION FROM RL TRAINING:
- Uses same CoinEx data source but separate data fetcher instances
- Separate paper trading system (algorithmic_paper_trading.py)
- Separate algorithm storage (algorithmic_trading/algorithms/)
- Separate status files and databases
- No shared state, variables, or resources with RL training
"""

import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque

from coinEx_getting_data import CoinExDataFetcher
from algorithmic_paper_trading import AlgorithmicPaperTrading
from algorithm_discovery import AlgorithmDiscovery, AlgorithmTemplate, MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy, MomentumStrategy
from utils.logger import ComponentLogger
from verify_patterns import normalize_symbol_to_ccxt


class PairMonitor:
    """Monitors a single trading pair and manages its algorithms"""
    
    def __init__(
        self,
        symbol: str,
        mode: str = "spot",
        starting_balance: float = 10000.0,
        min_trades_for_validation: int = 100,
        target_win_rate: float = 66.0,
        target_rr: float = 2.0,
    ):
        self.symbol = symbol
        self.mode = mode
        self.min_trades_for_validation = min_trades_for_validation
        self.target_win_rate = target_win_rate
        self.target_rr = target_rr
        
        # Components
        self.discovery = AlgorithmDiscovery(symbol, mode)
        self.paper_trading = AlgorithmicPaperTrading(symbol, starting_balance, mode)
        
        # Active algorithms (validated and currently trading)
        self.active_algorithms: Dict[str, Dict] = {}
        
        # Probationary algorithms (newly validated, being monitored)
        self.probationary_algorithms: Dict[str, Dict] = {}
        
        # Algorithm testing queue
        self.testing_algorithms: Dict[str, Dict] = {}
        
        # Probation settings (configurable)
        try:
            self.probation_min_trades = int(os.getenv('ALGO_PROBATION_MIN_TRADES', '50'))
            self.probation_min_win_rate = float(os.getenv('ALGO_PROBATION_MIN_WR', '60.0'))
            self.probation_min_rr = float(os.getenv('ALGO_PROBATION_MIN_RR', '1.8'))
        except Exception:
            self.probation_min_trades = 50  # Minimum trades in probation before promotion
            self.probation_min_win_rate = 60.0  # Minimum win rate during probation
            self.probation_min_rr = 1.8  # Minimum R:R during probation
        
        # Statistics
        self.stats = {
            'algorithms_tested': 0,
            'algorithms_validated': 0,
            'algorithms_active': 0,
            'total_trades': 0,
            'last_discovery': None,
            'last_market_update': None,
            'market_data_points': 0,
        }
        
        # Logger (initialize before loading algorithms so it can log)
        self.logger = ComponentLogger.algorithmic_logger()
        
        # Load existing validated algorithms
        self._load_validated_algorithms()
    
    def _load_validated_algorithms(self):
        """Load previously validated algorithms"""
        algorithms = self.discovery.load_algorithms()
        for algo_data in algorithms:
            eval_data = algo_data.get('evaluation', {})
            if eval_data.get('meets_criteria', False):
                self.active_algorithms[algo_data['id']] = {
                    'data': algo_data,
                    'trades_count': 0,
                    'last_signal_time': None,
                }
        
        self.stats['algorithms_active'] = len(self.active_algorithms)
        self.logger.info("algorithms_loaded", symbol=self.symbol, count=len(self.active_algorithms))
    
    def _create_algorithm_from_data(self, algo_data: Dict) -> Optional[AlgorithmTemplate]:
        """Recreate algorithm instance from saved data"""
        name = algo_data.get('name')
        params = algo_data.get('params', {})
        algo_id = algo_data.get('id')
        
        if name == 'MovingAverageCrossover':
            algo = MovingAverageCrossover(name, params, algo_id=algo_id)
        elif name == 'RSIStrategy':
            algo = RSIStrategy(name, params, algo_id=algo_id)
        elif name == 'BollingerBandsStrategy':
            algo = BollingerBandsStrategy(name, params, algo_id=algo_id)
        elif name == 'MomentumStrategy':
            algo = MomentumStrategy(name, params, algo_id=algo_id)
        else:
            return None
        
        return algo
    
    def discover_new_algorithm(self):
        """Generate and test a new algorithm - actively monitors market to find working algorithms"""
        try:
            # Ensure we have fresh market data before testing
            self.discovery.update_market_data()
            self.stats['last_market_update'] = datetime.utcnow().isoformat()
            
            # Check how much data we have
            df = self.discovery.get_dataframe(limit=500)
            if len(df) > 0:
                self.stats['market_data_points'] = len(df)
            
            # Generate new algorithm
            algorithm = self.discovery.generate_algorithm()
            
            # Evaluate it on current market data
            evaluation = self.discovery.evaluate_algorithm(
                algorithm,
                min_trades=self.min_trades_for_validation,
                target_win_rate=self.target_win_rate,
                target_rr=self.target_rr,
            )
            
            self.stats['algorithms_tested'] += 1
            self.stats['last_discovery'] = datetime.utcnow().isoformat()
            
            if evaluation.get('valid') and evaluation.get('meets_criteria'):
                # Save algorithm (but mark as probationary)
                self.discovery.save_algorithm(algorithm, evaluation)
                
                # Add to PROBATIONARY algorithms first (not fully active yet)
                self.probationary_algorithms[algorithm.id] = {
                    'data': {
                        'id': algorithm.id,
                        'name': algorithm.name,
                        'params': algorithm.params,
                        'evaluation': evaluation,
                    },
                    'trades_count': 0,
                    'last_signal_time': None,
                    'probation_start': datetime.utcnow().isoformat(),
                    'probation_trades': 0,
                    'probation_wins': 0,
                    'probation_losses': 0,
                    'probation_pnl': 0.0,
                }
                
                self.stats['algorithms_validated'] += 1
                
                self.logger.info(
                    "algorithm_on_probation",
                    symbol=self.symbol,
                    algorithm_id=algorithm.id,
                    win_rate=evaluation.get('win_rate'),
                    avg_rr=evaluation.get('avg_r_multiple'),
                    total_trades=evaluation.get('total_trades'),
                    note="Algorithm will be monitored in live trading before full activation",
                )
            else:
                reason = evaluation.get('reason', 'does_not_meet_criteria')
                win_rate = evaluation.get('win_rate', 0)
                avg_rr = evaluation.get('avg_r_multiple', 0)
                
                self.logger.info(
                    "algorithm_tested",
                    symbol=self.symbol,
                    algorithm_id=algorithm.id,
                    reason=reason,
                    win_rate=win_rate,
                    avg_rr=avg_rr,
                    total_trades=evaluation.get('total_trades', 0),
                )
        
        except Exception as e:
            self.logger.error("algorithm_discovery_error", symbol=self.symbol, error=str(e))
    
    def check_signals(self):
        """Check for trading signals from active and probationary algorithms"""
        try:
            df = self.discovery.get_dataframe(limit=200)
            if len(df) < 50:
                return
            
            # Check probationary algorithms (newly validated, being monitored)
            for algo_id, algo_info in list(self.probationary_algorithms.items()):
                try:
                    algo_data = algo_info['data']
                    algorithm = self._create_algorithm_from_data(algo_data)
                    
                    if not algorithm:
                        continue
                    
                    signal = algorithm.generate_signal(df)
                    
                    if signal:
                        # Check cooldown
                        last_signal = algo_info.get('last_signal_time')
                        if last_signal:
                            time_since = time.time() - last_signal
                            if time_since < 300:  # 5 minutes cooldown
                                continue
                        
                        entry = signal['entry']
                        stop = signal['stop']
                        tp = signal['tp']
                        side = signal['side']
                        
                        risk = abs(entry - stop)
                        reward = abs(tp - entry)
                        rr = reward / risk if risk > 0 else 0
                        
                        if rr < 2.0:
                            continue
                        
                        # Open trade (probationary algorithms can trade)
                        trade_id = self.paper_trading.open_trade(
                            algorithm_id=algo_id,
                            algorithm_name=algo_data.get('name', 'Unknown'),
                            side=side,
                            entry=entry,
                            stop=stop,
                            tp=tp,
                            risk_pct=0.01,
                        )
                        
                        if trade_id:
                            algo_info['trades_count'] += 1
                            algo_info['probation_trades'] += 1
                            algo_info['last_signal_time'] = time.time()
                            self.stats['total_trades'] += 1
                            
                            self.logger.info(
                                "probationary_signal_taken",
                                symbol=self.symbol,
                                algorithm_id=algo_id,
                                trade_id=trade_id,
                                side=side,
                                rr=round(rr, 2),
                                probation_trades=algo_info['probation_trades'],
                            )
                
                except Exception as e:
                    self.logger.error("probationary_signal_check_error", symbol=self.symbol, algorithm_id=algo_id, error=str(e))
            
            # Check each active algorithm (fully validated)
            for algo_id, algo_info in list(self.active_algorithms.items()):
                try:
                    algo_data = algo_info['data']
                    algorithm = self._create_algorithm_from_data(algo_data)
                    
                    if not algorithm:
                        continue
                    
                    signal = algorithm.generate_signal(df)
                    
                    if signal:
                        # Check if we should take this trade
                        # (avoid too frequent signals from same algorithm)
                        last_signal = algo_info.get('last_signal_time')
                        if last_signal:
                            time_since = time.time() - last_signal
                            if time_since < 300:  # 5 minutes cooldown
                                continue
                        
                        # Open trade
                        entry = signal['entry']
                        stop = signal['stop']
                        tp = signal['tp']
                        side = signal['side']
                        
                        # Calculate risk/reward
                        risk = abs(entry - stop)
                        reward = abs(tp - entry)
                        rr = reward / risk if risk > 0 else 0
                        
                        # Only take trades with good R:R
                        if rr < 2.0:
                            continue
                        
                        # Open trade
                        trade_id = self.paper_trading.open_trade(
                            algorithm_id=algo_id,
                            algorithm_name=algo_data.get('name', 'Unknown'),
                            side=side,
                            entry=entry,
                            stop=stop,
                            tp=tp,
                            risk_pct=0.01,  # 1% risk per trade
                        )
                        
                        if trade_id:
                            algo_info['trades_count'] += 1
                            algo_info['last_signal_time'] = time.time()
                            self.stats['total_trades'] += 1
                            
                            self.logger.info(
                                "signal_taken",
                                symbol=self.symbol,
                                algorithm_id=algo_id,
                                trade_id=trade_id,
                                side=side,
                                rr=round(rr, 2),
                            )
                
                except Exception as e:
                    self.logger.error("signal_check_error", symbol=self.symbol, algorithm_id=algo_id, error=str(e))
        
        except Exception as e:
            self.logger.error("signal_check_error", symbol=self.symbol, error=str(e))
    
    def update_trades(self):
        """Update open trades and check probationary algorithm performance"""
        self.paper_trading.step()
        
        # Check probationary algorithms performance
        self._evaluate_probationary_algorithms()
    
    def _evaluate_probationary_algorithms(self):
        """Evaluate probationary algorithms and promote/remove based on live performance"""
        for algo_id, algo_info in list(self.probationary_algorithms.items()):
            try:
                # Get live trading statistics for this algorithm
                stats = self.paper_trading.get_statistics(algorithm_id=algo_id)
                
                probation_trades = algo_info.get('probation_trades', 0)
                total_trades = stats.get('total_trades', 0)
                win_rate = stats.get('win_rate', 0)
                avg_r_multiple = stats.get('avg_r_multiple', 0)
                total_pnl = stats.get('total_pnl', 0)
                
                # Update probation tracking
                algo_info['probation_trades'] = total_trades
                algo_info['probation_pnl'] = total_pnl
                
                # Check if algorithm has enough trades and meets criteria
                if total_trades >= self.probation_min_trades:
                    if win_rate >= self.probation_min_win_rate and avg_r_multiple >= self.probation_min_rr:
                        # PROMOTE to active algorithms
                        self.active_algorithms[algo_id] = algo_info.copy()
                        del self.probationary_algorithms[algo_id]
                        
                        self.stats['algorithms_active'] = len(self.active_algorithms)
                        
                        self.logger.info(
                            "algorithm_promoted",
                            symbol=self.symbol,
                            algorithm_id=algo_id,
                            probation_trades=total_trades,
                            win_rate=win_rate,
                            avg_rr=avg_r_multiple,
                            pnl=total_pnl,
                            note="Algorithm proven in live trading, now fully active",
                        )
                    else:
                        # REMOVE - failed probation
                        del self.probationary_algorithms[algo_id]
                        
                        self.logger.warning(
                            "algorithm_failed_probation",
                            symbol=self.symbol,
                            algorithm_id=algo_id,
                            probation_trades=total_trades,
                            win_rate=win_rate,
                            avg_rr=avg_r_multiple,
                            pnl=total_pnl,
                            note="Algorithm removed - did not meet live trading criteria",
                        )
                
            except Exception as e:
                self.logger.error("probation_evaluation_error", symbol=self.symbol, algorithm_id=algo_id, error=str(e))
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        algo_stats = {}
        for algo_id, algo_info in self.active_algorithms.items():
            stats = self.paper_trading.get_statistics(algorithm_id=algo_id)
            algo_stats[algo_id] = {
                **stats,
                'name': algo_info['data'].get('name', 'Unknown'),
                'trades_count': algo_info['trades_count'],
                'status': 'active',
            }
        
        # Include probationary algorithms
        probation_stats = {}
        for algo_id, algo_info in self.probationary_algorithms.items():
            stats = self.paper_trading.get_statistics(algorithm_id=algo_id)
            probation_stats[algo_id] = {
                **stats,
                'name': algo_info['data'].get('name', 'Unknown'),
                'trades_count': algo_info.get('probation_trades', 0),
                'status': 'probationary',
                'probation_start': algo_info.get('probation_start'),
            }
        
        overall_stats = self.paper_trading.get_statistics()
        
        return {
            'symbol': self.symbol,
            'balance': round(self.paper_trading.balance, 2),
            'equity': round(self.paper_trading.equity, 2),
            'open_trades': len(self.paper_trading.open_trades),
            'closed_trades': len(self.paper_trading.closed_trades),
            'overall_stats': overall_stats,
            'algorithm_stats': algo_stats,
            'probationary_stats': probation_stats,
            'system_stats': {
                **self.stats,
                'probationary_count': len(self.probationary_algorithms),
            },
        }
    
    def write_status(self):
        """Write status to file"""
        self.paper_trading.write_status()
        
        # Write algorithm status
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': self.symbol,
            'statistics': self.get_statistics(),
        }
        
        # Use consistent symbol normalization to match paper_trading
        safe_symbol = self.symbol.replace('/', '_').replace('-', '_').upper()
        algo_dir = Path(__file__).resolve().parent / "algorithmic_trading" / safe_symbol
        os.makedirs(algo_dir, exist_ok=True)
        
        status_path = algo_dir / "algorithm_status.json"
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)


class AlgorithmicTradingSystem:
    """Main algorithmic trading system manager"""
    
    def __init__(
        self,
        pairs: List[str],
        mode: str = "spot",
        starting_balance_per_pair: float = 10000.0,
        discovery_interval: int = 3600,  # Discover new algorithms every hour
        signal_check_interval: int = 60,  # Check signals every minute
        min_trades_for_validation: int = 100,
        target_win_rate: float = 66.0,
        target_rr: float = 2.0,
    ):
        self.pairs = pairs
        self.mode = mode
        self.starting_balance_per_pair = starting_balance_per_pair
        self.discovery_interval = discovery_interval
        self.signal_check_interval = signal_check_interval
        self.min_trades_for_validation = min_trades_for_validation
        self.target_win_rate = target_win_rate
        self.target_rr = target_rr
        
        # Pair monitors - check for duplicate normalized symbols
        normalized_pairs = {}
        for pair in pairs:
            safe_symbol = pair.replace('/', '_').replace('-', '_').upper()
            if safe_symbol in normalized_pairs:
                raise ValueError(
                    f"Duplicate pair detected: '{pair}' and '{normalized_pairs[safe_symbol]}' "
                    f"normalize to the same symbol '{safe_symbol}'. Use consistent format."
                )
            normalized_pairs[safe_symbol] = pair
        
        self.monitors: Dict[str, PairMonitor] = {}
        for pair in pairs:
            self.monitors[pair] = PairMonitor(
                symbol=pair,
                mode=mode,
                starting_balance=starting_balance_per_pair,
                min_trades_for_validation=min_trades_for_validation,
                target_win_rate=target_win_rate,
                target_rr=target_rr,
            )
        
        # Threading
        self.running = False
        self.threads = []
        
        # Logger
        self.logger = ComponentLogger.algorithmic_logger()
    
    def start(self):
        """Start the algorithmic trading system"""
        if self.running:
            return
        
        self.running = True
        
        # Start market data monitoring thread (continuously updates market data)
        market_thread = threading.Thread(target=self._market_data_loop, daemon=True)
        market_thread.start()
        self.threads.append(market_thread)
        
        # Start discovery thread (runs periodically, tests multiple algorithms)
        discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        discovery_thread.start()
        self.threads.append(discovery_thread)
        
        # Start signal checking thread
        signal_thread = threading.Thread(target=self._signal_check_loop, daemon=True)
        signal_thread.start()
        self.threads.append(signal_thread)
        
        # Start trade update thread
        trade_thread = threading.Thread(target=self._trade_update_loop, daemon=True)
        trade_thread.start()
        self.threads.append(trade_thread)
        
        # Start status writer thread
        status_thread = threading.Thread(target=self._status_write_loop, daemon=True)
        status_thread.start()
        self.threads.append(status_thread)
        
        self.logger.info("system_started", pairs=self.pairs, mode=self.mode)
    
    def _market_data_loop(self):
        """
        Continuously monitor and update market data for all pairs.
        
        This ensures the system always has fresh market data for:
        - Algorithm testing and discovery
        - Signal generation
        - Trade execution
        
        Updates every 30 seconds to keep data current.
        """
        while self.running:
            try:
                for monitor in self.monitors.values():
                    # Update market data every 30 seconds to keep it fresh
                    monitor.discovery.update_market_data()
                    monitor.stats['last_market_update'] = datetime.utcnow().isoformat()
                
                time.sleep(30)  # Update every 30 seconds
            
            except Exception as e:
                self.logger.error("market_data_loop_error", error=str(e))
                time.sleep(30)
    
    def stop(self):
        """Stop the algorithmic trading system"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=5.0)
        self.logger.info("system_stopped")
    
    def _discovery_loop(self):
        """Continuously discover and test new algorithms"""
        last_discovery = {}
        # Test multiple algorithms per interval for better coverage
        algorithms_per_interval = 3
        
        while self.running:
            try:
                current_time = time.time()
                
                for pair, monitor in self.monitors.items():
                    last_time = last_discovery.get(pair, 0)
                    
                    # Discover new algorithms periodically
                    if current_time - last_time >= self.discovery_interval:
                        # Test multiple algorithms to find working ones faster
                        for _ in range(algorithms_per_interval):
                            monitor.discover_new_algorithm()
                            # Small delay between tests to avoid rate limits
                            time.sleep(2)
                        
                        last_discovery[pair] = current_time
                
                time.sleep(60)  # Check every minute
            
            except Exception as e:
                self.logger.error("discovery_loop_error", error=str(e))
                time.sleep(60)
    
    def _signal_check_loop(self):
        """Periodically check for trading signals - also updates market data"""
        while self.running:
            try:
                for monitor in self.monitors.values():
                    # Update market data before checking signals
                    monitor.discovery.update_market_data()
                    monitor.check_signals()
                
                time.sleep(self.signal_check_interval)
            
            except Exception as e:
                self.logger.error("signal_check_loop_error", error=str(e))
                time.sleep(self.signal_check_interval)
    
    def _trade_update_loop(self):
        """Update open trades"""
        while self.running:
            try:
                for monitor in self.monitors.values():
                    monitor.update_trades()
                
                time.sleep(5)  # Update every 5 seconds
            
            except Exception as e:
                self.logger.error("trade_update_loop_error", error=str(e))
                time.sleep(5)
    
    def _status_write_loop(self):
        """Write status periodically"""
        while self.running:
            try:
                for monitor in self.monitors.values():
                    monitor.write_status()
                
                time.sleep(10)  # Write every 10 seconds
            
            except Exception as e:
                self.logger.error("status_write_loop_error", error=str(e))
                time.sleep(10)
    
    def get_all_statistics(self) -> Dict:
        """Get statistics for all pairs"""
        return {
            pair: monitor.get_statistics()
            for pair, monitor in self.monitors.items()
        }

