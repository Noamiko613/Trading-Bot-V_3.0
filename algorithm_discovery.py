"""
Algorithm Discovery Engine
===========================

Generates, tests, and optimizes trading algorithms to find profitable strategies
with target win rates (66-75%) and risk/reward ratios (2:1 to 3:1).
"""

import json
import os
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from collections import deque

from coinEx_getting_data import CoinExDataFetcher
from core.indicators import add_atr, add_vwap
from verify_patterns import normalize_symbol_to_ccxt


class AlgorithmTemplate:
    """Base class for algorithm templates"""
    
    def __init__(self, name: str, params: Dict, algo_id: str = None):
        self.name = name
        self.params = params
        self.id = algo_id or f"{name}_{int(time.time() * 1000)}"
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal from dataframe"""
        raise NotImplementedError


class MovingAverageCrossover(AlgorithmTemplate):
    """Moving average crossover algorithm"""
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < max(self.params.get('fast_period', 10), self.params.get('slow_period', 20)):
            return None
        
        fast_ma = df['close'].rolling(self.params['fast_period']).mean()
        slow_ma = df['close'].rolling(self.params['slow_period']).mean()
        
        if len(fast_ma) < 2 or len(slow_ma) < 2:
            return None
        
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        
        # Bullish crossover
        if prev_fast <= prev_slow and current_fast > current_slow:
            entry = df['close'].iloc[-1]
            atr = df.get('atr', pd.Series([entry * 0.02] * len(df))).iloc[-1]
            stop = entry - (atr * self.params.get('stop_atr_mult', 2.0))
            tp = entry + (atr * self.params.get('tp_atr_mult', 4.0))
            return {
                'side': 'BUY',
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'confidence': 60,
            }
        
        # Bearish crossover
        elif prev_fast >= prev_slow and current_fast < current_slow:
            entry = df['close'].iloc[-1]
            atr = df.get('atr', pd.Series([entry * 0.02] * len(df))).iloc[-1]
            stop = entry + (atr * self.params.get('stop_atr_mult', 2.0))
            tp = entry - (atr * self.params.get('tp_atr_mult', 4.0))
            return {
                'side': 'SELL',
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'confidence': 60,
            }
        
        return None


class RSIStrategy(AlgorithmTemplate):
    """RSI-based mean reversion strategy"""
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        period = self.params.get('period', 14)
        oversold = self.params.get('oversold', 30)
        overbought = self.params.get('overbought', 70)
        
        if len(df) < period + 1:
            return None
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) < 1:
            return None
        
        current_rsi = rsi.iloc[-1]
        entry = df['close'].iloc[-1]
        atr = df.get('atr', pd.Series([entry * 0.02] * len(df))).iloc[-1]
        
        # Oversold - buy signal
        if current_rsi < oversold:
            stop = entry - (atr * self.params.get('stop_atr_mult', 2.0))
            tp = entry + (atr * self.params.get('tp_atr_mult', 4.0))
            return {
                'side': 'BUY',
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'confidence': 65,
            }
        
        # Overbought - sell signal
        elif current_rsi > overbought:
            stop = entry + (atr * self.params.get('stop_atr_mult', 2.0))
            tp = entry - (atr * self.params.get('tp_atr_mult', 4.0))
            return {
                'side': 'SELL',
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'confidence': 65,
            }
        
        return None


class BollingerBandsStrategy(AlgorithmTemplate):
    """Bollinger Bands mean reversion strategy"""
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        period = self.params.get('period', 20)
        std_dev = self.params.get('std_dev', 2.0)
        
        if len(df) < period:
            return None
        
        # Calculate Bollinger Bands
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        if len(sma) < 1:
            return None
        
        current_price = df['close'].iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_sma = sma.iloc[-1]
        atr = df.get('atr', pd.Series([current_price * 0.02] * len(df))).iloc[-1]
        
        # Price touches lower band - buy
        if current_price <= current_lower:
            entry = current_price
            stop = entry - (atr * self.params.get('stop_atr_mult', 2.0))
            tp = current_sma  # Target middle band
            return {
                'side': 'BUY',
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'confidence': 70,
            }
        
        # Price touches upper band - sell
        elif current_price >= current_upper:
            entry = current_price
            stop = entry + (atr * self.params.get('stop_atr_mult', 2.0))
            tp = current_sma  # Target middle band
            return {
                'side': 'SELL',
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'confidence': 70,
            }
        
        return None


class MomentumStrategy(AlgorithmTemplate):
    """Momentum-based breakout strategy"""
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        lookback = self.params.get('lookback', 20)
        momentum_threshold = self.params.get('momentum_threshold', 0.02)
        
        if len(df) < lookback + 1:
            return None
        
        # Calculate momentum
        current_price = df['close'].iloc[-1]
        past_price = df['close'].iloc[-lookback-1]
        momentum = (current_price - past_price) / past_price
        
        if abs(momentum) < momentum_threshold:
            return None
        
        entry = current_price
        atr = df.get('atr', pd.Series([entry * 0.02] * len(df))).iloc[-1]
        
        # Strong upward momentum - buy
        if momentum > momentum_threshold:
            stop = entry - (atr * self.params.get('stop_atr_mult', 2.0))
            tp = entry + (atr * self.params.get('tp_atr_mult', 4.0))
            return {
                'side': 'BUY',
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'confidence': 65,
            }
        
        # Strong downward momentum - sell
        elif momentum < -momentum_threshold:
            stop = entry + (atr * self.params.get('stop_atr_mult', 2.0))
            tp = entry - (atr * self.params.get('tp_atr_mult', 4.0))
            return {
                'side': 'SELL',
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'confidence': 65,
            }
        
        return None


class AlgorithmDiscovery:
    """Discovers profitable trading algorithms"""
    
    def __init__(self, symbol: str, mode: str = "spot"):
        self.symbol = symbol
        self.mode = mode
        self.project_root = Path(__file__).resolve().parent
        
        # Storage for discovered algorithms
        algo_dir = self.project_root / "algorithmic_trading" / "algorithms"
        os.makedirs(algo_dir, exist_ok=True)
        self.algorithms_dir = algo_dir
        
        # Data fetcher
        symbol_ccxt = normalize_symbol_to_ccxt(symbol)
        self.fetcher = CoinExDataFetcher(symbol=symbol_ccxt, timeframe_internal="5min", mode=mode)
        try:
            self.fetcher.update_initial(limit=500)
        except Exception:
            pass
        
        # Algorithm templates
        self.templates = [
            MovingAverageCrossover,
            RSIStrategy,
            BollingerBandsStrategy,
            MomentumStrategy,
        ]
    
    def generate_algorithm(self) -> AlgorithmTemplate:
        """Generate a random algorithm with random parameters"""
        template_class = random.choice(self.templates)
        
        # Generate random parameters based on template
        if template_class == MovingAverageCrossover:
            params = {
                'fast_period': random.randint(5, 20),
                'slow_period': random.randint(25, 50),
                'stop_atr_mult': random.uniform(1.5, 3.0),
                'tp_atr_mult': random.uniform(3.0, 6.0),
            }
        elif template_class == RSIStrategy:
            params = {
                'period': random.randint(10, 20),
                'oversold': random.randint(20, 35),
                'overbought': random.randint(65, 80),
                'stop_atr_mult': random.uniform(1.5, 3.0),
                'tp_atr_mult': random.uniform(3.0, 6.0),
            }
        elif template_class == BollingerBandsStrategy:
            params = {
                'period': random.randint(15, 30),
                'std_dev': random.uniform(1.5, 2.5),
                'stop_atr_mult': random.uniform(1.5, 3.0),
                'tp_atr_mult': random.uniform(2.0, 4.0),
            }
        else:  # MomentumStrategy
            params = {
                'lookback': random.randint(10, 30),
                'momentum_threshold': random.uniform(0.01, 0.05),
                'stop_atr_mult': random.uniform(1.5, 3.0),
                'tp_atr_mult': random.uniform(3.0, 6.0),
            }
        
        algorithm = template_class(name=template_class.__name__, params=params)
        algorithm.id = f"{algorithm.name}_{int(time.time() * 1000)}"
        return algorithm
    
    def update_market_data(self):
        """Update market data from exchange"""
        try:
            # Refresh data - get latest candles
            self.fetcher.update_initial(limit=500)
        except Exception as e:
            # Log error but continue
            pass
    
    def get_dataframe(self, limit: int = 200) -> pd.DataFrame:
        """Get OHLCV data as DataFrame - refreshes data first"""
        try:
            # Always refresh to get latest market data
            self.update_market_data()
            
            candles = list(self.fetcher.candles)
            if len(candles) < limit:
                # If still not enough, try again
                self.fetcher.update_initial(limit=limit)
                candles = list(self.fetcher.candles)
            
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(candles[-limit:])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Add indicators
            df = add_atr(df)
            df = add_vwap(df)
            
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def evaluate_algorithm(
        self,
        algorithm: AlgorithmTemplate,
        min_trades: int = 100,
        target_win_rate: float = 66.0,
        target_rr: float = 2.0,
    ) -> Dict:
        """
        Evaluate algorithm performance on current market data
        
        Uses live market data to test if algorithm would be profitable.
        Returns evaluation metrics.
        """
        # Get fresh market data for testing
        df = self.get_dataframe(limit=500)
        if len(df) < 100:
            return {'valid': False, 'reason': 'insufficient_data'}
        
        # Simulate trades
        trades = []
        for i in range(50, len(df)):
            window_df = df.iloc[:i+1]
            signal = algorithm.generate_signal(window_df)
            
            if signal:
                # Simulate trade outcome
                entry = signal['entry']
                stop = signal['stop']
                tp = signal['tp']
                side = signal['side']
                
                # Check if TP or SL hit in next N candles
                outcome = None
                for j in range(i+1, min(i+50, len(df))):
                    price = df['close'].iloc[j]
                    
                    if side == 'BUY':
                        if price >= tp:
                            outcome = 'WIN'
                            r_multiple = (tp - entry) / abs(entry - stop)
                            break
                        elif price <= stop:
                            outcome = 'LOSS'
                            r_multiple = -1.0
                            break
                    else:  # SELL
                        if price <= tp:
                            outcome = 'WIN'
                            r_multiple = (entry - tp) / abs(entry - stop)
                            break
                        elif price >= stop:
                            outcome = 'LOSS'
                            r_multiple = -1.0
                            break
                
                if outcome:
                    trades.append({
                        'outcome': outcome,
                        'r_multiple': r_multiple,
                    })
        
        if len(trades) < min_trades:
            return {
                'valid': False,
                'reason': 'insufficient_trades',
                'trades': len(trades),
            }
        
        # Calculate metrics
        wins = [t for t in trades if t['outcome'] == 'WIN']
        losses = [t for t in trades if t['outcome'] == 'LOSS']
        win_rate = (len(wins) / len(trades)) * 100
        avg_r_multiple = sum(t['r_multiple'] for t in trades) / len(trades)
        
        # Check if meets criteria
        meets_criteria = (
            win_rate >= target_win_rate and
            avg_r_multiple >= target_rr
        )
        
        return {
            'valid': True,
            'meets_criteria': meets_criteria,
            'total_trades': len(trades),
            'win_rate': round(win_rate, 2),
            'avg_r_multiple': round(avg_r_multiple, 2),
            'wins': len(wins),
            'losses': len(losses),
        }
    
    def save_algorithm(self, algorithm: AlgorithmTemplate, evaluation: Dict):
        """Save validated algorithm"""
        algo_data = {
            'id': algorithm.id,
            'name': algorithm.name,
            'params': algorithm.params,
            'symbol': self.symbol,
            'evaluation': evaluation,
            'created_at': datetime.utcnow().isoformat(),
        }
        
        file_path = self.algorithms_dir / f"{algorithm.id}.json"
        with open(file_path, 'w') as f:
            json.dump(algo_data, f, indent=2)
    
    def load_algorithms(self) -> List[Dict]:
        """Load all saved algorithms for this symbol"""
        algorithms = []
        for file_path in self.algorithms_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    algo_data = json.load(f)
                    if algo_data.get('symbol') == self.symbol:
                        algorithms.append(algo_data)
            except Exception:
                continue
        return algorithms

