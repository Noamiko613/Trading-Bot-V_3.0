"""
Reinforcement Learning Trading Environment
==========================================

Gym-style environment for training RL agents on trading tasks.
"""

import os
import sys

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sqlite3
from datetime import datetime, timedelta
from coinEx_getting_data import CoinExDataFetcher
from simulate_trading import TradeSimulator
from utils.analytics import PerformanceAnalytics


class TradingEnv(gym.Env):
    """Trading environment for reinforcement learning"""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        starting_balance: float = 100000.0,
        lookback_window: int = 100,
        max_trades: int = None,  # None means unlimited
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.symbol = symbol
        self.starting_balance = starting_balance
        self.lookback_window = lookback_window
        self.max_trades = max_trades  # Fix: Allow unlimited trades if None
        
        # Initialize data fetcher
        from verify_patterns import normalize_symbol_to_ccxt
        symbol_ccxt = normalize_symbol_to_ccxt(symbol)
        self.data_fetcher = CoinExDataFetcher(symbol=symbol_ccxt, timeframe_internal="1h")
        
        # Initialize simulator
        self.simulator = TradeSimulator(
            starting_balance=starting_balance,
            symbol=symbol,
            mode="spot"
        )
        
        # Analytics
        self.analytics = PerformanceAnalytics()
        
        # State space: OHLCV features + technical indicators + portfolio state
        # Features: open, high, low, close, volume, RSI(14), MACD, Bollinger Bands, portfolio balance, equity, open positions
        n_features = 5  # OHLCV
        n_indicators = 8  # RSI, MACD, BB upper, BB lower, BB middle, volume MA, price change, volatility
        n_portfolio = 3  # balance, equity, num_open_positions
        self.observation_dim = lookback_window * (n_features + n_indicators) + n_portfolio
        
        # Action space: 3 discrete actions (0: Hold, 1: Buy, 2: Sell)
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.total_trades_executed = 0
        self.episode_trades = []
        self.initial_balance = starting_balance
        self.render_mode = render_mode
        
        # Market data cache
        self.market_data = None
        self.current_price = None
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset simulator
        self.simulator = TradeSimulator(
            starting_balance=self.starting_balance,
            symbol=self.symbol,
            mode="spot"
        )
        
        # Reset tracking
        self.current_step = 0
        self.total_trades_executed = 0
        self.episode_trades = []
        
        # Load market data
        self._load_market_data()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Execute action
        reward = 0.0
        done = False
        truncated = False
        
        # Get current price
        current_price = self._get_current_price()
        if current_price is None:
            # End episode if no price data
            done = True
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, done, truncated, info
        
        # Action 0: Hold
        # Action 1: Buy (open long position)
        # Action 2: Sell (close position or open short - simplified to close for now)
        
        if action == 1:  # Buy
            # Check if we can trade (not at max trades limit)
            # FIX: Only check limit if max_trades is not None
            if self.max_trades is None or self.total_trades_executed < self.max_trades:
                # Simplified: try to open a position
                # In real implementation, this would use pattern signals
                if len(self.simulator.open_trades) == 0:  # Only one position at a time for simplicity
                    # Create a simple trade setup
                    setup = {
                        'pattern': 'RL_BUY',
                        'side': 'BUY',
                        'entry': current_price,
                        'stop': current_price * 0.98,  # 2% stop loss
                        'tp': current_price * 1.03,  # 3% take profit
                        'rr': 1.5,
                        'risk_pct': 0.002,  # 0.2% risk
                        'timeframe': '1h'
                    }
                    self.simulator.submit_signal(setup)
                    self.total_trades_executed += 1
                    self.episode_trades.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'price': current_price,
                        'type': 'OPEN'
                    })
        
        elif action == 2:  # Sell/Close
            # Close any open positions
            if len(self.simulator.open_trades) > 0:
                # Positions will be closed automatically by simulator on TP/SL hit
                # For RL, we can force close if needed
                pass
        
        # Update simulator (check for closed trades)
        self.simulator.step()
        
        # Calculate reward based on portfolio performance
        current_equity = self.simulator.equity
        previous_equity = getattr(self, '_previous_equity', self.starting_balance)
        
        # Reward = percentage change in equity
        reward = (current_equity - previous_equity) / previous_equity if previous_equity > 0 else 0.0
        
        # Bonus reward for closing profitable trades
        closed_trades = self.simulator.closed_trades[-5:]  # Check last 5 closed trades
        for trade in closed_trades:
            if trade.get('pnl', 0) > 0:
                reward += trade['pnl'] / self.starting_balance * 10  # Scale reward
        
        self._previous_equity = current_equity
        
        # Check if episode is done
        # FIX: Don't stop at max_trades, just track it
        if self.max_trades is not None and self.total_trades_executed >= self.max_trades:
            # Continue but don't allow new trades
            pass
        
        # End episode after many steps or if balance is too low
        if self.current_step >= 10000 or self.simulator.equity < self.starting_balance * 0.5:
            done = True
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _load_market_data(self):
        """Load market data for training"""
        try:
            # Fetch latest candles
            candles = self.data_fetcher.get_kline_data(limit=self.lookback_window + 50)
            if candles:
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                self.market_data = df
            else:
                # Fallback: use dummy data for testing
                self.market_data = pd.DataFrame({
                    'open': np.random.randn(self.lookback_window + 50) * 100 + 50000,
                    'high': np.random.randn(self.lookback_window + 50) * 100 + 50100,
                    'low': np.random.randn(self.lookback_window + 50) * 100 + 49900,
                    'close': np.random.randn(self.lookback_window + 50) * 100 + 50000,
                    'volume': np.random.rand(self.lookback_window + 50) * 1000
                })
        except Exception as e:
            print(f"Error loading market data: {e}")
            # Fallback to dummy data
            self.market_data = pd.DataFrame({
                'open': np.random.randn(self.lookback_window + 50) * 100 + 50000,
                'high': np.random.randn(self.lookback_window + 50) * 100 + 50100,
                'low': np.random.randn(self.lookback_window + 50) * 100 + 49900,
                'close': np.random.randn(self.lookback_window + 50) * 100 + 50000,
                'volume': np.random.rand(self.lookback_window + 50) * 1000
            })
    
    def _get_current_price(self) -> Optional[float]:
        """Get current market price"""
        if self.market_data is not None and len(self.market_data) > 0:
            return float(self.market_data['close'].iloc[-1])
        return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd'] = df['macd'].fillna(0)
        df['macd_signal'] = df['macd_signal'].fillna(0)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_middle'] = df['bb_middle'].fillna(df['close'])
        df['bb_upper'] = df['bb_upper'].fillna(df['close'] * 1.02)
        df['bb_lower'] = df['bb_lower'].fillna(df['close'] * 0.98)
        
        # Volume MA
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ma'] = df['volume_ma'].fillna(df['volume'])
        
        # Price change
        df['price_change'] = df['close'].pct_change()
        df['price_change'] = df['price_change'].fillna(0)
        
        # Volatility
        df['volatility'] = df['price_change'].rolling(window=20).std()
        df['volatility'] = df['volatility'].fillna(0.01)
        
        return df
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.market_data is None or len(self.market_data) < self.lookback_window:
            # Return zeros if no data
            return np.zeros(self.observation_dim, dtype=np.float32)
        
        # Get recent data
        recent_data = self.market_data.tail(self.lookback_window).copy()
        
        # Calculate indicators
        recent_data = self._calculate_indicators(recent_data)
        
        # Extract features
        features = []
        
        for idx, row in recent_data.iterrows():
            # OHLCV
            features.extend([
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ])
            
            # Indicators
            features.extend([
                row['rsi'] / 100.0,  # Normalize to 0-1
                row['macd'],
                row['macd_signal'],
                row['bb_upper'],
                row['bb_lower'],
                row['bb_middle'],
                row['volume_ma'],
                row['price_change'] * 100,  # Percentage
                row['volatility'] * 100
            ])
        
        # Portfolio state
        balance = self.simulator.balance / self.starting_balance  # Normalize
        equity = self.simulator.equity / self.starting_balance  # Normalize
        num_positions = len(self.simulator.open_trades) / 10.0  # Normalize (max 10)
        
        features.extend([balance, equity, num_positions])
        
        # Pad if necessary
        while len(features) < self.observation_dim:
            features.append(0.0)
        
        return np.array(features[:self.observation_dim], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get info dictionary"""
        return {
            'step': self.current_step,
            'balance': self.simulator.balance,
            'equity': self.simulator.equity,
            'total_trades': self.total_trades_executed,
            'open_positions': len(self.simulator.open_trades),
            'closed_trades_count': len(self.simulator.closed_trades),
            'pnl': self.simulator.equity - self.starting_balance,
            'pnl_pct': (self.simulator.equity - self.starting_balance) / self.starting_balance * 100
        }
    
    def render(self):
        """Render environment (optional)"""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"Step: {info['step']}, Balance: ${info['balance']:.2f}, "
                  f"Equity: ${info['equity']:.2f}, Trades: {info['total_trades']}, "
                  f"PnL: ${info['pnl']:.2f} ({info['pnl_pct']:.2f}%)")
