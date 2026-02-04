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
pd.set_option('future.no_silent_downcasting', True)
from typing import Dict, List, Optional, Tuple
import sqlite3
from datetime import datetime, timedelta
from collections import deque
from coinEx_getting_data import CoinExDataFetcher
from utils.analytics import PerformanceAnalytics
from core.dynamic_sl_calculator import get_sl_calculator

# Lazy import to avoid circular dependency - import only when needed
def _get_trade_simulator():
    """Lazy import of TradeSimulator to break circular dependency"""
    from simulate_trading import TradeSimulator
    return TradeSimulator


class TradingEnv(gym.Env):
    """Trading environment for reinforcement learning"""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        starting_balance: float = 1_000_000.0,
        lookback_window: int = 100,
        max_trades: int = None,  # None means unlimited
        render_mode: Optional[str] = None,
        symbol_index: Optional[int] = None,
        total_symbols: int = 1,
        use_continuous_actions: bool = False,  # New: enable continuous position-based actions
        domain_randomization: bool = True,  # New: randomize fees/slippage for generalization
        kill_switch_enabled: bool = True,  # Disable for RL training to allow exploration
        historical_replay: Optional[object] = None,  # HistoricalDataReplay instance for offline training
    ):
        super().__init__()
        
        self.symbol = symbol
        self.starting_balance = starting_balance
        self.lookback_window = lookback_window
        self.max_trades = max_trades  # Fix: Allow unlimited trades if None
        self.use_continuous_actions = use_continuous_actions
        self.domain_randomization = domain_randomization
        self.kill_switch_enabled = kill_switch_enabled
        self.historical_replay = historical_replay
        self.is_historical_mode = historical_replay is not None
        self.action_gating_enabled = os.getenv('ACTION_GATING_ENABLED', '0') == '1'
        try:
            self.gating_min_score = float(os.getenv('GATING_MIN_SCORE', '0.3'))
        except Exception:
            self.gating_min_score = 0.3
        self.gating_soft = os.getenv('GATING_SOFT', '1') == '1'
        
        # Domain randomization: vary fees and slippage each episode
        self.base_fee_pct = 0.0004  # 0.04% base taker fee
        self.base_slippage_pct = 0.0005  # 0.05% base slippage
        self.current_fee_pct = self.base_fee_pct
        self.current_slippage_pct = self.base_slippage_pct
        
        # Initialize multi-timeframe data fetchers for richer market context
        from verify_patterns import normalize_symbol_to_ccxt
        symbol_ccxt = normalize_symbol_to_ccxt(symbol)
        
        # Multi-timeframe analysis: comprehensive coverage from scalping to swing trading
        # 1m (scalping), 5m (short-term), 15m (medium-term), 1h (intraday), 6h (swing), 12h (trend), 1d (position)
        self.timeframes = ["1m", "5m", "15m", "1h", "6h", "12h", "1d"]
        self.data_fetchers = {}
        self._fetcher_configs = {}
        for tf in self.timeframes:
            self._fetcher_configs[tf] = {
                'symbol': symbol_ccxt,
                'timeframe_internal': tf,
                'max_candles': max(lookback_window + 50, 200)
            }
        
        # Primary timeframe for execution (1h)
        self.primary_timeframe = "1h"
        self.data_fetcher = None # Will be lazy-loaded
        
        # Lazy-initialize simulator and analytics to support multiprocessing
        # These will be created in the `reset` method, inside the child process
        self.simulator = None
        self.analytics = None
        
        # Encode symbol identity so a single shared policy can learn behaviour
        # across multiple markets. We add a small one-hot (or index) vector that
        # indicates which symbol this environment represents.
        self.symbol_index = symbol_index if symbol_index is not None else 0
        self.total_symbols = max(int(total_symbols or 1), 1)
        
        # Optimized state space: reduced dimensionality for better sample efficiency (< 500 features)
        # Strategy: Use consolidated features per timeframe instead of full history
        # Primary timeframe (1h): recent 50 bars with key indicators
        # Secondary timeframes: aggregated features (returns, volatility, trend) only
        
        # Primary timeframe: recent 50 bars (reduced from 100)
        n_primary_bars = 50
        n_features_per_bar = 5  # OHLCV only (indicators computed separately)
        n_key_indicators = 8  # RSI, MACD, ATR, volume_surge, trend_strength, volatility, price_change, choppiness
        n_primary = n_primary_bars * n_features_per_bar + n_key_indicators
        
        # Secondary timeframes: consolidated features only (not full bars)
        # Each timeframe: returns (5 periods), volatility, trend, volume_ratio, regime flags
        n_secondary_per_tf = 12  # 5 returns + volatility + trend + volume + 4 regime flags
        n_secondary = (len(self.timeframes) - 1) * n_secondary_per_tf  # Exclude primary
        
        n_portfolio = 5  # Enhanced: balance, equity, num_positions, unrealized_pnl, realized_pnl_today
        n_regime = 4  # Market regime: trend/range, volatility regime, liquidity, session
        n_symbol = self.total_symbols  # one-hot symbol encoding
        
        self.observation_dim = n_primary + n_secondary + n_portfolio + n_regime + n_symbol
        
        # Action space: continuous position-based or discrete
        if use_continuous_actions:
            # Continuous: action ∈ [-1, 1] maps to target position fraction
            # -1 = full short, 0 = flat, +1 = full long
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.max_position_fraction = 0.5  # Max 50% of equity in one position
            self.max_position_change_per_step = 0.1  # Max 10% change per step
        else:
            # Discrete: 0=Hold, 1=Buy, 2=Sell
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
        # Position-sizing guardrails
        self.min_trade_risk_pct = float(os.getenv("MIN_TRADE_RISK_PCT", "0.25"))  # Minimum % of equity risked per trade
        self.max_trade_risk_pct = float(os.getenv("MAX_TRADE_RISK_PCT", "0.35"))  # Maximum % of equity risked per trade
        self.small_trade_penalty = float(os.getenv("SMALL_TRADE_PENALTY", "0.6"))  # Reward penalty for undersized trades
        self.oversize_trade_penalty = float(os.getenv("OVERSIZE_TRADE_PENALTY", "0.35"))  # Penalty for attempting to oversize
        
        # Market data cache
        self.market_data = None
        self.current_price = None
        
        # Current position tracking (for continuous actions)
        self.current_position_fraction = 0.0  # -1 to +1
        
        # Cooldown to prevent rapid trade churning (minimum steps between trades)
        self.last_trade_step = -1000  # Initialize to allow first trade
        self.min_steps_between_trades = 4  # Minimum 4 steps between opening new trades
        
        # Maximum trade duration (in hours) - configurable to avoid false "stale" for long swings
        try:
            self.max_trade_duration_hours = float(os.getenv("MAX_TRADE_DURATION_HOURS", "96"))
        except Exception:
            self.max_trade_duration_hours = 96.0
        try:
            self.stale_check_interval_sec = int(os.getenv("STALE_CHECK_INTERVAL_SEC", "90"))
        except Exception:
            self.stale_check_interval_sec = 90

        # For Sharpe Ratio calculation
        self._episode_returns = []
        self._previous_sharpe = 0.0
        self.rolling_returns = deque(maxlen=2000)
        self.sharpe_annualization_factor = 24 * 365
        self.sharpe_history = deque(maxlen=100)
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Domain randomization: vary fees and slippage each episode
        if self.domain_randomization:
            # Randomize fees: 0.02% to 0.08% (0.5x to 2x base)
            self.current_fee_pct = self.base_fee_pct * np.random.uniform(0.5, 2.0)
            # Randomize slippage: 0.025% to 0.1% (0.5x to 2x base)
            self.current_slippage_pct = self.base_slippage_pct * np.random.uniform(0.5, 2.0)
        
        # Initialize simulator and analytics here, inside the process
        if self.simulator is None:
            TradeSimulator = _get_trade_simulator()
            # During historical training, mark trades so they don't get counted as live paper trades
            historical_mode = self.is_historical_mode
            self.simulator = TradeSimulator(
                starting_balance=self.starting_balance,
                symbol=self.symbol,
                mode="spot",
                fee_pct=self.current_fee_pct,
                slippage_pct=self.current_slippage_pct,
                kill_switch_enabled=self.kill_switch_enabled,
                max_trade_duration_hours=self.max_trade_duration_hours,
                stale_check_interval_sec=self.stale_check_interval_sec,
            )
            # Mark simulator as historical training mode
            if historical_mode:
                self.simulator.is_historical_training = True
                print(f"[RL_ENV] Historical training mode: Trades will be marked as training data")
        else:
            # If it exists, just reset it
            self.simulator.reset(
                starting_balance=self.starting_balance,
                fee_pct=self.current_fee_pct,
                slippage_pct=self.current_slippage_pct
            )
            # Keep simulator stale logic aligned with env settings
            try:
                self.simulator.max_trade_duration_hours = float(self.max_trade_duration_hours)
                self.simulator.stale_check_interval_sec = int(self.stale_check_interval_sec)
            except Exception:
                pass

        if self.analytics is None:
            self.analytics = PerformanceAnalytics()
        
        # Reset tracking
        self.current_step = 0
        self.total_trades_executed = 0
        self.episode_trades = []
        self.current_position_fraction = 0.0
        self._previous_position_fraction = 0.0
        self._previous_equity = self.starting_balance
        self.data_unavailable = False
        # Reset trade tracking for immediate SL penalty
        self._last_trade_entry_price = None
        self._last_trade_stop = None
        self._last_trade_side = None
        self._trades_before_step = 0
        self._last_closed_count = 0  # Track closed trades for reward calculation
        self._last_close_was_loss = False  # For consecutive-loss penalty in reward
        # Reset cooldown
        self.last_trade_step = -1000
        
        # Reset Sharpe Ratio calculation
        self._episode_returns = []
        self._previous_sharpe = 0.0
        self.rolling_returns.clear()
        self.sharpe_history.clear()
        
        # Load market data for all timeframes
        if self.is_historical_mode:
            # Historical replay mode: reset replay and load initial data
            self.historical_replay.reset()
            self._load_historical_market_data()
        else:
            # Live mode: use data fetchers
            self._lazy_init_fetchers()
            self._load_market_data()
        
        # Ensure market_data is dict format for multi-timeframe
        if not isinstance(self.market_data, dict):
            # Convert to dict format if needed
            if isinstance(self.market_data, pd.DataFrame):
                self.market_data = {self.primary_timeframe: self.market_data}
                for tf in self.timeframes:
                    if tf != self.primary_timeframe and tf not in self.market_data:
                        # Initialize empty for secondary timeframes if not loaded
                        self.market_data[tf] = pd.DataFrame()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Track trades before step to detect closures and openings
        self._trades_before_step = len(self.simulator.open_trades)
        
        # CRITICAL: Refresh market data and update simulator BEFORE processing action
        # This ensures we have the latest prices for TP/SL checking
        if self.is_historical_mode:
            # Historical replay mode: advance to next candle
            if not self.historical_replay.is_done():
                self.historical_replay.step()
                # Update market data from replay
                all_data = self.historical_replay.get_all_current_data()
                self.market_data = all_data
            else:
                # Reached end of historical data
                observation = self._get_observation()
                info = self._get_info()
                return observation, 0.0, True, False, info
        else:
            # Live mode: fetch latest data
            try:
                if self.primary_timeframe in self.data_fetchers:
                    fetcher = self.data_fetchers[self.primary_timeframe]
                    candles = fetcher.get_kline_data(limit=min(self.lookback_window + 50, 200))
                    if candles:
                        df = pd.DataFrame(candles)
                        df['timestamp'] = pd.to_datetime(
                            df['timestamp'],
                            format="ISO8601",
                            utc=True,
                            errors="coerce",
                        )
                        df = df.dropna(subset=['timestamp']).sort_values('timestamp')
                        if isinstance(self.market_data, dict):
                            self.market_data[self.primary_timeframe] = df
                        else:
                            self.market_data = {self.primary_timeframe: df}
            except Exception as e:
                # If refresh fails, continue with cached data
                pass
        
        # Update simulator to check TP/SL on existing trades BEFORE processing new action
        # This ensures trades are closed when TP/SL is hit, not just when new actions are taken
        self.simulator.step()
        
        # Reset trade tracking if no trades were opened in previous step
        # (This prevents false positives in immediate SL penalty check)
        if not hasattr(self, '_last_trade_entry_price'):
            self._last_trade_entry_price = None
            self._last_trade_stop = None
            self._last_trade_side = None
        
        # Execute action
        reward = 0.0
        done = False
        truncated = False
        
        action_penalty = -0.005  # Reduced penalty (was -0.01) to allow more exploration early in training
        raw_reward = 0.0  # Initialize reward accumulator for all action types
        
        # Get current price
        current_price = self._get_current_price()
        if current_price is None:
            # End episode if no price data
            done = True
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, done, truncated, info
        
        if getattr(self, "data_unavailable", False):
            # Fail fast if data could not be loaded; avoid training on random/noisy placeholders
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, True, truncated, info
        
        # Debug: Log action selection and trade status periodically
        if self.current_step % 50 == 0 and self.current_step > 0:
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL"} if not self.use_continuous_actions else "CONTINUOUS"
            open_trades_on_symbol = [t for t in self.simulator.open_trades 
                                     if t.get('symbol') == self.symbol and t.get('status') == 'OPEN']
            closed_trades_count = len([t for t in self.simulator.closed_trades if t.get('symbol') == self.symbol])
            price_str = f"{current_price:.2f}" if current_price else "N/A"
            print(f"[RL_ENV] Step {self.current_step}: Action={action_names.get(action, action) if not self.use_continuous_actions else f'{action[0]:.3f}'}, "
                  f"OpenTrades={len(open_trades_on_symbol)}/{len(self.simulator.open_trades)} (on {self.symbol}/total), "
                  f"ClosedTrades={closed_trades_count}, TotalExecuted={self.total_trades_executed}, "
                  f"Price=${price_str}, Equity=${self.simulator.equity:.2f}")
        
        # Execute action based on action space type
        if self.use_continuous_actions:
            # Continuous action: map [-1, 1] to target position fraction
            target_position = float(np.clip(action[0], -1.0, 1.0)) * self.max_position_fraction
            
            # Limit position change per step to prevent instant large swings
            delta = target_position - self.current_position_fraction
            delta = np.clip(delta, -self.max_position_change_per_step, self.max_position_change_per_step)
            new_position = self.current_position_fraction + delta
            
            # Execute position change
            if abs(new_position - self.current_position_fraction) > 0.01:  # Threshold to avoid micro-trades
                if new_position > 0 and self.current_position_fraction <= 0:
                    # Opening long or closing short
                    open_shorts = [t for t in self.simulator.open_trades if t.get('side') == 'SELL' and t.get('status') == 'OPEN']
                    if open_shorts:
                        for trade in open_shorts:
                            try:
                                close_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                entry_price = trade.get('entry', current_price)
                                trade_id = trade.get('id')
                                # Close the trade first to get actual PnL
                                self.simulator.close_trade(trade_id, reason="RL_CONT_CLOSE_SHORT")
                                # Get the actual PnL from closed trade
                                closed_trade = next((t for t in self.simulator.closed_trades if t.get('id') == trade_id), None)
                                actual_pnl = closed_trade.get('pnl', 0.0) if closed_trade else 0.0
                                exit_price = closed_trade.get('exit', current_price) if closed_trade else current_price
                                print(f"[RL_ENV] 🔵 CLOSING SHORT TRADE (Continuous) | Pair: {self.symbol} | Time: {close_time} | Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f} | PnL: ${actual_pnl:.2f}")
                            finally:
                                self.episode_trades.append({
                                    'step': self.current_step,
                                    'action': 'CLOSE_SHORT',
                                    'price': current_price,
                                    'type': 'CLOSE',
                                    'trade_id': trade.get('id')
                                })
                    if len(self.simulator.open_trades) == 0:
                        rr_ratio, stop_loss_pct, _tmp_risk = self._calculate_rr_and_risk(current_price)
                        # FIXED: Ensure valid stop loss
                        if stop_loss_pct <= 0 or stop_loss_pct < 0.005:
                            stop_loss_pct = 0.015  # Use 1.5% minimum
                        if stop_loss_pct > 0:
                            risk_pct_raw = abs(new_position) * 0.5
                            risk_pct = np.clip(risk_pct_raw, self.min_trade_risk_pct, self.max_trade_risk_pct)
                            if risk_pct_raw < self.min_trade_risk_pct:
                                raw_reward -= self.small_trade_penalty
                            if risk_pct_raw > self.max_trade_risk_pct:
                                raw_reward -= self.oversize_trade_penalty
                            setup = {
                                'pattern': 'RL_CONTINUOUS_LONG',
                                'side': 'BUY',
                                'entry': current_price,
                                'stop': current_price * (1 - stop_loss_pct),
                                'tp': current_price * (1 + stop_loss_pct * rr_ratio),
                                'rr': rr_ratio,
                                'risk_pct': risk_pct,  # Enforce minimum risk to avoid micro trades
                                'timeframe': '1h'
                            }
                            self._submit_with_gating(setup)
                            self.total_trades_executed += 1
                            self.last_trade_step = self.current_step
                            self._last_trade_entry_price = current_price
                            self._last_trade_stop = current_price * (1 - stop_loss_pct)
                            self._last_trade_side = 'BUY'
                elif new_position < 0 and self.current_position_fraction >= 0:
                    # Opening short or closing long
                    open_longs = [t for t in self.simulator.open_trades if t.get('side') == 'BUY' and t.get('status') == 'OPEN']
                    if open_longs:
                        for trade in open_longs:
                            try:
                                close_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                entry_price = trade.get('entry', current_price)
                                trade_id = trade.get('id')
                                # Close the trade first to get actual PnL
                                self.simulator.close_trade(trade_id, reason="RL_CONT_CLOSE_LONG")
                                # Get the actual PnL from closed trade
                                closed_trade = next((t for t in self.simulator.closed_trades if t.get('id') == trade_id), None)
                                actual_pnl = closed_trade.get('pnl', 0.0) if closed_trade else 0.0
                                exit_price = closed_trade.get('exit', current_price) if closed_trade else current_price
                                print(f"[RL_ENV] 🔵 CLOSING LONG TRADE (Continuous) | Pair: {self.symbol} | Time: {close_time} | Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f} | PnL: ${actual_pnl:.2f}")
                            finally:
                                self.episode_trades.append({
                                    'step': self.current_step,
                                    'action': 'CLOSE_LONG',
                                    'price': current_price,
                                    'type': 'CLOSE',
                                    'trade_id': trade.get('id')
                                })
                    if len(self.simulator.open_trades) == 0:
                        rr_ratio, stop_loss_pct, _tmp_risk = self._calculate_rr_and_risk(current_price)
                        # FIXED: Ensure valid stop loss
                        if stop_loss_pct <= 0 or stop_loss_pct < 0.005:
                            stop_loss_pct = 0.015  # Use 1.5% minimum
                        if stop_loss_pct > 0:
                            raw_stop = current_price * (1 + stop_loss_pct)
                            try:
                                primary_df = self.market_data.get(self.primary_timeframe) if isinstance(self.market_data, dict) else None
                                sl_calc = get_sl_calculator()
                                enforced_stop = sl_calc.enforce_minimum_sl(
                                    symbol=self.symbol,
                                    entry_price=current_price,
                                    stop_loss=raw_stop,
                                    side='SELL',
                                    df=primary_df if isinstance(primary_df, pd.DataFrame) else None
                                )
                            except Exception:
                                enforced_stop = raw_stop
                            tp_price = current_price - (enforced_stop - current_price) * rr_ratio

                            risk_pct_raw = abs(new_position) * 0.5
                            risk_pct = np.clip(risk_pct_raw, self.min_trade_risk_pct, self.max_trade_risk_pct)
                            if risk_pct_raw < self.min_trade_risk_pct:
                                raw_reward -= self.small_trade_penalty
                            if risk_pct_raw > self.max_trade_risk_pct:
                                raw_reward -= self.oversize_trade_penalty
                            setup = {
                                'pattern': 'RL_CONTINUOUS_SHORT',
                                'side': 'SELL',
                                'entry': current_price,
                                'stop': enforced_stop,
                                'tp': tp_price,
                                'rr': rr_ratio,
                                'risk_pct': risk_pct,  # Enforce minimum risk to avoid micro trades
                                'timeframe': '1h'
                            }
                            self._submit_with_gating(setup)
                            self.total_trades_executed += 1
                            self.last_trade_step = self.current_step
                            self._last_trade_entry_price = current_price
                            self._last_trade_stop = enforced_stop
                            self._last_trade_side = 'SELL'
            
            self.current_position_fraction = new_position
        else:
            # Discrete actions: 
            #   0 = Hold (wait for better setup)
            #   1 = Buy (Long) - Opens long if no position, closes short if short exists
            #   2 = Sell (Short) - Opens short if no position, closes long if long exists
            # This design allows the model to learn both long and short trading strategies
            # Initialize raw_reward - will be calculated below for BUY/SELL, set here for HOLD
            raw_reward = 0.0
            
            if action == 0:  # Hold
                # Small positive reward for holding when no good setup (encourages patience)
                # This helps the model learn that sometimes not trading is better
                if len(self.simulator.open_trades) == 0:
                    # No reward for holding when no positions - avoids discouraging trades
                    raw_reward = 0.0
                else:
                    # No extra reward when holding with open positions (let TP/SL handle it)
                    raw_reward = 0.0
            
            elif action == 1:  # Buy (Long or Close Short)
                raw_reward = action_penalty
                # If we have a SELL (short) position open, close it first
                # Otherwise, open a BUY (long) position
                open_sell_trades = [t for t in self.simulator.open_trades if t.get('side') == 'SELL' and t.get('status') == 'OPEN']
                
                if len(open_sell_trades) > 0:
                    # Close existing SELL (short) positions
                    for trade in open_sell_trades:
                        try:
                            close_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            entry_price = trade.get('entry', current_price)
                            trade_id = trade.get('id')
                            # Close the trade first to get actual PnL
                            self.simulator.close_trade(trade_id, reason="RL_CLOSE_SHORT")
                            # Get the actual PnL from closed trade
                            closed_trade = next((t for t in self.simulator.closed_trades if t.get('id') == trade_id), None)
                            actual_pnl = closed_trade.get('pnl', 0.0) if closed_trade else 0.0
                            exit_price = closed_trade.get('exit', current_price) if closed_trade else current_price
                            print(f"[RL_ENV] 🔵 CLOSING SHORT TRADE | Pair: {self.symbol} | Time: {close_time} | Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f} | PnL: ${actual_pnl:.2f}")
                        finally:
                            self.episode_trades.append({
                                'step': self.current_step,
                                'action': 'CLOSE_SHORT',
                                'price': current_price,
                                'type': 'CLOSE',
                                'trade_id': trade.get('id')
                            })
                # CRITICAL FIX: Check for open trades on THIS symbol specifically (same as SELL path)
                open_trades_on_symbol = [t for t in self.simulator.open_trades 
                                         if t.get('symbol') == self.symbol and t.get('status') == 'OPEN']
                
                # NEW: Prevent opening a new trade if one is already open on this symbol
                if len(open_trades_on_symbol) == 0:
                    steps_since_last_trade = self.current_step - self.last_trade_step
                    if steps_since_last_trade >= self.min_steps_between_trades:
                        if self.max_trades is None or self.total_trades_executed < self.max_trades:
                            # Calculate dynamic RR and stop loss for long position
                            rr_ratio, stop_loss_pct, risk_pct = self._calculate_rr_and_risk(current_price)
                            # FIXED: Changed threshold from 0.0100000001 to 0.005 to allow smaller stop losses
                            # Also ensure minimum stop loss is enforced
                            if stop_loss_pct <= 0 or stop_loss_pct < 0.005:
                                stop_loss_pct = 0.015  # Use 1.5% minimum
                            
                            # Always proceed with trade if we have valid stop loss
                            if stop_loss_pct > 0:
                                raw_stop = current_price * (1 - stop_loss_pct)
                                # Enforce pair-specific minimum SL using latest primary timeframe data
                                try:
                                    primary_df = self.market_data.get(self.primary_timeframe)
                                    sl_calc = get_sl_calculator()
                                    enforced_stop = sl_calc.enforce_minimum_sl(
                                        symbol=self.symbol,
                                        entry_price=current_price,
                                        stop_loss=raw_stop,
                                        side='BUY',
                                        df=primary_df if isinstance(primary_df, pd.DataFrame) else None
                                    )
                                except Exception:
                                    enforced_stop = raw_stop
                                # Recalculate TP to preserve RR ratio after enforcement
                                tp_price = current_price + (current_price - enforced_stop) * rr_ratio

                                raw_risk_pct = risk_pct
                                risk_pct = np.clip(risk_pct, self.min_trade_risk_pct, self.max_trade_risk_pct)
                                if raw_risk_pct < self.min_trade_risk_pct:
                                    raw_reward -= self.small_trade_penalty
                                if raw_risk_pct > self.max_trade_risk_pct:
                                    raw_reward -= self.oversize_trade_penalty

                                setup = {
                                    'pattern': 'RL_BUY',
                                    'side': 'BUY',
                                    'entry': current_price,
                                    'stop': enforced_stop,
                                    'tp': tp_price,
                                    'rr': rr_ratio,
                                    'risk_pct': risk_pct,
                                    'timeframe': '1h'
                                }
                                trade_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                mode_prefix = "[HISTORICAL TRAINING] " if self.is_historical_mode else ""
                                print(f"[RL_ENV] {mode_prefix}🟢 OPENING BUY TRADE | Pair: {self.symbol} | Time: {trade_time} | Entry: ${current_price:.2f} | Stop: ${enforced_stop:.2f} | TP: ${tp_price:.2f} | SL%: {stop_loss_pct*100:.2f}%")
                                self._submit_with_gating(setup)
                                self.total_trades_executed += 1
                                self.last_trade_step = self.current_step
                                self.episode_trades.append({
                                    'step': self.current_step,
                                    'action': 'BUY',
                                    'price': current_price,
                                    'type': 'OPEN'
                                })
                                self._last_trade_entry_price = current_price
                                self._last_trade_stop = enforced_stop
                                self._last_trade_side = 'BUY'
                                print(f"[RL_ENV] ✅ BUY trade submitted successfully! Total trades executed: {self.total_trades_executed}")
                            else:
                                print(f"[RL_ENV] BUY action blocked: Invalid stop_loss_pct={stop_loss_pct}")
                        else:
                            if self.current_step % 50 == 0:  # Log periodically
                                print(f"[RL_ENV] BUY action blocked: max_trades limit reached ({self.total_trades_executed}/{self.max_trades})")
                    else:
                        if self.current_step % 50 == 0:  # Log periodically
                            print(f"[RL_ENV] BUY action blocked: Cooldown active (steps since last trade: {steps_since_last_trade}/{self.min_steps_between_trades})")
                else:
                    # Penalize attempting to open when position already exists (helps model learn)
                    raw_reward -= 0.1  # Penalty for trying to open duplicate position
                    if self.current_step % 10 == 0:  # Log more frequently for debugging
                        open_count = len(open_trades_on_symbol)
                        print(f"[RL_ENV] ⚠️ BUY action blocked: Already have {open_count} open trade(s) on {self.symbol}")
            
            elif action == 2:  # Sell (Short or Close Long)
                raw_reward = action_penalty
                # If we have a BUY position open, close it
                # Otherwise, open a SELL (short) position
                open_buy_trades = [t for t in self.simulator.open_trades if t.get('side') == 'BUY' and t.get('status') == 'OPEN']
                
                if len(open_buy_trades) > 0:
                    # Close existing BUY positions
                    for trade in open_buy_trades:
                        try:
                            close_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            entry_price = trade.get('entry', current_price)
                            trade_id = trade.get('id')
                            # Close the trade first to get actual PnL
                            self.simulator.close_trade(trade_id, reason="RL_CLOSE_LONG")
                            # Get the actual PnL from closed trade
                            closed_trade = next((t for t in self.simulator.closed_trades if t.get('id') == trade_id), None)
                            actual_pnl = closed_trade.get('pnl', 0.0) if closed_trade else 0.0
                            exit_price = closed_trade.get('exit', current_price) if closed_trade else current_price
                            print(f"[RL_ENV] 🔵 CLOSING LONG TRADE | Pair: {self.symbol} | Time: {close_time} | Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f} | PnL: ${actual_pnl:.2f}")
                        finally:
                            self.episode_trades.append({
                                'step': self.current_step,
                                'action': 'CLOSE_LONG',
                                'price': current_price,
                                'type': 'CLOSE',
                                'trade_id': trade.get('id')
                            })
                # NEW: Prevent opening a new trade if one is already open on this symbol
                # Check for open trades on THIS symbol specifically
                open_trades_on_symbol = [t for t in self.simulator.open_trades 
                                         if t.get('symbol') == self.symbol and t.get('status') == 'OPEN']
                
                if len(open_trades_on_symbol) == 0:
                    steps_since_last_trade = self.current_step - self.last_trade_step
                    if steps_since_last_trade >= self.min_steps_between_trades:
                        if self.max_trades is None or self.total_trades_executed < self.max_trades:
                            # Calculate dynamic RR and stop loss for short position
                            rr_ratio, stop_loss_pct, risk_pct = self._calculate_rr_and_risk(current_price)
                            # FIXED: Changed threshold from 0.0100000001 to 0.005 to allow smaller stop losses
                            # Also ensure minimum stop loss is enforced
                            if stop_loss_pct <= 0 or stop_loss_pct < 0.005:
                                stop_loss_pct = 0.015  # Use 1.5% minimum
                            
                            # Always proceed with trade if we have valid stop loss
                            if stop_loss_pct > 0:
                                raw_stop = current_price * (1 + stop_loss_pct)
                                # Enforce pair-specific minimum SL using latest primary timeframe data
                                try:
                                    primary_df = self.market_data.get(self.primary_timeframe)
                                    sl_calc = get_sl_calculator()
                                    enforced_stop = sl_calc.enforce_minimum_sl(
                                        symbol=self.symbol,
                                        entry_price=current_price,
                                        stop_loss=raw_stop,
                                        side='SELL',
                                        df=primary_df if isinstance(primary_df, pd.DataFrame) else None
                                    )
                                except Exception:
                                    enforced_stop = raw_stop
                                # Recalculate TP to preserve RR ratio after enforcement
                                tp_price = current_price - (enforced_stop - current_price) * rr_ratio

                                raw_risk_pct = risk_pct
                                risk_pct = np.clip(risk_pct, self.min_trade_risk_pct, self.max_trade_risk_pct)
                                if raw_risk_pct < self.min_trade_risk_pct:
                                    raw_reward -= self.small_trade_penalty
                                if raw_risk_pct > self.max_trade_risk_pct:
                                    raw_reward -= self.oversize_trade_penalty

                                setup = {
                                    'pattern': 'RL_SELL',
                                    'side': 'SELL',
                                    'entry': current_price,
                                    'stop': enforced_stop,
                                    'tp': tp_price,
                                    'rr': rr_ratio,
                                    'risk_pct': risk_pct,
                                    'timeframe': '1h'
                                }
                                trade_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                mode_prefix = "[HISTORICAL TRAINING] " if self.is_historical_mode else ""
                                print(f"[RL_ENV] {mode_prefix}🔴 OPENING SELL TRADE | Pair: {self.symbol} | Time: {trade_time} | Entry: ${current_price:.2f} | Stop: ${enforced_stop:.2f} | TP: ${tp_price:.2f} | SL%: {stop_loss_pct*100:.2f}%")
                                self._submit_with_gating(setup)
                                self.total_trades_executed += 1
                                self.last_trade_step = self.current_step
                                self.episode_trades.append({
                                    'step': self.current_step,
                                    'action': 'SELL',
                                    'price': current_price,
                                    'type': 'OPEN'
                                })
                                self._last_trade_entry_price = current_price
                                self._last_trade_stop = enforced_stop
                                self._last_trade_side = 'SELL'
                                print(f"[RL_ENV] ✅ SELL trade submitted successfully! Total trades executed: {self.total_trades_executed}")
                            else:
                                print(f"[RL_ENV] SELL action blocked: Invalid stop_loss_pct={stop_loss_pct}")
                        else:
                            if self.current_step % 50 == 0:  # Log periodically
                                print(f"[RL_ENV] SELL action blocked: max_trades limit reached ({self.total_trades_executed}/{self.max_trades})")
                    else:
                        if self.current_step % 50 == 0:  # Log periodically
                            print(f"[RL_ENV] SELL action blocked: Cooldown active (steps since last trade: {steps_since_last_trade}/{self.min_steps_between_trades})")
                else:
                    # Penalize attempting to open when position already exists (helps model learn)
                    raw_reward -= 0.1  # Penalty for trying to open duplicate position
                    if self.current_step % 10 == 0:  # Log more frequently for debugging
                        open_count = len(open_trades_on_symbol)
                        print(f"[RL_ENV] ⚠️ SELL action blocked: Already have {open_count} open trade(s) on {self.symbol}")
        
        # CRITICAL: Update simulator FIRST (checks TP/SL, updates prices, etc.)
        # This must be called BEFORE checking stale trades to ensure TP/SL are processed
        # This also persists trades to database via TradeSimulator
        self.simulator.step()
        
        # Check for trades that have been open too long and force close them
        # This ensures trades don't stay open indefinitely (after TP/SL check)
        self._check_and_close_stale_trades()
        
        # CRITICAL: Check again after stale trade closure to ensure all closures are processed
        # This ensures that if we force-closed any trades, the simulator processes them
        if len(self.simulator.open_trades) < self._trades_before_step:
            # A trade was closed, refresh simulator state
            self.simulator.step()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        self._previous_equity = self.simulator.equity
        
        # Check if episode is done
        if self.max_trades is not None and self.total_trades_executed >= self.max_trades:
            pass
        
        # Optimized episode termination for faster learning and iteration
        # Shorter episodes = more frequent learning updates = faster convergence
        max_trades_per_episode = 100  # Reduced from 200 for faster iteration
        max_steps_per_episode = 5000  # Reduced from 10000 for faster episodes
        
        if (self.current_step >= max_steps_per_episode or 
            self.simulator.equity < self.starting_balance * 0.5 or
            self.total_trades_executed >= max_trades_per_episode):
            done = True
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info

    def _calculate_reward(self) -> float:
        """
        Optimized reward calculation for fast learning and real market performance.
        Uses multiple reward components: immediate PnL, risk-adjusted returns, trade quality.
        """
        previous_equity = getattr(self, '_previous_equity', self.starting_balance)
        current_equity = self.simulator.equity

        # Regime flags for reward shaping
        try:
            if isinstance(self.market_data, dict):
                _dfreg = self.market_data.get(self.primary_timeframe)
            elif isinstance(self.market_data, pd.DataFrame):
                _dfreg = self.market_data
            else:
                _dfreg = getattr(self, '_primary_market_data', None)
            if _dfreg is not None and len(_dfreg) > 0:
                _cls = float(pd.to_numeric(_dfreg['close'], errors='coerce').iloc[-1])
                _mean20 = float(pd.to_numeric(_dfreg['close'], errors='coerce').rolling(20).mean().iloc[-1]) if len(_dfreg) > 0 else 0.0
                _trend_strength = ((_cls - _mean20) / _mean20) if (_mean20 and _mean20 > 0) else 0.0
                _rets = pd.to_numeric(_dfreg['close'], errors='coerce').pct_change()
                _vol = float(_rets.tail(20).std()) if len(_rets) > 0 else 0.0
                is_trend = bool(abs(_trend_strength) > 0.003)
                is_high_vol = bool(_vol > 0.01)
            else:
                is_trend = False
                is_high_vol = False
        except Exception:
            is_trend = False
            is_high_vol = False

        # Calculate step return (immediate feedback)
        if previous_equity > 0:
            step_return = (current_equity - previous_equity) / previous_equity
        else:
            step_return = 0.0
        
        self.rolling_returns.append(step_return)

        # Primary reward: Immediate PnL signal (scaled for faster learning)
        # This gives immediate feedback on actions, critical for fast learning
        reward = step_return * 50.0  # Scale up for more signal (was 0.1, too weak)
        
        # Secondary reward: Risk-adjusted returns (Sharpe ratio) for long-term performance
        if len(self.rolling_returns) > 50:  # Reduced from 100 for faster signal
            returns_array = np.array(self.rolling_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array) + 1e-9
            
            # Sortino ratio (only penalizes downside volatility - better for trading)
            downside_returns = returns_array[returns_array < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9
            sortino_ratio = (mean_return / downside_std) * np.sqrt(self.sharpe_annualization_factor) if downside_std > 0 else 0.0
            
            # Annualized Sharpe Ratio (fallback)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(self.sharpe_annualization_factor)
            
            # Use Sortino (preferred) or Sharpe
            risk_adjusted_ratio = sortino_ratio if sortino_ratio > 0 else sharpe_ratio
            
            # Add to sharpe history
            self.sharpe_history.append(risk_adjusted_ratio)
            
            # Differential risk-adjusted return (improvement signal)
            if len(self.sharpe_history) > 1:
                prev_avg = np.mean(list(self.sharpe_history)[:-1])
                improvement = risk_adjusted_ratio - prev_avg
                reward += improvement * 5.0  # Scale improvement signal
            else:
                reward += risk_adjusted_ratio * 0.1  # Initial signal

            # Strong penalty for negative risk-adjusted returns
            if risk_adjusted_ratio < 0:
                reward -= 0.2  # Increased penalty
        else:
            # Early training: use simple return signal
            reward = step_return * 50.0

        # Dense MFE/MAE-based shaping on open trades
        try:
            if len(self.simulator.open_trades) > 0:
                _mfe = []
                _mae = []
                for _t in self.simulator.open_trades:
                    if _t.get('symbol') != self.symbol:
                        continue
                    _mfe.append(float(_t.get('min_runup', 0.0) or 0.0))
                    _mae.append(float(_t.get('max_drawdown', 0.0) or 0.0))
                if _mfe:
                    reward += float(np.clip(np.mean(_mfe) * 0.03, -0.2, 0.2))
                if _mae:
                    reward -= float(np.clip(np.mean(_mae) * 0.02, -0.2, 0.2))
        except Exception:
            pass

        # Reward for closing trades: asymmetric win vs loss to improve accuracy (win rate)
        closed_trades_this_step = len(self.simulator.closed_trades) - getattr(self, '_last_closed_count', 0)
        if closed_trades_this_step > 0:
            recent_closed = self.simulator.closed_trades[-closed_trades_this_step:]
            for trade in recent_closed:
                if trade.get('symbol') == self.symbol:
                    pnl = float(trade.get('pnl', 0.0) or 0.0)
                    r_multiple = float(trade.get('r_multiple', 0.0) or 0.0)
                    _reason = str(trade.get('close_reason', ''))
                    if pnl > 0:
                        reward += min(r_multiple * 0.15, 0.6) if r_multiple > 0 else 0.12
                        if _reason == 'TP':
                            reward += 0.08
                    else:
                        reward -= 0.22
                        if _reason == 'SL':
                            reward -= 0.06
                        # Consecutive-loss penalty: discourages repeating same mistake
                        if getattr(self, '_last_close_was_loss', False):
                            reward -= 0.04
                        self._last_close_was_loss = True
                    if pnl > 0:
                        self._last_close_was_loss = False
        self._last_closed_count = len(self.simulator.closed_trades)

        # Penalty for holding positions too long (encourages timely exits)
        if len(self.simulator.open_trades) > 0:
            from datetime import datetime
            now = datetime.utcnow()
            
            for trade in self.simulator.open_trades:
                if trade.get('symbol') != self.symbol:
                    continue
                try:
                    trade_time_str = trade.get('time', '')
                    if trade_time_str:
                        trade_time = datetime.fromisoformat(trade_time_str.replace('Z', '+00:00'))
                        if trade_time.tzinfo:
                            trade_time = trade_time.replace(tzinfo=None)
                        age = now - trade_time
                        age_hours = age.total_seconds() / 3600
                        
                        # Base holding penalty (small)
                        reward -= 0.00005

                        # Regime-specific allowance/penalty
                        try:
                            allowed_hours = self.max_trade_duration_hours * (1.25 if (is_trend or is_high_vol) else 0.75)
                            if age_hours > allowed_hours:
                                reward -= 0.002 * min(2.0, (age_hours - allowed_hours) / max(1e-6, self.max_trade_duration_hours))
                        except Exception:
                            pass
                        
                        # Increasing penalty as trade ages
                        if age_hours > self.max_trade_duration_hours * 0.75:
                            penalty_factor = (age_hours / self.max_trade_duration_hours) - 0.75
                            reward -= 0.002 * penalty_factor  # Increased penalty
                        
                        # Strong penalty if exceeding max duration
                        if age_hours > self.max_trade_duration_hours:
                            reward -= 0.02  # Increased penalty
                except Exception:
                    reward -= 0.00005
        
        # Penalty for excessive trading (overtrading - reduces win rate)
        # More aggressive penalty to prevent churn
        if self.total_trades_executed > 50:  # Reduced threshold from 100
            overtrading_penalty = (self.total_trades_executed - 50) * 0.00005
            reward -= overtrading_penalty
        
        # Bonus for maintaining positive equity; stronger penalty for drawdown (risk management)
        if current_equity > self.starting_balance:
            equity_bonus = (current_equity - self.starting_balance) / self.starting_balance * 0.12
            reward += equity_bonus
        elif current_equity < self.starting_balance * 0.95:
            # Progressive drawdown penalty so agent learns to cut losses and preserve capital
            drawdown_ratio = 1.0 - (current_equity / self.starting_balance)
            drawdown_penalty = min(1.0, drawdown_ratio * 1.2)  # Cap so one step isn't overwhelming
            reward -= drawdown_penalty

        # Clip reward to reasonable range (wider for more signal)
        return np.clip(reward, -2.0, 2.0)
    
    def _lazy_init_fetchers(self):
        """Initialize data fetchers on first use, inside the process."""
        if not self.data_fetchers:
            # Initialize market_data as dict if not already done
            if not hasattr(self, 'market_data') or not isinstance(self.market_data, dict):
                self.market_data = {}
            
            # Initialize each fetcher
            for tf, config in self._fetcher_configs.items():
                try:
                    # Import here to avoid circular imports and use the canonical fetcher
                    from coinEx_getting_data import CoinExDataFetcher
                    self.data_fetchers[tf] = CoinExDataFetcher(**config)
                    # Initialize empty dataframe for this timeframe
                    self.market_data[tf] = pd.DataFrame()
                except Exception as e:
                    print(f"Warning: Could not initialize {tf} fetcher: {e}")
                    self.market_data[tf] = pd.DataFrame()
            
            # Set the primary data fetcher
            self.data_fetcher = self.data_fetchers.get(self.primary_timeframe)
            
            # Mark as available for now; _load_market_data will validate real data
            self.data_unavailable = False
    
    def _load_market_data(self) -> None:
        """Load initial OHLCV history for all configured timeframes (live mode)."""
        try:
            for tf, fetcher in self.data_fetchers.items():
                try:
                    # Pull enough history for indicators; fetcher handles limits internally
                    fetcher.update_initial(limit=min(self.lookback_window + 50, fetcher.max_candles))
                except Exception as exc:
                    print(f"[RL_ENV] Warning: initial fetch failed for {tf}: {exc}")
                    self.market_data[tf] = pd.DataFrame()
                    continue

                df = pd.DataFrame(list(fetcher.candles))
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(
                        df['timestamp'],
                        utc=True,
                        errors='coerce',
                    )
                    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
                self.market_data[tf] = df

            primary_df = self.market_data.get(self.primary_timeframe, pd.DataFrame())
            self.data_unavailable = primary_df.empty
            if self.data_unavailable:
                print(f"[RL_ENV] Warning: no primary timeframe data loaded for {self.symbol}")
        except Exception as exc:
            print(f"[RL_ENV] Error while loading market data: {exc}")
            self.data_unavailable = True
    
    def _load_historical_market_data(self) -> None:
        """Load initial historical data from replay (historical mode)."""
        try:
            if self.historical_replay is None:
                self.data_unavailable = True
                return
            
            # Get initial data from replay
            all_data = self.historical_replay.get_all_current_data()
            self.market_data = all_data
            
            primary_df = self.market_data.get(self.primary_timeframe, pd.DataFrame())
            self.data_unavailable = primary_df.empty
            if self.data_unavailable:
                print(f"[RL_ENV] Warning: no primary timeframe data loaded for {self.symbol}")
            else:
                print(f"[RL_ENV] Loaded historical data: {len(primary_df)} candles in primary timeframe")
        except Exception as exc:
            print(f"[RL_ENV] Error while loading historical market data: {exc}")
            self.data_unavailable = True
    
    def _get_current_price(self) -> Optional[float]:
        """Get current market price from primary timeframe"""
        if isinstance(self.market_data, dict):
            primary_data = self.market_data.get(self.primary_timeframe)
        elif isinstance(self.market_data, pd.DataFrame):
            primary_data = self.market_data
        else:
            primary_data = getattr(self, '_primary_market_data', None)
        
        if primary_data is not None and len(primary_data) > 0:
            return float(primary_data['close'].iloc[-1])
        return None

    def _compute_signal_quality(self) -> float:
        """Compute a soft signal quality score [0,1] from primary timeframe features."""
        try:
            if isinstance(self.market_data, dict):
                df = self.market_data.get(self.primary_timeframe)
            elif isinstance(self.market_data, pd.DataFrame):
                df = self.market_data
            else:
                df = getattr(self, '_primary_market_data', None)
            if df is None or len(df) == 0:
                return 0.0
            df2 = df.copy()
            for c in ['open','high','low','close','volume']:
                if c in df2.columns:
                    df2[c] = pd.to_numeric(df2[c], errors='coerce').ffill().bfill()
            close_last = float(pd.to_numeric(df2['close'], errors='coerce').iloc[-1]) if 'close' in df2.columns and len(df2) > 0 else 0.0
            mean20 = float(pd.to_numeric(df2['close'], errors='coerce').rolling(20).mean().iloc[-1]) if 'close' in df2.columns and len(df2) > 0 else 0.0
            trend_strength = ((close_last - mean20) / mean20) if (mean20 and mean20 > 0) else 0.0
            rets = (pd.to_numeric(df2['close'], errors='coerce').pct_change() if 'close' in df2.columns else pd.Series([], dtype=float))
            volatility = float(rets.tail(20).std()) if len(rets) > 0 else 0.0
            vols = pd.to_numeric(df2.get('volume', pd.Series([], dtype=float)), errors='coerce')
            vol_mean20 = float(vols.rolling(20).mean().iloc[-1]) if len(vols) > 0 else 0.0
            volume_ratio = (float(vols.iloc[-1]) / vol_mean20) if (vol_mean20 and vol_mean20 > 0 and len(vols) > 0) else 0.0
            choppiness = float(np.tanh(volatility / (abs(trend_strength) + 1e-6))) if np.isfinite(volatility) else 1.0
            t = float(np.clip(abs(trend_strength) / 0.01, 0.0, 1.0))
            v = float(np.clip((volume_ratio - 1.0) / 1.0, 0.0, 1.0))
            c = float(np.clip(1.0 - choppiness, 0.0, 1.0))
            score = 0.4 * t + 0.3 * v + 0.3 * c
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    def _submit_with_gating(self, setup: dict) -> None:
        """Submit trade with action gating (soft or hard)."""
        if not getattr(self, 'action_gating_enabled', False):
            self.simulator.submit_signal(setup)
            return
        q = float(self._compute_signal_quality())
        if getattr(self, 'gating_soft', True):
            if q < getattr(self, 'gating_min_score', 0.3):
                try:
                    rp = float(setup.get('risk_pct', 0.25))
                    setup['risk_pct'] = max(0.05, rp * max(0.2, q))
                except Exception:
                    pass
            self.simulator.submit_signal(setup)
        else:
            if q >= getattr(self, 'gating_min_score', 0.3):
                self.simulator.submit_signal(setup)
            else:
                if self.current_step % 50 == 0:
                    try:
                        print('[RL_ENV] GATING block: quality={:.2f} < {:.2f}'.format(q, getattr(self, 'gating_min_score', 0.3)))
                    except Exception:
                        pass

    def _get_observation(self) -> np.ndarray:
        obs_parts: List[np.ndarray] = []

        # Resolve primary data
        if isinstance(self.market_data, dict):
            primary_data = self.market_data.get(self.primary_timeframe)
        elif isinstance(self.market_data, pd.DataFrame):
            primary_data = self.market_data
        else:
            primary_data = getattr(self, '_primary_market_data', None)

        n_primary_bars = 50
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']

        # Primary OHLCV (flattened)
        if primary_data is None or len(primary_data) == 0:
            primary_ohlcv = np.zeros((n_primary_bars, 5), dtype=np.float32)
            close_last = 0.0
            vol_series = pd.Series([], dtype=float)
            dfp = None
        else:
            dfp = primary_data.copy()
            for c in ohlcv_cols:
                if c in dfp.columns:
                    dfp[c] = pd.to_numeric(dfp[c], errors='coerce').ffill().bfill()
                else:
                    dfp[c] = 0.0
            try:
                dfp = self._calculate_indicators(dfp)
            except Exception:
                pass
            tail = dfp[ohlcv_cols].tail(n_primary_bars).to_numpy()
            if tail.shape[0] < n_primary_bars:
                pad = np.zeros((n_primary_bars - tail.shape[0], 5))
                primary_ohlcv = np.vstack([pad, tail]).astype(np.float32)
            else:
                primary_ohlcv = tail.astype(np.float32)
            close_last = float(dfp['close'].iloc[-1]) if 'close' in dfp.columns and len(dfp) > 0 else 0.0
            vol_series = pd.to_numeric(dfp['volume'], errors='coerce') if 'volume' in dfp.columns else pd.Series([], dtype=float)

        obs_parts.append(primary_ohlcv.flatten())

        # Key indicators (8)
        def _safe_last(series_name: str, default: float = 0.0) -> float:
            if isinstance(dfp, pd.DataFrame) and series_name in dfp.columns and len(dfp) > 0:
                v = dfp[series_name].iloc[-1]
                try:
                    v = float(v)
                    if np.isfinite(v):
                        return v
                except Exception:
                    pass
            return float(default)

        rsi_last = _safe_last('rsi', 50.0)
        macd_last = _safe_last('macd', 0.0)
        macd_sig = _safe_last('macd_signal', 0.0)
        atr_last = _safe_last('atr', 0.0)
        atr_pct = (atr_last / close_last) if (close_last and close_last > 0) else 0.0
        vol_mean20 = float(pd.to_numeric(vol_series, errors='coerce').rolling(20).mean().iloc[-1]) if len(vol_series) > 0 else 0.0
        volume_surge = (float(vol_series.iloc[-1]) / vol_mean20) if (vol_mean20 and vol_mean20 > 0 and len(vol_series) > 0) else 0.0
        mean20 = float(pd.to_numeric(dfp['close'], errors='coerce').rolling(20).mean().iloc[-1]) if isinstance(dfp, pd.DataFrame) and 'close' in dfp.columns and len(dfp) > 0 else 0.0
        trend_strength = ((close_last - mean20) / mean20) if (mean20 and mean20 > 0) else 0.0
        rets = (pd.to_numeric(dfp['close'], errors='coerce').pct_change() if isinstance(dfp, pd.DataFrame) and 'close' in dfp.columns else pd.Series([], dtype=float))
        volatility = float(rets.tail(20).std()) if len(rets) > 0 else 0.0
        price_change = float(rets.iloc[-1]) if len(rets) > 0 and np.isfinite(rets.iloc[-1]) else 0.0
        choppiness = float(np.tanh(volatility / (abs(trend_strength) + 1e-6))) if np.isfinite(volatility) else 0.0

        key_indicators = np.array([
            rsi_last,
            macd_last - macd_sig,
            atr_pct,
            volume_surge,
            trend_strength,
            volatility,
            price_change,
            choppiness,
        ], dtype=np.float32)
        key_indicators = np.nan_to_num(key_indicators, nan=0.0, posinf=0.0, neginf=0.0)
        obs_parts.append(key_indicators)

        # Secondary timeframes (12 each)
        for tf in self.timeframes:
            if tf == self.primary_timeframe:
                continue
            df = self.market_data.get(tf) if isinstance(self.market_data, dict) else None
            if df is None or len(df) == 0:
                obs_parts.append(np.zeros(12, dtype=np.float32))
                continue
            df2 = df.copy()
            for c in ohlcv_cols:
                if c in df2.columns:
                    df2[c] = pd.to_numeric(df2[c], errors='coerce').ffill().bfill()
            cls = pd.to_numeric(df2.get('close', pd.Series([], dtype=float)), errors='coerce')
            vols = pd.to_numeric(df2.get('volume', pd.Series([], dtype=float)), errors='coerce')
            rets_tf = cls.pct_change()
            last5 = rets_tf.tail(5).to_numpy()
            if last5.shape[0] < 5:
                last5 = np.concatenate([np.zeros(5 - last5.shape[0]), last5])
            vol_std = float(rets_tf.tail(20).std()) if len(rets_tf) > 0 else 0.0
            mean20_tf = float(cls.rolling(20).mean().iloc[-1]) if len(cls) > 0 else 0.0
            trend_tf = ((float(cls.iloc[-1]) - mean20_tf) / mean20_tf) if (mean20_tf and mean20_tf > 0) else 0.0
            vol_mean20_tf = float(vols.rolling(20).mean().iloc[-1]) if len(vols) > 0 else 0.0
            vol_ratio = (float(vols.iloc[-1]) / vol_mean20_tf) if (vol_mean20_tf and vol_mean20_tf > 0 and len(vols) > 0) else 0.0
            bullish = 1.0 if trend_tf > 0.002 else 0.0
            bearish = 1.0 if trend_tf < -0.002 else 0.0
            high_vol = 1.0 if vol_std > 0.01 else 0.0
            low_vol = 1.0 if vol_std < 0.003 else 0.0
            sec_vec = np.array([
                *last5.tolist(),
                vol_std,
                trend_tf,
                vol_ratio,
                bullish,
                bearish,
                high_vol,
                low_vol,
            ], dtype=np.float32)
            sec_vec = np.nan_to_num(sec_vec, nan=0.0, posinf=0.0, neginf=0.0)
            obs_parts.append(sec_vec)

        # Portfolio state (5)
        balance = float(getattr(self.simulator, 'balance', self.starting_balance))
        equity = float(getattr(self.simulator, 'equity', balance))
        num_positions = float(len(getattr(self.simulator, 'open_trades', [])))
        unrealized = 0.0
        for t in getattr(self.simulator, 'open_trades', []):
            entry = float(t.get('entry', close_last or 0.0))
            size = float(t.get('size', 0.0))
            side = t.get('side')
            if side == 'BUY':
                unrealized += (close_last - entry) * size
            else:
                unrealized += (entry - close_last) * size
        realized_recent = sum(float(t.get('pnl', 0.0)) for t in getattr(self.simulator, 'closed_trades', [])[-50:])
        portfolio_vec = np.array([balance, equity, num_positions, unrealized, realized_recent], dtype=np.float32)
        portfolio_vec = np.nan_to_num(portfolio_vec, nan=0.0, posinf=0.0, neginf=0.0)
        obs_parts.append(portfolio_vec)

        # Regime flags (4) from primary
        is_trend = 1.0 if abs(trend_strength) > 0.003 else 0.0
        is_range = 1.0 - is_trend
        is_high_vol = 1.0 if volatility > 0.01 else 0.0
        is_low_vol = 1.0 if volatility < 0.003 else 0.0
        regime_vec = np.array([is_trend, is_range, is_high_vol, is_low_vol], dtype=np.float32)
        obs_parts.append(regime_vec)

        # Symbol one-hot
        one_hot = np.zeros(int(self.total_symbols), dtype=np.float32)
        idx = int(self.symbol_index) if 0 <= int(self.symbol_index) < int(self.total_symbols) else None
        if idx is not None:
            one_hot[idx] = 1.0
        obs_parts.append(one_hot)

        obs = np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)
        # Ensure exact shape
        if obs.shape[0] != self.observation_dim:
            if obs.shape[0] < self.observation_dim:
                pad = np.zeros(self.observation_dim - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, pad], axis=0)
            else:
                obs = obs[: self.observation_dim]
        return obs

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with NaN/inf protection"""
        df = df.copy()
        
        # Ensure we have valid numeric data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    # First, forward-fill and back-fill to propagate valid values
                    df[col] = df[col].ffill().bfill()
                if df[col].isnull().any():
                    # If NaNs still exist (e.g., all values were NaN), fill with a neutral value
                    fill_value = df[col].mean() if not pd.isna(df[col].mean()) else 0
                    df[col] = df[col].fillna(fill_value)
        
        # RSI - with safe division
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Safe division to prevent inf
        rs = gain / loss.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].replace([np.inf, -np.inf], 50.0).fillna(50.0).clip(0, 100)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = (exp1 - exp2).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_signal'] = df['macd_signal'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        # Bollinger Bands - with safe std calculation
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std().fillna(0.0)
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_middle'] = df['bb_middle'].fillna(df['close'])

        # ATR (Average True Range) for volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
        df['atr'] = tr.rolling(window=14).mean().fillna(0)

        # Final cleanup
        df['atr'] = np.where(np.isfinite(df['atr']), df['atr'], 0)

        return df

    def _calculate_rr_and_risk(self, current_price: float) -> Tuple[float, float, float]:
        """
        Calculate dynamic RR ratio, stop loss percentage, and risk percentage.
        Uses pair-specific dynamic SL calculator with ATR, volatility, and volume adjustments.
        """
        stop_loss_pct = 0.02  # default fallback 2%
        sl_calc = None

        primary_data = self.market_data.get(self.primary_timeframe)
        
        # Use dynamic SL calculator for pair-specific minimum SL
        try:
            sl_calc = get_sl_calculator()
            if isinstance(primary_data, pd.DataFrame) and len(primary_data) >= 20 and current_price > 0:
                # Calculate pair-specific minimum SL
                stop_loss_pct = sl_calc.calculate_min_sl_pct(
                    symbol=self.symbol,
                    current_price=current_price,
                    df=primary_data
                )
            else:
                # Fallback: use pair config defaults
                stop_loss_pct = sl_calc.calculate_min_sl_pct(
                    symbol=self.symbol,
                    current_price=current_price,
                    df=None
                )
        except Exception as e:
            # If dynamic calculator fails, use ATR-based fallback
            atr_val = None
            if isinstance(primary_data, pd.DataFrame) and len(primary_data) >= 20:
                if 'atr' in primary_data.columns and not primary_data['atr'].empty and float(primary_data['atr'].iloc[-1]) > 0:
                    atr_val = float(primary_data['atr'].iloc[-1])
                else:
                    # Compute ATR from recent bars
                    recent = primary_data.tail(50).copy()
                    try:
                        hl = pd.to_numeric(recent['high'], errors='coerce') - pd.to_numeric(recent['low'], errors='coerce')
                        hc = (pd.to_numeric(recent['high'], errors='coerce') - pd.to_numeric(recent['close'], errors='coerce').shift()).abs()
                        lc = (pd.to_numeric(recent['low'], errors='coerce') - pd.to_numeric(recent['close'], errors='coerce').shift()).abs()
                        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1, skipna=False)
                        atr_series = tr.rolling(window=14).mean()
                        if not atr_series.empty and np.isfinite(atr_series.iloc[-1]):
                            atr_val = float(max(atr_series.iloc[-1], 0.0))
                    except Exception:
                        atr_val = None

            if atr_val is not None and current_price and current_price > 0:
                stop_loss_pct = (atr_val * 1.8) / current_price  # 1.8x ATR for RL training
            else:
                stop_loss_pct = 0.02  # Default 2%

        # Pair-specific clamp: use configured min/max instead of hard 1-5%
        try:
            if sl_calc is not None:
                cfg = sl_calc.get_pair_config(self.symbol)
                min_pct = float(cfg.get('min_pct', 1.0)) / 100.0
                max_pct = float(cfg.get('max_pct', 5.0)) / 100.0
            else:
                min_pct = 0.01
                max_pct = 0.05
        except Exception:
            min_pct = 0.01
            max_pct = 0.05
        # Ensure stop_loss_pct is valid and within bounds
        if stop_loss_pct <= 0 or not np.isfinite(stop_loss_pct):
            stop_loss_pct = max(min_pct, 0.015)  # Use at least 1.5% if invalid
        stop_loss_pct = float(np.clip(stop_loss_pct, min_pct, max_pct))

        rr_ratio = 1.5  # Target 1.5:1 reward-to-risk
        risk_pct = max(self.min_trade_risk_pct, 0.25)  # Enforce minimum risk size (percent of equity)

        return rr_ratio, stop_loss_pct, risk_pct

    def _check_and_close_stale_trades(self):
        """Check for trades that have exceeded maximum duration and force close them"""
        if not hasattr(self, 'max_trade_duration_hours'):
            return
        
        current_price = self._get_current_price()
        if current_price is None:
            return
        
        from datetime import datetime, timedelta
        max_duration = timedelta(hours=self.max_trade_duration_hours)
        now = datetime.utcnow()
        
        trades_to_close = []
        for trade in self.simulator.open_trades:
            # CRITICAL: Only check trades for THIS symbol
            if trade.get('symbol') != self.symbol:
                continue
            if trade.get('status') != 'OPEN':
                continue
            
            try:
                trade_time_str = trade.get('time', '')
                if not trade_time_str:
                    continue
                
                # Parse trade time (handle both with and without timezone)
                trade_time = datetime.fromisoformat(trade_time_str.replace('Z', '+00:00'))
                if trade_time.tzinfo:
                    trade_time = trade_time.replace(tzinfo=None)
                
                age = now - trade_time
                if age > max_duration:
                    trades_to_close.append(trade.get('id'))
                    print(f"[RL_ENV] ⏰ STALE TRADE DETECTED | Pair: {self.symbol} | Trade ID: {trade.get('id', 'unknown')[:8]}... | Age: {age.total_seconds()/3600:.1f}h | Max: {self.max_trade_duration_hours}h")
            except Exception as e:
                # If we can't parse the time, skip this trade
                continue
        
        # Close stale trades
        for trade_id in trades_to_close:
            try:
                print(f"[RL_ENV] 🔴 FORCE CLOSING STALE TRADE | Pair: {self.symbol} | Trade ID: {trade_id[:8] if trade_id else 'unknown'}... | Reason: MAX_DURATION_EXCEEDED")
                success = self.simulator.close_trade(trade_id, reason=f"MAX_DURATION_EXCEEDED_{self.max_trade_duration_hours}h")
                if success:
                    self.episode_trades.append({
                        'step': self.current_step,
                        'action': 'FORCE_CLOSE',
                        'price': current_price,
                        'type': 'CLOSE',
                        'trade_id': trade_id,
                        'reason': 'max_duration'
                    })
                    print(f"[RL_ENV] ✅ Stale trade closed successfully")
                else:
                    print(f"[RL_ENV] ⚠️ Failed to close stale trade (may have been closed already)")
            except Exception as e:
                print(f"[RL_ENV] ⚠️ Error closing stale trade: {e}")
    
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
            'pnl_pct': ((self.simulator.equity - self.starting_balance) / self.starting_balance * 100.0) if self.starting_balance > 0 else 0.0,
        }

    def render(self):
        """Render environment (optional)"""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"Step: {info['step']}, Balance: ${info['balance']:.2f}, "
                  f"Equity: ${info['equity']:.2f}, Trades: {info['total_trades']}, "
                  f"PnL: ${info['pnl']:.2f} ({info['pnl_pct']:.2f}%)")
