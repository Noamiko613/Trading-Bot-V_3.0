"""
Reinforcement Learning Training Script
======================================

Trains a neural network RL agent for trading with:
- Pause/Resume functionality
- Progress monitoring
- Model checkpointing
"""

import os
import sys
import json
import time
import signal
import threading
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from collections import deque

# Force UTF-8 output to avoid Windows codepage errors when printing symbols/emojis
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from datetime import datetime

# Try to import torch - handle DLL errors on Windows
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except (ImportError, OSError, RuntimeError) as e:
    HAS_TORCH = False
    print(f"Error: torch could not be loaded ({type(e).__name__}: {e})")
    print("\nTo fix this issue:")
    print("1. Install Visual C++ Redistributables: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("2. Reinstall torch: pip uninstall torch && pip install torch")
    print("3. Or use CPU-only version: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    raise

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError as e:
    HAS_SB3 = False
    print(f"Error: stable-baselines3 not available: {e}")
    print("Install with: pip install stable-baselines3[extra]")
    raise

# Try to source a RecurrentPPO implementation, if present
try:
    from stable_baselines3 import RecurrentPPO as RECUR_PPO_CLS
except Exception:
    try:
        from sb3_contrib import RecurrentPPO as RECUR_PPO_CLS
    except Exception:
        RECUR_PPO_CLS = None

from rl_trading_env import TradingEnv
from rl_progress_monitor import RLProgressMonitor
from rl_dashboard import RLDashboard
from rl_experience_buffer import PrioritizedExperienceBuffer
from rl_behavior_cloning import BehaviorCloningTrainer
from rl_metrics_evaluator import MultiMetricEvaluator
from utils.analytics import PerformanceAnalytics
from historical_data_replay import HistoricalDataReplay, get_common_cache_date_range


RL_TRAINING_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

RL_TRAINING_TIMEFRAMES = [
    "1m",
    "5m",
    "15m",
    "1h",
    "4h",
    "6h",
    "12h",
    "1d",
]


class TrainingState:
    """Manages training state for pause/resume (per symbol/model_dir)."""
    
    def __init__(self, state_file: str):
        """
        Parameters
        ----------
        state_file : str
            Path to the JSON file that will store training state for a single
            symbol/model directory. Callers should pass a symbol-specific path
            (e.g. `<model_dir>/<SYMBOL>/rl_training_state.json`) so that
            multi-symbol training does not share timesteps/checkpoints.
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load training state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Backwards‑compatible default mode
                    if "mode" not in state:
                        state["mode"] = "training"
                    return state
            except Exception:
                pass
        return {
            'episode': 0,
            'total_timesteps': 0,
            'last_checkpoint': None,
            'best_reward': float('-inf'),
            'total_trades': 0,
            'mode': 'training',
            'historical_pretraining_completed': False,
        }
    
    def save_state(
        self,
        episode: int,
        timesteps: int,
        checkpoint_path: str,
        best_reward: float,
        total_trades: int,
        mode: str = "training",
    ):
        """Save training state"""
        # Preserve existing state fields (like historical_pretraining_completed)
        self.state.update({
            'episode': episode,
            'total_timesteps': timesteps,
            'last_checkpoint': checkpoint_path,
            'best_reward': best_reward,
            'total_trades': total_trades,
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
        })
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def get_state(self) -> Dict:
        """Get current state"""
        return self.state.copy()


class PauseResumeCallback(BaseCallback):
    """Callback to handle pause/resume"""
    
    def __init__(self, state_manager: TrainingState, pause_event: threading.Event, verbose=0):
        super().__init__(verbose)
        self.state_manager = state_manager
        self.pause_event = pause_event
        self.paused = False
    
    def _on_step(self) -> bool:
        """Check if training should pause"""
        if self.pause_event.is_set():
            if not self.paused:
                self.paused = True
                print("\n[PAUSE] Training paused. Press Ctrl+C again to resume...")
            return False  # Stop training
        else:
            if self.paused:
                self.paused = False
                print("\n[RESUME] Training resumed...")
            return True


class ExperienceReplayCallback(BaseCallback):
    """
    Callback that integrates experience replay buffers with PPO training.
    After each rollout, samples from good/bad buffers and applies auxiliary
    supervised learning to accelerate learning from mistakes.
    """
    
    def __init__(
        self,
        experience_buffer: PrioritizedExperienceBuffer,
        bc_trainer: BehaviorCloningTrainer,
        bc_learning_rate: float = 1e-5,
        bc_batch_size: int = 32,
        bc_ratio: float = 0.1,  # 10% of updates come from BC
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.experience_buffer = experience_buffer
        self.bc_trainer = bc_trainer
        self.bc_learning_rate = bc_learning_rate
        self.bc_batch_size = bc_batch_size
        self.bc_ratio = bc_ratio
        self.episode_started = False
    
    def _on_rollout_start(self) -> None:
        """Called when a new rollout starts"""
        if not self.episode_started:
            self.experience_buffer.start_episode()
            self.episode_started = True
    
    def _on_step(self) -> bool:
        """Called on each step - collect transitions"""
        if self.episode_started:
            obs = self.training_env.get_original_obs()
            action = self.locals['actions']
            reward = self.locals['rewards']
            next_obs = self.locals['new_obs']
            done = self.locals['dones']
            info = self.locals['infos']
            self.experience_buffer.add_transition(obs, action, reward, next_obs, done, info)
        return True
    
    def _on_rollout_end(self) -> None:
        """Called when rollout ends - process episode"""
        if self.episode_started:
            # Get final pnl and equity
            final_pnl = self.training_env.get_attr("simulator")[0].equity - self.training_env.get_attr("starting_balance")[0]
            final_equity = self.training_env.get_attr("simulator")[0].equity
            
            # End the episode in the buffer
            self.experience_buffer.end_episode(final_pnl, final_equity)
            self.episode_started = False

            # Sample from buffer and train
            good_transitions = self.experience_buffer.sample_transitions_from_good(self.bc_batch_size)
            bad_transitions = self.experience_buffer.sample_transitions_from_bad(self.bc_batch_size)
            
            if self.bc_trainer is not None:
                if good_transitions:
                    self.bc_trainer.train_on_good_episodes(good_transitions)
                if bad_transitions:
                    self.bc_trainer.train_negative_correction(bad_transitions)



class CustomPPONetwork(nn.Module):
    """Custom neural network architecture for PPO"""
    
    def __init__(self, feature_dim: int, action_dim: int, net_arch=[512, 512, 256, 128]):
        super().__init__()
        self.feature_dim = feature_dim
        self.net_arch = net_arch
        
        # Build shared layers
        layers = []
        prev_dim = feature_dim
        for layer_size in net_arch:
            layers.append(nn.Linear(prev_dim, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = layer_size
        
        self.shared_net = nn.Sequential(*layers)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, features):
        shared = self.shared_net(features)
        value = self.value_head(shared)
        policy_logits = self.policy_head(shared)
        return policy_logits, value


class PolicyFeatureExtractor(nn.Module):
    """Feature extractor for stable-baselines3"""
    
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__()
        # Calculate input size
        n_input = observation_space.shape[0]
        
        # Custom network architecture
        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )
    
    def forward(self, observations):
        return self.net(observations)


class RLTrainer:
    """RL Training manager"""
    
    def __init__(
        self,
        symbol: Optional[str] = None,
        starting_balance: float = 1_000_000.0,
        max_trades: Optional[int] = None,  # None = unlimited
        model_dir: str = "models/rl_models",
        checkpoint_interval: int = 10000
    ):
        """
        If symbol is provided, the PPO model will be trained for only that
        market. If symbol is None, a single shared policy is trained across
        all symbols in RL_TRAINING_SYMBOLS.
        """
        self.symbol = symbol
        self.starting_balance = starting_balance
        self.max_trades = max_trades
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        
        # Training state: one state file per logical model directory. In
        # multi-symbol mode this is a shared state for the unified model.
        state_path = self.model_dir / "rl_training_state.json"
        self.state_manager = TrainingState(str(state_path))
        self.pause_event = threading.Event()
        self.is_paused = False
        self.shutdown_event = threading.Event()  # Flag for graceful shutdown
        
        # Historical pre-training configuration
        # Note: CoinEx typically only has data from ~2020-2021 onwards
        # For crypto bear/bull markets, we want:
        # - 2020-2021: Bull market start
        # - 2022: Bear market/crash
        # - 2023: Recovery
        # - 2024: Bull market
        # Adjust date range based on what's actually available
        from datetime import datetime
        today = datetime.now()
        self.historical_start_date = os.getenv('HISTORICAL_START_DATE', '2020-01-01')
        end_date_env = os.getenv('HISTORICAL_END_DATE', None)
        if end_date_env is None:
            # Default to today (don't request future dates)
            self.historical_end_date = today.strftime('%Y-%m-%d')
        else:
            # Validate end date isn't in the future
            try:
                end_dt = datetime.fromisoformat(end_date_env.replace('Z', '+00:00'))
                if end_dt > today:
                    print(f"[RL] Warning: End date {end_date_env} is in the future. Using today's date instead.")
                    self.historical_end_date = today.strftime('%Y-%m-%d')
                else:
                    self.historical_end_date = end_date_env
            except:
                self.historical_end_date = today.strftime('%Y-%m-%d')
        self.use_historical_pretraining = os.getenv('USE_HISTORICAL_PRETRAINING', '1') == '1'
        # Historical pre-training timesteps: 
        # - Default: 9,600,000 (full historical phase before live-style training)
        # - Overridable via HISTORICAL_PRETRAIN_TIMESTEPS env or --historical-pretrain-timesteps CLI
        self.historical_pretrain_timesteps = int(os.getenv('HISTORICAL_PRETRAIN_TIMESTEPS', '9600000'))
        self.historical_replay = None
        
        # Progress monitor
        self.progress_monitor = None
        self.monitor_thread = None
        
        # Dashboard
        self.dashboard = None
        
        # Environment
        self.env = None
        self.model = None

        # Analytics for global accuracy (win rate)
        self.analytics = PerformanceAnalytics()
        
        # Config path for persistence
        self.acceptance_config_path = Path("config/rl_acceptance_criteria.json")

        # Load acceptance criteria from config (conservative defaults)
        self.acceptance_config = self._load_acceptance_config()
        acceptance = self.acceptance_config.get("acceptance_criteria", {})
        buffer_config = self.acceptance_config.get("experience_buffer", {})
        norm_config = self.acceptance_config.get("normalization", {})
        paper_config = self.acceptance_config.get("paper_testing", {})
        training_config = self.acceptance_config.get("training_loop", {})
        auto_tighten_config = self.acceptance_config.get("auto_tighten", {})
        
        # Experience replay buffers for accelerated learning
        # Fixed capacities: 10k transitions each (top/bottom episodes)
        self.experience_buffer = PrioritizedExperienceBuffer(
            good_buffer_size=buffer_config.get("good_buffer_size", 10000),  # 5k-20k range
            bad_buffer_size=buffer_config.get("bad_buffer_size", 10000),
            good_percentile=buffer_config.get("good_percentile", 0.75),
            bad_percentile=buffer_config.get("bad_percentile", 0.25),
            persist_path=str(self.model_dir / "experience_buffers.json")
        )
        
        # Sampling ratios from config
        self.sampling_ratios = buffer_config.get("sampling_ratios", {
            "on_policy_ppo": 0.75,
            "bc_from_good": 0.15,
            "negative_correction": 0.10
        })
        
        # Multi-metric evaluator with stability windows (configurable)
        self.metrics_evaluator = MultiMetricEvaluator(
            win_rate_threshold=acceptance.get("win_rate_threshold", 68.0),
            profit_factor_threshold=acceptance.get("profit_factor_threshold", 1.3),
            sharpe_threshold=acceptance.get("sharpe_threshold", 1.0),
            max_drawdown_threshold=acceptance.get("max_drawdown_threshold", 12.0),
            min_trades_per_window=acceptance.get("min_trades_per_window", 500),
            stability_windows_required=acceptance.get("stability_windows_required", 3),
            window_size_trades=acceptance.get("window_size_trades", 500),
        )
        
        # Training loop config (from training_config block; fall back to defaults)
        # Core adaptive hyperparameters (needed even when loading from checkpoints)
        self.initial_ent_coef = training_config.get("initial_ent_coef", 0.08)
        self.min_ent_coef = training_config.get("min_ent_coef", 0.01)
        self.initial_lr = training_config.get("initial_lr", 5e-4)
        self.min_lr = training_config.get("min_lr", 1e-4)
        self.entropy_decay_steps = training_config.get("entropy_decay_steps", 1_000_000)
        # Respect constructor-provided checkpoint_interval unless overridden in config
        self.checkpoint_interval = training_config.get("checkpoint_interval", self.checkpoint_interval)
        self.max_trades_per_episode = training_config.get("max_trades_per_episode", 200)
        # Delay evaluation until enough experience is collected to reduce early overfitting/false positives
        self.min_timesteps_before_eval = training_config.get("min_timesteps_before_eval", 100_000)
        
        # Risk settings
        risk_settings = training_config.get("risk_settings", {})
        self.max_global_open_trades = risk_settings.get("max_global_open_trades", 20)
        
        # Behavior cloning trainer (initialized after model is created)
        self.bc_trainer = None
        
        # Reward normalization (running mean/std for stable value function learning)
        self.reward_clip_range = norm_config.get("reward_clip_range", [-1.0, 1.0])
        self.use_running_mean_std = norm_config.get("use_running_mean_std", True)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        self.reward_history = deque(maxlen=10000)
        
        # Observation normalization (running mean/std)
        self.observation_normalization = norm_config.get("observation_normalization", True)
        self.obs_mean = None
        self.obs_std = None
        self.obs_count = 0
        
        # Paper testing config
        self.keep_training_during_paper = paper_config.get("keep_training_during_paper", True)
        self.auto_rollback_enabled = paper_config.get("auto_rollback_enabled", True)
        self.rollback_threshold_drawdown = paper_config.get("rollback_threshold_drawdown", 15.0)
        self.rollback_threshold_sharpe = paper_config.get("rollback_threshold_sharpe", 0.5)
        self.paper_test_episodes = paper_config.get("paper_test_episodes", 5)
        self.last_safe_model_path = None
        self.current_mode = "training"  # Track current mode for rollback checks

        # Auto-tighten settings (if repeated failures/rollbacks occur)
        self.auto_tighten_enabled = auto_tighten_config.get("enabled", True)
        self.auto_tighten_on_rollback = auto_tighten_config.get("on_rollback", True)
        self.auto_tighten_on_eval_failures = auto_tighten_config.get("on_eval_failures", True)
        self.auto_tighten_failures_before_tighten = auto_tighten_config.get("failures_before_tighten", 3)
        self.auto_tighten_max_steps = auto_tighten_config.get("max_steps", 5)
        self.auto_tighten_win_rate_inc = auto_tighten_config.get("win_rate_increment", 1.0)
        self.auto_tighten_profit_factor_inc = auto_tighten_config.get("profit_factor_increment", 0.05)
        self.auto_tighten_sharpe_inc = auto_tighten_config.get("sharpe_increment", 0.1)
        self.auto_tighten_drawdown_dec = auto_tighten_config.get("drawdown_decrement", 0.5)
        self.auto_tighten_applied = 0
        self.eval_failures_since_tighten = 0
        
        # Setup signal handlers for pause/resume
        self._setup_signal_handlers()
    
    def _load_acceptance_config(self) -> Dict:
        """Load acceptance criteria and training config from file"""
        config_path = Path("config/rl_acceptance_criteria.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[RL] Warning: Could not load acceptance config: {e}. Using defaults.")
        # Return conservative defaults
        return {
            "acceptance_criteria": {
                "win_rate_threshold": 85.0,
                "profit_factor_threshold": 1.3,
                "sharpe_threshold": 1.0,
                "max_drawdown_threshold": 12.0,
                "min_trades_per_window": 500,
                "stability_windows_required": 3,
                "window_size_trades": 500,
            },
            "experience_buffer": {
                "good_buffer_size": 10000,
                "bad_buffer_size": 10000,
                "good_percentile": 0.75,
                "bad_percentile": 0.25,
                "sampling_ratios": {
                    "on_policy_ppo": 0.75,
                    "bc_from_good": 0.15,
                    "negative_correction": 0.10
                }
            },
            "normalization": {
                "reward_clip_range": [-1.0, 1.0],
                "use_running_mean_std": True,
                "observation_normalization": True
            },
            "paper_testing": {
                "keep_training_during_paper": True,
                "auto_rollback_enabled": True,
                "rollback_threshold_drawdown": 15.0,
                "rollback_threshold_sharpe": 0.5,
                "paper_test_episodes": 5
            },
            "training_loop": {
                "min_timesteps_before_eval": 100_000
            },
            "auto_tighten": {
                "enabled": True,
                "on_rollback": True,
                "on_eval_failures": True,
                "failures_before_tighten": 3,
                "max_steps": 5,
                "win_rate_increment": 1.0,
                "profit_factor_increment": 0.05,
                "sharpe_increment": 0.1,
                "drawdown_decrement": 0.5
            }
        }
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running mean/std and clip"""
        if self.use_running_mean_std:
            # Update running statistics
            self.reward_count += 1
            self.reward_history.append(reward)
            
            # Calculate running mean and std
            if len(self.reward_history) > 100:  # Need some history
                self.reward_mean = np.mean(self.reward_history)
                self.reward_std = np.std(self.reward_history)
                if self.reward_std < 1e-8:
                    self.reward_std = 1.0
            
            # Normalize
            if self.reward_std > 1e-8:
                normalized = (reward - self.reward_mean) / self.reward_std
            else:
                normalized = reward
        else:
            normalized = reward
        
        # Clip to reasonable range
        return np.clip(normalized, self.reward_clip_range[0], self.reward_clip_range[1])
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for pause/resume and graceful shutdown"""
        def signal_handler(sig, frame):
            if self.is_paused:
                print("\n[RESUME] Resuming training...")
                self.pause_event.clear()
                self.is_paused = False
            else:
                print("\n[PAUSE] Pausing training (press Ctrl+C again to resume, or Ctrl+Q to save and exit)...")
                self.pause_event.set()
                self.is_paused = True
        
        # Handle Ctrl+C for pause/resume
        signal.signal(signal.SIGINT, signal_handler)
        
        # Setup keyboard listener for Ctrl+Q (save and exit)
        self._setup_keyboard_listener()
    
    def _setup_keyboard_listener(self):
        """Setup keyboard listener for 'q' key to save and exit"""
        def keyboard_listener():
            """Listen for 'q' key press to trigger graceful shutdown"""
            import sys
            import platform
            
            # Windows-specific keyboard detection
            if platform.system() == 'Windows':
                try:
                    import msvcrt
                    while not self.shutdown_event.is_set():
                        # Check for 'q' key press (Windows)
                        if msvcrt.kbhit():
                            key = msvcrt.getch()
                            # Handle both regular 'q' and Ctrl+Q (0x11)
                            if key == b'q' or key == b'Q' or key == b'\x11':
                                print("\n[SHUTDOWN] 'q' key detected. Saving state and shutting down gracefully...")
                                self.shutdown_event.set()
                                break
                        time.sleep(0.1)  # Check every 100ms
                except ImportError:
                    # Fallback: use input thread
                    pass
            else:
                # Unix/Linux approach
                try:
                    import select
                    import termios
                    import tty
                    # Set terminal to raw mode
                    old_settings = termios.tcgetattr(sys.stdin)
                    try:
                        tty.setraw(sys.stdin.fileno())
                        while not self.shutdown_event.is_set():
                            if select.select([sys.stdin], [], [], 0.1)[0]:
                                key = sys.stdin.read(1)
                                if key.lower() == 'q' or ord(key) == 17:  # 'q' or Ctrl+Q
                                    print("\n[SHUTDOWN] 'q' key detected. Saving state and shutting down gracefully...")
                                    self.shutdown_event.set()
                                    break
                    finally:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except (ImportError, AttributeError, OSError):
                    # If terminal manipulation fails, use simple input
                    pass
        
        # Start keyboard listener in background thread
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()
    
    def create_environment(self, historical_replay=None):
        """Create trading environment.

        If self.symbol is set, a single-symbol environment is created.
        If self.symbol is None, we build one environment per symbol in
        RL_TRAINING_SYMBOLS and train a single shared PPO policy across
        all of them (multi-market learning).
        
        Args:
            historical_replay: Optional HistoricalDataReplay instance for offline pre-training
        """
        symbols = [self.symbol] if self.symbol else RL_TRAINING_SYMBOLS

        def make_env(idx: int, sym: str):
            def _thunk():
                # Create historical replay for this symbol if in historical mode
                symbol_replay = None
                if historical_replay is not None:
                    # Check if we have a replay map for multi-symbol training
                    replay_map = getattr(self, '_historical_replay_map', None)
                    if replay_map and sym in replay_map:
                        # Use symbol-specific replay
                        symbol_replay = replay_map[sym]
                    else:
                        # Fallback to single replay (backward compatibility)
                        symbol_replay = historical_replay
                
                env = TradingEnv(
                    symbol=sym,
                    starting_balance=self.starting_balance,
                    lookback_window=100,
                    max_trades=self.max_trades,  # None = unlimited
                    symbol_index=idx,
                    total_symbols=len(symbols),
                    use_continuous_actions=False,  # Keep discrete for now (can enable later)
                    domain_randomization=True,  # Enable domain randomization
                    kill_switch_enabled=False,  # Allow exploration; drawdown still penalized via reward
                    historical_replay=symbol_replay,  # Pass historical replay if available
                )
                # Wrap with Monitor for stats
                monitor_path = self.model_dir / f"training_monitor_{sym}.csv"
                env_mon = Monitor(env, filename=str(monitor_path))
                return env_mon

            return _thunk

        env_fns = [make_env(i, s) for i, s in enumerate(symbols)]
        # NOTE: On Windows, SubprocVecEnv often fails due to pickling issues (thread locks, etc.).
        # To keep training stable, we default to DummyVecEnv on Windows and only enable
        # SubprocVecEnv when explicitly requested AND the OS is not Windows.
        use_subproc = (
            os.getenv('USE_SUBPROC_VEC', '0') == '1'
            and len(env_fns) > 1
            and os.name != 'nt'
        )
        if use_subproc:
            vec = SubprocVecEnv(env_fns)
        else:
            vec = DummyVecEnv(env_fns)
        self.env = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.)
        return self.env
    
    def create_model(self, load_checkpoint: bool = True):
        """Create or load PPO model with custom network"""
        state = self.state_manager.get_state()
        
        # Custom feature extractor
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        from stable_baselines3.common.policies import ActorCriticPolicy
        
        class CustomFeatureExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space, features_dim: int = 512):
                super().__init__(observation_space, features_dim)
                n_input = observation_space.shape[0]
                
                self.net = nn.Sequential(
                    nn.Linear(n_input, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, features_dim)
                )
            
            def forward(self, observations):
                return self.net(observations)
        
        # Optimized network architecture for trading (faster learning, better generalization)
        # Slightly smaller but deeper network for better feature extraction
        policy_kwargs = {
            "net_arch": dict(
                pi=[256, 256, 128, 64],  # Policy network: optimized for action selection
                vf=[256, 256, 128, 64]   # Value network: optimized for value estimation
            )
        }
        
        # Try to load existing model - check multiple locations
        if load_checkpoint:
            checkpoint_path = None
            
            # First, try the checkpoint from state file
            if state.get('last_checkpoint'):
                if os.path.exists(state['last_checkpoint']):
                    checkpoint_path = state['last_checkpoint']
                else:
                    print(f"[RL] Warning: Checkpoint from state file not found: {state['last_checkpoint']}")
            
            # If state checkpoint doesn't exist, look for latest checkpoint in checkpoints directory
            if not checkpoint_path:
                checkpoints_dir = self.model_dir / "checkpoints"
                if checkpoints_dir.exists():
                    # Check for both historical and regular checkpoints
                    all_checkpoints = sorted(checkpoints_dir.glob("rl_model_*.zip"))
                    historical_checkpoints = sorted(checkpoints_dir.glob("rl_model_historical_*.zip"))
                    
                    # Prefer historical checkpoints if we're in historical mode, otherwise use any
                    if historical_checkpoints:
                        checkpoint_path = str(historical_checkpoints[-1])
                        print(f"[RL] Found latest historical checkpoint: {checkpoint_path}")
                    elif all_checkpoints:
                        checkpoint_path = str(all_checkpoints[-1])
                        print(f"[RL] Found latest checkpoint: {checkpoint_path}")
            
            # Try to load the checkpoint
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    print(f"[RL] Loading model from {checkpoint_path}")
                    self.model = PPO.load(
                        checkpoint_path,
                        env=self.env,
                        device='auto'
                    )
                    # Update entropy coefficient adaptively based on training progress
                    total_timesteps = state.get('total_timesteps', 0)
                    progress = min(1.0, total_timesteps / 1_000_000)  # Decay over 1M steps
                    adaptive_ent = 0.08 * (1.0 - progress * 0.875) + 0.01  # Decay from 0.08 to 0.01
                    self.model.ent_coef = max(0.01, adaptive_ent)
                    print(f"[RL] Model loaded successfully. Continuing from episode {state['episode']}, timesteps: {state['total_timesteps']}")
                    print(f"[RL] Adaptive entropy coefficient: {self.model.ent_coef:.4f} (progress: {progress*100:.1f}%)")
                    return self.model
                except Exception as e:
                    print(f"[RL] Error loading model: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"[RL] Creating new model instead...")
            else:
                print(f"[RL] No checkpoint found. Creating new model...")
        
        # Create new model with optimized hyperparameters for fast learning and real market performance
        print("[RL] Creating new PPO model with optimized hyperparameters for fast learning...")
        
        # Optimized hyperparameters based on research and trading RL best practices:
        # - Higher learning rate for faster convergence
        # - More frequent updates (smaller n_steps)
        # - Better gradient estimates (larger batch size)
        # - Balanced exploration/exploitation (adaptive entropy)
        
        # Choose algorithm: RecurrentPPO if available and enabled, else PPO
        use_recurrent = os.getenv('USE_RECURRENT_PPO', '1') == '1' and RECUR_PPO_CLS is not None
        algo_cls = RECUR_PPO_CLS if use_recurrent else PPO
        policy_name = 'MlpLstmPolicy' if use_recurrent else 'MlpPolicy'

        self.model = algo_cls(
            policy_name,
            self.env,
            policy_kwargs=policy_kwargs if not use_recurrent else None,
            learning_rate=5e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=8,
            gamma=0.995,
            gae_lambda=0.98,
            clip_range=0.15,
            ent_coef=0.08,
            vf_coef=0.5,
            max_grad_norm=1.0,
            verbose=1,
            device='auto',
            tensorboard_log=str(self.model_dir / "tensorboard")
        )
        
        # Store initial entropy for adaptive decay
        self.initial_ent_coef = 0.08
        self.min_ent_coef = 0.01
        self.entropy_decay_steps = 1_000_000  # Decay over 1M steps
        self.initial_lr = 5e-4
        self.min_lr = 1e-4
        
        # Initialize behavior cloning trainer
        try:
            self.bc_trainer = BehaviorCloningTrainer(
                policy=self.model.policy,
                learning_rate=1e-5,
                bc_weight=0.1,
                negative_weight=0.05
            )
            print("[RL] Behavior cloning trainer initialized")
        except Exception as e:
            print(f"[RL] Warning: Could not initialize BC trainer: {e}")
            self.bc_trainer = None
        
        return self.model
    
    def start_progress_monitor(self):
        """Start progress monitoring in separate thread"""
        if self.progress_monitor is None:
            self.progress_monitor = RLProgressMonitor(
                model_dir=str(self.model_dir),
                model=self.model,
                state_manager=self.state_manager
            )
            self.monitor_thread = threading.Thread(
                target=self.progress_monitor.start,
                daemon=True
            )
            self.monitor_thread.start()
            print("[RL] Progress monitor started")
    
    def start_dashboard(self):
        """Start comprehensive dashboard"""
        if self.dashboard is None:
            self.dashboard = RLDashboard(
                model_dir=str(self.model_dir),
                state_file=str(self.state_manager.state_file),
                update_interval=2.0
            )
            self.dashboard.set_model(self.model)
            self.dashboard.start()
            print("[RL] Dashboard started")
    
    def _get_env_with_simulator(self, env):
        """Unwrap env until we get TradingEnv (has .simulator). Handles both .unwrapped (Gymnasium) and .env (gym.Wrapper)."""
        current = env
        seen = set()
        while id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, 'simulator'):
                return current
            if getattr(current, 'unwrapped', None) is not None and getattr(current, 'unwrapped') is not current:
                current = current.unwrapped
                continue
            if getattr(current, 'env', None) is not None:
                current = current.env
                continue
            break
        return current

    def start_trade_update_thread(self):
        """Start background thread to continuously update trades and check TP/SL"""
        def trade_update_loop():
            """Continuously update all simulators to check for TP/SL hits"""
            import time
            import json
            import os
            log_path = r"c:\Users\Mini Echo09\Desktop\Trading-Bot-V_2.0-feature-enhanced-logging-metrics\.cursor\debug.log"
            loop_count = 0
            while not self.shutdown_event.is_set():
                try:
                    # #region agent log
                    loop_count += 1
                    try:
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_trade_update_loop","timestamp":int(time.time()*1000),"location":"rl_training.py:904","message":"Trade update loop iteration","data":{"loop_count":loop_count,"env_is_none":self.env is None,"shutdown_set":self.shutdown_event.is_set()},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
                    except: pass
                    # #endregion
                    if self.env is None:
                        # #region agent log
                        try:
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_env_none","timestamp":int(time.time()*1000),"location":"rl_training.py:907","message":"Env is None, sleeping","data":{"loop_count":loop_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
                        except: pass
                        # #endregion
                        time.sleep(5.0)
                        continue
                    
                    # VecNormalize wraps VecEnv, need to unwrap
                    vec_env = self.env
                    if hasattr(self.env, 'venv'):
                        vec_env = self.env.venv
                    
                    # Get all environments and their simulators
                    simulators_found = 0
                    simulators_updated = 0
                    try:
                        if hasattr(vec_env, 'envs'):
                            # VecEnv - iterate through all environments
                            for env_idx, env in enumerate(vec_env.envs):
                                # Unwrap Monitor and any wrappers to reach TradingEnv (has .simulator)
                                unwrapped = self._get_env_with_simulator(env)
                                
                                if hasattr(unwrapped, 'simulator'):
                                    simulators_found += 1
                                    # #region agent log
                                    try:
                                        symbol = getattr(unwrapped.simulator, 'symbol', 'UNKNOWN')
                                        open_count_before = len(unwrapped.simulator.open_trades) if hasattr(unwrapped.simulator, 'open_trades') else 0
                                        with open(log_path, 'a') as f:
                                            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_before_step","timestamp":int(time.time()*1000),"location":"rl_training.py:927","message":"Before simulator.step()","data":{"env_idx":env_idx,"symbol":symbol,"open_trades_before":open_count_before,"loop_count":loop_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,D"}) + "\n")
                                    except: pass
                                    # #endregion
                                    try:
                                        # Update simulator to check TP/SL and persist to DB
                                        unwrapped.simulator.step()
                                        # Also run env's stale-trade close (max duration)
                                        if hasattr(unwrapped, '_check_and_close_stale_trades'):
                                            try:
                                                unwrapped._check_and_close_stale_trades()
                                            except Exception:
                                                pass
                                        simulators_updated += 1
                                        # #region agent log
                                        try:
                                            open_count_after = len(unwrapped.simulator.open_trades) if hasattr(unwrapped.simulator, 'open_trades') else 0
                                            with open(log_path, 'a') as f:
                                                f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_after_step","timestamp":int(time.time()*1000),"location":"rl_training.py:927","message":"After simulator.step()","data":{"env_idx":env_idx,"symbol":symbol,"open_trades_before":open_count_before,"open_trades_after":open_count_after,"trades_closed":open_count_before-open_count_after,"loop_count":loop_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B,D"}) + "\n")
                                        except: pass
                                        # #endregion
                                    except Exception as e:
                                        # #region agent log
                                        try:
                                            with open(log_path, 'a') as f:
                                                f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_step_error","timestamp":int(time.time()*1000),"location":"rl_training.py:928","message":"simulator.step() exception","data":{"env_idx":env_idx,"error":str(e),"loop_count":loop_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B"}) + "\n")
                                        except: pass
                                        # #endregion
                                        # Log but don't crash - continue with next simulator
                                        pass
                        elif hasattr(vec_env, 'simulator'):
                            # Single environment
                            simulators_found = 1
                            # #region agent log
                            try:
                                symbol = getattr(vec_env.simulator, 'symbol', 'UNKNOWN')
                                open_count_before = len(vec_env.simulator.open_trades) if hasattr(vec_env.simulator, 'open_trades') else 0
                                with open(log_path, 'a') as f:
                                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_before_step_single","timestamp":int(time.time()*1000),"location":"rl_training.py:933","message":"Before simulator.step() (single env)","data":{"symbol":symbol,"open_trades_before":open_count_before,"loop_count":loop_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,D"}) + "\n")
                            except: pass
                            # #endregion
                            try:
                                vec_env.simulator.step()
                                simulators_updated += 1
                                # #region agent log
                                try:
                                    open_count_after = len(vec_env.simulator.open_trades) if hasattr(vec_env.simulator, 'open_trades') else 0
                                    with open(log_path, 'a') as f:
                                        f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_after_step_single","timestamp":int(time.time()*1000),"location":"rl_training.py:934","message":"After simulator.step() (single env)","data":{"symbol":symbol,"open_trades_before":open_count_before,"open_trades_after":open_count_after,"trades_closed":open_count_before-open_count_after,"loop_count":loop_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B,D"}) + "\n")
                                except: pass
                                # #endregion
                            except Exception as e:
                                # #region agent log
                                try:
                                    with open(log_path, 'a') as f:
                                        f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_step_error_single","timestamp":int(time.time()*1000),"location":"rl_training.py:935","message":"simulator.step() exception (single env)","data":{"error":str(e),"loop_count":loop_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B"}) + "\n")
                                except: pass
                                # #endregion
                                pass
                        else:
                            # No simulators found - log this but continue
                            # #region agent log
                            try:
                                with open(log_path, 'a') as f:
                                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_no_simulators","timestamp":int(time.time()*1000),"location":"rl_training.py:936","message":"No simulators found in environment","data":{"vec_env_type":type(vec_env).__name__,"has_envs":hasattr(vec_env, 'envs'),"has_simulator":hasattr(vec_env, 'simulator'),"loop_count":loop_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,D"}) + "\n")
                            except: pass
                            # #endregion
                    except Exception as e:
                        # #region agent log
                        try:
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_env_access_error","timestamp":int(time.time()*1000),"location":"rl_training.py:937","message":"Error accessing environment/simulators","data":{"error":str(e),"loop_count":loop_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,D"}) + "\n")
                        except: pass
                        # #endregion
                        # Continue loop even if we can't access simulators
                        pass
                    # #region agent log
                    try:
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_loop_summary","timestamp":int(time.time()*1000),"location":"rl_training.py:939","message":"Trade update loop summary","data":{"loop_count":loop_count,"simulators_found":simulators_found,"simulators_updated":simulators_updated},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,D"}) + "\n")
                    except: pass
                    # #endregion
                    # Sleep for 2 seconds between updates (frequent enough to catch TP/SL quickly)
                    time.sleep(2.0)
                    
                except Exception as e:
                    # #region agent log
                    try:
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_loop_exception","timestamp":int(time.time()*1000),"location":"rl_training.py:941","message":"Trade update loop exception","data":{"error":str(e),"loop_count":loop_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
                    except: pass
                    # #endregion
                    # Log error but continue
                    time.sleep(5.0)  # Longer sleep on error
        
        # Start the thread
        self.trade_update_thread = threading.Thread(target=trade_update_loop, daemon=True)
        self.trade_update_thread.start()
        print("[RL] Background trade update thread started (checks TP/SL every 2 seconds)")
    
    def train(
        self,
        total_timesteps: int = 0,
        use_monitor: bool = False,
        use_dashboard: bool = False,
        accuracy_threshold: float = 85.0,  # Legacy parameter (now using multi-metric gate)
        min_trades_for_threshold: int = 500,  # Minimum trades per evaluation window
    ):
        """Train the RL agent with two-phase approach:
        
        Phase 1: Pre-train on historical data (2019-2024) to build robust baseline
        Phase 2: Fine-tune on live paper trading for adaptation to current conditions
        
        If total_timesteps is <= 0, the trainer will run in an open‑ended
        loop and continue improving until the multi-metric gate is passed
        (win rate + profit factor + Sharpe + drawdown across 3 consecutive
        stability windows). This still fully supports pause/resume via the
        TrainingState and Ctrl+C handler.
        """
        # #region agent log - train() entry
        import json
        import time as time_module
        log_path = r"c:\Users\Mini Echo09\Desktop\Trading-Bot-V_2.0-feature-enhanced-logging-metrics\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_train_entry","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1051","message":"train() method entered","data":{"total_timesteps":total_timesteps,"use_historical_pretraining":self.use_historical_pretraining},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW1"}) + "\n")
        except: pass
        # #endregion
        
        # Phase 1: Historical pre-training (if enabled)
        if self.use_historical_pretraining:
            print("\n" + "="*80)
            print("PHASE 1: HISTORICAL PRE-TRAINING")
            print("="*80)
            print(f"Training on historical data: {self.historical_start_date} to {self.historical_end_date}")
            print(f"Target timesteps: {self.historical_pretrain_timesteps:,}")
            print("="*80 + "\n")
            
            # Check if historical pre-training already completed
            state = self.state_manager.get_state()
            historical_completed = state.get('historical_pretraining_completed', False)
            
            if not historical_completed:
                # #region agent log - starting historical phase
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_historical_phase_start","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1077","message":"Starting historical pre-training phase","data":{},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW1"}) + "\n")
                except: pass
                # #endregion
                
                # Run historical training phase - returns True if completed, False if interrupted
                historical_finished = self._train_historical_phase()
                
                # #region agent log - historical phase done
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_historical_phase_done","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1078","message":"Historical pre-training phase completed/interrupted","data":{"historical_finished":historical_finished},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW1"}) + "\n")
                except: pass
                # #endregion
                
                # Check if shutdown was requested during historical training
                if self.shutdown_event.is_set():
                    print("\n[RL] ⚠️ Shutdown requested during historical pre-training.")
                    print("[RL] Exiting. Historical pre-training was NOT completed.")
                    print("[RL] Resume training later to continue historical pre-training.")
                    # Save state but DON'T mark as completed
                    state = self.state_manager.get_state()
                    self.state_manager.save_state(
                        episode=state.get('episode', 0),
                        timesteps=state.get('total_timesteps', 0),
                        checkpoint_path=state.get('last_checkpoint'),
                        best_reward=state.get('best_reward', float('-inf')),
                        total_trades=state.get('total_trades', 0),
                        mode=state.get('mode', 'training'),
                    )
                    return  # Exit - don't proceed to Phase 2
                
                # If historical training was interrupted (didn't finish), exit
                if not historical_finished:
                    print("\n[RL] ⚠️ Historical pre-training was interrupted.")
                    print("[RL] Exiting. Resume training later to continue.")
                    return  # Exit - don't proceed to Phase 2
                
                # Mark historical pre-training as completed (only if it actually finished)
                state = self.state_manager.get_state()
                # Update state with completion flag
                state['historical_pretraining_completed'] = True
                # Save updated state
                self.state_manager.save_state(
                    episode=state.get('episode', 0),
                    timesteps=state.get('total_timesteps', 0),
                    checkpoint_path=state.get('last_checkpoint'),
                    best_reward=state.get('best_reward', float('-inf')),
                    total_trades=state.get('total_trades', 0),
                    mode=state.get('mode', 'training'),
                )
                # Also update the state dict directly for immediate access
                self.state_manager.state['historical_pretraining_completed'] = True
            else:
                print("[RL] Historical pre-training already completed. Skipping to live fine-tuning.")
                # #region agent log - skipping historical
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_skip_historical","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1119","message":"Skipping historical pre-training (already completed)","data":{},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW1"}) + "\n")
                except: pass
                # #endregion
        
        # Phase 2: Live paper trading fine-tuning
        print("\n" + "="*80)
        print("PHASE 2: LIVE PAPER TRADING FINE-TUNING")
        print("="*80)
        print("Fine-tuning on live market data for adaptation to current conditions")
        print("="*80 + "\n")
        
        # #region agent log - Phase 2 entry
        import json
        import time as time_module
        log_path = r"c:\Users\Mini Echo09\Desktop\Trading-Bot-V_2.0-feature-enhanced-logging-metrics\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_phase2_entry","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1121","message":"Entering Phase 2 - about to create environment","data":{"use_historical_pretraining":self.use_historical_pretraining},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW1"}) + "\n")
        except: pass
        # #endregion
        
        # Create environment for live training (no historical replay)
        self.create_environment(historical_replay=None)
        
        # #region agent log - environment created
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_env_created","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1130","message":"Environment created successfully","data":{"env_type":type(self.env).__name__ if self.env else "None"},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW1"}) + "\n")
        except: pass
        # #endregion
        
        # Create or load model (skip checkpoint if fresh_start is True)
        load_checkpoint = not getattr(self, '_fresh_start', False)
        self.create_model(load_checkpoint=load_checkpoint)
        
        # Debug: Print model info
        if self.model:
            print(f"[RL] Model created/loaded. Policy type: {type(self.model.policy)}")
            print(f"[RL] Entropy coefficient: {self.model.ent_coef}")
            print(f"[RL] Action space: {self.env.action_space}")
            print(f"[RL] Observation space shape: {self.env.observation_space.shape}")
        
        # #region agent log - model created
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_model_created","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1135","message":"Model created/loaded successfully","data":{"model_type":type(self.model).__name__ if self.model else "None"},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW1"}) + "\n")
        except: pass
        # #endregion
        
        # Note: Progress monitor and dashboard are now optional so that
        # rl_training.py can run with simple, non-glitchy terminal output
        # by default. They can be enabled via CLI flags.

        # #region agent log - about to start trade update thread
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_starting_trade_thread","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1147","message":"About to start trade update thread","data":{},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW1"}) + "\n")
        except: pass
        # #endregion
        
        # Start background trade update thread (CRITICAL: ensures trades close on TP/SL)
        self.start_trade_update_thread()
        
        # #region agent log - trade update thread started
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_trade_thread_started","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1148","message":"Trade update thread started successfully","data":{"thread_alive":self.trade_update_thread.is_alive() if hasattr(self, 'trade_update_thread') and self.trade_update_thread else False},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW1"}) + "\n")
        except: pass
        # #endregion
        
        # Start optional progress monitor
        if use_monitor:
            self.start_progress_monitor()
        
        # Start optional comprehensive dashboard
        if use_dashboard:
            self.start_dashboard()
        
        # If starting fresh, reset persisted training state to zero so we don't skip work
        if getattr(self, "_fresh_start", False):
            self.state_manager.save_state(
                episode=0,
                timesteps=0,
                checkpoint_path=None,
                best_reward=float("-inf"),
                total_trades=0,
                mode="training",
            )
        
        # Load state
        state = self.state_manager.get_state()
        current_timesteps = state['total_timesteps']
        infinite_mode = total_timesteps <= 0
        remaining_timesteps = max(0, total_timesteps - current_timesteps) if not infinite_mode else -1
        current_mode = state.get("mode", "training")
        
        # Update mode tracking for rollback checks
        self.current_mode = current_mode
        
        print(f"\n[RL] Starting training...")
        if infinite_mode:
            print(f"[RL] Mode: open‑ended (until accuracy threshold reached)")
        else:
            print(f"[RL] Total timesteps: {total_timesteps}")
            print(f"[RL] Already trained: {current_timesteps}")
            print(f"[RL] Remaining: {remaining_timesteps}")
        print(f"[RL] Max trades limit: {self.max_trades if self.max_trades else 'Unlimited'}")
        print(f"[RL] Press Ctrl+C to pause/resume training")
        print(f"[RL] Press 'q' to save state and exit gracefully\n")
        
        
        callbacks = [
            PauseResumeCallback(self.state_manager, self.pause_event),
            CheckpointCallback(
                save_freq=self.checkpoint_interval,
                save_path=str(self.model_dir / "checkpoints"),
                name_prefix="rl_model",
                save_replay_buffer=True,
                save_vecnormalize=True
            ),
            ExperienceReplayCallback(
                experience_buffer=self.experience_buffer,
                bc_trainer=self.bc_trainer,
            )
        ]
        
        episode = 0
        best_reward = state.get('best_reward', float('-inf'))
        
        # CRITICAL FIX: Save initial state BEFORE training starts
        # This ensures we can detect if training loop crashes immediately
        print(f"[RL] Saving initial state (before training starts)...")
        try:
            self.state_manager.save_state(
                episode=episode,
                timesteps=current_timesteps,
                checkpoint_path=state.get('last_checkpoint'),
                best_reward=best_reward,
                total_trades=state.get('total_trades', 0),
                mode=self.current_mode,
            )
            print(f"[RL] ✅ Initial state saved (episode={episode}, timesteps={current_timesteps:,})")
        except Exception as e:
            print(f"[RL] ⚠️ Warning: Could not save initial state: {e}")
        
        while True:
            try:
                # Train in chunks to allow for pause/resume. In open‑ended mode
                # we simply keep stepping forward in fixed chunks until the
                # accuracy threshold logic stops us.
                default_chunk = 50_000
                chunk_size = default_chunk if infinite_mode else min(remaining_timesteps, default_chunk)
            
                # Track last state save time for heartbeat
                import time
                last_state_save = time.time()
                state_save_interval = 300  # Save state every 5 minutes as heartbeat
            
                # #region agent log - entering training loop
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_training_loop_entry","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1290","message":"Entering main training loop","data":{"infinite_mode":infinite_mode,"remaining_timesteps":remaining_timesteps,"chunk_size":chunk_size},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW2"}) + "\n")
                except: pass
                # #endregion
            
                while infinite_mode or remaining_timesteps > 0:
                    # Check for graceful shutdown
                    if self.shutdown_event.is_set():
                        print("\n[RL] Graceful shutdown requested. Saving state...")
                        break
                
                    # Check if paused
                    if self.pause_event.is_set():
                        print("[RL] Training paused. Waiting... (Press 'q' to save and exit)")
                        while self.pause_event.is_set():
                            if self.shutdown_event.is_set():
                                print("\n[RL] Graceful shutdown requested during pause. Saving state...")
                                break
                            time.sleep(0.5)
                        if self.shutdown_event.is_set():
                            break
                        print("[RL] Training resumed")
                
                    # Adaptive entropy and learning rate decay: reduce exploration/learning rate as training progresses
                    total_timesteps_before = current_timesteps
                    progress = min(1.0, total_timesteps_before / self.entropy_decay_steps)
                
                    # Adaptive entropy: decay from 0.08 to 0.01 over 1M steps
                    adaptive_ent = self.initial_ent_coef * (1.0 - progress * 0.875) + self.min_ent_coef
                    self.model.ent_coef = max(self.min_ent_coef, adaptive_ent)
                
                    # Adaptive learning rate: decay from 5e-4 to 1e-4 over 1M steps
                    adaptive_lr = self.initial_lr * (1.0 - progress * 0.8) + self.min_lr
                    self.model.learning_rate = max(self.min_lr, adaptive_lr)
                
                    # Train for a chunk
                    try:
                        print(f"[RL] Starting training chunk: {chunk_size} timesteps (Episode {episode}, Total: {current_timesteps:,})")
                    
                        # #region agent log - before model.learn()
                        try:
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_before_learn","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1322","message":"About to call model.learn()","data":{"episode":episode,"chunk_size":chunk_size,"current_timesteps":current_timesteps},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW2"}) + "\n")
                        except: pass
                        # #endregion
                    
                        # CRITICAL FIX: Set timeout/checkpoints during model.learn() to prevent infinite hangs
                        # If model.learn() hangs, we need to detect it
                        chunk_start_time = time.time()
                    
                        self.model.learn(
                            total_timesteps=chunk_size,
                            callback=callbacks,
                            reset_num_timesteps=False,
                            tb_log_name="PPO"
                        )
                    
                        # #region agent log - after model.learn()
                        try:
                            chunk_duration = time.time() - chunk_start_time
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"id":f"log_{int(time_module.time()*1000)}_after_learn","timestamp":int(time_module.time()*1000),"location":"rl_training.py:1333","message":"model.learn() completed","data":{"episode":episode,"chunk_duration_sec":chunk_duration},"sessionId":"debug-session","runId":"run1","hypothesisId":"NEW2"}) + "\n")
                        except: pass
                        # #endregion
                    
                        chunk_duration = time.time() - chunk_start_time
                        print(f"[RL] Training chunk took {chunk_duration:.1f} seconds ({chunk_duration/60:.1f} minutes)")
                    
                        # Log adaptive parameters
                        if episode % 10 == 0:
                            print(f"[RL] Adaptive parameters: LR={self.model.learning_rate:.6f}, Entropy={self.model.ent_coef:.4f}, Progress={progress*100:.1f}%")
                    
                        # Update state AFTER successful training
                        current_timesteps += chunk_size
                        print(f"[RL] ✅ Training chunk completed. New total: {current_timesteps:,} timesteps")
                    
                        # Update heartbeat timestamp
                        last_state_save = time.time()
                    except Exception as e:
                        print(f"[RL] ❌ ERROR during training: {e}")
                        import traceback
                        traceback.print_exc()
                        # Save state even on error so we can diagnose
                        self.state_manager.save_state(
                            episode=episode,
                            timesteps=current_timesteps,
                            checkpoint_path=str(self.model_dir / "checkpoints" / f"rl_model_{current_timesteps}_steps.zip"),
                            best_reward=best_reward,
                            total_trades=state.get('total_trades', 0),
                            mode=self.current_mode,
                        )
                        # Re-raise to stop training (user can fix issue and resume)
                        raise
                    if not infinite_mode:
                        remaining_timesteps -= chunk_size
                    episode += 1
                
                    # Get latest episode stats
                    if hasattr(self.env, 'get_attr'):
                        try:
                            info = self.env.get_attr('total_trades_executed', [0])[0] if hasattr(self.env, 'get_attr') else 0
                            total_trades = info if isinstance(info, (int, float)) else state.get('total_trades', 0)
                        except:
                            total_trades = state.get('total_trades', 0)
                    else:
                        total_trades = state.get('total_trades', 0)
                
                    # Save state (CRITICAL: Save after each chunk to prevent data loss on crash)
                    checkpoint_path = str(self.model_dir / "checkpoints" / f"rl_model_{current_timesteps}_steps.zip")
                    if not os.path.exists(checkpoint_path):
                        # Use latest checkpoint
                        checkpoints_dir = self.model_dir / "checkpoints"
                        if checkpoints_dir.exists():
                            checkpoints = sorted(checkpoints_dir.glob("rl_model_*.zip"))
                            if checkpoints:
                                checkpoint_path = str(checkpoints[-1])
                
                    # Persist state with the current mode (training or paper) so dashboards/status
                    # reflect what the loop is actually doing.
                    try:
                        self.state_manager.save_state(
                            episode=episode,
                            timesteps=current_timesteps,
                            checkpoint_path=checkpoint_path,
                            best_reward=best_reward,
                            total_trades=total_trades,
                            mode=self.current_mode,
                        )
                        print(f"[RL] 💾 State saved: Episode {episode}, Timesteps {current_timesteps:,}, Trades {total_trades}")
                        last_state_save = time.time()  # Update heartbeat
                    except Exception as e:
                        print(f"[RL] ⚠️ Warning: Failed to save state: {e}")
                
                    # HEARTBEAT: Periodically save state even if no training chunk completed
                    # This helps detect if training loop is alive but stuck
                    if time.time() - last_state_save > state_save_interval:
                        try:
                            print(f"[RL] 💓 Heartbeat: Saving state (no training chunk in {state_save_interval//60} minutes)")
                            self.state_manager.save_state(
                                episode=episode,
                                timesteps=current_timesteps,
                                checkpoint_path=checkpoint_path,
                                best_reward=best_reward,
                                total_trades=total_trades,
                                mode=self.current_mode,
                            )
                            last_state_save = time.time()
                        except Exception as e:
                            print(f"[RL] ⚠️ Warning: Failed heartbeat save: {e}")
                
                    # Update progress monitor
                    if self.progress_monitor:
                        self.progress_monitor.update_stats({
                            'episode': episode,
                            'timesteps': current_timesteps,
                            'total_trades': total_trades
                        })
                
                    # Update dashboard model and verify connection
                    if self.dashboard:
                        self.dashboard.update_model(self.model)
                        # Verify dashboard is running
                        if not self.dashboard.running:
                            print("[RL] ⚠️ Dashboard is not running! Restarting...")
                            self.dashboard.start()
                        elif self.dashboard.update_thread and not self.dashboard.update_thread.is_alive():
                            print("[RL] ⚠️ Dashboard thread died! Restarting...")
                            self.dashboard.start()
                
                    # Get episode rewards and update best_reward
                    try:
                        if hasattr(self.env, 'get_episode_rewards'):
                            episode_rewards = self.env.get_episode_rewards()
                            if episode_rewards:
                                latest_reward = episode_rewards[-1]
                                if latest_reward > best_reward:
                                    best_reward = latest_reward
                                    print(f"[RL] New best reward: {best_reward:.4f}")
                    except Exception as e:
                        print(f"[RL] Warning: Could not get episode rewards: {e}")

                    if infinite_mode:
                        print(f"\n[RL] Progress: {current_timesteps:,} timesteps - Trades: {total_trades}")
                    else:
                        print(f"\n[RL] Progress: {current_timesteps}/{total_timesteps} timesteps "
                              f"({current_timesteps/total_timesteps*100:.1f}%) - Trades: {total_trades}")

                    # Multi-metric evaluation with stability windows (skip early to reduce overfitting)
                    try:
                        if current_timesteps >= self.min_timesteps_before_eval:
                            # Get all closed trades from database (exclude historical training trades)
                            trades = self.analytics.get_closed_trades(symbol=None, days=None, exclude_historical_training=True)
                        
                            # Debug: Print trade count for troubleshooting
                            if len(trades) == 0:
                                print(f"[RL] Warning: No closed trades found in database (path: {self.analytics.db_path})")
                                print(f"[RL] This may be normal early in training. Trades will appear as they are closed.")
                            else:
                                print(f"[RL] Found {len(trades)} closed trades in database for evaluation")
                        
                            if trades and len(trades) >= min_trades_for_threshold:
                                # Check stability using multi-metric gate
                                stable, summary = self.metrics_evaluator.check_stability(trades)
                            
                                current_metrics = summary.get('current_window', {})
                                # When check_stability returns early (e.g. insufficient_trades), there is no current_window;
                                # show actual metrics from available trades so they match the dashboard (e.g. 48% win rate)
                                if not current_metrics and trades:
                                    current_metrics = self.analytics.calculate_metrics(trades)
                                    current_metrics['win_rate_ok'] = current_metrics.get('win_rate', 0) >= self.metrics_evaluator.win_rate_threshold
                                    current_metrics['profit_factor_ok'] = current_metrics.get('profit_factor', 0) >= self.metrics_evaluator.profit_factor_threshold
                                    current_metrics['sharpe_ok'] = False  # not computed in simple path
                                    current_metrics['drawdown_ok'] = current_metrics.get('max_drawdown_pct', 100) <= self.metrics_evaluator.max_drawdown_threshold
                                    summary = dict(summary, current_window=current_metrics, windows_passed=0, windows_evaluated=0)
                                win_rate = current_metrics.get('win_rate', 0.0)
                                profit_factor = current_metrics.get('profit_factor', 0.0)
                                sharpe = current_metrics.get('sharpe_ratio', 0.0)
                                drawdown = current_metrics.get('max_drawdown_pct', 100.0)
                            
                                print(f"[RL] Multi-metric evaluation:")
                                if summary.get('reason') == 'insufficient_trades':
                                    print(f"[RL]   (Stability not evaluated: {summary.get('total_trades', 0)} trades, need {summary.get('required', 0)} for stability windows)")
                                print(f"  Win Rate: {win_rate:.2f}% (target: ≥{self.metrics_evaluator.win_rate_threshold}%) {'✓' if current_metrics.get('win_rate_ok') else '✗'}")
                                print(f"  Profit Factor: {profit_factor:.2f} (target: ≥{self.metrics_evaluator.profit_factor_threshold}) {'✓' if current_metrics.get('profit_factor_ok') else '✗'}")
                                print(f"  Sharpe Ratio: {sharpe:.2f} (target: ≥{self.metrics_evaluator.sharpe_threshold}) {'✓' if current_metrics.get('sharpe_ok') else '✗'}")
                                print(f"  Max Drawdown: {drawdown:.2f}% (target: ≤{self.metrics_evaluator.max_drawdown_threshold}%) {'✓' if current_metrics.get('drawdown_ok') else '✗'}")
                                print(f"  Stability: {summary.get('windows_passed', 0)}/{summary.get('windows_evaluated', 0)} windows passed (need {self.metrics_evaluator.stability_windows_required} consecutive)")

                                # If stable across multiple windows, save model and run paper test
                                if stable:
                                    # Reset failure counter on success
                                    self.eval_failures_since_tighten = 0
                                    best_model_path = self.model_dir / f"best_model_{win_rate:.2f}wr_{profit_factor:.2f}pf.zip"
                                    self.model.save(str(best_model_path))
                                    self.last_safe_model_path = str(best_model_path)  # Track for rollback
                                    print(f"\n[RL] ✅ Multi-metric gate PASSED with stability!")
                                    print(f"[RL] Best model saved to {best_model_path}")
                                    print(f"[RL] Metrics: WR={win_rate:.2f}%, PF={profit_factor:.2f}, Sharpe={sharpe:.2f}, DD={drawdown:.2f}%")

                                    # Switch to paper testing mode (but keep training running if configured)
                                    if self.keep_training_during_paper:
                                        print(f"[RL] Switching to paper testing mode (training continues in background for rollback)")
                                        self.state_manager.save_state(
                                            episode=episode,
                                            timesteps=current_timesteps,
                                            checkpoint_path=checkpoint_path,
                                            best_reward=best_reward,
                                            total_trades=total_trades,
                                            mode="paper",
                                        )
                                        self.current_mode = "paper"  # Update mode tracking
                                        self._send_accuracy_reached_notification(
                                            summary, current_timesteps, total_trades, to_email="noamiko613@gmail.com"
                                        )
                                        # Continue training loop but monitor for rollback
                                        # Paper testing happens in parallel via dashboard/monitor
                                    else:
                                        # Run paper-testing mode with the best model (legacy behavior)
                                        self._run_paper_test(str(best_model_path), test_episodes=getattr(self, "paper_test_episodes", 5))
                                        print("[RL] Paper testing completed. Stopping training loop.")
                                        return
                                else:
                                    # Count failure and maybe tighten thresholds
                                    self._record_eval_failure()
                            
                                # Auto-rollback check: if metrics degrade significantly during paper testing
                                if self.current_mode == "paper" and self.auto_rollback_enabled:
                                    if drawdown > self.rollback_threshold_drawdown or sharpe < self.rollback_threshold_sharpe:
                                        print(f"\n[RL] ⚠️  AUTO-ROLLBACK TRIGGERED!")
                                        print(f"[RL] Metrics degraded: DD={drawdown:.2f}% (threshold: {self.rollback_threshold_drawdown}%), "
                                              f"Sharpe={sharpe:.2f} (threshold: {self.rollback_threshold_sharpe})")
                                        if self.last_safe_model_path and os.path.exists(self.last_safe_model_path):
                                            print(f"[RL] Rolling back to last safe model: {self.last_safe_model_path}")
                                            try:
                                                self.model = PPO.load(self.last_safe_model_path, env=self.env, device='auto')
                                                self.state_manager.save_state(
                                                    episode=episode,
                                                    timesteps=current_timesteps,
                                                    checkpoint_path=self.last_safe_model_path,
                                                    best_reward=best_reward,
                                                    total_trades=total_trades,
                                                    mode="training",  # Back to training mode
                                                )
                                                self.current_mode = "training"  # Update mode tracking
                                                print(f"[RL] Rollback successful. Resuming training.")
                                            except Exception as e:
                                                print(f"[RL] Rollback failed: {e}")
                                        else:
                                            print(f"[RL] No safe model found for rollback. Continuing with current model.")
                                        # Auto-tighten if configured
                                        self._auto_tighten_thresholds(reason="rollback")
                        else:
                            # Too early to evaluate; defer to avoid overfitting/false positives
                            if current_timesteps % 50_000 == 0:
                                print(f"[RL] Skipping evaluation until {self.min_timesteps_before_eval} timesteps (current: {current_timesteps})")
                    except Exception as e:
                        print(f"[RL] Error while evaluating metrics: {e}")
                        import traceback
                        traceback.print_exc()
        
            except KeyboardInterrupt:
                print("\n[RL] Training interrupted. Saving final checkpoint...")
                # Get current state before saving
                try:
                    if hasattr(self.env, 'get_attr'):
                        info = self.env.get_attr('total_trades_executed', [0])[0] if hasattr(self.env, 'get_attr') else 0
                        total_trades = info if isinstance(info, (int, float)) else state.get('total_trades', 0)
                    else:
                        total_trades = state.get('total_trades', 0)
                except Exception:
                    total_trades = state.get('total_trades', 0)
                self._save_final_state(episode, current_timesteps, best_reward, total_trades)
                break
        
            # Handle graceful shutdown
            if self.shutdown_event.is_set():
                print("\n[RL] Performing graceful shutdown...")
                try:
                    if hasattr(self.env, 'get_attr'):
                        info = self.env.get_attr('total_trades_executed', [0])[0] if hasattr(self.env, 'get_attr') else 0
                        total_trades = info if isinstance(info, (int, float)) else state.get('total_trades', 0)
                    else:
                        total_trades = state.get('total_trades', 0)
                except Exception:
                    total_trades = state.get('total_trades', 0)
                self._save_final_state(episode, current_timesteps, best_reward, total_trades)
                break
            
            # When fixed timesteps just completed (no shutdown): check accuracy
            if not self.shutdown_event.is_set() and not infinite_mode and remaining_timesteps <= 0:
                try:
                    if hasattr(self.env, 'get_attr'):
                        info = self.env.get_attr('total_trades_executed', [0])[0] if hasattr(self.env, 'get_attr') else 0
                        total_trades = info if isinstance(info, (int, float)) else state.get('total_trades', 0)
                    else:
                        total_trades = state.get('total_trades', 0)
                except Exception:
                    total_trades = state.get('total_trades', 0)
                trades = self.analytics.get_closed_trades(symbol=None, days=None, exclude_historical_training=True)
                if len(trades) >= min_trades_for_threshold:
                    stable, summary = self.metrics_evaluator.check_stability(trades)
                    if stable:
                        self.current_mode = "paper"
                        checkpoint_path = str(self.model_dir / "checkpoints" / f"rl_model_{current_timesteps}_steps.zip")
                        if not os.path.exists(checkpoint_path):
                            checkpoints_dir = self.model_dir / "checkpoints"
                            if checkpoints_dir.exists():
                                cps = sorted(checkpoints_dir.glob("rl_model_*.zip"))
                                if cps:
                                    checkpoint_path = str(cps[-1])
                        self.state_manager.save_state(
                            episode=episode,
                            timesteps=current_timesteps,
                            checkpoint_path=checkpoint_path,
                            best_reward=best_reward,
                            total_trades=total_trades,
                            mode="paper",
                        )
                        self._save_final_state(episode, current_timesteps, best_reward, total_trades)
                        self._send_accuracy_reached_notification(summary, current_timesteps, total_trades, to_email="noamiko613@gmail.com")
                        print("\n[RL] Accuracy threshold reached. Switched to paper trading on live data. Notification sent.")
                        return
                print("\n[RL] Accuracy threshold not yet reached. Continuing training on live data (paper mode) until threshold is met...")
                infinite_mode = True
                remaining_timesteps = 50_000
                continue
            break
        
        print("\n[RL] Training completed!")
    
    def _find_oldest_available_date(self, test_symbol: str) -> Optional[str]:
        """Find the oldest available historical data date.
        
        User confirmed CoinEx has data from 2019-12, so we start there.
        Will not go earlier than 2017-01-01 (minimum limit).
        """
        from datetime import datetime, timezone
        from verify_patterns import normalize_symbol_to_ccxt
        from coinEx_getting_data import CoinExDataFetcher
        
        print(f"[RL] Finding oldest available data for {test_symbol}...")
        print(f"[RL] Starting from 2019-12-01 (CoinEx confirmed start date)")
        print(f"[RL] Minimum date limit: 2017-01-01 (will not search earlier)")
        
        symbol_ccxt = normalize_symbol_to_ccxt(test_symbol)
        
        # Start from 2019-12-01 (user confirmed CoinEx has data from 2019-12)
        # Fetch full range to find actual oldest date in the data
        try:
            fetcher = CoinExDataFetcher(
                symbol=symbol_ccxt,
                timeframe_internal="1h",
                max_candles=100000,  # Large enough for full historical range
                mode='spot',
            )
            
            print(f"[RL] Fetching full historical range from 2019-12-01 to {self.historical_end_date}...")
            print(f"[RL] This may take a few minutes...")
            
            candles = fetcher.fetch_historical_range("2019-12-01", self.historical_end_date)
            
            if candles and len(candles) > 10:
                import pandas as pd
                df = pd.DataFrame(candles)
                if not df.empty and 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
                    first_date = df['timestamp'].min()
                    last_date = df['timestamp'].max()
                    first_date_str = first_date.strftime('%Y-%m-%d')
                    last_date_str = last_date.strftime('%Y-%m-%d')
                    
                    print(f"[RL] ✅ Found {len(candles)} candles")
                    print(f"[RL] Date range: {first_date_str} to {last_date_str}")
                    
                    # Verify it's actually old data (not recent)
                    now = datetime.now(timezone.utc)  # Make timezone-aware
                    if isinstance(first_date, pd.Timestamp):
                        first_dt = first_date.to_pydatetime()
                        # Ensure timezone-aware
                        if first_dt.tzinfo is None:
                            first_dt = first_dt.replace(tzinfo=timezone.utc)
                    elif isinstance(first_date, datetime):
                        first_dt = first_date
                        if first_dt.tzinfo is None:
                            first_dt = first_dt.replace(tzinfo=timezone.utc)
                    else:
                        first_dt = pd.Timestamp(first_date).to_pydatetime()
                        if first_dt.tzinfo is None:
                            first_dt = first_dt.replace(tzinfo=timezone.utc)
                    days_ago = (now - first_dt).days
                    
                    if days_ago > 365:  # At least 1 year old
                        print(f"[RL] ✅ Verified: Data is {days_ago} days old (historical data)")
                        return first_date_str
                    else:
                        print(f"[RL] ⚠️ Warning: Data appears recent ({days_ago} days old), may not be historical")
                        # Still return it, but warn
                        return first_date_str
            else:
                print(f"[RL] ⚠️ No data found from 2019-12-01")
        except Exception as e:
            print(f"[RL] Error fetching from 2019-12-01: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback: return 2019-12-01 (user confirmed it exists)
        print(f"[RL] Using 2019-12-01 as start date (CoinEx confirmed start)")
        return "2019-12-01"
    
    def _train_historical_phase(self) -> bool:
        """
        Phase 1: Pre-train on historical data.
        
        Returns:
            bool: True if historical training completed successfully, False if interrupted
        """
        try:
            # Determine symbols to train on
            symbols = [self.symbol] if self.symbol else RL_TRAINING_SYMBOLS
            
            print(f"[RL] Historical pre-training on {len(symbols)} symbols: {', '.join(symbols)}")
            
            # Prioritize timeframes: shorter timeframes = more samples
            preferred_timeframes = ["15m", "1h", "4h", "6h", "12h", "1d"]
            print(f"[RL] Using timeframes (prioritizing shorter for more samples): {preferred_timeframes}")
            
            # Use a COMMON date range across ALL symbols so we don't get "some pairs 2025, others other times"
            print(f"[RL] Computing common cache date range for all symbols...")
            common_start, common_end, per_symbol_info = get_common_cache_date_range(
                symbols, preferred_timeframes, self.historical_start_date, self.historical_end_date
            )
            if common_start and common_end:
                effective_start_date = common_start
                effective_end_date = common_end
                print(f"[RL] ✅ Common date range (all symbols): {effective_start_date} to {effective_end_date}")
                for sym, info in per_symbol_info.items():
                    if info.get("ok"):
                        print(f"[RL]   {sym}: {info.get('start')} to {info.get('end')} ({info.get('candles', 0)} candles)")
                    else:
                        print(f"[RL]   {sym}: ⚠️ No cache data - will be skipped")
                # Only train on symbols that have data in the common range
                symbols_to_load = [s for s in symbols if per_symbol_info.get(s, {}).get("ok")]
                if not symbols_to_load:
                    print(f"[RL] ⚠️ No symbol has cache in common range; falling back to single-symbol oldest date")
                    oldest_available_date = self._find_oldest_available_date(symbols[0])
                    effective_start_date = oldest_available_date or self.historical_start_date
                    effective_end_date = self.historical_end_date
                    symbols_to_load = symbols
                else:
                    # Warn if common range is short (inconsistent caches e.g. some pairs 2025-only)
                    try:
                        ds = datetime.fromisoformat(effective_start_date.replace('Z', '+00:00'))
                        de = datetime.fromisoformat(effective_end_date.replace('Z', '+00:00'))
                        days = (de - ds).days
                        if days < 365:
                            print(f"[RL] ⚠️ Common range is only {days} days. For best results, re-download data for all symbols with the same range:")
                            print(f"[RL]   python run_rl_training.py  (without --skip-download) or use --force-download")
                    except Exception:
                        pass
            else:
                print(f"[RL] No common range in cache; using oldest available from first symbol")
                oldest_available_date = self._find_oldest_available_date(symbols[0])
                effective_start_date = oldest_available_date or self.historical_start_date
                effective_end_date = self.historical_end_date
                symbols_to_load = symbols
            
            # For multi-symbol training, create a replay for each symbol with the SAME date range
            historical_replays = {}
            for symbol in symbols_to_load:
                print(f"\n[RL] Loading historical data for {symbol}...")
                try:
                    replay = HistoricalDataReplay(
                        symbol=symbol,
                        timeframes=preferred_timeframes,
                        start_date=effective_start_date,
                        end_date=effective_end_date,
                        mode='spot',
                        lookback_window=100,
                    )
                    historical_replays[symbol] = replay
                    print(f"[RL] ✅ {symbol}: Loaded {sum(len(df) for df in replay.historical_data.values())} total candles")
                except Exception as e:
                    print(f"[RL] ⚠️  {symbol}: Failed to load historical data: {e}")
                    continue
            
            if not historical_replays:
                raise ValueError("No historical data loaded for any symbol!")
            
            print(f"\n[RL] Successfully loaded historical data for {len(historical_replays)} symbols")
            
            # Store replay map for multi-symbol training
            self._historical_replay_map = historical_replays
            
            # Use the symbol with the most data for primary replay
            # (Other symbols' replays will be used via the map in create_environment)
            best_symbol = max(historical_replays.keys(), 
                            key=lambda s: sum(len(df) for df in historical_replays[s].historical_data.values()))
            self.historical_replay = historical_replays[best_symbol]
            
            # Show data summary
            print(f"\n[RL] Historical Data Summary:")
            for symbol, replay in historical_replays.items():
                total_candles = sum(len(df) for df in replay.historical_data.values())
                primary_tf = replay.primary_timeframe
                primary_candles = len(replay.historical_data.get(primary_tf, pd.DataFrame()))
                print(f"  {symbol}: {total_candles} total candles ({primary_candles} in {primary_tf})")
            
            print(f"[RL] Primary replay: {best_symbol} (most data)")
            print(f"[RL] Multi-symbol training: {len(historical_replays)} symbols will be used")
            
            # Check data coverage and warn if insufficient
            primary_df = self.historical_replay.historical_data.get(self.historical_replay.primary_timeframe, pd.DataFrame())
            if not primary_df.empty:
                first_date = primary_df['timestamp'].iloc[0]
                last_date = primary_df['timestamp'].iloc[-1]
                from datetime import datetime, timezone
                start_dt = datetime.fromisoformat(effective_start_date.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(effective_end_date.replace('Z', '+00:00'))
                
                # Ensure timezone-aware for date calculations
                if isinstance(first_date, pd.Timestamp):
                    actual_start = first_date.to_pydatetime()
                    if actual_start.tzinfo is None:
                        actual_start = actual_start.replace(tzinfo=timezone.utc)
                elif isinstance(first_date, datetime):
                    actual_start = first_date
                    if actual_start.tzinfo is None:
                        actual_start = actual_start.replace(tzinfo=timezone.utc)
                else:
                    actual_start = pd.Timestamp(first_date).to_pydatetime()
                    if actual_start.tzinfo is None:
                        actual_start = actual_start.replace(tzinfo=timezone.utc)
                
                if isinstance(last_date, pd.Timestamp):
                    actual_end = last_date.to_pydatetime()
                    if actual_end.tzinfo is None:
                        actual_end = actual_end.replace(tzinfo=timezone.utc)
                elif isinstance(last_date, datetime):
                    actual_end = last_date
                    if actual_end.tzinfo is None:
                        actual_end = actual_end.replace(tzinfo=timezone.utc)
                else:
                    actual_end = pd.Timestamp(last_date).to_pydatetime()
                    if actual_end.tzinfo is None:
                        actual_end = actual_end.replace(tzinfo=timezone.utc)
                
                expected_days = (end_dt - start_dt).days
                actual_days = (actual_end - actual_start).days
                coverage = (actual_days / expected_days * 100) if expected_days > 0 else 0
                
                print(f"\n[RL] Historical Data Coverage Summary:")
                print(f"  Range: {effective_start_date} to {effective_end_date} ({expected_days} days)")
                print(f"  Available: {actual_start.strftime('%Y-%m-%d %H:%M:%S')} to {actual_end.strftime('%Y-%m-%d %H:%M:%S')} ({actual_days} days)")
                print(f"  Coverage: {coverage:.1f}%")
                
                if coverage < 50:
                    print(f"\n[RL] ⚠️  WARNING: Low data coverage ({coverage:.1f}%)")
                    print(f"[RL] CoinEx may not have historical data for the requested date range.")
                    print(f"[RL] Consider:")
                    print(f"[RL]   - Using a more recent start date (e.g., 2021-01-01)")
                    print(f"[RL]   - Or using a different data source for older historical data")
                    print(f"[RL] Training will continue with available data ({actual_days} days)")
                elif coverage < 80:
                    print(f"\n[RL] ⚠️  Note: Partial data coverage ({coverage:.1f}%)")
                    print(f"[RL] Some historical periods may be missing, but training will proceed.")
                else:
                    print(f"\n[RL] ✅ Good data coverage ({coverage:.1f}%)")
            
            # Create environment with historical replay
            self.create_environment(historical_replay=self.historical_replay)
            
            # Create or load model
            load_checkpoint = not getattr(self, '_fresh_start', False)
            self.create_model(load_checkpoint=load_checkpoint)
            
            # Start background trade update thread
            self.start_trade_update_thread()
            
            # Training loop for historical data
            state = self.state_manager.get_state()
            current_timesteps = state.get('total_timesteps', 0)
            target_timesteps = self.historical_pretrain_timesteps
            remaining = max(0, target_timesteps - current_timesteps)
            
            print(f"[RL] Historical pre-training: {current_timesteps:,}/{target_timesteps:,} timesteps")
            
            callbacks = [
                PauseResumeCallback(self.state_manager, self.pause_event),
                CheckpointCallback(
                    save_freq=self.checkpoint_interval,
                    save_path=str(self.model_dir / "checkpoints"),
                    name_prefix="rl_model_historical",
                    save_replay_buffer=True,
                    save_vecnormalize=True
                ),
                ExperienceReplayCallback(
                    experience_buffer=self.experience_buffer,
                    bc_trainer=self.bc_trainer,
                )
            ]
            
            episode = 0
            chunk_size = 50000
            
            while remaining > 0:
                # Check for graceful shutdown
                if self.shutdown_event.is_set():
                    print("\n[RL] Graceful shutdown requested during historical pre-training.")
                    # Save final state before exiting
                    state = self.state_manager.get_state()
                    # Find actual latest checkpoint
                    checkpoints_dir = self.model_dir / "checkpoints"
                    checkpoint_path = None
                    if checkpoints_dir.exists():
                        checkpoints = sorted(checkpoints_dir.glob("rl_model_historical_*.zip"))
                        if checkpoints:
                            checkpoint_path = str(checkpoints[-1])
                    if not checkpoint_path:
                        checkpoint_path = str(self.model_dir / "checkpoints" / f"rl_model_historical_{current_timesteps}_steps.zip")
                    self.state_manager.save_state(
                        episode=episode,
                        timesteps=current_timesteps,
                        checkpoint_path=checkpoint_path,
                        best_reward=state.get('best_reward', float('-inf')),
                        total_trades=state.get('total_trades', 0),
                        mode="training",
                    )
                    print(f"[RL] State saved: Episode {episode}, Timesteps {current_timesteps:,}/{target_timesteps:,}")
                    return False  # Return False to indicate interruption
                
                # Check if paused
                if self.pause_event.is_set():
                    while self.pause_event.is_set():
                        if self.shutdown_event.is_set():
                            break
                        time.sleep(0.5)
                    if self.shutdown_event.is_set():
                        break
                
                # Train for a chunk
                train_chunk = min(chunk_size, remaining)
                self.model.learn(
                    total_timesteps=train_chunk,
                    callback=callbacks,
                    reset_num_timesteps=False,
                    tb_log_name="PPO_Historical"
                )
                
                current_timesteps += train_chunk
                remaining -= train_chunk
                episode += 1
                
                # Save state - find ACTUAL latest checkpoint (CheckpointCallback may have saved at different timestep)
                checkpoints_dir = self.model_dir / "checkpoints"
                checkpoint_path = None
                
                # Always find the actual latest checkpoint file (don't assume path based on timesteps)
                if checkpoints_dir.exists():
                    checkpoints = sorted(checkpoints_dir.glob("rl_model_historical_*.zip"))
                    if checkpoints:
                        checkpoint_path = str(checkpoints[-1])
                        # Extract actual timesteps from checkpoint filename to verify
                        import re
                        match = re.search(r'_(\d+)_steps\.zip$', checkpoint_path)
                        if match:
                            checkpoint_timesteps = int(match.group(1))
                            if checkpoint_timesteps != current_timesteps:
                                print(f"[RL] ⚠️  Note: Latest checkpoint is at {checkpoint_timesteps:,} timesteps, but state shows {current_timesteps:,}")
                                print(f"[RL] Using checkpoint: {checkpoint_path}")
                
                # Fallback to expected path if no checkpoint found
                if not checkpoint_path:
                    checkpoint_path = str(self.model_dir / "checkpoints" / f"rl_model_historical_{current_timesteps}_steps.zip")
                
                # Update state (use 'training' mode for historical phase too)
                state = self.state_manager.get_state()
                self.state_manager.save_state(
                    episode=episode,
                    timesteps=current_timesteps,
                    checkpoint_path=checkpoint_path,
                    best_reward=state.get('best_reward', float('-inf')),
                    total_trades=state.get('total_trades', 0),
                    mode="training",  # Use 'training' mode (historical is just a phase)
                )
                
                progress_pct = (current_timesteps / target_timesteps) * 100
                print(f"[RL] Historical pre-training progress: {current_timesteps:,}/{target_timesteps:,} ({progress_pct:.1f}%)")
                
                # Reset historical replay for next episode
                if self.historical_replay:
                    self.historical_replay.reset()
            
            print(f"\n[RL] ✅ Historical pre-training completed: {current_timesteps:,} timesteps")
            print(f"[RL] Model ready for live paper trading fine-tuning")
            
            # Save final historical model
            historical_model_path = self.model_dir / "historical_pretrained_model.zip"
            self.model.save(str(historical_model_path))
            print(f"[RL] Historical pre-trained model saved to {historical_model_path}")
            
            return True  # Return True to indicate successful completion
            
        except Exception as e:
            print(f"[RL] Error during historical pre-training: {e}")
            import traceback
            traceback.print_exc()
            # Save state even on error
            state = self.state_manager.get_state()
            checkpoint_path = str(self.model_dir / "checkpoints" / f"rl_model_historical_{current_timesteps}_steps.zip")
            if not os.path.exists(checkpoint_path):
                checkpoints_dir = self.model_dir / "checkpoints"
                if checkpoints_dir.exists():
                    checkpoints = sorted(checkpoints_dir.glob("rl_model_historical_*.zip"))
                    if checkpoints:
                        checkpoint_path = str(checkpoints[-1])
            try:
                self.state_manager.save_state(
                    episode=episode,
                    timesteps=current_timesteps,
                    checkpoint_path=checkpoint_path,
                    best_reward=state.get('best_reward', float('-inf')),
                    total_trades=state.get('total_trades', 0),
                    mode="training",
                )
            except:
                pass
            raise
    
    def _save_final_state(self, episode, current_timesteps, best_reward, total_trades):
        """Save final model and state before shutdown"""
        try:
            # Save final model checkpoint
            final_path = self.model_dir / "final_model.zip"
            if self.model:
                self.model.save(str(final_path))
                print(f"[RL] Final model saved to {final_path}")
            
            # Save final state
            checkpoint_path = str(self.model_dir / "checkpoints" / f"rl_model_{current_timesteps}_steps.zip")
            if not os.path.exists(checkpoint_path):
                # Use latest checkpoint if final doesn't exist
                checkpoints_dir = self.model_dir / "checkpoints"
                if checkpoints_dir.exists():
                    checkpoints = sorted(checkpoints_dir.glob("rl_model_*.zip"))
                    if checkpoints:
                        checkpoint_path = str(checkpoints[-1])
            
            self.state_manager.save_state(
                episode=episode,
                timesteps=current_timesteps,
                checkpoint_path=checkpoint_path,
                best_reward=best_reward,
                total_trades=total_trades,
                mode=self.current_mode,
            )
            print(f"[RL] Training state saved: Episode {episode}, Timesteps {current_timesteps:,}, Trades {total_trades}")
            print(f"[RL] You can resume training from this checkpoint next time.")
        except Exception as e:
            print(f"[RL] Error saving final state: {e}")
            import traceback
            traceback.print_exc()

    def _send_accuracy_reached_notification(
        self,
        summary: Dict,
        current_timesteps: int,
        total_trades: int,
        to_email: str = "noamiko613@gmail.com",
    ) -> None:
        """Send notification email when accuracy threshold is reached.
        Uses env: RL_NOTIFY_EMAIL_TO (default noamiko613@gmail.com),
        RL_SMTP_HOST, RL_SMTP_PORT, RL_SMTP_USER, RL_SMTP_PASSWORD.
        If SMTP is not configured, logs a message and skips sending.
        """
        to_email = os.getenv("RL_NOTIFY_EMAIL_TO", to_email).strip()
        host = os.getenv("RL_SMTP_HOST", "smtp.gmail.com").strip()
        port = int(os.getenv("RL_SMTP_PORT", "587"))
        user = os.getenv("RL_SMTP_USER", "").strip()
        password = os.getenv("RL_SMTP_PASSWORD", "").strip()
        if not user or not password:
            print(f"[RL] Email notification skipped (set RL_SMTP_USER and RL_SMTP_PASSWORD to enable).")
            print(f"[RL] Would have sent to: {to_email}")
            return
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
        except ImportError:
            print("[RL] Email notification skipped (smtplib/email not available).")
            return
        try:
            current = summary.get("current_window", {})
            win_rate = current.get("win_rate", 0)
            profit_factor = current.get("profit_factor", 0)
            sharpe = current.get("sharpe_ratio", 0)
            drawdown = current.get("max_drawdown_pct", 0)
            subject = "[RL Trading] Accuracy threshold reached - switched to paper trading"
            body_lines = [
                "RL training has reached the accuracy threshold and switched to paper trading on live data.",
                "",
                "Details:",
                f"  Win rate: {win_rate:.2f}%",
                f"  Profit factor: {profit_factor:.2f}",
                f"  Sharpe ratio: {sharpe:.2f}",
                f"  Max drawdown: {drawdown:.2f}%",
                f"  Total timesteps: {current_timesteps:,}",
                f"  Total trades: {total_trades}",
                f"  Model dir: {self.model_dir}",
                "",
                "The bot is now in paper trading mode on live data.",
            ]
            body = "\n".join(body_lines)
            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"] = user
            msg["To"] = to_email
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP(host, port) as server:
                server.starttls()
                server.login(user, password)
                server.sendmail(user, [to_email], msg.as_string())
            print(f"[RL] Notification email sent to {to_email}")
        except Exception as e:
            print(f"[RL] Failed to send notification email: {e}")

    def _auto_tighten_thresholds(self, reason: str = ""):
        """Automatically tighten acceptance thresholds after repeated failures or rollbacks."""
        if not self.auto_tighten_enabled:
            return
        if self.auto_tighten_applied >= self.auto_tighten_max_steps:
            print(f"[RL] Auto-tighten max steps reached ({self.auto_tighten_max_steps}). No further tightening.")
            return

        self.auto_tighten_applied += 1
        self.eval_failures_since_tighten = 0

        # Tighten acceptance criteria
        self.metrics_evaluator.win_rate_threshold += self.auto_tighten_win_rate_inc
        self.metrics_evaluator.profit_factor_threshold += self.auto_tighten_profit_factor_inc
        self.metrics_evaluator.sharpe_threshold += self.auto_tighten_sharpe_inc
        self.metrics_evaluator.max_drawdown_threshold = max(
            1.0,  # prevent too strict (never 0)
            self.metrics_evaluator.max_drawdown_threshold - self.auto_tighten_drawdown_dec
        )

        # Persist to config file to survive restarts
        try:
            self.acceptance_config["acceptance_criteria"]["win_rate_threshold"] = self.metrics_evaluator.win_rate_threshold
            self.acceptance_config["acceptance_criteria"]["profit_factor_threshold"] = self.metrics_evaluator.profit_factor_threshold
            self.acceptance_config["acceptance_criteria"]["sharpe_threshold"] = self.metrics_evaluator.sharpe_threshold
            self.acceptance_config["acceptance_criteria"]["max_drawdown_threshold"] = self.metrics_evaluator.max_drawdown_threshold
            self.acceptance_config["auto_tighten"] = self.acceptance_config.get("auto_tighten", {})
            self.acceptance_config["auto_tighten"]["applied"] = self.auto_tighten_applied
            self.acceptance_config["auto_tighten"]["last_reason"] = reason
            with open(self.acceptance_config_path, "w") as f:
                json.dump(self.acceptance_config, f, indent=2)
            print(f"[RL] Auto-tighten applied (reason: {reason}). New thresholds -> "
                  f"WR≥{self.metrics_evaluator.win_rate_threshold:.2f}%, "
                  f"PF≥{self.metrics_evaluator.profit_factor_threshold:.2f}, "
                  f"Sharpe≥{self.metrics_evaluator.sharpe_threshold:.2f}, "
                  f"DD≤{self.metrics_evaluator.max_drawdown_threshold:.2f}% "
                  f"(step {self.auto_tighten_applied}/{self.auto_tighten_max_steps})")
        except Exception as exc:
            print(f"[RL] Warning: Failed to persist auto-tighten config: {exc}")

    def _record_eval_failure(self):
        """Count consecutive evaluation failures and trigger auto-tighten if needed."""
        if not self.auto_tighten_enabled or not self.auto_tighten_on_eval_failures:
            return
        self.eval_failures_since_tighten += 1
        if self.eval_failures_since_tighten >= self.auto_tighten_failures_before_tighten:
            self._auto_tighten_thresholds(reason="eval_failures")

    def _run_paper_test(self, model_path: str, test_episodes: int = 5):
        """Run paper-testing mode using a saved PPO model.

        This uses the trading environment in inference mode only (no further learning)
        and allows the TradeSimulator/analytics stack to record paper trades.
        
        Note: If keep_training_during_paper is True, this method is not called
        and paper testing happens in parallel with training.
        """
        try:
            print(f"[RL] Starting paper-testing run with model: {model_path}")

            # Mark state as paper-testing so dashboards can clearly display mode
            state = self.state_manager.get_state()
            self.state_manager.save_state(
                episode=state.get("episode", 0),
                timesteps=state.get("total_timesteps", 0),
                checkpoint_path=state.get("last_checkpoint") or model_path,
                best_reward=state.get("best_reward", float("-inf")),
                total_trades=state.get("total_trades", 0),
                mode="paper",
            )

            # Create a fresh environment for testing
            self.create_environment()

            # Load model for inference
            test_model = PPO.load(model_path, env=self.env, device="auto")

            episodes_run = 0
            while episodes_run < test_episodes:
                obs = self.env.reset()
                done = False

                while not done:
                    action, _ = test_model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = self.env.step(action)

                    # DummyVecEnv returns arrays; consider episode done when first env is done
                    done = bool(dones[0]) if isinstance(dones, (list, tuple, np.ndarray)) else bool(dones)

                episodes_run += 1
                print(f"[RL] Paper test episode {episodes_run}/{test_episodes} completed")

            print("[RL] Paper-testing run completed")
        except Exception as e:
            print(f"[RL] Error during paper-testing run: {e}")
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model"""
        if path is None:
            path = str(self.model_dir / "final_rl_model.zip")
        self.model.save(path)
        print(f"[RL] Model saved to {path}")


def main():
    """Main training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol (default: all RL training symbols)")
    parser.add_argument("--balance", type=float, default=1_000_000.0, help="Starting balance")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help="Total training timesteps. Use 0 or negative for open‑ended "
             "training that runs until the accuracy threshold is reached.",
    )
    parser.add_argument("--max-trades", type=int, default=None, help="Max trades per episode (None=unlimited)")
    parser.add_argument("--model-dir", type=str, default="models/rl_models", help="Model directory")
    parser.add_argument("--monitor", action="store_true", help="Enable simple progress monitor")
    parser.add_argument("--dashboard", action="store_true", help="Enable full RL dashboard")
    parser.add_argument("--accuracy-threshold", type=float, default=85.0,
                        help="Win rate threshold (legacy - now using multi-metric gate with win rate + profit factor + Sharpe + drawdown)")
    parser.add_argument("--min-trades", type=int, default=50,
                        help="Minimum number of closed trades required before applying accuracy threshold (default: 50)")
    parser.add_argument("--fresh-start", action="store_true",
                        help="Force fresh start: ignore existing checkpoints and start new training")
    parser.add_argument("--historical-pretrain-timesteps", type=int, default=None,
                        help="Historical pre-training timesteps (default: 9,600,000)")
    
    args = parser.parse_args()

    # If a symbol is provided, train only that market. Otherwise, create one
    # shared PPO model trained across all RL_TRAINING_SYMBOLS.
    symbols = [args.symbol] if args.symbol else RL_TRAINING_SYMBOLS

    print("\n[RL] =====================================================")
    print("[RL] RL TRAINING MODE - PAPER SIMULATION ONLY (NO REAL MONEY)")
    print("[RL] Training symbols: " + ", ".join(symbols))
    print("[RL] Logical training timeframes: " + ", ".join(RL_TRAINING_TIMEFRAMES))
    print("[RL] =====================================================\n")

    # In shared‑model mode (no explicit --symbol), we train one PPO policy
    # across all symbols using parallel environments. In single‑symbol mode
    # we still store that symbol’s model in a dedicated subdirectory.
    if args.symbol:
        print(f"\n[RL] === Starting training for single symbol {args.symbol} ===")
        symbol_model_dir = os.path.join(args.model_dir, args.symbol)
        trainer = RLTrainer(
            symbol=args.symbol,
            starting_balance=args.balance,
            max_trades=args.max_trades,  # None = unlimited
            model_dir=symbol_model_dir,
        )
    else:
        print(f"\n[RL] === Starting shared-model training for symbols: {', '.join(symbols)} ===")
        trainer = RLTrainer(
            symbol=None,
            starting_balance=args.balance,
            max_trades=args.max_trades,
            model_dir=args.model_dir,
        )

    # Set fresh_start flag if requested
    if args.fresh_start:
        trainer._fresh_start = True
        print("\n[RL] ⚠️  FRESH START MODE: Ignoring existing checkpoints")
        print("[RL] Starting new training with optimized hyperparameters\n")

    # Override historical pre-training timesteps if provided via CLI
    if args.historical_pretrain_timesteps is not None:
        trainer.historical_pretrain_timesteps = args.historical_pretrain_timesteps
        print(f"[RL] Historical pre-training target: {trainer.historical_pretrain_timesteps:,} timesteps\n")
    
    trainer.train(
        total_timesteps=args.timesteps,
        use_monitor=args.monitor,
        use_dashboard=args.dashboard,
        accuracy_threshold=args.accuracy_threshold,
        min_trades_for_threshold=args.min_trades,
    )


if __name__ == "__main__":
    main()

