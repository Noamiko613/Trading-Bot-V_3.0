"""
Reinforcement Learning Training Script
======================================

Trains a neural network RL agent for trading with:
- Pause/Resume functionality
- Progress monitoring
- Model checkpointing
"""

import os
import json
import time
import signal
import threading
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl_trading_env import TradingEnv
from rl_progress_monitor import RLProgressMonitor


class TrainingState:
    """Manages training state for pause/resume"""
    
    def __init__(self, state_file: str = "models/rl_training_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load training state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'episode': 0,
            'total_timesteps': 0,
            'last_checkpoint': None,
            'best_reward': float('-inf'),
            'total_trades': 0
        }
    
    def save_state(self, episode: int, timesteps: int, checkpoint_path: str, best_reward: float, total_trades: int):
        """Save training state"""
        self.state = {
            'episode': episode,
            'total_timesteps': timesteps,
            'last_checkpoint': checkpoint_path,
            'best_reward': best_reward,
            'total_trades': total_trades,
            'timestamp': datetime.now().isoformat()
        }
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
                print("\n[PAUSE] Training paused. Press 'r' to resume...")
            return False  # Stop training
        else:
            if self.paused:
                self.paused = False
                print("\n[RESUME] Training resumed...")
            return True


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
        symbol: str = "BTCUSDT",
        starting_balance: float = 100000.0,
        max_trades: Optional[int] = None,  # None = unlimited
        model_dir: str = "models/rl_models",
        checkpoint_interval: int = 10000
    ):
        self.symbol = symbol
        self.starting_balance = starting_balance
        self.max_trades = max_trades
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        
        # Training state
        self.state_manager = TrainingState()
        self.pause_event = threading.Event()
        self.is_paused = False
        
        # Progress monitor
        self.progress_monitor = None
        self.monitor_thread = None
        
        # Environment
        self.env = None
        self.model = None
        
        # Setup signal handlers for pause/resume
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for pause/resume"""
        def signal_handler(sig, frame):
            if self.is_paused:
                print("\n[RESUME] Resuming training...")
                self.pause_event.clear()
                self.is_paused = False
            else:
                print("\n[PAUSE] Pausing training (press Ctrl+C again to resume)...")
                self.pause_event.set()
                self.is_paused = True
        
        # Handle Ctrl+C for pause/resume
        signal.signal(signal.SIGINT, signal_handler)
    
    def create_environment(self):
        """Create trading environment"""
        def make_env():
            env = TradingEnv(
                symbol=self.symbol,
                starting_balance=self.starting_balance,
                lookback_window=100,
                max_trades=self.max_trades  # Pass max_trades (None = unlimited)
            )
            # Wrap with Monitor for stats
            env = Monitor(env, filename=str(self.model_dir / "training_monitor.csv"))
            return env
        
        # Create vectorized environment
        self.env = DummyVecEnv([make_env])
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
        
        # Policy kwargs with custom network
        policy_kwargs = {
            "features_extractor_class": CustomFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [dict(pi=[512, 512, 256, 128], vf=[512, 512, 256, 128])]
        }
        
        # Try to load existing model
        if load_checkpoint and state['last_checkpoint']:
            checkpoint_path = state['last_checkpoint']
            if os.path.exists(checkpoint_path):
                try:
                    print(f"[RL] Loading model from {checkpoint_path}")
                    self.model = PPO.load(
                        checkpoint_path,
                        env=self.env,
                        device='auto'
                    )
                    print(f"[RL] Model loaded successfully. Continuing from episode {state['episode']}")
                    return self.model
                except Exception as e:
                    print(f"[RL] Error loading model: {e}. Creating new model...")
        
        # Create new model
        print("[RL] Creating new PPO model with custom neural network...")
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device='auto',
            tensorboard_log=str(self.model_dir / "tensorboard")
        )
        
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
    
    def train(self, total_timesteps: int = 1000000):
        """Train the RL agent"""
        # Create environment
        self.create_environment()
        
        # Create or load model
        self.create_model(load_checkpoint=True)
        
        # Start progress monitor
        self.start_progress_monitor()
        
        # Load state
        state = self.state_manager.get_state()
        current_timesteps = state['total_timesteps']
        remaining_timesteps = max(0, total_timesteps - current_timesteps)
        
        print(f"\n[RL] Starting training...")
        print(f"[RL] Total timesteps: {total_timesteps}")
        print(f"[RL] Already trained: {current_timesteps}")
        print(f"[RL] Remaining: {remaining_timesteps}")
        print(f"[RL] Max trades limit: {self.max_trades if self.max_trades else 'Unlimited'}")
        print(f"[RL] Press Ctrl+C to pause/resume training\n")
        
        # Custom callback for pause/resume and checkpointing
        callbacks = [
            PauseResumeCallback(self.state_manager, self.pause_event),
            CheckpointCallback(
                save_freq=self.checkpoint_interval,
                save_path=str(self.model_dir / "checkpoints"),
                name_prefix="rl_model",
                save_replay_buffer=True,
                save_vecnormalize=True
            )
        ]
        
        episode = 0
        best_reward = state.get('best_reward', float('-inf'))
        
        try:
            # Train in chunks to allow for pause/resume
            chunk_size = min(remaining_timesteps, 50000)
            
            while remaining_timesteps > 0:
                # Check if paused
                if self.pause_event.is_set():
                    print("[RL] Training paused. Waiting...")
                    while self.pause_event.is_set():
                        time.sleep(1)
                    print("[RL] Training resumed")
                
                # Train for a chunk
                self.model.learn(
                    total_timesteps=chunk_size,
                    callback=callbacks,
                    reset_num_timesteps=False,
                    tb_log_name="PPO"
                )
                
                # Update state
                current_timesteps += chunk_size
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
                
                # Save state
                checkpoint_path = str(self.model_dir / "checkpoints" / f"rl_model_{current_timesteps}_steps.zip")
                if not os.path.exists(checkpoint_path):
                    # Use latest checkpoint
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
                    total_trades=total_trades
                )
                
                # Update progress monitor
                if self.progress_monitor:
                    self.progress_monitor.update_stats({
                        'episode': episode,
                        'timesteps': current_timesteps,
                        'total_trades': total_trades
                    })
                
                print(f"\n[RL] Progress: {current_timesteps}/{total_timesteps} timesteps "
                      f"({current_timesteps/total_timesteps*100:.1f}%) - Trades: {total_trades}")
        
        except KeyboardInterrupt:
            print("\n[RL] Training interrupted. Saving final checkpoint...")
            final_path = self.model_dir / "final_model.zip"
            self.model.save(str(final_path))
            print(f"[RL] Final model saved to {final_path}")
        
        print("\n[RL] Training completed!")
    
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
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--balance", type=float, default=100000.0, help="Starting balance")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--max-trades", type=int, default=None, help="Max trades per episode (None=unlimited)")
    parser.add_argument("--model-dir", type=str, default="models/rl_models", help="Model directory")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RLTrainer(
        symbol=args.symbol,
        starting_balance=args.balance,
        max_trades=args.max_trades,  # None = unlimited
        model_dir=args.model_dir
    )
    
    # Train
    trainer.train(total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()

