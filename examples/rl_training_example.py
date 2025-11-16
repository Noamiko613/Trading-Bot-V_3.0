"""
Example: Running RL Training
============================

This script demonstrates how to run RL training with all features:
- Neural network-based reinforcement learning
- Pause/Resume functionality
- Progress monitoring with NN visualization
- Automatic checkpointing
"""

from rl_training import RLTrainer
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    """Run RL training example"""
    
    print("=" * 80)
    print("RL TRADING AGENT TRAINING")
    print("=" * 80)
    print("\nFeatures:")
    print("  - Neural Network Reinforcement Learning (PPO)")
    print("  - Live Neural Network Architecture Visualization")
    print("  - Pause/Resume with Ctrl+C")
    print("  - Automatic Checkpointing")
    print("  - Progress Monitoring")
    print("\nControls:")
    print("  - Press Ctrl+C once to pause training")
    print("  - Press Ctrl+C again to resume training")
    print("  - Progress monitor shows NN architecture and metrics")
    print("=" * 80)
    
    # Create trainer
    trainer = RLTrainer(
        symbol="BTCUSDT",
        starting_balance=100000.0,
        max_trades=None,  # None = unlimited trades (fixes 229 limit issue)
        model_dir="models/rl_models",
        checkpoint_interval=10000  # Save checkpoint every 10k steps
    )
    
    # Train for 1 million timesteps (can pause/resume anytime)
    print("\nStarting training...")
    trainer.train(total_timesteps=1000000)
    
    # Save final model
    trainer.save_model()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

