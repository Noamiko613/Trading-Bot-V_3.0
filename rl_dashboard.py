"""
RL Training Dashboard (Terminal-Only)
======================================

Comprehensive live dashboard for RL training showing:
- Global and per-pair accuracy
- Neural network architecture visualization (ASCII art)
- Training metrics and statistics
- Performance analytics

All visualization is done in the terminal using ASCII art and ANSI colors.
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError, RuntimeError):
    HAS_TORCH = False

from utils.analytics import PerformanceAnalytics

# ANSI color codes for terminal
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    GRAY = '\033[90m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_RED = '\033[91m'


class RLDashboard:
    """Comprehensive RL training dashboard"""
    
    def __init__(
        self,
        model_dir: str = "models/rl_models",
        state_file: str = "models/rl_training_state.json",
        db_path: str = "sim_results/trades.db",
        update_interval: float = 2.0
    ):
        self.model_dir = Path(model_dir)
        self.state_file = Path(state_file)
        self.db_path = db_path
        self.update_interval = update_interval
        
        # Analytics
        self.analytics = PerformanceAnalytics(db_path=db_path)
        
        # Model and state
        self.model = None
        self.state = {}
        
        # Statistics
        self.stats = {
            'episode': 0,
            'timesteps': 0,
            'total_trades': 0,
            'rewards': [],
            'episode_rewards': [],
            'losses': [],
            'learning_rate': 0.0,
            'accuracy_global': 0.0,
            'accuracy_by_pair': {},
            'win_rate_global': 0.0,
            'win_rate_by_pair': {},
            'avg_reward': 0.0,
            'best_episode_reward': float('-inf'),
            'total_pnl': 0.0,
            'pnl_by_pair': {}
        }
        
        # Neural network info
        self.nn_info = {
            'layers': [],
            'layer_sizes': [],
            'weights': [],
            'weight_history': [],
            'total_parameters': 0,
            'learning_progress': []
        }
        
        # Threading
        self.running = False
        self.update_thread = None
        self.lock = threading.Lock()
        
        # Terminal width for formatting
        try:
            self.terminal_width = os.get_terminal_size().columns
        except:
            self.terminal_width = 120
    
    def set_model(self, model):
        """Set the RL model for visualization"""
        self.model = model
        if model is not None:
            self._extract_nn_info()
    
    def _extract_nn_info(self):
        """Extract neural network architecture and weights"""
        if not HAS_TORCH or self.model is None:
            return
        
        try:
            policy = self.model.policy
            self.nn_info['layers'] = []
            self.nn_info['layer_sizes'] = []
            self.nn_info['weights'] = []
            
            # Extract feature extractor
            if hasattr(policy, 'features_extractor') and hasattr(policy.features_extractor, 'net'):
                for module in policy.features_extractor.net.modules():
                    if isinstance(module, torch.nn.Linear):
                        in_size = module.in_features
                        out_size = module.out_features
                        self.nn_info['layer_sizes'].append({'in': in_size, 'out': out_size})
                        
                        # Get weight statistics
                        weight_mean = float(module.weight.data.abs().mean().cpu().numpy())
                        weight_std = float(module.weight.data.std().cpu().numpy())
                        self.nn_info['weights'].append({
                            'mean': weight_mean,
                            'std': weight_std,
                            'min': float(module.weight.data.min().cpu().numpy()),
                            'max': float(module.weight.data.max().cpu().numpy())
                        })
            
            # Extract MLP layers
            if hasattr(policy, 'mlp_extractor'):
                mlp = policy.mlp_extractor
                for module_list in [mlp.policy_net, mlp.value_net]:
                    if hasattr(module_list, '__iter__'):
                        for module in module_list:
                            if isinstance(module, torch.nn.Linear):
                                in_size = module.in_features
                                out_size = module.out_features
                                self.nn_info['layer_sizes'].append({'in': in_size, 'out': out_size})
                                
                                weight_mean = float(module.weight.data.abs().mean().cpu().numpy())
                                weight_std = float(module.weight.data.std().cpu().numpy())
                                self.nn_info['weights'].append({
                                    'mean': weight_mean,
                                    'std': weight_std,
                                    'min': float(module.weight.data.min().cpu().numpy()),
                                    'max': float(module.weight.data.max().cpu().numpy())
                                })
            
            # Count parameters
            self.nn_info['total_parameters'] = sum(
                p.numel() for p in policy.parameters() if p.requires_grad
            )
            
            # Track learning progress (weight changes)
            current_weights = [w['mean'] for w in self.nn_info['weights']]
            if len(self.nn_info['weight_history']) > 0:
                prev_weights = self.nn_info['weight_history'][-1]
                weight_change = np.mean([abs(c - p) for c, p in zip(current_weights, prev_weights)])
                self.nn_info['learning_progress'].append(weight_change)
            self.nn_info['weight_history'].append(current_weights)
            
            # Keep only recent history
            if len(self.nn_info['weight_history']) > 100:
                self.nn_info['weight_history'] = self.nn_info['weight_history'][-100:]
            if len(self.nn_info['learning_progress']) > 100:
                self.nn_info['learning_progress'] = self.nn_info['learning_progress'][-100:]
                
        except Exception as e:
            print(f"Error extracting NN info: {e}")
    
    def _update_stats(self):
        """Update statistics from state file and database"""
        # Load state
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                    self.stats['episode'] = self.state.get('episode', 0)
                    self.stats['timesteps'] = self.state.get('total_timesteps', 0)
                    self.stats['total_trades'] = self.state.get('total_trades', 0)
            except Exception:
                pass
        
        # Get trades from database
        try:
            all_trades = self.analytics.get_closed_trades(days=None)
            
            if all_trades:
                # Global metrics
                metrics = self.analytics.calculate_metrics(all_trades)
                self.stats['win_rate_global'] = metrics.get('win_rate', 0.0)
                self.stats['accuracy_global'] = metrics.get('win_rate', 0.0)  # Using win rate as accuracy proxy
                self.stats['total_pnl'] = metrics.get('total_pnl', 0.0)
                self.stats['avg_reward'] = metrics.get('avg_pnl', 0.0) / 100  # Scale for reward
                
                # Per-pair metrics
                pairs = {}
                for trade in all_trades:
                    symbol = trade.get('symbol', 'UNKNOWN')
                    if symbol not in pairs:
                        pairs[symbol] = []
                    pairs[symbol].append(trade)
                
                self.stats['accuracy_by_pair'] = {}
                self.stats['win_rate_by_pair'] = {}
                self.stats['pnl_by_pair'] = {}
                
                for symbol, trades in pairs.items():
                    pair_metrics = self.analytics.calculate_metrics(trades)
                    self.stats['accuracy_by_pair'][symbol] = pair_metrics.get('win_rate', 0.0)
                    self.stats['win_rate_by_pair'][symbol] = pair_metrics.get('win_rate', 0.0)
                    self.stats['pnl_by_pair'][symbol] = pair_metrics.get('total_pnl', 0.0)
        
        except Exception as e:
            pass  # Database might not exist yet
        
        # Update NN info if model available
        if self.model is not None:
            self._extract_nn_info()
    
    def _draw_neural_network_ascii(self) -> str:
        """Draw neural network as ASCII art"""
        try:
            layers = self.nn_info.get('layer_sizes', [])
            weights = self.nn_info.get('weights', [])
            
            if not layers:
                # Default structure
                layers = [
                    {'in': 100, 'out': 512},
                    {'in': 512, 'out': 512},
                    {'in': 512, 'out': 256},
                    {'in': 256, 'out': 128},
                    {'in': 128, 'out': 3}
                ]
                weights = [{'mean': 0.1} for _ in layers]
            
            lines = []
            lines.append(f"{Colors.BOLD}{Colors.CYAN}Neural Network Architecture (Live Learning){Colors.RESET}")
            lines.append("=" * min(80, self.terminal_width))
            
            # ASCII representation of layers
            max_display_nodes = 15  # Max nodes to display per layer
            layer_spacing = 12
            
            for layer_idx, layer in enumerate(layers):
                num_nodes = layer['out']
                display_nodes = min(num_nodes, max_display_nodes)
                weight_info = weights[layer_idx] if layer_idx < len(weights) else {'mean': 0.1}
                weight_intensity = min(1.0, weight_info['mean'] * 10)
                
                # Choose color based on learning progress
                if weight_intensity > 0.5:
                    color = Colors.BRIGHT_GREEN
                    symbol = '●'
                elif weight_intensity > 0.2:
                    color = Colors.BRIGHT_YELLOW
                    symbol = '○'
                else:
                    color = Colors.GRAY
                    symbol = '·'
                
                # Layer header
                lines.append(f"\n{Colors.BOLD}Layer {layer_idx+1}{Colors.RESET} ({num_nodes} nodes):")
                
                # Display nodes
                node_line = "  "
                for i in range(display_nodes):
                    node_line += f"{color}{symbol}{Colors.RESET} "
                if num_nodes > max_display_nodes:
                    node_line += f"{Colors.GRAY}... (+{num_nodes - max_display_nodes}){Colors.RESET}"
                lines.append(node_line)
                
                # Weight info
                lines.append(f"  Weight: {weight_info.get('mean', 0):.4f} | "
                           f"Learning: {'Active' if weight_intensity > 0.2 else 'Low'}")
            
            # Network info
            lines.append(f"\n{Colors.BOLD}Network Info:{Colors.RESET}")
            lines.append(f"  Parameters: {self.nn_info.get('total_parameters', 0):,}")
            if self.nn_info.get('learning_progress'):
                recent = np.mean(self.nn_info['learning_progress'][-10:]) if self.nn_info['learning_progress'] else 0
                lines.append(f"  Learning Progress: {recent:.6f}")
            
            return "\n".join(lines)
        except Exception as e:
            return f"Error displaying NN: {e}"
    
    def _format_metrics_text(self) -> str:
        """Format global metrics as text"""
        acc = self.stats['accuracy_global']
        acc_color = Colors.BRIGHT_GREEN if acc >= 50 else Colors.BRIGHT_YELLOW if acc >= 40 else Colors.BRIGHT_RED
        
        lines = []
        lines.append(f"{Colors.BOLD}{Colors.CYAN}Global Accuracy & Performance{Colors.RESET}")
        lines.append("=" * min(50, self.terminal_width))
        lines.append(f"{acc_color}Global Accuracy: {acc:.2f}%{Colors.RESET}")
        lines.append(f"Win Rate: {self.stats['win_rate_global']:.2f}%")
        lines.append(f"Total Trades: {self.stats['total_trades']}")
        lines.append(f"Total P&L: ${self.stats['total_pnl']:,.2f}")
        lines.append(f"Avg Reward: {self.stats['avg_reward']:.4f}")
        lines.append(f"Best Episode: {self.stats['best_episode_reward']:.2f}")
        lines.append(f"Learning Rate: {self.stats['learning_rate']:.6f}")
        lines.append(f"Timesteps: {self.stats['timesteps']:,}")
        return "\n".join(lines)
    
    def _format_per_pair_text(self) -> str:
        """Format per-pair metrics as text"""
        lines = []
        lines.append(f"{Colors.BOLD}{Colors.CYAN}Per-Pair Performance{Colors.RESET}")
        lines.append("=" * min(50, self.terminal_width))
        
        pairs = sorted(self.stats['accuracy_by_pair'].items(), 
                      key=lambda x: x[1], reverse=True)[:8]
        
        if not pairs:
            lines.append("No trades yet...")
        else:
            for symbol, accuracy in pairs:
                win_rate = self.stats['win_rate_by_pair'].get(symbol, 0.0)
                pnl = self.stats['pnl_by_pair'].get(symbol, 0.0)
                pnl_color = Colors.BRIGHT_GREEN if pnl >= 0 else Colors.BRIGHT_RED
                
                lines.append(f"{symbol}:")
                lines.append(f"  Acc: {accuracy:.1f}% | WR: {win_rate:.1f}% | "
                           f"{pnl_color}P&L: ${pnl:,.0f}{Colors.RESET}")
        
        return "\n".join(lines)
    
    def _format_reward_chart(self) -> str:
        """Format reward chart as ASCII"""
        if not self.stats['episode_rewards']:
            return "No reward data yet"
        
        lines = []
        lines.append(f"{Colors.BOLD}Episode Rewards{Colors.RESET}")
        lines.append("=" * min(50, self.terminal_width))
        
        rewards = self.stats['episode_rewards'][-30:]  # Last 30 episodes
        if rewards:
            max_reward = max(rewards) if max(rewards) > 0 else 1
            min_reward = min(rewards)
            chart_width = 40
            
            for i, reward in enumerate(rewards[-10:]):  # Show last 10
                bar_length = int((reward - min_reward) / (max_reward - min_reward) * chart_width) if max_reward != min_reward else 0
                bar = "█" * bar_length
                color = Colors.BRIGHT_GREEN if reward > 0 else Colors.BRIGHT_RED
                lines.append(f"Ep {len(rewards)-10+i:3d}: {color}{bar}{Colors.RESET} {reward:.2f}")
        
        return "\n".join(lines)
    
    def _format_accuracy_chart(self) -> str:
        """Format accuracy chart as ASCII"""
        if not hasattr(self, '_accuracy_history'):
            self._accuracy_history = []
        
        self._accuracy_history.append(self.stats['accuracy_global'])
        if len(self._accuracy_history) > 50:
            self._accuracy_history = self._accuracy_history[-50:]
        
        lines = []
        lines.append(f"{Colors.BOLD}Accuracy Over Time{Colors.RESET}")
        lines.append("=" * min(50, self.terminal_width))
        
        if len(self._accuracy_history) > 1:
            chart_width = 40
            for i, acc in enumerate(self._accuracy_history[-10:]):  # Last 10
                bar_length = int(acc / 100 * chart_width)
                bar = "█" * bar_length
                color = Colors.BRIGHT_GREEN if acc >= 50 else Colors.BRIGHT_YELLOW if acc >= 40 else Colors.BRIGHT_RED
                lines.append(f"Ep {len(self._accuracy_history)-10+i:3d}: {color}{bar}{Colors.RESET} {acc:.1f}%")
        
        return "\n".join(lines)
    
    def _format_training_stats_text(self) -> str:
        """Format training statistics as text"""
        lines = []
        lines.append(f"{Colors.BOLD}{Colors.CYAN}Training Statistics{Colors.RESET}")
        lines.append("=" * min(50, self.terminal_width))
        lines.append(f"Episode: {self.stats['episode']}")
        lines.append(f"Timesteps: {self.stats['timesteps']:,}")
        lines.append(f"Total Trades: {self.stats['total_trades']}")
        lines.append(f"Avg Reward: {self.stats['avg_reward']:.4f}")
        lines.append(f"Learning Rate: {self.stats['learning_rate']:.6f}")
        lines.append(f"Best Reward: {self.stats['best_episode_reward']:.2f}")
        lines.append(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
        return "\n".join(lines)
    
    def _format_pnl_chart(self) -> str:
        """Format P&L chart as ASCII"""
        if not hasattr(self, '_pnl_history'):
            self._pnl_history = []
        
        self._pnl_history.append(self.stats['total_pnl'])
        if len(self._pnl_history) > 50:
            self._pnl_history = self._pnl_history[-50:]
        
        lines = []
        lines.append(f"{Colors.BOLD}Cumulative P&L{Colors.RESET}")
        lines.append("=" * min(50, self.terminal_width))
        
        if len(self._pnl_history) > 1:
            max_pnl = max(abs(p) for p in self._pnl_history) if self._pnl_history else 1
            chart_width = 40
            
            for i, pnl in enumerate(self._pnl_history[-10:]):  # Last 10
                bar_length = int(abs(pnl) / max_pnl * chart_width) if max_pnl > 0 else 0
                bar = "█" * bar_length
                color = Colors.BRIGHT_GREEN if pnl >= 0 else Colors.BRIGHT_RED
                lines.append(f"Ep {len(self._pnl_history)-10+i:3d}: {color}{bar}{Colors.RESET} ${pnl:,.0f}")
        
        return "\n".join(lines)
    
    def _update_all(self):
        """Update all dashboard elements - terminal only"""
        self._print_text_stats()
    
    def _print_text_stats(self):
        """Print comprehensive statistics in terminal"""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Header
        width = min(self.terminal_width, 120)
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}RL TRAINING DASHBOARD - Live Monitoring{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.RESET}\n")
        
        # Two column layout
        left_col = []
        right_col = []
        
        # Left column: NN and Global metrics
        left_col.append(self._draw_neural_network_ascii())
        left_col.append("")
        left_col.append(self._format_metrics_text())
        left_col.append("")
        left_col.append(self._format_training_stats_text())
        
        # Right column: Per-pair, charts
        right_col.append(self._format_per_pair_text())
        right_col.append("")
        right_col.append(self._format_reward_chart())
        right_col.append("")
        right_col.append(self._format_accuracy_chart())
        right_col.append("")
        right_col.append(self._format_pnl_chart())
        
        # Print side by side if terminal is wide enough, otherwise stacked
        if width >= 120:
            left_lines = "\n".join(left_col).split('\n')
            right_lines = "\n".join(right_col).split('\n')
            max_lines = max(len(left_lines), len(right_lines))
            
            for i in range(max_lines):
                left = left_lines[i] if i < len(left_lines) else ""
                right = right_lines[i] if i < len(right_lines) else ""
                # Pad to create columns
                left_padded = left.ljust(60)
                right_padded = right[:60] if len(right) > 60 else right
                print(f"{left_padded}  {right_padded}")
        else:
            # Stacked layout for narrow terminals
            print("\n".join(left_col))
            print("\n")
            print("\n".join(right_col))
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.RESET}")
        print(f"{Colors.GRAY}Press Ctrl+C to stop{Colors.RESET}\n")
    
    def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                with self.lock:
                    self._update_stats()
                    self._update_all()
            except Exception as e:
                print(f"Error in update loop: {e}")
            
            time.sleep(self.update_interval)
    
    def start(self):
        """Start dashboard"""
        self.running = True
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            print("[Dashboard] Started")
    
    def stop(self):
        """Stop dashboard"""
        self.running = False
    
    def update_model(self, model):
        """Update model reference"""
        self.model = model
        if model is not None:
            self._extract_nn_info()


def main():
    """Test dashboard standalone"""
    dashboard = RLDashboard()
    dashboard.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dashboard.stop()
        print("\nDashboard stopped")


if __name__ == "__main__":
    main()

