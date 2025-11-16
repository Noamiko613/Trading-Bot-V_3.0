"""
RL Training Dashboard
=====================

Comprehensive live dashboard for RL training showing:
- Global and per-pair accuracy
- Live neural network visualization with learning updates
- Training metrics and statistics
- Performance analytics
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

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
    import matplotlib.patches as mpatches
    import matplotlib
    matplotlib.use('TkAgg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Dashboard will be text-only.")

from utils.analytics import PerformanceAnalytics


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
        
        # Setup visualization
        if HAS_MATPLOTLIB:
            self._setup_figure()
        else:
            self.fig = None
    
    def _setup_figure(self):
        """Setup matplotlib figure with multiple subplots"""
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('RL Training Dashboard - Live Monitoring', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3, figure=self.fig)
        
        # Neural Network Visualization (top-left, spans 2 rows)
        self.ax_nn = self.fig.add_subplot(gs[0:2, 0])
        self.ax_nn.set_title('Neural Network Architecture (Live Learning)', fontsize=12, fontweight='bold')
        self.ax_nn.axis('off')
        
        # Global Metrics (top-center)
        self.ax_global = self.fig.add_subplot(gs[0, 1])
        self.ax_global.set_title('Global Accuracy & Performance', fontsize=11, fontweight='bold')
        self.ax_global.axis('off')
        
        # Per-Pair Metrics (top-right)
        self.ax_pairs = self.fig.add_subplot(gs[0, 2])
        self.ax_pairs.set_title('Per-Pair Performance', fontsize=11, fontweight='bold')
        self.ax_pairs.axis('off')
        
        # Reward Plot (middle-center)
        self.ax_reward = self.fig.add_subplot(gs[1, 1])
        self.ax_reward.set_title('Episode Rewards', fontsize=10)
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)
        
        # Loss Plot (middle-right)
        self.ax_loss = self.fig.add_subplot(gs[1, 2])
        self.ax_loss.set_title('Training Loss', fontsize=10)
        self.ax_loss.set_xlabel('Update')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Accuracy Plot (bottom-left)
        self.ax_accuracy = self.fig.add_subplot(gs[2, 0])
        self.ax_accuracy.set_title('Accuracy Over Time', fontsize=10)
        self.ax_accuracy.set_xlabel('Episode')
        self.ax_accuracy.set_ylabel('Accuracy %')
        self.ax_accuracy.grid(True, alpha=0.3)
        
        # Training Stats (bottom-center)
        self.ax_stats = self.fig.add_subplot(gs[2, 1])
        self.ax_stats.set_title('Training Statistics', fontsize=10, fontweight='bold')
        self.ax_stats.axis('off')
        
        # PnL Plot (bottom-right)
        self.ax_pnl = self.fig.add_subplot(gs[2, 2])
        self.ax_pnl.set_title('Cumulative P&L', fontsize=10)
        self.ax_pnl.set_xlabel('Episode')
        self.ax_pnl.set_ylabel('P&L ($)')
        self.ax_pnl.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        self.ax_pnl.grid(True, alpha=0.3)
        
        plt.ion()
        plt.show(block=False)
    
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
    
    def _draw_neural_network(self):
        """Draw live neural network with learning visualization"""
        if not HAS_MATPLOTLIB or self.ax_nn is None:
            return
        
        self.ax_nn.clear()
        self.ax_nn.axis('off')
        self.ax_nn.set_title('Neural Network Architecture (Live Learning)', 
                            fontsize=12, fontweight='bold', pad=20)
        
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
            
            num_layers = len(layers)
            max_nodes = max([l['out'] for l in layers] + [100])
            
            # Position layers horizontally
            layer_x = np.linspace(0.1, 0.9, num_layers)
            
            # Draw layers and connections
            prev_layer_nodes = None
            node_radius = 0.015
            
            for layer_idx, layer in enumerate(layers):
                num_nodes = layer['out']
                display_nodes = min(num_nodes, 25)  # Max 25 nodes per layer for clarity
                spacing = 0.8 / display_nodes if display_nodes > 1 else 0.4
                
                layer_nodes = []
                weight_info = weights[layer_idx] if layer_idx < len(weights) else {'mean': 0.1}
                weight_intensity = min(1.0, weight_info['mean'] * 10)  # Scale for visualization
                
                for node_idx in range(display_nodes):
                    x = layer_x[layer_idx]
                    y = 0.1 + (node_idx + 0.5) * spacing
                    layer_nodes.append((x, y))
                    
                    # Node color based on weight (learning progress)
                    # Green = learned, Yellow = learning, Red = needs learning
                    if weight_intensity > 0.5:
                        node_color = plt.cm.Greens(0.3 + weight_intensity * 0.5)
                    elif weight_intensity > 0.2:
                        node_color = plt.cm.YlOrRd(0.3 + weight_intensity * 0.7)
                    else:
                        node_color = 'lightgray'
                    
                    # Draw node with size based on weight
                    node_size = node_radius * (1 + weight_intensity * 0.5)
                    circle = Circle((x, y), node_size, color=node_color, zorder=3, edgecolor='black', linewidth=0.5)
                    self.ax_nn.add_patch(circle)
                
                # Draw connections to previous layer
                if prev_layer_nodes is not None:
                    for prev_node in prev_layer_nodes:
                        for curr_node in layer_nodes:
                            # Connection opacity based on weight strength
                            alpha = min(0.4, weight_intensity * 0.3 + 0.1)
                            self.ax_nn.plot(
                                [prev_node[0], curr_node[0]],
                                [prev_node[1], curr_node[1]],
                                'b-', alpha=alpha, linewidth=0.3, zorder=1
                            )
                
                prev_layer_nodes = layer_nodes
                
                # Label layer
                layer_label = f"L{layer_idx+1}\n{num_nodes}n"
                self.ax_nn.text(layer_x[layer_idx], 0.95, layer_label,
                               ha='center', va='top', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # Add info box
            info_text = (
                f"Parameters: {self.nn_info.get('total_parameters', 0):,}\n"
                f"Episodes: {self.stats['episode']}\n"
                f"Steps: {self.stats['timesteps']:,}\n"
                f"Trades: {self.stats['total_trades']}\n"
            )
            if self.nn_info.get('learning_progress'):
                recent_progress = np.mean(self.nn_info['learning_progress'][-10:]) if self.nn_info['learning_progress'] else 0
                info_text += f"Learning: {recent_progress:.6f}"
            
            self.ax_nn.text(0.02, 0.02, info_text, transform=self.ax_nn.transAxes,
                           fontsize=9, verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            self.ax_nn.set_xlim(-0.05, 1.05)
            self.ax_nn.set_ylim(0, 1)
            
        except Exception as e:
            self.ax_nn.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=self.ax_nn.transAxes)
    
    def _draw_global_metrics(self):
        """Draw global accuracy and performance metrics"""
        if not HAS_MATPLOTLIB or self.ax_global is None:
            return
        
        self.ax_global.clear()
        self.ax_global.axis('off')
        self.ax_global.set_title('Global Accuracy & Performance', fontsize=11, fontweight='bold', pad=10)
        
        metrics_text = (
            f"Global Accuracy: {self.stats['accuracy_global']:.2f}%\n"
            f"Win Rate: {self.stats['win_rate_global']:.2f}%\n"
            f"Total Trades: {self.stats['total_trades']}\n"
            f"Total P&L: ${self.stats['total_pnl']:,.2f}\n"
            f"Avg Reward: {self.stats['avg_reward']:.4f}\n"
            f"Best Episode: {self.stats['best_episode_reward']:.2f}\n"
            f"Learning Rate: {self.stats['learning_rate']:.6f}\n"
            f"Timesteps: {self.stats['timesteps']:,}"
        )
        
        # Color code accuracy
        acc_color = 'green' if self.stats['accuracy_global'] >= 50 else 'orange' if self.stats['accuracy_global'] >= 40 else 'red'
        
        self.ax_global.text(0.1, 0.9, metrics_text, transform=self.ax_global.transAxes,
                           fontsize=10, verticalalignment='top', family='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Highlight accuracy
        self.ax_global.text(0.1, 0.9, f"Global Accuracy: {self.stats['accuracy_global']:.2f}%",
                           transform=self.ax_global.transAxes, fontsize=11, fontweight='bold',
                           color=acc_color, verticalalignment='top')
    
    def _draw_per_pair_metrics(self):
        """Draw per-pair performance metrics"""
        if not HAS_MATPLOTLIB or self.ax_pairs is None:
            return
        
        self.ax_pairs.clear()
        self.ax_pairs.axis('off')
        self.ax_pairs.set_title('Per-Pair Performance', fontsize=11, fontweight='bold', pad=10)
        
        pairs_text = "Pair Performance:\n\n"
        pairs = sorted(self.stats['accuracy_by_pair'].items(), 
                      key=lambda x: x[1], reverse=True)[:8]  # Top 8 pairs
        
        if not pairs:
            pairs_text += "No trades yet..."
        else:
            for symbol, accuracy in pairs:
                win_rate = self.stats['win_rate_by_pair'].get(symbol, 0.0)
                pnl = self.stats['pnl_by_pair'].get(symbol, 0.0)
                pnl_color = 'green' if pnl >= 0 else 'red'
                
                pairs_text += f"{symbol}:\n"
                pairs_text += f"  Acc: {accuracy:.1f}% | "
                pairs_text += f"WR: {win_rate:.1f}% | "
                pairs_text += f"P&L: ${pnl:,.0f}\n"
        
        self.ax_pairs.text(0.05, 0.95, pairs_text, transform=self.ax_pairs.transAxes,
                          fontsize=9, verticalalignment='top', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def _draw_reward_plot(self):
        """Draw episode rewards plot"""
        if not HAS_MATPLOTLIB or self.ax_reward is None:
            return
        
        self.ax_reward.clear()
        if self.stats['episode_rewards']:
            episodes = range(len(self.stats['episode_rewards']))
            self.ax_reward.plot(episodes, self.stats['episode_rewards'], 'b-', linewidth=1.5, label='Reward')
            
            # Moving average
            if len(self.stats['episode_rewards']) > 10:
                window = min(20, len(self.stats['episode_rewards']) // 2)
                ma = np.convolve(self.stats['episode_rewards'], 
                               np.ones(window)/window, mode='valid')
                self.ax_reward.plot(range(window-1, len(self.stats['episode_rewards'])), 
                                   ma, 'r--', linewidth=2, label='MA')
            
            self.ax_reward.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            self.ax_reward.legend()
        
        self.ax_reward.set_title(f'Episode Rewards (Ep: {self.stats["episode"]})', fontsize=10)
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)
    
    def _draw_loss_plot(self):
        """Draw training loss plot"""
        if not HAS_MATPLOTLIB or self.ax_loss is None:
            return
        
        self.ax_loss.clear()
        if self.stats['losses']:
            self.ax_loss.plot(self.stats['losses'], 'r-', linewidth=1.5)
            self.ax_loss.set_title('Training Loss', fontsize=10)
        else:
            self.ax_loss.text(0.5, 0.5, 'No loss data yet', ha='center', va='center',
                            transform=self.ax_loss.transAxes)
        
        self.ax_loss.set_xlabel('Update')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
    
    def _draw_accuracy_plot(self):
        """Draw accuracy over time"""
        if not HAS_MATPLOTLIB or self.ax_accuracy is None:
            return
        
        self.ax_accuracy.clear()
        
        # Track accuracy history
        if not hasattr(self, '_accuracy_history'):
            self._accuracy_history = []
        
        self._accuracy_history.append(self.stats['accuracy_global'])
        if len(self._accuracy_history) > 200:
            self._accuracy_history = self._accuracy_history[-200:]
        
        if len(self._accuracy_history) > 1:
            self.ax_accuracy.plot(self._accuracy_history, 'g-', linewidth=2, label='Global Accuracy')
            self.ax_accuracy.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50%')
            self.ax_accuracy.legend()
        
        self.ax_accuracy.set_title('Accuracy Over Time', fontsize=10)
        self.ax_accuracy.set_xlabel('Episode')
        self.ax_accuracy.set_ylabel('Accuracy %')
        self.ax_accuracy.set_ylim(0, 100)
        self.ax_accuracy.grid(True, alpha=0.3)
    
    def _draw_training_stats(self):
        """Draw training statistics"""
        if not HAS_MATPLOTLIB or self.ax_stats is None:
            return
        
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Training Statistics', fontsize=10, fontweight='bold', pad=10)
        
        stats_text = (
            f"Episode: {self.stats['episode']}\n"
            f"Timesteps: {self.stats['timesteps']:,}\n"
            f"Total Trades: {self.stats['total_trades']}\n"
            f"Avg Reward: {self.stats['avg_reward']:.4f}\n"
            f"Learning Rate: {self.stats['learning_rate']:.6f}\n"
            f"Best Reward: {self.stats['best_episode_reward']:.2f}\n"
            f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
        )
        
        self.ax_stats.text(0.1, 0.9, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=9, verticalalignment='top', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def _draw_pnl_plot(self):
        """Draw cumulative P&L plot"""
        if not HAS_MATPLOTLIB or self.ax_pnl is None:
            return
        
        self.ax_pnl.clear()
        
        # Track PnL history
        if not hasattr(self, '_pnl_history'):
            self._pnl_history = []
        
        self._pnl_history.append(self.stats['total_pnl'])
        if len(self._pnl_history) > 200:
            self._pnl_history = self._pnl_history[-200:]
        
        if len(self._pnl_history) > 1:
            color = 'green' if self._pnl_history[-1] >= 0 else 'red'
            self.ax_pnl.plot(self._pnl_history, color=color, linewidth=2, label='Cumulative P&L')
            self.ax_pnl.fill_between(range(len(self._pnl_history)), 0, self._pnl_history,
                                    alpha=0.3, color=color)
            self.ax_pnl.legend()
        
        self.ax_pnl.set_title('Cumulative P&L', fontsize=10)
        self.ax_pnl.set_xlabel('Episode')
        self.ax_pnl.set_ylabel('P&L ($)')
        self.ax_pnl.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        self.ax_pnl.grid(True, alpha=0.3)
    
    def _update_all(self):
        """Update all dashboard elements"""
        if HAS_MATPLOTLIB and self.fig is not None:
            try:
                self._draw_neural_network()
                self._draw_global_metrics()
                self._draw_per_pair_metrics()
                self._draw_reward_plot()
                self._draw_loss_plot()
                self._draw_accuracy_plot()
                self._draw_training_stats()
                self._draw_pnl_plot()
                
                # Refresh
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except Exception as e:
                print(f"Error updating dashboard: {e}")
        else:
            self._print_text_stats()
    
    def _print_text_stats(self):
        """Print statistics as text (fallback)"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 100)
        print("RL TRAINING DASHBOARD")
        print("=" * 100)
        print(f"\nGlobal Accuracy: {self.stats['accuracy_global']:.2f}%")
        print(f"Win Rate: {self.stats['win_rate_global']:.2f}%")
        print(f"Total Trades: {self.stats['total_trades']}")
        print(f"Episode: {self.stats['episode']}")
        print(f"Timesteps: {self.stats['timesteps']:,}")
        print(f"\nPer-Pair Performance:")
        for symbol, acc in sorted(self.stats['accuracy_by_pair'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {symbol}: {acc:.1f}%")
        print("=" * 100)
    
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
        if HAS_MATPLOTLIB and self.fig is not None:
            plt.close(self.fig)
    
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

