"""
RL Progress Monitor with Neural Network Visualization
======================================================

Displays real-time training progress including:
- Live neural network architecture visualization
- Training metrics
- Episode statistics
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import numpy as np

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError, RuntimeError) as e:
    HAS_TORCH = False
    error_msg = str(e)
    if "DLL" in error_msg or "c10.dll" in error_msg or "1114" in error_msg:
        print("\n" + "="*70)
        print("ERROR: Torch DLL initialization failed (Windows)")
        print("="*70)
        print(f"Error: {type(e).__name__}: {error_msg}")
        print("\nTo fix this issue, run:")
        print("  python fix_torch_dll_error.py")
        print("\nOr manually:")
        print("1. Install Visual C++ Redistributables:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("2. Reinstall torch (CPU-only):")
        print("   pip uninstall torch -y")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("="*70 + "\n")
    else:
        print(f"Warning: torch not available ({type(e).__name__}: {e})")
        print("RL training will work but NN visualization may be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation
    import matplotlib
    matplotlib.use('TkAgg' if os.name != 'nt' else 'TkAgg')  # Use TkAgg for live updates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. NN visualization will be text-only.")


class RLProgressMonitor:
    """Monitors RL training progress with NN visualization"""
    
    def __init__(
        self,
        model_dir: str,
        model,
        state_manager,
        update_interval: float = 2.0
    ):
        self.model_dir = Path(model_dir)
        self.model = model
        self.state_manager = state_manager
        self.update_interval = update_interval
        
        # Statistics
        self.stats = {
            'episode': 0,
            'timesteps': 0,
            'total_trades': 0,
            'rewards': [],
            'episode_lengths': [],
            'losses': [],
            'learning_rate': 0.0
        }
        
        # Neural network architecture info
        self.nn_architecture = {
            'layers': [],
            'layer_sizes': [],
            'activation_functions': [],
            'total_parameters': 0
        }
        
        # Threading
        self.running = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Extract NN architecture
        self._extract_architecture()
        
        # Setup visualization if matplotlib available
        if HAS_MATPLOTLIB:
            self._setup_visualization()
        else:
            self.fig = None
            self.ax = None
    
    def _extract_architecture(self):
        """Extract neural network architecture from model"""
        try:
            if self.model is None:
                return
            
            # Get policy network
            policy = self.model.policy
            
            if HAS_TORCH:
                # Extract feature extractor layers
                if hasattr(policy, 'features_extractor'):
                    feat_ext = policy.features_extractor
                    if hasattr(feat_ext, 'net'):
                        # Extract layers from Sequential
                        for name, module in feat_ext.net.named_modules():
                            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                                self.nn_architecture['layers'].append(name)
                                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                                    self.nn_architecture['layer_sizes'].append({
                                        'in_features': module.in_features,
                                        'out_features': module.out_features
                                    })
                
                # Extract policy network layers
                if hasattr(policy, 'mlp_extractor'):
                    mlp = policy.mlp_extractor
                    # Policy network
                    if hasattr(mlp, 'policy_net'):
                        for module in mlp.policy_net:
                            if isinstance(module, torch.nn.Linear):
                                self.nn_architecture['layer_sizes'].append({
                                    'in_features': module.in_features,
                                    'out_features': module.out_features
                                })
                    
                    # Value network
                    if hasattr(mlp, 'value_net'):
                        for module in mlp.value_net:
                            if isinstance(module, torch.nn.Linear):
                                self.nn_architecture['layer_sizes'].append({
                                    'in_features': module.in_features,
                                    'out_features': module.out_features
                                })
                
                # Count total parameters
                total_params = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
                self.nn_architecture['total_parameters'] = total_params
            
        except Exception as e:
            print(f"Error extracting architecture: {e}")
            # Default architecture if extraction fails
            self.nn_architecture = {
                'layers': ['Input', 'Hidden1', 'Hidden2', 'Hidden3', 'Output'],
                'layer_sizes': [
                    {'in_features': 100, 'out_features': 512},
                    {'in_features': 512, 'out_features': 512},
                    {'in_features': 512, 'out_features': 256},
                    {'in_features': 256, 'out_features': 128},
                    {'in_features': 128, 'out_features': 3}
                ],
                'total_parameters': 0
            }
    
    def _setup_visualization(self):
        """Setup matplotlib figure for visualization"""
        if not HAS_MATPLOTLIB:
            return
        
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('RL Training Progress Monitor', fontsize=16, fontweight='bold')
        
        # Create subplots: NN visualization (left) and metrics (right)
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # NN Architecture visualization (left side, takes 2 rows)
        self.ax_nn = self.fig.add_subplot(gs[:, 0])
        self.ax_nn.set_title('Neural Network Architecture', fontsize=12, fontweight='bold')
        self.ax_nn.axis('off')
        
        # Metrics plots (right side)
        self.ax_reward = self.fig.add_subplot(gs[0, 1])
        self.ax_reward.set_title('Episode Rewards', fontsize=10)
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        
        self.ax_loss = self.fig.add_subplot(gs[1, 1])
        self.ax_loss.set_title('Training Loss', fontsize=10)
        self.ax_loss.set_xlabel('Update')
        self.ax_loss.set_ylabel('Loss')
        
        plt.ion()  # Interactive mode
        plt.show(block=False)
    
    def _draw_neural_network(self):
        """Draw neural network architecture"""
        if not HAS_MATPLOTLIB or self.ax_nn is None:
            return
        
        self.ax_nn.clear()
        self.ax_nn.axis('off')
        self.ax_nn.set_title('Neural Network Architecture (Live)', fontsize=12, fontweight='bold')
        
        try:
            # Get layer sizes
            layers = self.nn_architecture.get('layer_sizes', [])
            if not layers:
                # Default structure
                layers = [
                    {'in_features': 100, 'out_features': 512},
                    {'in_features': 512, 'out_features': 512},
                    {'in_features': 512, 'out_features': 256},
                    {'in_features': 256, 'out_features': 128},
                    {'in_features': 128, 'out_features': 3}
                ]
            
            # Add input layer if not present
            if not any('in_features' in str(l) and l.get('in_features', 0) > 100 for l in layers):
                layers.insert(0, {'in_features': 0, 'out_features': layers[0]['in_features'] if layers else 100})
            
            # Draw network
            num_layers = len(layers)
            max_nodes = max([l.get('out_features', l.get('in_features', 100)) for l in layers] + [100])
            
            # Position layers
            layer_positions = np.linspace(0.1, 0.9, num_layers)
            node_radius = 0.015
            
            # Get live weights if available
            live_weights = self._get_layer_weights()
            
            # Draw layers and connections
            prev_nodes = None
            node_positions_by_layer = []
            
            for layer_idx, layer in enumerate(layers):
                num_nodes = layer.get('out_features', layer.get('in_features', 100))
                
                # Scale node count for visualization (max 20 nodes per layer)
                display_nodes = min(num_nodes, 20)
                spacing = 0.8 / display_nodes if display_nodes > 1 else 0
                
                nodes = []
                for node_idx in range(display_nodes):
                    x = layer_positions[layer_idx]
                    y = 0.1 + (node_idx + 0.5) * spacing
                    nodes.append((x, y))
                    
                    # Draw node
                    circle = plt.Circle((x, y), node_radius, 
                                       color=self._get_node_color(layer_idx, node_idx, live_weights),
                                       zorder=3)
                    self.ax_nn.add_patch(circle)
                
                node_positions_by_layer.append(nodes)
                
                # Draw connections to previous layer
                if prev_nodes is not None:
                    for prev_node in prev_nodes:
                        for curr_node in nodes:
                            # Get weight strength for coloring
                            weight_alpha = 0.3  # Default
                            if live_weights and layer_idx < len(live_weights):
                                weight_alpha = min(0.7, abs(live_weights[layer_idx]) * 0.5 + 0.2)
                            
                            self.ax_nn.plot(
                                [prev_node[0], curr_node[0]],
                                [prev_node[1], curr_node[1]],
                                'b-', alpha=weight_alpha, linewidth=0.5, zorder=1
                            )
                
                prev_nodes = nodes
                
                # Label layer
                layer_name = f"Layer {layer_idx + 1}\n({num_nodes} nodes)"
                self.ax_nn.text(layer_positions[layer_idx], 0.95, layer_name,
                               ha='center', va='top', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Add info text
            info_text = (
                f"Total Parameters: {self.nn_architecture.get('total_parameters', 0):,}\n"
                f"Episodes: {self.stats['episode']}\n"
                f"Timesteps: {self.stats['timesteps']:,}\n"
                f"Total Trades: {self.stats['total_trades']}"
            )
            self.ax_nn.text(0.02, 0.02, info_text, transform=self.ax_nn.transAxes,
                           fontsize=9, verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            self.ax_nn.set_xlim(-0.1, 1.1)
            self.ax_nn.set_ylim(-0.1, 1.1)
            
        except Exception as e:
            print(f"Error drawing NN: {e}")
    
    def _get_node_color(self, layer_idx: int, node_idx: int, live_weights: Optional[List]) -> str:
        """Get color for node based on activation/weight"""
        if live_weights and layer_idx < len(live_weights):
            weight = abs(live_weights[layer_idx])
            # Color intensity based on weight
            intensity = min(1.0, weight * 2)
            return plt.cm.RdYlGn(0.3 + intensity * 0.4)  # Green to yellow to red
        return 'lightblue'
    
    def _get_layer_weights(self) -> Optional[List]:
        """Get current layer weights for visualization"""
        try:
            if self.model is None or not HAS_TORCH:
                return None
            
            weights = []
            policy = self.model.policy
            
            # Get feature extractor weights
            if hasattr(policy, 'features_extractor'):
                feat_ext = policy.features_extractor
                if hasattr(feat_ext, 'net'):
                    for module in feat_ext.net.modules():
                        if isinstance(module, torch.nn.Linear):
                            weight_mean = float(module.weight.data.abs().mean().cpu().numpy())
                            weights.append(weight_mean)
            
            # Get MLP weights
            if hasattr(policy, 'mlp_extractor'):
                mlp = policy.mlp_extractor
                for module_list in [mlp.policy_net, mlp.value_net]:
                    if hasattr(module_list, '__iter__'):
                        for module in module_list:
                            if isinstance(module, torch.nn.Linear):
                                weight_mean = float(module.weight.data.abs().mean().cpu().numpy())
                                weights.append(weight_mean)
            
            return weights if weights else None
            
        except Exception as e:
            return None
    
    def _update_metrics_plots(self):
        """Update metrics plots"""
        if not HAS_MATPLOTLIB:
            return
        
        try:
            # Rewards plot
            self.ax_reward.clear()
            if self.stats['rewards']:
                self.ax_reward.plot(self.stats['rewards'], 'b-', linewidth=1)
                self.ax_reward.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                self.ax_reward.set_title(f'Episode Rewards (Avg: {np.mean(self.stats["rewards"][-100:]):.2f})', fontsize=10)
                self.ax_reward.set_xlabel('Episode')
                self.ax_reward.set_ylabel('Reward')
                self.ax_reward.grid(True, alpha=0.3)
            
            # Loss plot
            self.ax_loss.clear()
            if self.stats['losses']:
                self.ax_loss.plot(self.stats['losses'], 'r-', linewidth=1)
                self.ax_loss.set_title('Training Loss', fontsize=10)
                self.ax_loss.set_xlabel('Update')
                self.ax_loss.set_ylabel('Loss')
                self.ax_loss.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def update_stats(self, new_stats: Dict):
        """Update statistics"""
        with self.lock:
            self.stats.update(new_stats)
    
    def _update_display(self):
        """Update display with current stats"""
        if HAS_MATPLOTLIB and self.fig is not None:
            try:
                # Draw neural network
                self._draw_neural_network()
                
                # Update metrics
                self._update_metrics_plots()
                
                # Refresh display
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
            except Exception as e:
                print(f"Error updating display: {e}")
        else:
            # Text-only mode
            self._print_text_stats()
    
    def _print_text_stats(self):
        """Print statistics as text"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 80)
        print("RL TRAINING PROGRESS MONITOR")
        print("=" * 80)
        print(f"\nEpisode: {self.stats['episode']}")
        print(f"Timesteps: {self.stats['timesteps']:,}")
        print(f"Total Trades: {self.stats['total_trades']}")
        print(f"\nNeural Network Architecture:")
        print(f"  Total Parameters: {self.nn_architecture.get('total_parameters', 0):,}")
        print(f"  Layers: {len(self.nn_architecture.get('layer_sizes', []))}")
        for i, layer in enumerate(self.nn_architecture.get('layer_sizes', [])):
            size = layer.get('out_features', layer.get('in_features', 0))
            print(f"    Layer {i+1}: {size} nodes")
        print(f"\nLatest Rewards: {self.stats['rewards'][-10:] if self.stats['rewards'] else 'N/A'}")
        print("=" * 80)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Update stats from state manager
                state = self.state_manager.get_state()
                with self.lock:
                    self.stats['episode'] = state.get('episode', 0)
                    self.stats['timesteps'] = state.get('total_timesteps', 0)
                    self.stats['total_trades'] = state.get('total_trades', 0)
                
                # Try to get training metrics from model
                try:
                    if self.model and hasattr(self.model, 'logger'):
                        # Get recent rewards/losses from logger
                        pass  # Would need to access logger internals
                except:
                    pass
                
                # Update display
                self._update_display()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                time.sleep(self.update_interval)
    
    def start(self):
        """Start monitoring"""
        self.running = True
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if HAS_MATPLOTLIB and self.fig is not None:
            plt.close(self.fig)

