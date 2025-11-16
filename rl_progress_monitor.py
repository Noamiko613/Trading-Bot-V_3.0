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

# No matplotlib - terminal only
HAS_MATPLOTLIB = False


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
    
    # All matplotlib visualization removed - using terminal-only dashboard
    
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
    
    # Metrics plots removed - handled by dashboard
    
    def update_stats(self, new_stats: Dict):
        """Update statistics"""
        with self.lock:
            self.stats.update(new_stats)
    
    def _update_display(self):
        """Update display with current stats - terminal only"""
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

