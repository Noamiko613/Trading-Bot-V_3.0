"""
RL Model Loader - Load and use trained reinforcement learning models in the trading system.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np

from models.rl_trainer import NeuralNetworkRLAgent


class RLTradingModel:
    """
    Wrapper for trained RL models that can be used in the trading system.
    Provides a clean interface for making trading decisions based on RL model predictions.
    """
    
    def __init__(self, symbol: str, model_path: Optional[str] = None):
        """
        Initialize RL trading model
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USDT")
            model_path: Path to trained model file. If None, searches in default locations.
        """
        self.symbol = symbol.upper().replace("/", "-")
        self.model_path = model_path or self._find_model_path()
        self.agent: Optional[NeuralNetworkRLAgent] = None
        self.is_loaded = False
        
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()
    
    def _find_model_path(self) -> Optional[str]:
        """Find trained model in default locations"""
        # Check production directory first
        production_path = os.path.join(
            "models", "rl_production", f"{self.symbol}_rl_model.pkl"
        )
        if os.path.exists(production_path):
            return production_path
        
        # Check checkpoint directory
        checkpoint_path = os.path.join(
            "sim_results", "rl_checkpoints", f"{self.symbol}_model.pkl"
        )
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        
        return None
    
    def _load_model(self):
        """Load the trained RL model"""
        if not self.model_path or not os.path.exists(self.model_path):
            return False
        
        try:
            # Create a dummy agent to load the model
            # We need the actions list - use default
            actions = ["hold", "long", "short"]
            self.agent = NeuralNetworkRLAgent(
                actions=actions,
                learning_rate=0.1,  # Not used for inference
                discount_factor=0.9,  # Not used for inference
                initial_epsilon=0.0,  # No exploration during inference
                epsilon_min=0.0,
                epsilon_decay=1.0,
                model_path=self.model_path,
            )
            
            if self.agent.load_model(self.model_path):
                self.is_loaded = True
                return True
        except Exception as e:
            print(f"[RL Model Loader] Failed to load model for {self.symbol}: {e}")
        
        return False
    
    def predict_action(self, state: Tuple) -> Tuple[str, Dict[str, float]]:
        """
        Predict trading action based on current state
        
        Args:
            state: State tuple (same format as used during training)
        
        Returns:
            Tuple of (action, confidence_scores)
            - action: "hold", "long", or "short"
            - confidence_scores: Dictionary with Q-values for each action
        """
        if not self.is_loaded or not self.agent:
            return "hold", {"hold": 0.0, "long": 0.0, "short": 0.0}
        
        # Get Q-values for all actions
        state_features = self.agent._state_to_features(state)
        confidence_scores = {}
        
        for action in self.agent.actions:
            try:
                q_val = self.agent.models[action].predict(state_features)[0]
                confidence_scores[action] = float(q_val)
            except Exception:
                confidence_scores[action] = 0.0
        
        # Select action with highest Q-value
        best_action = max(confidence_scores, key=confidence_scores.get)
        
        return best_action, confidence_scores
    
    def get_confidence(self, state: Tuple, action: str) -> float:
        """
        Get confidence score for a specific action
        
        Args:
            state: State tuple
            action: Action to get confidence for
        
        Returns:
            Confidence score (Q-value) for the action
        """
        if not self.is_loaded or not self.agent:
            return 0.0
        
        try:
            state_features = self.agent._state_to_features(state)
            if action in self.agent.models:
                q_val = self.agent.models[action].predict(state_features)[0]
                return float(q_val)
        except Exception:
            pass
        
        return 0.0


class RLModelManager:
    """
    Manager for loading and using multiple RL models across different symbols.
    """
    
    def __init__(self):
        self.models: Dict[str, RLTradingModel] = {}
        self.production_dir = os.path.join("models", "rl_production")
        self.checkpoint_dir = os.path.join("sim_results", "rl_checkpoints")
    
    def get_model(self, symbol: str, auto_load: bool = True) -> Optional[RLTradingModel]:
        """
        Get RL model for a symbol
        
        Args:
            symbol: Trading symbol
            auto_load: If True, automatically load model if not already loaded
        
        Returns:
            RLTradingModel instance or None if not available
        """
        symbol = symbol.upper().replace("/", "-")
        
        if symbol in self.models:
            return self.models[symbol]
        
        if auto_load:
            model = RLTradingModel(symbol)
            if model.is_loaded:
                self.models[symbol] = model
                return model
        
        return None
    
    def list_available_models(self) -> List[str]:
        """List all available trained RL models"""
        available = []
        
        # Check production directory
        if os.path.exists(self.production_dir):
            for filename in os.listdir(self.production_dir):
                if filename.endswith("_rl_model.pkl"):
                    symbol = filename.replace("_rl_model.pkl", "")
                    available.append(symbol)
        
        # Check checkpoint directory
        if os.path.exists(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith("_model.pkl"):
                    symbol = filename.replace("_model.pkl", "")
                    if symbol not in available:
                        available.append(symbol)
        
        return available
    
    def is_model_available(self, symbol: str) -> bool:
        """Check if a trained model is available for a symbol"""
        symbol = symbol.upper().replace("/", "-")
        
        # Check if already loaded
        if symbol in self.models and self.models[symbol].is_loaded:
            return True
        
        # Check file system
        production_path = os.path.join(
            self.production_dir, f"{symbol}_rl_model.pkl"
        )
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{symbol}_model.pkl"
        )
        
        return os.path.exists(production_path) or os.path.exists(checkpoint_path)


# Global instance for easy access
_global_manager: Optional[RLModelManager] = None


def get_rl_model_manager() -> RLModelManager:
    """Get global RL model manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = RLModelManager()
    return _global_manager


def get_rl_model(symbol: str) -> Optional[RLTradingModel]:
    """Convenience function to get RL model for a symbol"""
    manager = get_rl_model_manager()
    return manager.get_model(symbol)

