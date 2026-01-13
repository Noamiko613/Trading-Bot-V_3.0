"""
Behavior Cloning for RL Training
=================================

Implements supervised learning on good episodes to accelerate learning
and negative correction on bad episodes to avoid repeated mistakes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from collections import deque

try:
    from stable_baselines3.common.policies import ActorCriticPolicy
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class BehaviorCloningTrainer:
    """
    Applies behavior cloning (supervised learning) on good episodes
    and negative correction on bad episodes to accelerate RL learning.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 1e-5,
        bc_weight: float = 0.1,  # Weight for BC loss in combined objective
        negative_weight: float = 0.05,  # Weight for negative correction
    ):
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required for behavior cloning")
        
        self.policy = policy
        self.learning_rate = learning_rate
        self.bc_weight = bc_weight
        self.negative_weight = negative_weight
        
        # Optimizer for BC updates
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss() if hasattr(policy, 'action_net') else nn.MSELoss()
    
    def train_on_good_episodes(
        self,
        good_transitions: List[Dict],
        epochs: int = 1
    ) -> float:
        """
        Train policy to imitate actions from good episodes (behavior cloning).
        
        Parameters
        ----------
        good_transitions : List[Dict]
            List of transitions from good episodes, each with 'obs', 'action', etc.
        epochs : int
            Number of training epochs
        
        Returns
        -------
        float
            Average loss over training
        """
        if not good_transitions or len(good_transitions) < 10:
            return 0.0
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            # Shuffle transitions
            np.random.shuffle(good_transitions)
            
            # Mini-batch training
            batch_size = min(32, len(good_transitions))
            for i in range(0, len(good_transitions), batch_size):
                batch = good_transitions[i:i + batch_size]
                
                # Extract observations and actions
                obs_batch = torch.FloatTensor([t['obs'] for t in batch])
                action_batch = torch.LongTensor([t['action'] for t in batch])
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Get policy logits
                if hasattr(self.policy, 'forward'):
                    try:
                        features = self.policy.extract_features(obs_batch) if hasattr(self.policy, 'extract_features') else obs_batch
                        logits, _ = self.policy.forward(features)
                        
                        # Compute loss
                        if len(logits.shape) == 2 and logits.shape[1] > 1:
                            # Discrete actions
                            loss = self.criterion(logits, action_batch)
                        else:
                            # Continuous actions - use MSE
                            loss = nn.MSELoss()(logits.squeeze(), action_batch.float())
                        
                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                        self.optimizer.step()
                        
                        total_loss += loss.item()
                        num_batches += 1
                    except Exception as e:
                        # Skip if policy structure doesn't match
                        pass
        
        return total_loss / max(num_batches, 1)
    
    def train_negative_correction(
        self,
        bad_transitions: List[Dict],
        epochs: int = 1
    ) -> float:
        """
        Train policy to avoid actions from bad episodes (negative correction).
        
        This applies a negative gradient to discourage actions that led to losses.
        """
        if not bad_transitions or len(bad_transitions) < 10:
            return 0.0
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            np.random.shuffle(bad_transitions)
            
            batch_size = min(32, len(bad_transitions))
            for i in range(0, len(bad_transitions), batch_size):
                batch = bad_transitions[i:i + batch_size]
                
                obs_batch = torch.FloatTensor([t['obs'] for t in batch])
                action_batch = torch.LongTensor([t['action'] for t in batch])
                
                self.optimizer.zero_grad()
                
                try:
                    features = self.policy.extract_features(obs_batch) if hasattr(self.policy, 'extract_features') else obs_batch
                    logits, _ = self.policy.forward(features)
                    
                    # Negative correction: reduce probability of bad actions
                    if len(logits.shape) == 2 and logits.shape[1] > 1:
                        # For discrete: encourage different actions
                        probs = torch.softmax(logits, dim=1)
                        # Penalize the action that was taken
                        action_probs = probs.gather(1, action_batch.unsqueeze(1)).squeeze()
                        loss = action_probs.mean()  # Minimize probability of bad actions
                    else:
                        # For continuous: use MSE with opposite target
                        loss = nn.MSELoss()(logits.squeeze(), -action_batch.float())
                    
                    loss = loss * self.negative_weight  # Scale down negative correction
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                except Exception:
                    pass
        
        return total_loss / max(num_batches, 1)

