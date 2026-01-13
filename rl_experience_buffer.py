"""
Experience Replay Buffers for RL Training
==========================================

Implements prioritized experience replay with good/bad episode buffers
to accelerate learning from mistakes and reinforce successful behaviors.
"""

import numpy as np
import json
import os
from collections import deque
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import random


class EpisodeBuffer:
    """Stores a complete episode trajectory"""
    
    def __init__(self, episode_id: int, final_pnl: float, final_equity: float):
        self.episode_id = episode_id
        self.final_pnl = final_pnl
        self.final_equity = final_equity
        self.transitions: List[Dict] = []  # List of (obs, action, reward, next_obs, done) tuples
        
    def add_transition(self, obs, action, reward, next_obs, done, info):
        """Add a single transition to this episode"""
        self.transitions.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'info': info.copy() if info else {}
        })
    
    def get_length(self):
        """Get number of transitions in this episode"""
        return len(self.transitions)


class PrioritizedExperienceBuffer:
    """
    Maintains separate buffers for good and bad episodes.
    Good episodes: top percentile by final PnL
    Bad episodes: bottom percentile by final PnL
    """
    
    def __init__(
        self,
        good_buffer_size: int = 10000,
        bad_buffer_size: int = 10000,
        good_percentile: float = 0.75,  # Top 25% are "good"
        bad_percentile: float = 0.25,   # Bottom 25% are "bad"
        persist_path: Optional[str] = "models/rl_models/experience_buffers.json",
    ):
        self.good_buffer_size = good_buffer_size
        self.bad_buffer_size = bad_buffer_size
        self.good_percentile = good_percentile
        self.bad_percentile = bad_percentile
        self.persist_path = Path(persist_path) if persist_path else None
        
        # FIFO buffers for good and bad episodes
        self.good_episodes: deque = deque(maxlen=good_buffer_size)
        self.bad_episodes: deque = deque(maxlen=bad_buffer_size)
        
        # Temporary storage for current episode being built
        self.current_episode: Optional[EpisodeBuffer] = None
        self.episode_counter = 0
        
        # Track all episode PnLs for percentile calculation
        self.recent_pnls: deque = deque(maxlen=1000)
        
        # Load persisted buffers if they exist
        self._load_buffers()
    
    def start_episode(self):
        """Start a new episode"""
        self.current_episode = EpisodeBuffer(
            episode_id=self.episode_counter,
            final_pnl=0.0,
            final_equity=0.0
        )
        self.episode_counter += 1
    
    def add_transition(self, obs, action, reward, next_obs, done, info):
        """Add a transition to the current episode"""
        if self.current_episode is None:
            self.start_episode()
        self.current_episode.add_transition(obs, action, reward, next_obs, done, info)
    
    def end_episode(self, final_pnl: float, final_equity: float):
        """End the current episode and classify it as good/bad"""
        if self.current_episode is None:
            return
        
        self.current_episode.final_pnl = final_pnl
        self.current_episode.final_equity = final_equity
        
        # Add to recent PnLs for percentile calculation
        self.recent_pnls.append(final_pnl)
        
        # Classify episode
        if len(self.recent_pnls) >= 10:  # Need at least 10 episodes for percentiles
            pnl_threshold_good = np.percentile(list(self.recent_pnls), self.good_percentile * 100)
            pnl_threshold_bad = np.percentile(list(self.recent_pnls), self.bad_percentile * 100)
            
            if final_pnl >= pnl_threshold_good:
                self.good_episodes.append(self.current_episode)
            elif final_pnl <= pnl_threshold_bad:
                self.bad_episodes.append(self.current_episode)
        else:
            # Early episodes: use simple threshold
            if final_pnl > 0:
                self.good_episodes.append(self.current_episode)
            else:
                self.bad_episodes.append(self.current_episode)
        
        self.current_episode = None
    
    def sample_good_episodes(self, n: int) -> List[EpisodeBuffer]:
        """Sample n episodes from good buffer"""
        if len(self.good_episodes) == 0:
            return []
        n = min(n, len(self.good_episodes))
        return random.sample(list(self.good_episodes), n)
    
    def sample_bad_episodes(self, n: int) -> List[EpisodeBuffer]:
        """Sample n episodes from bad buffer"""
        if len(self.bad_episodes) == 0:
            return []
        n = min(n, len(self.bad_episodes))
        return random.sample(list(self.bad_episodes), n)
    
    def sample_transitions_from_good(self, n: int) -> List[Dict]:
        """Sample n transitions from good episodes"""
        episodes = self.sample_good_episodes(max(1, n // 10))  # Sample from multiple episodes
        transitions = []
        for ep in episodes:
            transitions.extend(ep.transitions)
        if len(transitions) > n:
            return random.sample(transitions, n)
        return transitions
    
    def sample_transitions_from_bad(self, n: int) -> List[Dict]:
        """Sample n transitions from bad episodes"""
        episodes = self.sample_bad_episodes(max(1, n // 10))
        transitions = []
        for ep in episodes:
            transitions.extend(ep.transitions)
        if len(transitions) > n:
            return random.sample(transitions, n)
        return transitions
    
    def _load_buffers(self):
        """Load buffers from disk"""
        if self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                    self.episode_counter = data.get('episode_counter', 0)
                    # Note: We don't reload full transitions to save memory
                    # Only reload episode metadata if needed
            except Exception:
                pass
    
    def _save_buffers(self):
        """Save buffer metadata to disk (not full transitions to save space)"""
        if self.persist_path:
            try:
                self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    'episode_counter': self.episode_counter,
                    'good_episodes_count': len(self.good_episodes),
                    'bad_episodes_count': len(self.bad_episodes),
                    'recent_pnls': list(self.recent_pnls)[-100:],  # Save recent PnLs for percentile calc
                }
                with open(self.persist_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass
    
    def get_stats(self) -> Dict:
        """Get statistics about the buffers"""
        stats = {
            'good_episodes': len(self.good_episodes),
            'bad_episodes': len(self.bad_episodes),
            'total_episodes': self.episode_counter,
            'avg_good_pnl': np.mean([ep.final_pnl for ep in self.good_episodes]) if self.good_episodes else 0.0,
            'avg_bad_pnl': np.mean([ep.final_pnl for ep in self.bad_episodes]) if self.bad_episodes else 0.0,
        }
        # Save buffers when getting stats (called periodically)
        self._save_buffers()
        return stats

