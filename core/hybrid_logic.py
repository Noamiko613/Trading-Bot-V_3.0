"""
Hybrid logic scaffold combining pattern confidence with session confidence.
"""

from typing import Dict, Any


def blend_confidence(pattern_confidence: float, session_confidence: float, pattern_weight: float = 0.6, session_weight: float = 0.4) -> float:
    total = max(1e-9, pattern_weight + session_weight)
    return (pattern_confidence * pattern_weight + session_confidence * session_weight) / total


def decide_by_blend(side: str, pattern_confidence: float, session_confidence: float, threshold: float) -> bool:
    """Return True if blended confidence exceeds threshold."""
    final_conf = blend_confidence(pattern_confidence, session_confidence)
    return final_conf >= threshold


