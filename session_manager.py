"""
Session Manager for Trading Bot

Implements session awareness for cryptocurrency trading based on traditional finance sessions.
Even though crypto trades 24/7, liquidity and volatility follow traditional finance patterns.

Session Types:
- High Liquidity: US Open (13:30-15:00 UTC) and London/US Overlap (12:00-16:00 UTC)
- Medium Liquidity: Asia Open (23:00-02:00 UTC)
- Low Liquidity: Dead hours (late Asia session, Sunday night)

Trading Modes:
- Safe: Trade only US/London sessions, avoid first 30m of US open, skip quiet hours
- Balanced: Trade US/London, allow Asia trades only with multiple confirmations
- Aggressive: Trade everything, but scale down size in weak sessions
"""

from datetime import datetime, timezone, time, timedelta
from typing import Dict, Tuple, Optional
from enum import Enum


class SessionType(Enum):
    """Trading session types based on liquidity and volatility"""
    HIGH_LIQUIDITY = "high_liquidity"      # US Open & London/US Overlap
    MEDIUM_LIQUIDITY = "medium_liquidity"  # Asia Open
    LOW_LIQUIDITY = "low_liquidity"        # Dead hours
    WEEKEND = "weekend"                    # Sunday night


class TradingMode(Enum):
    """Trading mode configurations"""
    SAFE = "safe"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class SessionManager:
    """Manages session awareness and risk adjustments for trading"""
    
    def __init__(self, trading_mode: str = "balanced"):
        self.trading_mode = TradingMode(trading_mode.lower())
        
        # Session time definitions (UTC)
        self.session_times = {
            # High liquidity sessions
            "us_open": (time(13, 30), time(15, 0)),           # US Open
            "london_us_overlap": (time(12, 0), time(16, 0)),  # London/US Overlap
            
            # Medium liquidity sessions  
            "asia_open": (time(23, 0), time(2, 0)),           # Asia Open (crosses midnight)
            
            # Low liquidity periods
            "asia_late": (time(2, 0), time(8, 0)),            # Late Asia session
            "us_premarket": (time(8, 0), time(13, 30)),       # US Pre-market
            "us_afterhours": (time(16, 0), time(23, 0)),      # US After hours
        }
        
        # LAYER 5: Load session multipliers and adjustments from config
        try:
            import json
            import os
            config_path = os.path.join('config', 'sessions.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    session_mult = config.get('session_multipliers', {})
                    pos_mult = config.get('position_multipliers', {})
                    conf_adj = config.get('confidence_adjustments', {})
                    conf_thresh = config.get('confidence_thresholds', {})
                    
                    # Update risk factors (position multipliers)
                    self.risk_factors = {
                        SessionType.HIGH_LIQUIDITY: pos_mult.get('high_liquidity', 1.0),
                        SessionType.MEDIUM_LIQUIDITY: pos_mult.get('medium_liquidity', 0.9),
                        SessionType.LOW_LIQUIDITY: pos_mult.get('low_liquidity', 0.7),
                        SessionType.WEEKEND: pos_mult.get('weekend', 0.5),
                    }
                    
                    # Store session multipliers for confidence adjustment
                    self.session_multipliers = {
                        SessionType.HIGH_LIQUIDITY: session_mult.get('high_liquidity', 1.0),
                        SessionType.MEDIUM_LIQUIDITY: session_mult.get('medium_liquidity', 0.95),
                        SessionType.LOW_LIQUIDITY: session_mult.get('low_liquidity', 0.92),
                        SessionType.WEEKEND: session_mult.get('weekend', 0.9),
                    }
                    
                    # Store confidence adjustments
                    self.confidence_adjustments = {
                        SessionType.HIGH_LIQUIDITY: conf_adj.get('high_liquidity', 0),
                        SessionType.MEDIUM_LIQUIDITY: conf_adj.get('medium_liquidity', 5),
                        SessionType.LOW_LIQUIDITY: conf_adj.get('low_liquidity', 8),
                        SessionType.WEEKEND: conf_adj.get('weekend', 10),
                    }
                    
                    # Update confidence thresholds
                    self.confidence_thresholds = {
                        SessionType.HIGH_LIQUIDITY: conf_thresh.get('high_liquidity', 40),
                        SessionType.MEDIUM_LIQUIDITY: conf_thresh.get('medium_liquidity', 45),
                        SessionType.LOW_LIQUIDITY: conf_thresh.get('low_liquidity', 50),
                        SessionType.WEEKEND: conf_thresh.get('weekend', 55),
                    }
            else:
                # Defaults if config not found
                self.risk_factors = {
                    SessionType.HIGH_LIQUIDITY: 1.0,
                    SessionType.MEDIUM_LIQUIDITY: 0.9,
                    SessionType.LOW_LIQUIDITY: 0.7,
                    SessionType.WEEKEND: 0.5,
                }
                self.session_multipliers = {
                    SessionType.HIGH_LIQUIDITY: 1.0,
                    SessionType.MEDIUM_LIQUIDITY: 0.95,
                    SessionType.LOW_LIQUIDITY: 0.92,
                    SessionType.WEEKEND: 0.9,
                }
                self.confidence_adjustments = {
                    SessionType.HIGH_LIQUIDITY: 0,
                    SessionType.MEDIUM_LIQUIDITY: 5,
                    SessionType.LOW_LIQUIDITY: 8,
                    SessionType.WEEKEND: 10,
                }
                self.confidence_thresholds = {
                    SessionType.HIGH_LIQUIDITY: 40,
                    SessionType.MEDIUM_LIQUIDITY: 45,
                    SessionType.LOW_LIQUIDITY: 50,
                    SessionType.WEEKEND: 55,
                }
        except Exception:
            # Fallback defaults
            self.risk_factors = {
                SessionType.HIGH_LIQUIDITY: 1.0,
                SessionType.MEDIUM_LIQUIDITY: 0.9,
                SessionType.LOW_LIQUIDITY: 0.7,
                SessionType.WEEKEND: 0.5,
            }
            self.session_multipliers = {
                SessionType.HIGH_LIQUIDITY: 1.0,
                SessionType.MEDIUM_LIQUIDITY: 0.95,
                SessionType.LOW_LIQUIDITY: 0.92,
                SessionType.WEEKEND: 0.9,
            }
            self.confidence_adjustments = {
                SessionType.HIGH_LIQUIDITY: 0,
                SessionType.MEDIUM_LIQUIDITY: 5,
                SessionType.LOW_LIQUIDITY: 8,
                SessionType.WEEKEND: 10,
            }
            self.confidence_thresholds = {
                SessionType.HIGH_LIQUIDITY: 40,
                SessionType.MEDIUM_LIQUIDITY: 45,
                SessionType.LOW_LIQUIDITY: 50,
                SessionType.WEEKEND: 55,
            }

    def get_current_session(self, utc_time: Optional[datetime] = None) -> SessionType:
        """Determine current trading session type"""
        if utc_time is None:
            utc_time = datetime.now(timezone.utc)
        
        # OPTIMIZED: Cryptocurrency markets trade 24/7, including weekends
        # Only treat Saturday night (very early Sunday) as low liquidity
        if utc_time.weekday() == 6 and utc_time.hour < 6:  # Sunday before 6 AM UTC
            return SessionType.WEEKEND
        
        current_time = utc_time.time()
        
        # Check high liquidity sessions first
        if self._is_time_in_range(current_time, self.session_times["us_open"]):
            return SessionType.HIGH_LIQUIDITY
        if self._is_time_in_range(current_time, self.session_times["london_us_overlap"]):
            return SessionType.HIGH_LIQUIDITY
            
        # Check medium liquidity sessions
        if self._is_time_in_range(current_time, self.session_times["asia_open"]):
            return SessionType.MEDIUM_LIQUIDITY
            
        # Check specific low liquidity sessions
        if self._is_time_in_range(current_time, self.session_times["asia_late"]):
            return SessionType.LOW_LIQUIDITY
        if self._is_time_in_range(current_time, self.session_times["us_premarket"]):
            return SessionType.LOW_LIQUIDITY
        if self._is_time_in_range(current_time, self.session_times["us_afterhours"]):
            return SessionType.LOW_LIQUIDITY
            
        # Everything else is low liquidity
        return SessionType.LOW_LIQUIDITY

    def _is_time_in_range(self, current_time: time, time_range: Tuple[time, time]) -> bool:
        """Check if current time is within a time range (handles midnight crossing)"""
        start_time, end_time = time_range
        
        if start_time <= end_time:
            # Normal range (doesn't cross midnight)
            return start_time <= current_time <= end_time
        else:
            # Range crosses midnight (e.g., 23:00 to 02:00)
            return current_time >= start_time or current_time <= end_time

    def should_trade(self, confidence: int, pattern_name: str = "", 
                    has_confluence: bool = False, has_volume_confirmation: bool = False) -> Tuple[bool, str]:
        """
        Determine if a trade should be executed based on session and trading mode
        
        Returns:
            Tuple[bool, str]: (should_trade, reason)
        """
        session = self.get_current_session()
        
        # Get session-specific requirements
        required_confidence = self.confidence_thresholds[session]
        
        # Check basic confidence threshold
        if confidence < required_confidence:
            return False, f"Confidence {confidence}% below session threshold {required_confidence}%"
        
        # Apply mode-specific rules
        if self.trading_mode == TradingMode.SAFE:
            return self._safe_mode_rules(session, confidence, pattern_name, has_confluence, has_volume_confirmation)
        elif self.trading_mode == TradingMode.BALANCED:
            return self._balanced_mode_rules(session, confidence, pattern_name, has_confluence, has_volume_confirmation)
        elif self.trading_mode == TradingMode.AGGRESSIVE:
            return self._aggressive_mode_rules(session, confidence, pattern_name, has_confluence, has_volume_confirmation)
        
        return False, "Unknown trading mode"

    def _safe_mode_rules(self, session: SessionType, confidence: int, pattern_name: str, 
                        has_confluence: bool, has_volume_confirmation: bool) -> Tuple[bool, str]:
        """Safe mode: Trade only US/London sessions, avoid first 30m of US open, skip quiet hours"""
        
        if session == SessionType.HIGH_LIQUIDITY:
            # Check if we're in the first 30 minutes of US open (avoid volatility)
            utc_now = datetime.now(timezone.utc)
            if self._is_in_us_open_first_30min(utc_now):
                return False, "Avoiding first 30 minutes of US open (high volatility)"
            
            # High liquidity sessions - allow trades with standard requirements
            if confidence >= 70 and has_volume_confirmation:
                return True, f"Safe mode: High liquidity session, confidence {confidence}%"
            else:
                return False, f"Safe mode: Missing volume confirmation or low confidence"
        
        elif session == SessionType.MEDIUM_LIQUIDITY:
            # Asia session - require extra confirmation
            if confidence >= 85 and has_confluence and has_volume_confirmation:
                return True, f"Safe mode: Asia session with confluence, confidence {confidence}%"
            else:
                return False, "Safe mode: Asia session requires confluence + volume confirmation"
        
        else:
            # Low liquidity and weekend - skip
            return False, f"Safe mode: Skipping {session.value} session"

    def _balanced_mode_rules(self, session: SessionType, confidence: int, pattern_name: str,
                           has_confluence: bool, has_volume_confirmation: bool) -> Tuple[bool, str]:
        """Balanced mode: Trade US/London, allow Asia trades with multiple confirmations"""
        
        if session == SessionType.HIGH_LIQUIDITY:
            # High liquidity sessions - relaxed to session threshold with volume OR confluence
            req = self.confidence_thresholds[session]
            if confidence >= req and (has_volume_confirmation or has_confluence):
                return True, f"Balanced mode: High liquidity session, confidence {confidence}%"
            return False, "Balanced mode: Needs volume or confluence in high-liquidity"
        
        elif session == SessionType.MEDIUM_LIQUIDITY:
            # Asia session - require multiple confirmations (OPTIMIZED)
            confirmations = 0
            if has_confluence:
                confirmations += 1
            if has_volume_confirmation:
                confirmations += 1
            if confidence >= 60:  # Reduced from 80 to 60
                confirmations += 1
            
            if confirmations >= 1:  # Reduced from 2 to 1 for more trades
                return True, f"Balanced mode: Asia session with {confirmations} confirmations, confidence {confidence}%"
            else:
                return False, f"Balanced mode: Asia session needs 1+ confirmations (has {confirmations})"
        
        elif session == SessionType.LOW_LIQUIDITY:
            # Low liquidity - slightly conservative but allow with confirmation
            req = self.confidence_thresholds[session]
            if confidence >= req and (has_confluence or has_volume_confirmation):
                return True, f"Balanced mode: Low liquidity with confirmation, confidence {confidence}%"
            return False, "Balanced mode: Low liquidity needs confluence or volume"
        
        else:  # WEEKEND
            return False, f"Balanced mode: Skipping {session.value} session"

    def _aggressive_mode_rules(self, session: SessionType, confidence: int, pattern_name: str,
                             has_confluence: bool, has_volume_confirmation: bool) -> Tuple[bool, str]:
        """Aggressive mode: Trade everything, but scale down size in weak sessions"""
        
        if session == SessionType.HIGH_LIQUIDITY:
            req = max(60, self.confidence_thresholds[session])
            if confidence >= req:
                return True, f"Aggressive mode: High liquidity session, confidence {confidence}%"
        
        elif session == SessionType.MEDIUM_LIQUIDITY:
            if confidence >= 58:  # Slightly relaxed
                return True, f"Aggressive mode: Asia session, confidence {confidence}%"
        
        elif session == SessionType.LOW_LIQUIDITY:
            if confidence >= 65:
                return True, f"Aggressive mode: Low liquidity session, confidence {confidence}%"
        
        elif session == SessionType.WEEKEND:
            # Weekend - OPTIMIZED for more trades
            if confidence >= 75:
                return True, f"Aggressive mode: Weekend session, confidence {confidence}%"
        
        return False, f"Aggressive mode: Insufficient conditions for {session.value} session"

    def _is_in_us_open_first_30min(self, utc_time: datetime) -> bool:
        """Check if current time is within first 30 minutes of US open"""
        us_open_start = time(13, 30)
        us_open_first_30min_end = time(14, 0)
        
        current_time = utc_time.time()
        return us_open_start <= current_time <= us_open_first_30min_end

    def get_position_size_multiplier(self, base_size: float) -> float:
        """Get position size multiplier based on current session"""
        session = self.get_current_session()
        multiplier = self.risk_factors[session]
        
        # Apply additional scaling based on trading mode
        if self.trading_mode == TradingMode.SAFE:
            multiplier *= 0.8  # Conservative sizing
        elif self.trading_mode == TradingMode.AGGRESSIVE:
            multiplier *= 1.2  # More aggressive sizing (capped by risk factors)
            multiplier = min(multiplier, 1.0)  # Never exceed 100%
        
        return base_size * multiplier

    def get_stop_loss_multiplier(self) -> float:
        """Get stop loss multiplier based on current session (tighter stops in low liquidity)"""
        session = self.get_current_session()
        
        if session == SessionType.HIGH_LIQUIDITY:
            return 1.0  # Standard stops
        elif session == SessionType.MEDIUM_LIQUIDITY:
            return 0.9  # Slightly tighter stops
        elif session == SessionType.LOW_LIQUIDITY:
            return 0.8  # Tighter stops
        else:  # WEEKEND
            return 0.7  # Very tight stops

    def apply_session_confidence_adjustment(self, confidence: float) -> float:
        """
        LAYER 5: Apply session multiplier to confidence.
        effective_conf = confidence_pct * session_mult[session_type]
        """
        session = self.get_current_session()
        multiplier = self.session_multipliers.get(session, 1.0)
        adjusted = confidence * multiplier
        return max(0, min(100, adjusted))
    
    def get_session_info(self) -> Dict[str, any]:
        """Get current session information for logging and monitoring"""
        session = self.get_current_session()
        utc_now = datetime.now(timezone.utc)
        
        # Get detailed session status
        session_status = self._get_detailed_session_status(utc_now)
        
        return {
            "session_type": session.value,
            "trading_mode": self.trading_mode.value,
            "utc_time": utc_now.isoformat(),
            "position_size_multiplier": self.risk_factors[session],
            "stop_loss_multiplier": self.get_stop_loss_multiplier(),
            "confidence_threshold": self.confidence_thresholds[session],
            "is_us_open_first_30min": self._is_in_us_open_first_30min(utc_now),
            "session_status": session_status,
        }

    def _get_detailed_session_status(self, utc_time: datetime) -> Dict[str, any]:
        """Get detailed information about all sessions and their current status"""
        current_time = utc_time.time()
        
        sessions_status = {}
        
        # Check each session
        for session_name, (start_time, end_time) in self.session_times.items():
            is_active = self._is_time_in_range(current_time, (start_time, end_time))
            
            # Calculate time until start/end
            if is_active:
                # Session is active - calculate time until end
                if start_time <= end_time:
                    # Normal session (doesn't cross midnight)
                    end_datetime = utc_time.replace(hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0)
                    if end_datetime <= utc_time:
                        end_datetime += timedelta(days=1)
                    time_until_end = (end_datetime - utc_time).total_seconds()
                    status_text = f"Active (ends in {int(time_until_end//60)}m {int(time_until_end%60)}s)"
                else:
                    # Session crosses midnight
                    if current_time >= start_time:
                        # We're in the first part (before midnight)
                        end_datetime = utc_time.replace(hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0) + timedelta(days=1)
                    else:
                        # We're in the second part (after midnight)
                        end_datetime = utc_time.replace(hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0)
                    time_until_end = (end_datetime - utc_time).total_seconds()
                    status_text = f"Active (ends in {int(time_until_end//60)}m {int(time_until_end%60)}s)"
            else:
                # Session is not active - calculate time until start
                if start_time <= end_time:
                    # Normal session
                    start_datetime = utc_time.replace(hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0)
                    if start_datetime <= utc_time:
                        start_datetime += timedelta(days=1)
                    time_until_start = (start_datetime - utc_time).total_seconds()
                    status_text = f"Inactive (starts in {int(time_until_start//3600)}h {int((time_until_start%3600)//60)}m)"
                else:
                    # Session crosses midnight
                    if current_time < end_time:
                        # We're before the session (in the gap)
                        start_datetime = utc_time.replace(hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0)
                        if start_datetime <= utc_time:
                            start_datetime += timedelta(days=1)
                    else:
                        # We're after the session (in the gap)
                        start_datetime = utc_time.replace(hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0) + timedelta(days=1)
                    time_until_start = (start_datetime - utc_time).total_seconds()
                    status_text = f"Inactive (starts in {int(time_until_start//3600)}h {int((time_until_start%3600)//60)}m)"
            
            # Determine session category
            if session_name in ["us_open", "london_us_overlap"]:
                category = "High Liquidity"
            elif session_name == "asia_open":
                category = "Medium Liquidity"
            else:
                category = "Low Liquidity"
            
            sessions_status[session_name] = {
                "name": session_name.replace("_", " ").title(),
                "category": category,
                "start_time": start_time.strftime("%H:%M"),
                "end_time": end_time.strftime("%H:%M"),
                "is_active": is_active,
                "status_text": status_text
            }
        
        return sessions_status

    def log_session_decision(self, pattern_name: str, confidence: int, 
                           has_confluence: bool, has_volume_confirmation: bool) -> str:
        """Generate a detailed log message for session-based trading decisions"""
        session_info = self.get_session_info()
        should_trade, reason = self.should_trade(confidence, pattern_name, has_confluence, has_volume_confirmation)
        
        log_msg = (
            f"Session Decision: {session_info['session_type']} | "
            f"Mode: {session_info['trading_mode']} | "
            f"Pattern: {pattern_name} | "
            f"Confidence: {confidence}% | "
            f"Confluence: {has_confluence} | "
            f"Volume: {has_volume_confirmation} | "
            f"Decision: {'TRADE' if should_trade else 'SKIP'} | "
            f"Reason: {reason}"
        )
        
        return log_msg

    def get_session_summary(self) -> str:
        """Get a concise summary of current session status for display"""
        session_info = self.get_session_info()
        session_type = session_info['session_type']
        trading_mode = session_info['trading_mode']
        confidence_threshold = session_info['confidence_threshold']
        position_multiplier = session_info['position_size_multiplier']
        
        # Find active sessions
        active_sessions = []
        for session_name, session_data in session_info['session_status'].items():
            if session_data['is_active']:
                active_sessions.append(session_data['name'])
        
        if active_sessions:
            active_text = f"Active: {', '.join(active_sessions)}"
        else:
            active_text = "No active sessions"
        
        return (f"{session_type.upper()} | {trading_mode.upper()} | "
                f"Conf: {confidence_threshold}% | Pos: {position_multiplier:.1f}x | {active_text}")


# Example usage and testing
if __name__ == "__main__":
    # Test different trading modes
    for mode in ["safe", "balanced", "aggressive"]:
        print(f"\n=== Testing {mode.upper()} mode ===")
        session_mgr = SessionManager(mode)
        
        # Test different scenarios
        test_cases = [
            (85, "Golden Cross", True, True),   # High confidence with confluence
            (75, "MACD Crossover", False, True), # Medium confidence, no confluence
            (65, "RSI Divergence", True, False), # Low confidence, no volume
            (95, "Confluence Engine", True, True), # Very high confidence
        ]
        
        for confidence, pattern, confluence, volume in test_cases:
            should_trade, reason = session_mgr.should_trade(confidence, pattern, confluence, volume)
            print(f"  {pattern}: {confidence}% conf, confluence={confluence}, volume={volume} -> {'TRADE' if should_trade else 'SKIP'}")
            print(f"    Reason: {reason}")
        
        # Show session info
        session_info = session_mgr.get_session_info()
        print(f"  Current session: {session_info['session_type']}")
        print(f"  Position multiplier: {session_info['position_size_multiplier']}")
        print(f"  Confidence threshold: {session_info['confidence_threshold']}%")
