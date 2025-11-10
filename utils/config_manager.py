"""
Unified Configuration Management System
========================================

Centralized configuration management for the trading bot with:
- Environment variable support
- JSON configuration files
- Configuration validation
- Dynamic reloading
- Default values with override capability
"""

import json
import os
from typing import Any, Dict, Optional
from pathlib import Path
import copy


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files from config directory"""
        config_files = [
            'modes.json',
            'sessions.json',
            'trading.json',
            'risk.json',
            'indicators.json',
            'patterns.json'
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_name = config_file.replace('.json', '')
                    self._configs[config_name] = json.load(f)
    
    def get(self, key: str, default: Any = None, config_name: str = 'trading') -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'risk.max_drawdown')
            default: Default value if key not found
            config_name: Name of the configuration file
        
        Returns:
            Configuration value
        """
        # Check environment variable first (uppercase with prefix)
        env_key = f"TRADING_BOT_{key.replace('.', '_').upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._parse_env_value(env_value)
        
        # Check configuration file
        if config_name not in self._configs:
            return default
        
        # Support dot notation for nested keys
        keys = key.split('.')
        value = self._configs[config_name]
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any, config_name: str = 'trading', persist: bool = False):
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
            config_name: Name of the configuration file
            persist: Whether to save to file
        """
        if config_name not in self._configs:
            self._configs[config_name] = {}
        
        # Support dot notation for nested keys
        keys = key.split('.')
        config = self._configs[config_name]
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        if persist:
            self._save_config(config_name)
    
    def _save_config(self, config_name: str):
        """Save configuration to file"""
        config_path = self.config_dir / f"{config_name}.json"
        with open(config_path, 'w') as f:
            json.dump(self._configs[config_name], f, indent=2)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except:
            pass
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except:
            pass
        
        # Try to parse as boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Return as string
        return value
    
    def get_all(self, config_name: str = 'trading') -> Dict:
        """Get all configuration for a specific config file"""
        return copy.deepcopy(self._configs.get(config_name, {}))
    
    def reload(self):
        """Reload all configurations from files"""
        self._load_all_configs()
    
    def create_default_configs(self):
        """Create default configuration files if they don't exist"""
        
        # Trading configuration
        if not (self.config_dir / 'trading.json').exists():
            trading_config = {
                "symbol": "BTC-USDT",
                "mode": "spot",
                "timeframes": ["1min", "5min", "15min", "30min", "1h", "4h"],
                "operating_mode": "hybrid",
                "moderation_mode": "balanced",
                "trading_mode": "balanced",
                "paper_trading": True,
                "leverage": 20,
                "margin_mode": "isolated"
            }
            self._configs['trading'] = trading_config
            self._save_config('trading')
        
        # Risk configuration
        if not (self.config_dir / 'risk.json').exists():
            risk_config = {
                "max_drawdown_pct": 20.0,
                "risk_per_trade_pct": 1.0,
                "max_concurrent_trades": 5,
                "max_daily_trades": 20,
                "max_daily_loss_pct": 5.0,
                "stop_loss": {
                    "atr_multiplier": 2.0,
                    "min_pct": 0.5,
                    "max_pct": 3.0
                },
                "take_profit": {
                    "min_rr_ratio": 1.5,
                    "target_rr_ratio": 2.0,
                    "max_rr_ratio": 4.0
                },
                "position_sizing": {
                    "method": "risk_based",  # risk_based, fixed_pct, kelly
                    "kelly_fraction": 0.25,
                    "max_position_pct": 10.0
                }
            }
            self._configs['risk'] = risk_config
            self._save_config('risk')
        
        # Indicators configuration
        if not (self.config_dir / 'indicators.json').exists():
            indicators_config = {
                "ma": {
                    "short_period": 50,
                    "long_period": 200
                },
                "ema": {
                    "periods": [5, 8, 13, 21, 50, 200]
                },
                "rsi": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                },
                "macd": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                },
                "atr": {
                    "period": 14
                },
                "bollinger_bands": {
                    "period": 20,
                    "std_dev": 2.0
                },
                "volume": {
                    "sma_period": 20,
                    "threshold_factor": 0.6
                }
            }
            self._configs['indicators'] = indicators_config
            self._save_config('indicators')
        
        # Patterns configuration
        if not (self.config_dir / 'patterns.json').exists():
            patterns_config = {
                "confidence_thresholds": {
                    "strict": {
                        "buy": 55,
                        "sell": 55
                    },
                    "balanced": {
                        "buy": 35,
                        "sell": 35
                    },
                    "aggressive": {
                        "buy": 25,
                        "sell": 25
                    }
                },
                "pattern_weights": {
                    "golden_cross": 1.2,
                    "death_cross": 1.2,
                    "macd_cross": 1.0,
                    "rsi_divergence": 1.3,
                    "head_shoulders": 1.1,
                    "double_top_bottom": 1.1,
                    "bull_flag": 1.0,
                    "triangle_breakout": 1.0,
                    "bullish_engulfing": 0.9,
                    "hammer": 0.8,
                    "confluence_engine": 1.4
                },
                "pattern_enabled": {
                    "golden_cross": True,
                    "death_cross": True,
                    "macd_cross": True,
                    "rsi_divergence": True,
                    "head_shoulders": True,
                    "double_top_bottom": True,
                    "bull_flag": True,
                    "triangle_breakout": True,
                    "bullish_engulfing": True,
                    "hammer": True,
                    "confluence_engine": True,
                    "scalp_breakout": True,
                    "ema_pullback": True,
                    "vwap_bounce": True
                },
                "multi_timeframe": {
                    "enabled": True,
                    "confirmation_required": True,
                    "min_timeframes": 2
                }
            }
            self._configs['patterns'] = patterns_config
            self._save_config('patterns')


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.create_default_configs()
    return _config_manager


def get_config(key: str, default: Any = None, config_name: str = 'trading') -> Any:
    """
    Convenience function to get configuration value
    
    Args:
        key: Configuration key
        default: Default value if key not found
        config_name: Name of the configuration file
    
    Returns:
        Configuration value
    """
    return get_config_manager().get(key, default, config_name)


def set_config(key: str, value: Any, config_name: str = 'trading', persist: bool = False):
    """
    Convenience function to set configuration value
    
    Args:
        key: Configuration key
        value: Configuration value
        config_name: Name of the configuration file
        persist: Whether to save to file
    """
    get_config_manager().set(key, value, config_name, persist)


if __name__ == "__main__":
    # Test configuration manager
    config = get_config_manager()
    config.create_default_configs()
    
    print("Trading mode:", get_config('mode', config_name='trading'))
    print("Max drawdown:", get_config('max_drawdown_pct', config_name='risk'))
    print("RSI period:", get_config('rsi.period', config_name='indicators'))
    
    # Test setting config
    set_config('test_value', 123, persist=False)
    print("Test value:", get_config('test_value'))
    
    print("\nConfiguration test complete. Check config/ directory for files.")
