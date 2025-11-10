"""
Trading Bot Utilities Package
==============================

Comprehensive utilities for:
- Professional logging
- Configuration management
- Performance analytics
- ML-based pattern scoring
- Strategy optimization
"""

from .logger import (
    TradingLogger,
    ComponentLogger,
    PerformanceLogger,
    get_logger
)

from .config_manager import (
    ConfigManager,
    get_config_manager,
    get_config,
    set_config
)

from .analytics import (
    PerformanceAnalytics,
    RealTimeMonitor
)

from .ml_pattern_scorer import (
    MLPatternScorer,
    PatternDataCollector
)

from .strategy_optimizer import (
    StrategyOptimizer,
    BacktestEngine
)

__all__ = [
    # Logging
    'TradingLogger',
    'ComponentLogger',
    'PerformanceLogger',
    'get_logger',
    
    # Configuration
    'ConfigManager',
    'get_config_manager',
    'get_config',
    'set_config',
    
    # Analytics
    'PerformanceAnalytics',
    'RealTimeMonitor',
    
    # ML
    'MLPatternScorer',
    'PatternDataCollector',
    
    # Optimization
    'StrategyOptimizer',
    'BacktestEngine'
]

__version__ = '2.0.0'
