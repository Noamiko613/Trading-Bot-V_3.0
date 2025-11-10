"""
Quick Setup Script for Trading Bot Enhancements
================================================

This script sets up all the new enhancements:
- Creates directory structure
- Initializes configuration files
- Sets up logging
- Tests all components
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'utils',
        'examples',
        'config',
        'config/backup',
        'logs',
        'models',
        'analytics',
        'optimization_results'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {directory}/")
    
    print()


def initialize_configuration():
    """Initialize configuration files"""
    print("Initializing configuration files...")
    
    try:
        from utils.config_manager import get_config_manager
        
        config = get_config_manager()
        config.create_default_configs()
        
        print("  ✓ Trading configuration (config/trading.json)")
        print("  ✓ Risk management configuration (config/risk.json)")
        print("  ✓ Indicators configuration (config/indicators.json)")
        print("  ✓ Patterns configuration (config/patterns.json)")
        print()
        
        return True
    except Exception as e:
        print(f"  ✗ Error creating configuration files: {e}")
        return False


def test_logging():
    """Test logging system"""
    print("Testing logging system...")
    
    try:
        from utils.logger import get_logger
        
        logger = get_logger('setup_test')
        logger.info("Logging system initialized successfully")
        logger.debug("Debug message test")
        logger.warning("Warning message test")
        
        # Check if log files were created
        log_files = [
            'logs/setup_test.log',
            'logs/setup_test_errors.log'
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                print(f"  ✓ {log_file} created")
            else:
                print(f"  ⚠ {log_file} not found")
        
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error testing logging: {e}")
        return False


def test_analytics():
    """Test analytics system"""
    print("Testing analytics system...")
    
    try:
        from utils.analytics import PerformanceAnalytics, RealTimeMonitor
        
        analytics = PerformanceAnalytics()
        monitor = RealTimeMonitor(analytics)
        
        print("  ✓ PerformanceAnalytics initialized")
        print("  ✓ RealTimeMonitor initialized")
        print()
        
        return True
    except Exception as e:
        print(f"  ✗ Error testing analytics: {e}")
        return False


def test_ml_scorer():
    """Test ML pattern scorer"""
    print("Testing ML pattern scorer...")
    
    try:
        from utils.ml_pattern_scorer import MLPatternScorer
        
        scorer = MLPatternScorer()
        
        # Test feature extraction
        pattern_data = {
            'pattern': 'Test Pattern',
            'confidence': 50,
            'price': 45000,
            'volume_ratio': 1.0,
            'indicators': {
                'rsi': 50,
                'macd': 0,
                'atr': 500,
                'ma50': 45000,
                'ma200': 44000
            },
            'session': {'type': 'medium_liquidity'},
            'multi_timeframe': {'aligned': False},
            'setup': {'rr': 2.0, 'risk_pct': 1.0}
        }
        
        features = scorer.extract_features(pattern_data)
        
        print("  ✓ MLPatternScorer initialized")
        print(f"  ✓ Extracted {len(features)} features from test pattern")
        
        if scorer.is_trained:
            print("  ✓ Pre-trained model loaded")
        else:
            print("  ℹ No pre-trained model (will train after collecting trades)")
        
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error testing ML scorer: {e}")
        return False


def test_optimizer():
    """Test strategy optimizer"""
    print("Testing strategy optimizer...")
    
    try:
        from utils.strategy_optimizer import StrategyOptimizer, BacktestEngine
        
        optimizer = StrategyOptimizer()
        backtest = BacktestEngine()
        
        print("  ✓ StrategyOptimizer initialized")
        print("  ✓ BacktestEngine initialized")
        print()
        
        return True
    except Exception as e:
        print(f"  ✗ Error testing optimizer: {e}")
        return False


def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib (optional)',
        'plotly': 'plotly (optional)'
    }
    
    missing = []
    
    for package, display_name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - MISSING")
            missing.append(display_name)
    
    print()
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        print()
        return False
    
    return True


def create_example_env():
    """Create example .env file if it doesn't exist"""
    env_file = '.env'
    
    if not os.path.exists(env_file):
        print("Creating example .env file...")
        
        example_env = """# Trading Bot Configuration
# Copy this file and customize for your needs

# Trading Settings
DEFAULT_SYMBOL=BTC-USDT
DEFAULT_MODE=spot
DEFAULT_OPERATING_MODE=hybrid
DEFAULT_MODERATION_MODE=balanced
DEFAULT_TRADING_MODE=balanced

# Live Trading (0=paper, 1=live)
TRADE_LIVE=0

# Exchange API Keys (only needed for live trading)
COINEX_API_KEY=your_api_key_here
COINEX_API_SECRET=your_api_secret_here

# Risk Management Overrides (optional)
# TRADING_BOT_MAX_DRAWDOWN_PCT=20.0
# TRADING_BOT_RISK_PER_TRADE_PCT=1.0

# Accuracy & selectivity (balanced defaults - LAYER 1 & 6)
# Require higher timeframe confirmation for non-daily timeframes (directional alignment only)
REQUIRE_MTF_CONFIRMATION=1
# Minimum confidence score for non-daily signals (adjusted for realism)
MIN_CONFIDENCE=55
# Minimum risk-reward for non-daily signals (adjusted for crypto scalps)
MIN_RR=1.3
# Apply the above filters to daily timeframes? (0=no, 1=yes) - NOW ENABLED
APPLY_FILTERS_TO_DAILY=1
# Maximum risk per trade (% of equity) - 0.3% per trade
MAX_RISK_PCT=0.003
# ML adjustment bound (±5% initially)
ML_NUDGE_LIMIT=0.05
# Base risk per trade (% of equity) - 0.2% default
RISK_PER_TRADE_PCT=0.002
# Account-level protections (LAYER 6)
MAX_CONCURRENT_TRADES=6
MAX_TOTAL_EXPOSURE_PCT=0.02
DAILY_LOSS_LIMIT_PCT=0.03
KILL_SWITCH_HARD_DD=0.25
KILL_SWITCH_SOFT_DD=0.15

# Logging Level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
"""
        
        with open(env_file, 'w') as f:
            f.write(example_env)
        
        print(f"  ✓ Created {env_file}")
        print()
    else:
        print(f"  ℹ {env_file} already exists")
        print()


def display_summary(results):
    """Display setup summary"""
    print("=" * 80)
    print("SETUP SUMMARY")
    print("=" * 80)
    
    all_passed = all(results.values())
    
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{component:<40} {status}")
    
    print("=" * 80)
    
    if all_passed:
        print("\n✅ All enhancements successfully set up!")
        print("\nNext steps:")
        print("  1. Review and customize config files in config/ directory")
        print("  2. Update .env file with your settings")
        print("  3. Run: python examples/integrated_bot_example.py")
        print("  4. Monitor performance: python examples/performance_monitor.py")
        print("  5. Read ENHANCEMENT_GUIDE.md for detailed usage")
    else:
        print("\n⚠️  Some components failed to initialize.")
        print("Please check the errors above and ensure all dependencies are installed.")
        print("Run: pip install -r requirements.txt")
    
    print()


def main():
    """Main setup function"""
    print()
    print("=" * 80)
    print("TRADING BOT ENHANCEMENTS SETUP")
    print("=" * 80)
    print()
    
    results = {}
    
    # Create directories
    create_directory_structure()
    
    # Check dependencies
    results['Dependencies'] = check_dependencies()
    
    # Only continue if dependencies are available
    if not results['Dependencies']:
        print("\n⚠️  Please install missing dependencies before continuing.")
        print("Run: pip install -r requirements.txt")
        print()
        return
    
    # Initialize configuration
    results['Configuration'] = initialize_configuration()
    
    # Test components
    results['Logging'] = test_logging()
    results['Analytics'] = test_analytics()
    results['ML Pattern Scorer'] = test_ml_scorer()
    results['Strategy Optimizer'] = test_optimizer()
    
    # Create example .env
    create_example_env()
    
    # Display summary
    display_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
