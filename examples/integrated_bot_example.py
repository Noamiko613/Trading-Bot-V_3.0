"""
Integrated Trading Bot Example
================================

This example shows how to integrate the new utilities into the existing trading bot.
It demonstrates proper logging, configuration management, and analytics.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import get_logger, ComponentLogger
from utils.config_manager import get_config, get_config_manager
from utils.analytics import PerformanceAnalytics, RealTimeMonitor
from utils.ml_pattern_scorer import MLPatternScorer

# Initialize components
logger = get_logger('integrated_bot')
config_manager = get_config_manager()
analytics = PerformanceAnalytics()
monitor = RealTimeMonitor(analytics)
ml_scorer = MLPatternScorer()


def setup_bot():
    """Initialize bot with enhanced features"""
    logger.info("=" * 80)
    logger.info("TRADING BOT INITIALIZATION")
    logger.info("=" * 80)
    
    # Create default configurations
    config_manager.create_default_configs()
    logger.info("Configuration files created/loaded")
    
    # Load trading parameters
    symbol = get_config('symbol', config_name='trading')
    mode = get_config('mode', config_name='trading')
    operating_mode = get_config('operating_mode', config_name='trading')
    
    logger.info("Trading parameters loaded",
                symbol=symbol,
                mode=mode,
                operating_mode=operating_mode)
    
    # Load risk parameters
    max_drawdown = get_config('max_drawdown_pct', config_name='risk')
    risk_per_trade = get_config('risk_per_trade_pct', config_name='risk')
    
    logger.info("Risk parameters loaded",
                max_drawdown=max_drawdown,
                risk_per_trade=risk_per_trade)
    
    return {
        'symbol': symbol,
        'mode': mode,
        'operating_mode': operating_mode,
        'max_drawdown': max_drawdown,
        'risk_per_trade': risk_per_trade
    }


def process_pattern_with_ml(pattern_data):
    """Process pattern detection with ML enhancement"""
    
    # Get original confidence
    original_confidence = pattern_data.get('confidence', 50)
    
    # Log pattern detection
    logger.log_pattern_detection({
        'pattern': pattern_data.get('pattern'),
        'timeframe': pattern_data.get('timeframe'),
        'confidence': original_confidence,
        'price': pattern_data.get('price')
    })
    
    # Enhance with ML if model is trained
    if ml_scorer.is_trained:
        ml_confidence = ml_scorer.predict_confidence(pattern_data)
        logger.info("ML enhancement applied",
                    original=original_confidence,
                    ml_enhanced=ml_confidence,
                    pattern=pattern_data.get('pattern'))
        return ml_confidence
    else:
        logger.debug("ML model not trained, using original confidence")
        return original_confidence


def execute_trade(setup):
    """Execute trade with enhanced logging"""
    
    # Get risk parameters
    risk_per_trade = get_config('risk_per_trade_pct', config_name='risk')
    
    # Log trade execution
    logger.log_trade({
        'symbol': setup.get('symbol', 'BTC/USDT'),
        'side': setup.get('side', 'BUY'),
        'price': setup.get('entry'),
        'size': setup.get('size'),
        'pattern': setup.get('name'),
        'stop_loss': setup.get('stop'),
        'take_profit': setup.get('tp'),
        'rr_ratio': setup.get('rr'),
        'risk_pct': risk_per_trade
    })
    
    logger.info("Trade executed",
                symbol=setup.get('symbol'),
                side=setup.get('side'),
                entry=setup.get('entry'),
                rr=setup.get('rr'))


def monitor_performance():
    """Monitor performance with real-time alerts"""
    
    logger.info("Running performance checks...")
    
    # Check for alerts
    alerts = []
    
    # Drawdown alert
    max_allowed_drawdown = get_config('max_drawdown_pct', config_name='risk')
    drawdown_alert = monitor.check_drawdown_alert(max_allowed_drawdown)
    if drawdown_alert:
        alerts.append(drawdown_alert)
        logger.warning(drawdown_alert)
    
    # Win rate alert
    win_rate_alert = monitor.check_win_rate_alert(min_trades=10, min_win_rate=40.0)
    if win_rate_alert:
        alerts.append(win_rate_alert)
        logger.warning(win_rate_alert)
    
    # Consecutive losses alert
    loss_streak_alert = monitor.check_consecutive_losses_alert(5)
    if loss_streak_alert:
        alerts.append(loss_streak_alert)
        logger.warning(loss_streak_alert)
    
    # Get recent trades and metrics
    trades = analytics.get_closed_trades(days=7)
    if trades:
        metrics = analytics.calculate_metrics(trades)
        logger.info("Performance metrics (7 days)",
                    total_trades=metrics['total_trades'],
                    win_rate=metrics['win_rate'],
                    total_pnl=metrics['total_pnl'],
                    sharpe_ratio=metrics['sharpe_ratio'])
    
    return alerts


def generate_performance_report():
    """Generate and save performance report"""
    
    logger.info("Generating performance report...")
    
    # Generate report for last 30 days
    report = analytics.generate_report(days=30)
    
    logger.info("Performance report generated")
    print(report)
    
    # Export trades to CSV
    csv_file = analytics.export_to_csv(days=30)
    if csv_file:
        logger.info("Trades exported to CSV", file=csv_file)
    
    return report


def main():
    """Main function demonstrating integrated bot"""
    
    try:
        # Setup bot
        config = setup_bot()
        logger.info("Bot setup complete")
        
        # Example: Process a pattern detection
        example_pattern = {
            'pattern': 'Golden Cross',
            'timeframe': '1h',
            'confidence': 75,
            'price': 45000,
            'volume_ratio': 1.5,
            'indicators': {
                'rsi': 65,
                'macd': 150,
                'macd_signal': 100,
                'atr': 500,
                'ma50': 44000,
                'ma200': 43000,
                'ema5': 45200
            },
            'session': {'type': 'high_liquidity'},
            'multi_timeframe': {'aligned': True},
            'setup': {'rr': 2.5, 'risk_pct': 1.0}
        }
        
        # Process with ML
        enhanced_confidence = process_pattern_with_ml(example_pattern)
        logger.info("Pattern processed", confidence=enhanced_confidence)
        
        # Example: Execute a trade
        example_setup = {
            'symbol': 'BTC/USDT',
            'name': 'Golden Cross',
            'side': 'BUY',
            'entry': 45000,
            'stop': 44000,
            'tp': 47500,
            'rr': 2.5,
            'size': 0.1
        }
        
        execute_trade(example_setup)
        
        # Monitor performance
        alerts = monitor_performance()
        
        # Generate report
        if len(analytics.get_closed_trades(days=30)) > 0:
            generate_performance_report()
        
        logger.info("=" * 80)
        logger.info("Example execution complete")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.exception("Error in main execution", error=str(e))
        raise


if __name__ == "__main__":
    main()
