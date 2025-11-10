"""
Fully Automated Multi-Symbol Trading System
============================================

Automatically:
- Loads symbols from symbols_config.json
- Trains ML model if enough data exists
- Monitors and trades all configured symbols
- Handles everything without manual intervention
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import List, Dict
from dotenv import load_dotenv

# Load .env file with override to ensure paper trading mode is respected
load_dotenv(override=True)

# Import utilities
from utils.logger import get_logger, ComponentLogger
from utils.config_manager import get_config_manager, get_config
from utils.analytics import PerformanceAnalytics, RealTimeMonitor
from utils.ml_pattern_scorer import MLPatternScorer, PatternDataCollector

# Import existing bot components
from bot_manager import BotManager
from verify_patterns import watch_unverified

# Initialize logger
logger = get_logger('auto_trader')


class AutoTrader:
    """Automated multi-symbol trading system"""
    
    def __init__(self, config_file: str = "symbols_config.json"):
        """
        Initialize automated trader
        
        Args:
            config_file: Path to symbols configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.bot_managers = []
        self.analytics = PerformanceAnalytics()
        self.monitor = RealTimeMonitor(self.analytics)
        self.ml_scorer = MLPatternScorer()
        
        # Setup
        self._initialize_system()
    
    def _load_config(self) -> Dict:
        """Load symbols configuration from JSON file"""
        if not os.path.exists(self.config_file):
            logger.error(f"Configuration file not found: {self.config_file}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Configuration loaded from {self.config_file}")
        return config
    
    def _initialize_system(self):
        """Initialize the trading system"""
        logger.info("=" * 80)
        logger.info("AUTOMATED TRADING SYSTEM INITIALIZATION")
        logger.info("=" * 80)
        
        # Initialize configuration manager
        config_manager = get_config_manager()
        config_manager.create_default_configs()
        logger.info("Configuration system initialized")
        
        # Auto-train ML model if enabled
        if self.config['trading_settings'].get('auto_train_ml', True):
            self._auto_train_ml()
        
        logger.info("System initialization complete")
        logger.info("=" * 80)
    
    def _auto_train_ml(self):
        """Automatically train ML model if enough data exists"""
        logger.info("Checking ML model training status...")
        
        if self.ml_scorer.is_trained:
            logger.info("ML model already trained and loaded")
            return
        
        # Check if we have enough trades
        min_trades = self.config['trading_settings'].get('min_trades_for_ml', 50)
        collector = PatternDataCollector()
        
        try:
            training_data = collector.collect_from_trades()
            
            if len(training_data) >= min_trades:
                logger.info(f"Found {len(training_data)} trades. Training ML model...")
                self.ml_scorer.train(training_data)
                logger.info("ML model training complete")
            else:
                logger.info(f"Not enough trades for ML training ({len(training_data)}/{min_trades})")
                logger.info("ML model will train automatically after collecting more trades")
        except Exception as e:
            logger.warning(f"Could not train ML model: {e}")
            logger.info("ML model will train once trade data is available")
    
    def _get_enabled_symbols(self) -> List[Dict]:
        """Get list of enabled symbols from configuration"""
        enabled = [s for s in self.config['symbols'] if s.get('enabled', True)]
        logger.info(f"Found {len(enabled)} enabled symbols: {[s['symbol'] for s in enabled]}")
        return enabled
    
    def _start_symbol_bot(self, symbol_config: Dict):
        """Start bot for a specific symbol"""
        symbol = symbol_config['symbol']
        timeframes = symbol_config['timeframes']
        
        logger.info(f"Starting bot for {symbol} with timeframes: {timeframes}")
        
        try:
            # Get trading settings
            settings = self.config['trading_settings']
            
            # Initialize bot manager for this symbol
            bot_manager = BotManager(
                symbol=symbol,
                mode=get_config('mode', default='spot', config_name='trading'),
                timeframes=timeframes,
                operating_mode=settings.get('operating_mode', 'hybrid'),
                moderation_mode=settings.get('moderation_mode', 'balanced'),
                trading_mode=settings.get('trading_mode', 'balanced')
                # Note: paper trading is controlled by TRADE_LIVE environment variable
            )
            
            self.bot_managers.append({
                'symbol': symbol,
                'manager': bot_manager,
                'config': symbol_config
            })
            
            # Start bot in separate thread
            bot_thread = Thread(target=bot_manager.run, daemon=True)
            bot_thread.start()
            
            logger.info(f"Bot started for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to start bot for {symbol}: {e}")
    
    def _start_pattern_verification(self):
        """Start pattern verification threads for all symbols"""
        logger.info("Starting pattern verification system...")
        
        try:
            # Start verification thread for each enabled symbol
            for bot_info in self.bot_managers:
                symbol = bot_info['symbol']
                verification_thread = Thread(
                    target=watch_unverified, 
                    args=(symbol,),
                    daemon=True,
                    name=f"verification-{symbol}"
                )
                verification_thread.start()
                logger.info(f"Pattern verification started for {symbol}")
            
            logger.info("Pattern verification system started for all symbols")
        except Exception as e:
            logger.error(f"Failed to start pattern verification: {e}")
    
    def _monitor_performance(self):
        """Monitor performance and send alerts"""
        logger.info("Starting performance monitoring...")
        
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                
                # Check alerts
                alerts = []
                
                max_drawdown = get_config('max_drawdown_pct', default=20.0, config_name='risk')
                dd_alert = self.monitor.check_drawdown_alert(max_drawdown)
                if dd_alert:
                    alerts.append(dd_alert)
                
                wr_alert = self.monitor.check_win_rate_alert(min_trades=10, min_win_rate=40.0)
                if wr_alert:
                    alerts.append(wr_alert)
                
                loss_alert = self.monitor.check_consecutive_losses_alert(5)
                if loss_alert:
                    alerts.append(loss_alert)
                
                # Log alerts
                for alert in alerts:
                    logger.warning(alert)
                
                # Check if we should train ML model
                if not self.ml_scorer.is_trained:
                    collector = PatternDataCollector()
                    training_data = collector.collect_from_trades()
                    min_trades = self.config['trading_settings'].get('min_trades_for_ml', 50)
                    
                    if len(training_data) >= min_trades:
                        logger.info(f"Reached {len(training_data)} trades. Auto-training ML model...")
                        self.ml_scorer.train(training_data)
                        logger.info("ML model auto-training complete")
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
    
    def _print_status(self):
        """Print system status"""
        while True:
            try:
                time.sleep(3600)  # Every hour
                
                logger.info("=" * 80)
                logger.info("SYSTEM STATUS UPDATE")
                logger.info("=" * 80)
                
                # Get recent trades
                trades_24h = self.analytics.get_closed_trades(days=1)
                if trades_24h:
                    metrics = self.analytics.calculate_metrics(trades_24h)
                    logger.info(f"Last 24h: {metrics['total_trades']} trades, "
                              f"{metrics['win_rate']:.1f}% win rate, "
                              f"${metrics['total_pnl']:.2f} P&L")
                else:
                    logger.info("No trades in the last 24 hours")
                
                # Symbol status
                logger.info(f"Active symbols: {len(self.bot_managers)}")
                for bm in self.bot_managers:
                    logger.info(f"  - {bm['symbol']}: Running")
                
                # ML status
                if self.ml_scorer.is_trained:
                    logger.info("ML Model: Trained and active")
                else:
                    logger.info("ML Model: Not yet trained (collecting data)")
                
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"Error printing status: {e}")
    
    def start(self):
        """Start the automated trading system"""
        logger.info("=" * 80)
        logger.info("STARTING AUTOMATED TRADING SYSTEM")
        logger.info("=" * 80)
        
        # Get enabled symbols
        enabled_symbols = self._get_enabled_symbols()
        
        if not enabled_symbols:
            logger.error("No enabled symbols found in configuration!")
            return
        
        # Start bots for each symbol
        for symbol_config in enabled_symbols:
            self._start_symbol_bot(symbol_config)
            time.sleep(2)  # Stagger starts to avoid API rate limits
        
        # Start pattern verification
        self._start_pattern_verification()
        
        # Start performance monitoring in background
        monitor_thread = Thread(target=self._monitor_performance, daemon=True)
        monitor_thread.start()
        
        # Start status updates
        status_thread = Thread(target=self._print_status, daemon=True)
        status_thread.start()
        
        logger.info("=" * 80)
        logger.info("AUTOMATED TRADING SYSTEM RUNNING")
        logger.info(f"Monitoring {len(enabled_symbols)} symbols")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("STOPPING AUTOMATED TRADING SYSTEM")
            logger.info("=" * 80)
            
            # Generate final report
            self._generate_final_report()
            
            logger.info("System stopped")
    
    def _generate_final_report(self):
        """Generate final performance report"""
        try:
            logger.info("Generating final performance report...")
            report = self.analytics.generate_report(days=7)
            print("\n" + report)
        except Exception as e:
            logger.error(f"Error generating final report: {e}")


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("AUTOMATED MULTI-SYMBOL TRADING SYSTEM")
    print("=" * 80 + "\n")
    
    # Check if config file exists
    config_file = "symbols_config.json"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        print(f"\nPlease create {config_file} with your symbol configurations.")
        print("Example configuration has been created for you.")
        sys.exit(1)
    
    try:
        # Initialize and start automated trader
        trader = AutoTrader(config_file)
        trader.start()
        
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
