"""
Optimized Fully Automated Multi-Symbol Trading System
======================================================

Optimizations:
- Lower RAM usage (limited logging, data buffers)
- Clean console output (no jumping text)
- Forced paper trading verification
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock
from typing import List, Dict

# CRITICAL: Force paper trading mode
os.environ['TRADE_LIVE'] = '0'
os.environ['PAPER_TRADING'] = 'true'

# Import utilities
from utils.logger import get_logger
from utils.config_manager import get_config_manager, get_config
from utils.analytics import PerformanceAnalytics, RealTimeMonitor
from utils.ml_pattern_scorer import MLPatternScorer, PatternDataCollector

# Import existing bot components
from bot_manager import BotManager
from verify_patterns import watch_unverified

# Initialize logger with reduced verbosity
logger = get_logger('auto_trader')

# Console output lock for clean printing
console_lock = Lock()


class OptimizedAutoTrader:
    """Optimized automated multi-symbol trading system"""
    
    def __init__(self, config_file: str = "symbols_config.json"):
        """Initialize optimized automated trader"""
        self.config_file = config_file
        self.config = self._load_config()
        self.bot_managers = []
        self.analytics = PerformanceAnalytics()
        self.monitor = RealTimeMonitor(self.analytics)
        self.ml_scorer = MLPatternScorer()
        
        # Verify paper trading
        self._verify_paper_trading()
        
        # Setup
        self._initialize_system()
    
    def _verify_paper_trading(self):
        """Verify that paper trading is enabled"""
        trade_live = os.getenv('TRADE_LIVE', '1')
        
        if trade_live != '0':
            print("\n" + "=" * 80)
            print("⚠️  WARNING: TRADE_LIVE is not set to 0!")
            print("=" * 80)
            print(f"Current value: TRADE_LIVE={trade_live}")
            print("\nTo fix this, run: python force_paper_trading.py")
            print("\nForcing paper trading mode now...")
            os.environ['TRADE_LIVE'] = '0'
            os.environ['PAPER_TRADING'] = 'true'
            print("✅ Paper trading mode enforced for this session")
            print("=" * 80 + "\n")
            time.sleep(3)
        else:
            print("✅ Paper trading mode confirmed (TRADE_LIVE=0)")
    
    def _load_config(self) -> Dict:
        """Load symbols configuration from JSON file"""
        if not os.path.exists(self.config_file):
            logger.error(f"Configuration file not found: {self.config_file}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        return config
    
    def _initialize_system(self):
        """Initialize the trading system"""
        with console_lock:
            print("\n" + "=" * 80)
            print("AUTOMATED TRADING SYSTEM INITIALIZATION")
            print("=" * 80)
        
        # Initialize configuration manager
        config_manager = get_config_manager()
        config_manager.create_default_configs()
        
        # Auto-train ML model if enabled
        if self.config['trading_settings'].get('auto_train_ml', True):
            self._auto_train_ml()
        
        with console_lock:
            print("System initialization complete")
            print("=" * 80 + "\n")
    
    def _auto_train_ml(self):
        """Automatically train ML model if enough data exists"""
        if self.ml_scorer.is_trained:
            with console_lock:
                print("✅ ML model already trained and loaded")
            return
        
        min_trades = self.config['trading_settings'].get('min_trades_for_ml', 50)
        collector = PatternDataCollector()
        
        try:
            training_data = collector.collect_from_trades()
            
            if len(training_data) >= min_trades:
                with console_lock:
                    print(f"Found {len(training_data)} trades. Training ML model...")
                self.ml_scorer.train(training_data)
                with console_lock:
                    print("✅ ML model training complete")
            else:
                with console_lock:
                    print(f"ℹ️  ML training needs {min_trades - len(training_data)} more trades")
        except Exception as e:
            with console_lock:
                print(f"⚠️  ML training will occur once trade data is available")
    
    def _get_enabled_symbols(self) -> List[Dict]:
        """Get list of enabled symbols from configuration"""
        enabled = [s for s in self.config['symbols'] if s.get('enabled', True)]
        return enabled
    
    def _start_symbol_bot(self, symbol_config: Dict):
        """Start bot for a specific symbol"""
        symbol = symbol_config['symbol']
        timeframes = symbol_config['timeframes']
        
        try:
            settings = self.config['trading_settings']
            
            # Initialize bot manager for this symbol
            bot_manager = BotManager(
                symbol=symbol,
                mode=get_config('mode', default='spot', config_name='trading'),
                timeframes=timeframes,
                operating_mode=settings.get('operating_mode', 'hybrid'),
                moderation_mode=settings.get('moderation_mode', 'balanced'),
                trading_mode=settings.get('trading_mode', 'balanced')
            )
            
            self.bot_managers.append({
                'symbol': symbol,
                'manager': bot_manager,
                'config': symbol_config
            })
            
            # Start bot in separate thread
            bot_thread = Thread(target=bot_manager.run, daemon=True, name=f"bot-{symbol}")
            bot_thread.start()
            
            with console_lock:
                print(f"✅ Started: {symbol} ({len(timeframes)} timeframes)")
            
        except Exception as e:
            with console_lock:
                print(f"❌ Failed to start {symbol}: {e}")
    
    def _start_pattern_verification(self):
        """Start pattern verification threads for all symbols"""
        try:
            # Start verification thread for each enabled symbol
            for bot_info in self.bot_managers:
                symbol = bot_info['symbol']
                verification_thread = Thread(
                    target=watch_unverified, 
                    args=(symbol,),
                    daemon=True,
                    name=f"verify-{symbol}"
                )
                verification_thread.start()
            
            with console_lock:
                print(f"✅ Pattern verification started for {len(self.bot_managers)} symbols")
            
        except Exception as e:
            with console_lock:
                print(f"❌ Pattern verification error: {e}")
    
    def _monitor_performance(self):
        """Monitor performance and send alerts (minimal output)"""
        check_count = 0
        
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                check_count += 1
                
                # Only log every 4th check (20 minutes)
                if check_count % 4 == 0:
                    # Check if we should train ML model
                    if not self.ml_scorer.is_trained:
                        collector = PatternDataCollector()
                        training_data = collector.collect_from_trades()
                        min_trades = self.config['trading_settings'].get('min_trades_for_ml', 50)
                        
                        if len(training_data) >= min_trades:
                            with console_lock:
                                print(f"\n🤖 Auto-training ML model with {len(training_data)} trades...")
                            self.ml_scorer.train(training_data)
                            with console_lock:
                                print("✅ ML model trained successfully\n")
                
                # Check critical alerts only
                max_drawdown = get_config('max_drawdown_pct', default=20.0, config_name='risk')
                dd_alert = self.monitor.check_drawdown_alert(max_drawdown)
                if dd_alert:
                    with console_lock:
                        print(f"\n⚠️  {dd_alert}\n")
                
                loss_alert = self.monitor.check_consecutive_losses_alert(5)
                if loss_alert:
                    with console_lock:
                        print(f"\n⚠️  {loss_alert}\n")
                
            except Exception as e:
                pass  # Silent errors to reduce console spam
    
    def _print_status(self):
        """Print hourly status updates"""
        while True:
            try:
                time.sleep(3600)  # Every hour
                
                with console_lock:
                    print("\n" + "=" * 80)
                    print(f"STATUS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
                    print("=" * 80)
                    
                    # Get recent trades
                    trades_24h = self.analytics.get_closed_trades(days=1)
                    if trades_24h:
                        metrics = self.analytics.calculate_metrics(trades_24h)
                        print(f"Last 24h: {metrics['total_trades']} trades | "
                              f"{metrics['win_rate']:.1f}% WR | "
                              f"${metrics['total_pnl']:.2f} P&L")
                    else:
                        print("No trades in last 24 hours")
                    
                    # Symbol status
                    print(f"Active: {len(self.bot_managers)} symbols")
                    
                    # ML status
                    if self.ml_scorer.is_trained:
                        print("ML Model: ✅ Active")
                    else:
                        print("ML Model: 📊 Collecting data")
                    
                    print("=" * 80 + "\n")
                
            except Exception as e:
                pass  # Silent errors
    
    def start(self):
        """Start the automated trading system"""
        with console_lock:
            print("\n" + "=" * 80)
            print("STARTING AUTOMATED TRADING SYSTEM")
            print("=" * 80)
        
        # Get enabled symbols
        enabled_symbols = self._get_enabled_symbols()
        
        if not enabled_symbols:
            print("❌ No enabled symbols found in configuration!")
            return
        
        with console_lock:
            print(f"\n📊 Starting {len(enabled_symbols)} symbols...")
            print()
        
        # Start bots for each symbol
        for symbol_config in enabled_symbols:
            self._start_symbol_bot(symbol_config)
            time.sleep(1)  # Small delay to stagger starts
        
        # Start pattern verification
        time.sleep(2)
        self._start_pattern_verification()
        
        # Start monitoring in background
        monitor_thread = Thread(target=self._monitor_performance, daemon=True, name="monitor")
        monitor_thread.start()
        
        # Start status updates
        status_thread = Thread(target=self._print_status, daemon=True, name="status")
        status_thread.start()
        
        with console_lock:
            print("\n" + "=" * 80)
            print("🚀 SYSTEM RUNNING")
            print("=" * 80)
            print(f"Monitoring: {len(enabled_symbols)} symbols")
            print(f"Paper Trading: {'✅ ENABLED' if os.getenv('TRADE_LIVE') == '0' else '❌ CHECK .ENV'}")
            print("Press Ctrl+C to stop")
            print("=" * 80 + "\n")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            with console_lock:
                print("\n" + "=" * 80)
                print("STOPPING SYSTEM")
                print("=" * 80)
            
            # Generate final report
            self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate final performance report"""
        try:
            with console_lock:
                print("\nGenerating final report...")
            
            trades = self.analytics.get_closed_trades(days=7)
            if trades:
                metrics = self.analytics.calculate_metrics(trades)
                
                with console_lock:
                    print("\n" + "=" * 80)
                    print("FINAL REPORT (Last 7 Days)")
                    print("=" * 80)
                    print(f"Total Trades: {metrics['total_trades']}")
                    print(f"Win Rate: {metrics['win_rate']:.1f}%")
                    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
                    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
                    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
                    print("=" * 80 + "\n")
        except Exception as e:
            pass


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("OPTIMIZED AUTOMATED MULTI-SYMBOL TRADING SYSTEM")
    print("Low RAM Usage | Clean Output | Verified Paper Trading")
    print("=" * 80 + "\n")
    
    # Check config file
    config_file = "symbols_config.json"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        sys.exit(1)
    
    try:
        # Initialize and start
        trader = OptimizedAutoTrader(config_file)
        trader.start()
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
