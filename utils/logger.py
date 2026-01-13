"""
Professional Logging System for Trading Bot
============================================

Implements comprehensive logging with:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Separate log files for different components
- Rotating file handlers to prevent disk space issues
- Console and file output
- Structured logging with timestamps and context
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional
from pathlib import Path
import json


class TradingLogger:
    """Centralized logging system for the trading bot"""
    
    _loggers = {}  # Cache for loggers
    
    def __init__(self, name: str, log_dir: str = None):
        """
        Initialize a logger instance
        
        Args:
            name: Logger name (typically module name)
            log_dir: Directory for log files
        """
        self.name = name
        # Anchor logs to project root so they are consistent regardless of cwd
        if log_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            self.log_dir = str(project_root / "logs")
        else:
            self.log_dir = log_dir
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger with file and console handlers"""
        
        # Check if logger already exists
        if self.name in TradingLogger._loggers:
            return TradingLogger._loggers[self.name]
        
        # Create logs directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console Handler (INFO and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # Main Log File (All levels)
        main_log = os.path.join(self.log_dir, f"{self.name}.log")
        main_handler = RotatingFileHandler(
            main_log,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(detailed_formatter)
        logger.addHandler(main_handler)
        
        # Error Log File (ERROR and above)
        error_log = os.path.join(self.log_dir, f"{self.name}_errors.log")
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # Cache logger
        TradingLogger._loggers[self.name] = logger
        
        return logger
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(self._format_message(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(self._format_message(message, kwargs))
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(self._format_message(message, kwargs))
    
    def _format_number(self, value):
        """Format numbers to avoid scientific notation and ensure readability."""
        try:
            if isinstance(value, float):
                # Use 8 decimals for prices/small floats, trim trailing zeros
                s = f"{value:.8f}"
                # Normalize - remove trailing zeros and dot
                s = s.rstrip('0').rstrip('.') if '.' in s else s
                return s
            if isinstance(value, (int,)):
                return str(value)
            return value
        except Exception:
            return value

    def _format_message(self, message: str, context: dict) -> str:
        """Format message with context data"""
        if context:
            safe_items = []
            for k, v in context.items():
                vv = self._format_number(v)
                safe_items.append(f"{k}={vv}")
            context_str = " | ".join(safe_items)
            return f"{message} | {context_str}"
        return message
    
    def log_trade(self, trade_data: dict):
        """Log trade execution with structured data (open or generic trade event)"""
        trade_log = os.path.join(self.log_dir, "trades.jsonl")
        trade_data['timestamp'] = datetime.utcnow().isoformat()
        
        with open(trade_log, 'a') as f:
            f.write(json.dumps(trade_data) + '\n')

        # Console summary
        sym = trade_data.get('symbol')
        side = trade_data.get('side')
        price = trade_data.get('price') or trade_data.get('entry')
        price_str = self._format_number(float(price)) if isinstance(price, (int, float)) or (isinstance(price, str) and price.replace('.','',1).isdigit()) else price
        self.info("Trade executed", symbol=sym, side=side, price=price_str)

    def log_trade_close(self, trade: dict):
        """Log a closed trade with enhanced metrics and update per-pair aggregates.

        Expects keys: symbol, side, entry, exit, size, pnl, r_multiple, pattern, timeframe, closed_time
        """
        # Calculate per-trade return % relative to notional
        entry = float(trade.get('entry', 0) or 0)
        exit_price = float(trade.get('exit', 0) or 0)
        size = float(trade.get('size', 0) or 0)
        pnl = float(trade.get('pnl', 0) or 0)
        notional = abs(entry * size) if entry and size else 0.0
        return_pct = (pnl / notional * 100.0) if notional > 0 else 0.0
        outcome = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BREAKEVEN')

        enhanced = {
            **trade,
            'timestamp': datetime.utcnow().isoformat(),
            'return_pct': round(return_pct, 4),
            'outcome': outcome
        }

        # Write enhanced trade JSONL
        out_path = os.path.join(self.log_dir, "trades_enhanced.jsonl")
        with open(out_path, 'a') as f:
            f.write(json.dumps(enhanced) + '\n')

        # Update per-pair aggregates
        agg_path = os.path.join(self.log_dir, "pair_metrics.json")
        try:
            if os.path.exists(agg_path):
                with open(agg_path, 'r') as f:
                    agg = json.load(f)
            else:
                agg = {}
        except Exception:
            agg = {}

        sym = trade.get('symbol') or trade.get('Symbol')
        if sym:
            stats = agg.get(sym, {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'breakevens': 0,
                'total_pnl': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'avg_pnl': 0.0,
                'win_rate_pct': 0.0,
                'profit_factor': 0.0
            })

            stats['trades'] += 1
            if pnl > 0:
                stats['wins'] += 1
                stats['gross_profit'] += pnl
            elif pnl < 0:
                stats['losses'] += 1
                stats['gross_loss'] += abs(pnl)
            else:
                stats['breakevens'] += 1
            stats['total_pnl'] = round(float(stats.get('total_pnl', 0.0)) + pnl, 2)
            # avg_pnl over all trades
            if stats['trades'] > 0:
                stats['avg_pnl'] = round(stats['total_pnl'] / stats['trades'], 2)
                stats['win_rate_pct'] = round(stats['wins'] / stats['trades'] * 100.0, 2)
                stats['profit_factor'] = round((stats['gross_profit'] / stats['gross_loss']) if stats['gross_loss'] > 0 else (stats['gross_profit'] > 0 and 999.0 or 0.0), 4)

            agg[sym] = stats

            try:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(agg_path, 'w') as f:
                    json.dump(agg, f, indent=2)
            except Exception:
                pass

        # Human-readable console summary
        self.info(
            "Trade closed",
            symbol=sym,
            side=trade.get('side'),
            entry=self._format_number(entry),
            exit=self._format_number(exit_price),
            size=self._format_number(size),
            pnl=f"{pnl:.2f}",
            return_pct=f"{return_pct:.4f}%",
            outcome=outcome,
            r_multiple=trade.get('r_multiple')
        )
    
    def log_pattern_detection(self, pattern_data: dict):
        """Log pattern detection with structured data"""
        pattern_log = os.path.join(self.log_dir, "patterns.jsonl")
        pattern_data['timestamp'] = datetime.utcnow().isoformat()
        
        with open(pattern_log, 'a') as f:
            f.write(json.dumps(pattern_data) + '\n')
        
        self.info(f"Pattern detected: {pattern_data.get('pattern')} on {pattern_data.get('timeframe')}")
    
    def log_performance(self, metrics: dict):
        """Log performance metrics"""
        perf_log = os.path.join(self.log_dir, "performance.jsonl")
        metrics['timestamp'] = datetime.utcnow().isoformat()
        
        with open(perf_log, 'a') as f:
            f.write(json.dumps(metrics) + '\n')


class ComponentLogger:
    """Specialized loggers for different trading bot components"""
    
    @staticmethod
    def get_logger(component: str) -> TradingLogger:
        """Get a logger for a specific component"""
        return TradingLogger(component)
    
    @staticmethod
    def data_logger():
        """Logger for data fetching and processing"""
        return TradingLogger("data")
    
    @staticmethod
    def pattern_logger():
        """Logger for pattern detection"""
        return TradingLogger("pattern")
    
    @staticmethod
    def trading_logger():
        """Logger for trade execution"""
        return TradingLogger("trading")
    
    @staticmethod
    def risk_logger():
        """Logger for risk management"""
        return TradingLogger("risk")
    
    @staticmethod
    def session_logger():
        """Logger for session management"""
        return TradingLogger("session")
    
    @staticmethod
    def bot_manager_logger():
        """Logger for bot manager"""
        return TradingLogger("bot_manager")
    
    @staticmethod
    def simulator_logger():
        """Logger for trade simulator"""
        return TradingLogger("simulator")
    
    @staticmethod
    def algorithmic_logger():
        """Logger for algorithmic trading system"""
        return TradingLogger("algorithmic")


def get_logger(name: str) -> TradingLogger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        TradingLogger instance
    """
    return TradingLogger(name)


# Performance tracking logger
class PerformanceLogger:
    """Track and log performance metrics"""
    
    def __init__(self):
        self.logger = TradingLogger("performance")
        self.metrics_file = os.path.join("logs", "metrics.json")
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> dict:
        """Load existing metrics"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metrics(self):
        """Save metrics to file"""
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_metric(self, name: str, value: float, category: str = "general"):
        """Log a performance metric"""
        timestamp = datetime.utcnow().isoformat()
        
        if category not in self.metrics:
            self.metrics[category] = {}
        
        if name not in self.metrics[category]:
            self.metrics[category][name] = []
        
        self.metrics[category][name].append({
            'timestamp': timestamp,
            'value': value
        })
        
        # Keep only last 1000 entries
        if len(self.metrics[category][name]) > 1000:
            self.metrics[category][name] = self.metrics[category][name][-1000:]
        
        self._save_metrics()
        self.logger.debug(f"Metric logged: {category}.{name} = {value}")
    
    def get_metrics(self, category: Optional[str] = None) -> dict:
        """Get metrics by category"""
        if category:
            return self.metrics.get(category, {})
        return self.metrics
    
    def get_latest_metric(self, name: str, category: str = "general") -> Optional[float]:
        """Get latest value of a metric"""
        if category in self.metrics and name in self.metrics[category]:
            if self.metrics[category][name]:
                return self.metrics[category][name][-1]['value']
        return None


# Initialize default logger
default_logger = TradingLogger("trading_bot")

if __name__ == "__main__":
    # Test logging system
    logger = get_logger("test")
    logger.info("Logging system initialized")
    logger.debug("Debug message", symbol="BTC/USDT", timeframe="1h")
    logger.warning("Warning message", risk_level="high")
    logger.error("Error message", error_code=500)
    
    # Test trade logging
    logger.log_trade({
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'price': 45000,
        'size': 0.1,
        'pattern': 'Golden Cross'
    })
    
    # Test performance logging
    perf = PerformanceLogger()
    perf.log_metric("win_rate", 0.65, "trading")
    perf.log_metric("sharpe_ratio", 1.8, "risk")
    
    print("Logging test complete. Check logs/ directory for output.")
