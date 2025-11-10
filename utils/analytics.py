"""
Advanced Analytics and Performance Tracking
============================================

Comprehensive analytics system for tracking:
- Trade performance metrics
- Win/loss rates
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Pattern-specific performance
- Timeframe-specific performance
- Session-based performance
- Real-time monitoring and alerts
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


class PerformanceAnalytics:
    """Comprehensive performance analytics for trading bot"""
    
    def __init__(self, db_path: str = "sim_results/trades.db"):
        """
        Initialize analytics system
        
        Args:
            db_path: Path to trades database
        """
        self.db_path = db_path
        self.analytics_dir = "analytics"
        os.makedirs(self.analytics_dir, exist_ok=True)
    
    def get_closed_trades(self, symbol: Optional[str] = None, days: Optional[int] = None) -> List[Dict]:
        """Get closed trades from database"""
        if not os.path.exists(self.db_path):
            return []
        
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM trades_closed"
            conditions = []
            params = []
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if days:
                start_date = (datetime.now() - timedelta(days=days)).isoformat()
                conditions.append("closed_time >= ?")
                params.append(start_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY closed_time DESC"
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            columns = [desc[0] for desc in cursor.description]
            trades = []
            for row in cursor.fetchall():
                trades.append(dict(zip(columns, row)))
            
            return trades
        finally:
            conn.close()
    
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return self._empty_metrics()
        
        df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        breakeven_trades = len(df[df['pnl'] == 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # R-multiple analysis
        if 'r_multiple' in df.columns:
            avg_r_multiple = df['r_multiple'].mean()
            expectancy = df['r_multiple'].mean()  # Average R-multiple is expectancy
        else:
            avg_r_multiple = 0
            expectancy = 0
        
        # Profit factor
        gross_profit = df[df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Drawdown analysis
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max()
        max_drawdown_pct = (max_drawdown / running_max.max() * 100) if running_max.max() > 0 else 0
        
        # Sharpe ratio (assuming 252 trading days per year for crypto)
        if len(df) > 1:
            returns = df['pnl'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (downside deviation)
        if len(df) > 1:
            returns = df['pnl'].pct_change().dropna()
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = 0
        
        # Consecutive wins/losses
        pnl_signs = (df['pnl'] > 0).astype(int)
        max_consecutive_wins = self._max_consecutive(pnl_signs, 1)
        max_consecutive_losses = self._max_consecutive(pnl_signs, 0)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_r_multiple': round(avg_r_multiple, 2),
            'expectancy': round(expectancy, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def _max_consecutive(self, series, value):
        """Calculate maximum consecutive occurrences of a value"""
        max_count = 0
        current_count = 0
        
        for v in series:
            if v == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_r_multiple': 0,
            'expectancy': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
    
    def pattern_performance(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Analyze performance by pattern type"""
        if not trades:
            return {}
        
        df = pd.DataFrame(trades)
        pattern_stats = {}
        
        for pattern in df['pattern'].unique():
            pattern_trades = df[df['pattern'] == pattern]
            pattern_stats[pattern] = self.calculate_metrics(pattern_trades.to_dict('records'))
        
        return pattern_stats
    
    def timeframe_performance(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Analyze performance by timeframe"""
        # This would require adding timeframe to trades table
        # For now, return empty dict
        return {}
    
    def session_performance(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Analyze performance by trading session"""
        # This would require adding session type to trades table
        # For now, return empty dict
        return {}
    
    def generate_report(self, symbol: Optional[str] = None, days: int = 30) -> str:
        """Generate comprehensive performance report"""
        trades = self.get_closed_trades(symbol=symbol, days=days)
        
        if not trades:
            return "No trades found for the specified period."
        
        overall_metrics = self.calculate_metrics(trades)
        pattern_metrics = self.pattern_performance(trades)
        
        report = []
        report.append("=" * 80)
        report.append(f"TRADING BOT PERFORMANCE REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if symbol:
            report.append(f"Symbol: {symbol}")
        report.append(f"Period: Last {days} days")
        report.append("=" * 80)
        
        report.append("\n📊 OVERALL PERFORMANCE")
        report.append("-" * 80)
        report.append(f"Total Trades:           {overall_metrics['total_trades']}")
        report.append(f"Winning Trades:         {overall_metrics['winning_trades']} ({overall_metrics['win_rate']:.1f}%)")
        report.append(f"Losing Trades:          {overall_metrics['losing_trades']}")
        report.append(f"")
        report.append(f"Total P&L:              ${overall_metrics['total_pnl']:,.2f}")
        report.append(f"Average P&L per Trade:  ${overall_metrics['avg_pnl']:,.2f}")
        report.append(f"Average Win:            ${overall_metrics['avg_win']:,.2f}")
        report.append(f"Average Loss:           ${overall_metrics['avg_loss']:,.2f}")
        report.append(f"")
        report.append(f"Profit Factor:          {overall_metrics['profit_factor']:.2f}")
        report.append(f"Average R-Multiple:     {overall_metrics['avg_r_multiple']:.2f}R")
        report.append(f"Expectancy:             {overall_metrics['expectancy']:.2f}R")
        report.append(f"")
        report.append(f"Max Drawdown:           ${overall_metrics['max_drawdown']:,.2f} ({overall_metrics['max_drawdown_pct']:.1f}%)")
        report.append(f"Sharpe Ratio:           {overall_metrics['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio:          {overall_metrics['sortino_ratio']:.2f}")
        report.append(f"")
        report.append(f"Max Consecutive Wins:   {overall_metrics['max_consecutive_wins']}")
        report.append(f"Max Consecutive Losses: {overall_metrics['max_consecutive_losses']}")
        
        if pattern_metrics:
            report.append("\n\n📈 PATTERN-SPECIFIC PERFORMANCE")
            report.append("-" * 80)
            
            # Sort patterns by total PnL
            sorted_patterns = sorted(pattern_metrics.items(), 
                                    key=lambda x: x[1]['total_pnl'], 
                                    reverse=True)
            
            for pattern, metrics in sorted_patterns:
                report.append(f"\n{pattern}:")
                report.append(f"  Trades: {metrics['total_trades']} | Win Rate: {metrics['win_rate']:.1f}% | "
                            f"P&L: ${metrics['total_pnl']:,.2f} | Avg R: {metrics['avg_r_multiple']:.2f}R")
        
        report.append("\n" + "=" * 80)
        
        # Save report to file
        report_file = os.path.join(self.analytics_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)
    
    def export_to_csv(self, symbol: Optional[str] = None, days: Optional[int] = None):
        """Export trades to CSV for analysis"""
        trades = self.get_closed_trades(symbol=symbol, days=days)
        
        if not trades:
            return None
        
        df = pd.DataFrame(trades)
        csv_file = os.path.join(self.analytics_dir, 
                               f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(csv_file, index=False)
        
        return csv_file
    
    def get_equity_curve(self, symbol: Optional[str] = None, days: Optional[int] = None) -> pd.DataFrame:
        """Generate equity curve data"""
        trades = self.get_closed_trades(symbol=symbol, days=days)
        
        if not trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(trades)
        df['closed_time'] = pd.to_datetime(df['closed_time'])
        df = df.sort_values('closed_time')
        
        # Assuming starting balance from first trade (would need to track this better)
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        return df[['closed_time', 'cumulative_pnl']]


class RealTimeMonitor:
    """Real-time performance monitoring and alerts"""
    
    def __init__(self, analytics: PerformanceAnalytics):
        """
        Initialize real-time monitor
        
        Args:
            analytics: PerformanceAnalytics instance
        """
        self.analytics = analytics
        self.alerts = []
    
    def check_drawdown_alert(self, max_allowed_pct: float = 20.0) -> Optional[str]:
        """Check if drawdown exceeds threshold"""
        trades = self.analytics.get_closed_trades(days=1)
        if not trades:
            return None
        
        metrics = self.analytics.calculate_metrics(trades)
        
        if metrics['max_drawdown_pct'] > max_allowed_pct:
            alert = f"⚠️ ALERT: Drawdown ({metrics['max_drawdown_pct']:.1f}%) exceeds threshold ({max_allowed_pct}%)"
            self.alerts.append({'timestamp': datetime.now(), 'alert': alert})
            return alert
        
        return None
    
    def check_win_rate_alert(self, min_trades: int = 10, min_win_rate: float = 40.0) -> Optional[str]:
        """Check if win rate is below threshold"""
        trades = self.analytics.get_closed_trades(days=7)
        if not trades or len(trades) < min_trades:
            return None
        
        metrics = self.analytics.calculate_metrics(trades)
        
        if metrics['win_rate'] < min_win_rate:
            alert = f"⚠️ ALERT: Win rate ({metrics['win_rate']:.1f}%) below threshold ({min_win_rate}%)"
            self.alerts.append({'timestamp': datetime.now(), 'alert': alert})
            return alert
        
        return None
    
    def check_consecutive_losses_alert(self, max_consecutive: int = 5) -> Optional[str]:
        """Check for excessive consecutive losses"""
        trades = self.analytics.get_closed_trades(days=1)
        if not trades:
            return None
        
        metrics = self.analytics.calculate_metrics(trades)
        
        if metrics['max_consecutive_losses'] >= max_consecutive:
            alert = f"⚠️ ALERT: {metrics['max_consecutive_losses']} consecutive losses"
            self.alerts.append({'timestamp': datetime.now(), 'alert': alert})
            return alert
        
        return None
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts if a['timestamp'] > cutoff]


if __name__ == "__main__":
    # Test analytics
    analytics = PerformanceAnalytics()
    
    # Generate report
    report = analytics.generate_report(days=30)
    print(report)
    
    # Export to CSV
    csv_file = analytics.export_to_csv(days=30)
    if csv_file:
        print(f"\nTrades exported to: {csv_file}")
    
    # Test real-time monitor
    monitor = RealTimeMonitor(analytics)
    alerts = [
        monitor.check_drawdown_alert(),
        monitor.check_win_rate_alert(),
        monitor.check_consecutive_losses_alert()
    ]
    
    for alert in alerts:
        if alert:
            print(alert)
