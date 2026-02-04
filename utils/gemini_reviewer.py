"""
Gemini-Powered Daily Review System
==================================

Generates AI-powered reviews of bot/training status using Google's Gemini API
and sends them via email at scheduled times (start/middle/end of day).
"""

import os
import json
import time
import threading
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add parent directory to path for imports
import sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils.analytics import PerformanceAnalytics
from rl_dashboard import RLDashboard


class StatusCollector:
    """Collects comprehensive bot and training status"""
    
    def __init__(
        self,
        model_dir: str = "models/rl_models",
        state_file: str = "models/rl_models/rl_training_state.json",
        db_path: str = "sim_results/trades.db",
    ):
        self.model_dir = Path(model_dir)
        self.state_file = Path(state_file)
        self.db_path = db_path
        self.analytics = PerformanceAnalytics(db_path=db_path)
        self.dashboard = RLDashboard(
            model_dir=model_dir,
            state_file=state_file,
            db_path=db_path,
        )
    
    def collect_status(self) -> Dict:
        """Collect all status information"""
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'training': {},
            'trading': {},
            'performance': {},
            'stale_status': {},
        }
        
        # Training status
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    status['training'] = {
                        'episode': state.get('episode', 0),
                        'total_timesteps': state.get('total_timesteps', 0),
                        'total_trades_in_state': state.get('total_trades', 0),  # trades in current run (state file)
                        'mode': state.get('mode', 'unknown'),
                        'best_reward': state.get('best_reward', 0),
                        'last_checkpoint': state.get('last_checkpoint'),
                        'historical_pretraining_completed': state.get('historical_pretraining_completed', False),
                        'timestamp': state.get('timestamp'),
                        'learning_rate': state.get('learning_rate'),  # so reports show actual LR when run outside training
                    }
                    # Calculate age
                    if state.get('timestamp'):
                        try:
                            state_time = datetime.fromisoformat(state.get('timestamp').replace('Z', '+00:00'))
                            if state_time.tzinfo:
                                state_time = state_time.replace(tzinfo=None)
                            now = datetime.now(timezone.utc).replace(tzinfo=None)
                            age_minutes = (now - state_time).total_seconds() / 60
                            status['training']['state_age_minutes'] = age_minutes
                        except:
                            status['training']['state_age_minutes'] = None
            except Exception as e:
                status['training']['error'] = str(e)
        
        # Update dashboard stats
        try:
            self.dashboard._update_stats()
            stats = self.dashboard.stats
            
            # Trading status
            status['trading'] = {
                'open_trades_count': stats.get('open_trades_count', 0),
                'total_closed_trades': stats.get('total_closed_trades', 0),
                'winning_trades': stats.get('winning_trades', 0),
                'losing_trades': stats.get('losing_trades', 0),
            }
            
            # Performance metrics (learning_rate from dashboard; fallback to state when report run outside active training)
            lr = stats.get('learning_rate', 0)
            if lr == 0:
                lr = status.get('training', {}).get('learning_rate') or 0
                try:
                    lr = float(lr)
                except (TypeError, ValueError):
                    lr = 0
            status['performance'] = {
                'win_rate_global': stats.get('win_rate_global', 0),
                'accuracy_global': stats.get('accuracy_global', 0),
                'total_pnl': stats.get('total_pnl', 0),
                'avg_reward': stats.get('avg_reward', 0),
                'best_episode_reward': stats.get('best_episode_reward', 0),
                'learning_rate': lr,
                'learning_rate_note': 'N/A (report run outside active training; actual LR during training is adaptive 5e-4 to 1e-4)' if lr == 0 else None,
            }
            
            # Stale status
            status['stale_status'] = {
                'is_stale': stats.get('is_stale', False),
                'stale_minutes': stats.get('stale_minutes'),
                'in_historical_phase': stats.get('in_historical_phase', False),
                'state_file_age_minutes': stats.get('state_file_age_minutes'),
            }
            
            # Per-pair metrics
            status['performance']['accuracy_by_pair'] = stats.get('accuracy_by_pair', {})
            status['performance']['win_rate_by_pair'] = stats.get('win_rate_by_pair', {})
            status['performance']['pnl_by_pair'] = stats.get('pnl_by_pair', {})
            
        except Exception as e:
            status['error'] = f"Failed to collect dashboard stats: {e}"
        
        # Additional analytics
        try:
            closed_trades = self.analytics.get_closed_trades(exclude_historical_training=True)
            if closed_trades:
                metrics = self.analytics.calculate_metrics(closed_trades)
                status['performance']['detailed'] = {
                    'profit_factor': metrics.get('profit_factor', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
                    'avg_win': metrics.get('avg_win', 0),
                    'avg_loss': metrics.get('avg_loss', 0),
                    'largest_win': metrics.get('largest_win', 0),
                    'largest_loss': metrics.get('largest_loss', 0),
                }
        except Exception as e:
            status['performance']['analytics_error'] = str(e)
        
        return status


class GeminiReviewer:
    """Generates AI reviews using Google Gemini API"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    
    def generate_review(self, status: Dict, review_type: str = "daily") -> str:
        """
        Generate review using Gemini API
        
        Args:
            status: Status dictionary from StatusCollector
            review_type: "start", "middle", or "end" of day
        """
        # Build prompt
        prompt = self._build_prompt(status, review_type)
        
        try:
            # Use the correct API endpoint format
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            response = requests.post(
                url,
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0].get('content', {})
                parts = content.get('parts', [])
                if parts and 'text' in parts[0]:
                    return parts[0]['text']
            
            # Check for errors in response
            if 'error' in result:
                error_msg = result['error'].get('message', 'Unknown error')
                return f"Error from Gemini API: {error_msg}"
            
            return "Error: Unexpected response format from Gemini API"
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_response = e.response.json()
                if 'error' in error_response:
                    error_detail = f" - {error_response['error'].get('message', '')}"
            except:
                pass
            return f"HTTP Error calling Gemini API ({e.response.status_code}): {e}{error_detail}"
        except requests.exceptions.RequestException as e:
            return f"Error calling Gemini API: {e}"
        except Exception as e:
            return f"Error generating review: {e}"
    
    def _build_prompt(self, status: Dict, review_type: str) -> str:
        """Build the prompt for Gemini"""
        time_of_day = {
            "start": "morning (start of day)",
            "middle": "midday (middle of day)",
            "end": "evening (end of day)"
        }.get(review_type, "daily")
        
        training = status.get('training', {})
        trading = status.get('trading', {})
        performance = status.get('performance', {})
        stale = status.get('stale_status', {})
        
        lr_note = performance.get('learning_rate_note') or ''
        prompt = f"""You are an expert trading bot analyst. Generate a comprehensive, detailed review of the RL trading bot's status for the {time_of_day} review.

IMPORTANT CONTEXT:
- "Total Trades in State" = trades executed in the current training run only (from state file). May be much lower than DB count.
- "Closed Trades in DB" = all closed trades ever (all time). Performance metrics (win rate, PnL, largest win/loss, drawdown) use this.
- Learning Rate: {lr_note or 'Actual LR during training is adaptive (5e-4 to 1e-4).'}
- Max Drawdown % is capped at 100%. Largest Win/Loss and drawdown are computed from chronological trade history.

Current Status:
- Training Mode: {training.get('mode', 'unknown')}
- Episode: {training.get('episode', 0)}
- Total Timesteps: {training.get('total_timesteps', 0):,}
- Total Trades in State (current run): {training.get('total_trades_in_state', 0)}
- Closed Trades in DB (all time): {trading.get('total_closed_trades', 0)}
- Best Episode Reward: {training.get('best_reward', 0):.4f}
- Historical Pre-training Completed: {training.get('historical_pretraining_completed', False)}
- State File Age: {training.get('state_age_minutes', 'N/A')} minutes ago

Trading Activity:
- Open Trades: {trading.get('open_trades_count', 0)}
- Closed Trades (DB): {trading.get('total_closed_trades', 0)}
- Winning Trades: {trading.get('winning_trades', 0)}
- Losing Trades: {trading.get('losing_trades', 0)}

Performance Metrics:
- Global Win Rate: {performance.get('win_rate_global', 0):.2f}%
- Global Accuracy: {performance.get('accuracy_global', 0):.2f}%
- Total PnL: ${performance.get('total_pnl', 0):.2f}
- Average Reward: {performance.get('avg_reward', 0):.4f}
- Learning Rate: {performance.get('learning_rate', 0):.6f}{' (' + lr_note + ')' if lr_note else ''}
"""
        
        # Add detailed metrics if available
        if 'detailed' in performance:
            det = performance['detailed']
            prompt += f"""
Detailed Performance:
- Profit Factor: {det.get('profit_factor', 0):.2f}
- Sharpe Ratio: {det.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {det.get('max_drawdown_pct', 0):.2f}%
- Average Win: ${det.get('avg_win', 0):.2f}
- Average Loss: ${det.get('avg_loss', 0):.2f}
- Largest Win: ${det.get('largest_win', 0):.2f}
- Largest Loss: ${det.get('largest_loss', 0):.2f}
"""
        
        # Add stale status
        is_stale = stale.get('is_stale', False)
        stale_min = stale.get('stale_minutes')
        prompt += f"""
Stale Status:
- Bot is STALE: {'YES ⚠️' if is_stale else 'NO ✅'}
- Minutes Since Last Activity: {stale_min if stale_min is not None else 'N/A'}
- In Historical Phase: {stale.get('in_historical_phase', False)}
"""
        
        # Add per-pair metrics if available
        if 'accuracy_by_pair' in performance and performance['accuracy_by_pair']:
            prompt += "\nPer-Pair Performance:\n"
            for symbol, acc in performance['accuracy_by_pair'].items():
                wr = performance.get('win_rate_by_pair', {}).get(symbol, 0)
                pnl = performance.get('pnl_by_pair', {}).get(symbol, 0)
                prompt += f"- {symbol}: Accuracy {acc:.2f}%, Win Rate {wr:.2f}%, PnL ${pnl:.2f}\n"
        
        prompt += """
Please provide a comprehensive review that includes:
1. Overall status assessment (healthy, needs attention, critical issues)
2. Training progress analysis (is it progressing well, stuck, or regressing?)
3. Trading performance evaluation (profitable, break-even, losing)
4. Stale status analysis (if stale, explain why and what might be wrong)
5. Key insights and recommendations
6. Action items if any issues are detected

Be specific, detailed, and actionable. If the bot is stale, explain what might have caused it and what should be checked.
"""
        
        return prompt


class EmailSender:
    """Sends email notifications"""
    
    def __init__(self, to_email: str = "noamiko613@gmail.com"):
        self.to_email = os.getenv("RL_NOTIFY_EMAIL_TO", to_email).strip()
        self.host = os.getenv("RL_SMTP_HOST", "smtp.gmail.com").strip()
        self.port = int(os.getenv("RL_SMTP_PORT", "587"))
        self.user = os.getenv("RL_SMTP_USER", "").strip()
        self.password = os.getenv("RL_SMTP_PASSWORD", "").strip()
    
    def send_review(self, review: str, review_type: str, status: Dict) -> bool:
        """Send review email"""
        if not self.user or not self.password:
            print(f"[GeminiReviewer] Email skipped (set RL_SMTP_USER and RL_SMTP_PASSWORD to enable)")
            print(f"[GeminiReviewer] Would have sent to: {self.to_email}")
            return False
        
        try:
            time_of_day = {
                "start": "Morning",
                "middle": "Midday",
                "end": "Evening"
            }.get(review_type, "Daily")
            
            subject = f"[RL Trading Bot] {time_of_day} Status Review - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Build email body
            body_lines = [
                f"{time_of_day} Status Review",
                "=" * 60,
                "",
                review,
                "",
                "=" * 60,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                "",
                "Quick Status Summary:",
                f"- Mode: {status.get('training', {}).get('mode', 'unknown')}",
                f"- Timesteps: {status.get('training', {}).get('total_timesteps', 0):,}",
                f"- Trades (state): {status.get('training', {}).get('total_trades_in_state', 0)} | Closed (DB): {status.get('trading', {}).get('total_closed_trades', 0)}",
                f"- Win Rate: {status.get('performance', {}).get('win_rate_global', 0):.2f}%",
                f"- Stale: {'YES ⚠️' if status.get('stale_status', {}).get('is_stale') else 'NO ✅'}",
            ]
            body = "\n".join(body_lines)
            
            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"] = self.user
            msg["To"] = self.to_email
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.sendmail(self.user, [self.to_email], msg.as_string())
            
            print(f"[GeminiReviewer] Review email sent to {self.to_email}")
            return True
        except Exception as e:
            print(f"[GeminiReviewer] Failed to send email: {e}")
            return False


class ReviewScheduler:
    """Schedules and runs reviews at start/middle/end of day"""
    
    def __init__(
        self,
        gemini_api_key: str,
        model_dir: str = "models/rl_models",
        state_file: str = "models/rl_models/rl_training_state.json",
        db_path: str = "sim_results/trades.db",
        to_email: str = "noamiko613@gmail.com",
        model_name: str = "gemini-2.5-flash",
    ):
        self.collector = StatusCollector(model_dir, state_file, db_path)
        self.reviewer = GeminiReviewer(gemini_api_key, model_name=model_name)
        self.emailer = EmailSender(to_email)
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the scheduler in a background thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        print("[GeminiReviewer] Scheduler started (reviews at start/middle/end of day)")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                current_hour = now.hour
                
                # Determine which review to send
                review_type = None
                if current_hour == 0:  # Start of day (midnight UTC)
                    review_type = "start"
                elif current_hour == 12:  # Middle of day (noon UTC)
                    review_type = "middle"
                elif current_hour == 23:  # End of day (11 PM UTC)
                    review_type = "end"
                
                if review_type:
                    print(f"[GeminiReviewer] Generating {review_type} of day review...")
                    self._generate_and_send_review(review_type)
                    # Wait 2 hours to avoid duplicate sends
                    time.sleep(2 * 3600)
                else:
                    # Check every hour
                    time.sleep(3600)
            except Exception as e:
                print(f"[GeminiReviewer] Error in scheduler: {e}")
                time.sleep(3600)
    
    def _generate_and_send_review(self, review_type: str):
        """Generate and send a review"""
        try:
            # Collect status
            status = self.collector.collect_status()
            
            # Generate review
            review = self.reviewer.generate_review(status, review_type)
            
            # Send email
            self.emailer.send_review(review, review_type, status)
        except Exception as e:
            print(f"[GeminiReviewer] Error generating/sending review: {e}")
    
    def generate_review_now(self, review_type: str = "daily") -> str:
        """Generate a review immediately (for testing)"""
        status = self.collector.collect_status()
        review = self.reviewer.generate_review(status, review_type)
        self.emailer.send_review(review, review_type, status)
        return review


def main():
    """CLI entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Gemini-powered bot status review")
    parser.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY", "AIzaSyAtxfgJJQT4FnfhTQ5S-5iBKmVOaDy1eQc"),
                       help="Gemini API key")
    parser.add_argument("--model-name", type=str, default=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
                       help="Gemini model name (e.g., gemini-1.5-flash, gemini-1.5-pro, gemini-2.5-flash)")
    parser.add_argument("--type", type=str, choices=["start", "middle", "end", "daily"], default="daily",
                       help="Review type")
    parser.add_argument("--start-scheduler", action="store_true",
                       help="Start the scheduler (runs continuously)")
    parser.add_argument("--model-dir", type=str, default="models/rl_models",
                       help="Model directory")
    parser.add_argument("--state-file", type=str, default="models/rl_models/rl_training_state.json",
                       help="Training state file")
    parser.add_argument("--db-path", type=str, default="sim_results/trades.db",
                       help="Trades database path")
    
    args = parser.parse_args()
    
    scheduler = ReviewScheduler(
        gemini_api_key=args.api_key,
        model_dir=args.model_dir,
        state_file=args.state_file,
        db_path=args.db_path,
        model_name=args.model_name,
    )
    
    if args.start_scheduler:
        print("[GeminiReviewer] Starting scheduler...")
        scheduler.start()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n[GeminiReviewer] Stopping scheduler...")
            scheduler.stop()
    else:
        # Generate one review
        print(f"[GeminiReviewer] Generating {args.type} review...")
        review = scheduler.generate_review_now(args.type)
        print("\n" + "="*60)
        print("REVIEW:")
        print("="*60)
        print(review)


if __name__ == "__main__":
    main()
