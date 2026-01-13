import json
import os
import pickle
import random
import threading
import time
import numpy as np
from collections import deque
from datetime import datetime
from statistics import pstdev
from typing import Deque, Dict, List, Optional, Tuple

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from coinEx_getting_data import CoinExDataFetcher
from session_manager import SessionManager
from utils.logger import ComponentLogger
from verify_patterns import normalize_symbol_to_ccxt


class NeuralNetworkRLAgent:
    """
    Neural network-based RL agent using Deep Q-Network (DQN) approach.
    Uses MLPRegressor to learn Q-values for each action and learns from mistakes.
    """

    def __init__(
        self,
        actions: List[str],
        learning_rate: float,
        discount_factor: float,
        initial_epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        model_path: Optional[str] = None,
    ):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model_path = model_path
        
        # Experience replay buffer for learning from past mistakes
        self.experience_buffer: Deque[Dict] = deque(maxlen=10000)
        self.min_experience_for_training = 50  # Train more frequently to learn faster
        self.training_frequency = 10  # Train every N new experiences
        self.experiences_since_training = 0
        
        # Neural network models for each action (one model per action)
        self.models: Dict[str, Pipeline] = {}
        self.scaler = StandardScaler()
        self.feature_dim = None
        self._initialize_models()
        
        # Training statistics
        self.training_steps = 0
        self.last_training_loss = 0.0

    def _initialize_models(self):
        """Initialize neural network models for each action"""
        for action in self.actions:
            # Use MLPRegressor with multiple hidden layers for better learning
            model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2 regularization
                learning_rate='adaptive',
                learning_rate_init=self.learning_rate * 0.1,  # Scale down for neural net
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
                warm_start=True,  # Enable incremental learning
            )
            self.models[action] = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

    def _state_to_features(self, state: Tuple) -> np.ndarray:
        """Convert state tuple to numerical feature vector"""
        # Convert string features to numerical encoding
        features = []
        for item in state:
            if isinstance(item, (int, float)):
                features.append(float(item))
            elif isinstance(item, str):
                # Simple hash-based encoding for string features
                features.append(float(hash(item) % 1000) / 1000.0)
            else:
                features.append(0.0)
        
        if not features:
            features = [0.0] * 10  # Default feature vector
        
        return np.array(features).reshape(1, -1)

    def select_action(self, state: Tuple) -> str:
        """Select action using epsilon-greedy policy with neural network"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Use neural network to predict Q-values
        state_features = self._state_to_features(state)
        
        if self.feature_dim is None:
            self.feature_dim = state_features.shape[1]
        
        q_values = {}
        for action in self.actions:
            if action in self.models:
                try:
                    # Predict Q-value for this action
                    q_val = self.models[action].predict(state_features)[0]
                    q_values[action] = float(q_val)
                except Exception:
                    # Model not trained yet, use default
                    q_values[action] = 0.0
            else:
                q_values[action] = 0.0
        
        # Select action with highest Q-value
        best_action = max(q_values, key=q_values.get)
        return best_action

    def update(self, state: Tuple, action: str, reward: float, next_state: Optional[Tuple]):
        """
        Update Q-network using experience replay.
        Learns from mistakes by training on past experiences.
        """
        # Store experience in replay buffer
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
        }
        self.experience_buffer.append(experience)
        self.experiences_since_training += 1
        
        # Train the model periodically using experience replay
        # Train more frequently to learn faster from mistakes
        if (len(self.experience_buffer) >= self.min_experience_for_training and 
            self.experiences_since_training >= self.training_frequency):
            self._train_from_experience()
            self.experiences_since_training = 0

    def _train_from_experience(self):
        """Train neural network from experience replay buffer"""
        if len(self.experience_buffer) < self.min_experience_for_training:
            return
        
        # Sample recent experiences (prioritize recent mistakes)
        sample_size = min(500, len(self.experience_buffer))
        recent_experiences = list(self.experience_buffer)[-sample_size:]
        
        # Prepare training data for each action
        for action in self.actions:
            action_experiences = [exp for exp in recent_experiences if exp['action'] == action]
            if len(action_experiences) < 10:
                continue
            
            X = []
            y = []
            
            for exp in action_experiences:
                state_features = self._state_to_features(exp['state']).flatten()
                X.append(state_features)
                
                # Calculate target Q-value using Bellman equation
                reward = exp['reward']
                if exp['next_state'] is not None:
                    # Estimate future value using current models
                    next_state_features = self._state_to_features(exp['next_state'])
                    max_future_q = max([
                        self.models[a].predict(next_state_features)[0] 
                        if a in self.models else 0.0
                        for a in self.actions
                    ])
                    target_q = reward + (self.discount_factor * max_future_q)
                else:
                    target_q = reward
                
                y.append(target_q)
            
            if len(X) >= 10:
                X = np.array(X)
                y = np.array(y)
                
                # Update feature dimension
                if self.feature_dim is None:
                    self.feature_dim = X.shape[1]
                
                try:
                    # Refit model with accumulated experience (learns from mistakes)
                    # This approach allows the model to learn from all past experiences
                    # and improve its predictions over time
                    self.models[action].fit(X, y)
                    self.training_steps += 1
                except Exception as e:
                    # If training fails, try with smaller sample
                    try:
                        if len(X) > 50:
                            # Sample subset if too large
                            indices = np.random.choice(len(X), size=50, replace=False)
                            X_sample = X[indices]
                            y_sample = y[indices]
                            self.models[action].fit(X_sample, y_sample)
                            self.training_steps += 1
                    except Exception:
                        pass

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path: str):
        """Save trained neural network models"""
        try:
            model_data = {
                'models': {},
                'epsilon': self.epsilon,
                'training_steps': self.training_steps,
                'feature_dim': self.feature_dim,
                'actions': self.actions,
            }
            
            # Save each model
            for action, model in self.models.items():
                model_data['models'][action] = pickle.dumps(model)
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            raise Exception(f"Failed to save RL model: {e}")

    def load_model(self, path: str):
        """Load trained neural network models"""
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.training_steps = model_data.get('training_steps', 0)
            self.feature_dim = model_data.get('feature_dim', None)
            
            # Load each model
            for action, model_bytes in model_data.get('models', {}).items():
                if action in self.actions:
                    self.models[action] = pickle.loads(model_bytes)
            
            return True
        except Exception as e:
            return False


class TrainingContext:
    """Holds state and resources for a specific symbol/timeframe RL loop."""

    def __init__(
        self,
        symbol: str,
        timeframes: List[str],
        agent: NeuralNetworkRLAgent,
        fetchers: Dict[str, CoinExDataFetcher],
        histories: Dict[str, Deque[Dict]],
        last_timestamps: Dict[str, Optional[str]],
        data_paths: Dict[str, str],
        data_last_written: Dict[str, Optional[str]],
        status_path: str,
        checkpoint_path: str,
        model_path: str,
        poll_seconds: int,
        precision_window: int,
    ):
        self.symbol = symbol
        self.timeframes = timeframes
        self.primary_timeframe = timeframes[0]
        self.agent = agent
        self.fetchers = fetchers
        self.histories = histories
        self.last_timestamps = last_timestamps
        self.data_paths = data_paths
        self.data_last_written = data_last_written
        self.status_path = status_path
        self.checkpoint_path = checkpoint_path
        self.model_path = model_path
        self.poll_seconds = poll_seconds
        self.training_history: Deque[Dict] = deque(maxlen=precision_window)
        self.metrics: Dict[str, Optional[float]] = {
            "steps": 0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "epsilon": agent.epsilon,
            "last_reward": 0.0,
            "total_reward": 0.0,
            "ready": False,
            "ready_timestamp": None,
            "model_saved": False,
            "model_saved_timestamp": None,
            "accuracy_pct": 0.0,
            "stage": "initializing",
            "balance_snapshot": None,
            "training_steps": 0,
        }
        self.previous_state: Optional[Tuple] = None
        self.previous_action: Optional[str] = None
        self.previous_price: Optional[float] = None
        self.previous_volatility: float = 0.0
        self.steps_since_log = 0
        self.status_counter = 0
        self.next_poll_time = 0.0
        self.last_session_info: Dict = {}


class ReinforcementLearningTrainer:
    """Background trainer that runs multi-symbol reinforcement learning in paper mode."""

    ACTIONS = ["hold", "long", "short"]
    TIMEFRAME_SECONDS = {
        "1min": 60,
        "3min": 180,
        "5min": 300,
        "15min": 900,
        "30min": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "12h": 43200,
        "1d": 86400,
    }
    VOL_BUCKET_THRESHOLDS = [0.0005, 0.0012, 0.0025, 0.005, 0.01]

    def __init__(
        self,
        symbol: str,
        trading_mode: str = "balanced",
        simulator=None,
        mode: str = "spot",
        config_path: str = "config/rl_training.json",
    ):
        self.default_symbol = symbol
        self.mode = (mode or "spot").lower()
        self.trading_mode = (trading_mode or "balanced").lower()
        self.simulator = simulator
        self.config = self._load_config(config_path)
        self.logger = ComponentLogger.get_logger("rl_trainer")
        self.session_manager = SessionManager(self.trading_mode)

        self.learning_rate = float(self.config.get("learning_rate", 0.12))
        self.discount_factor = float(self.config.get("discount_factor", 0.9))
        self.initial_epsilon = float(self.config.get("initial_epsilon", 1.0))
        self.epsilon_min = float(self.config.get("epsilon_min", 0.05))
        self.epsilon_decay = float(self.config.get("epsilon_decay", 0.995))
        self.max_history = int(self.config.get("max_history", 400))
        self.vol_window = int(self.config.get("volatility_window", 20))
        self.reward_scaling = float(self.config.get("reward_scaling", 100.0))
        self.hold_penalty = float(self.config.get("hold_penalty", 0.2))
        self.risk_penalty = float(self.config.get("risk_penalty", 25.0))
        self.target_precision = float(self.config.get("target_precision", 0.75))
        self.precision_window = int(self.config.get("precision_window", 150))
        self.status_write_interval = max(1, int(self.config.get("status_write_interval", 5)))
        self.log_interval = max(1, int(self.config.get("log_interval", 20)))
        self.poll_min_seconds = int(self.config.get("poll_min_seconds", 30))
        self.poll_max_seconds = int(self.config.get("poll_max_seconds", 120))
        self.checkpoint_dir = self.config.get("checkpoint_dir", "sim_results/rl_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.status_lock = threading.Lock()
        self.contexts: List[TrainingContext] = []
        self._build_contexts()

        self.running = False
        self._thread: Optional[threading.Thread] = None

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #

    def start(self):
        if os.getenv("TRADE_LIVE", "0") == "1":
            raise RuntimeError("ReinforcementLearningTrainer can only operate in paper trading mode.")
        if not self.contexts:
            self.logger.warning("RL trainer has no contexts to process; check rl_training.json configuration.")
            return
        if self.running:
            return

        symbols = ",".join(sorted({ctx.symbol for ctx in self.contexts}))
        self.logger.info(
            "Starting reinforcement learning trainer",
            mode=self.mode,
            trading_mode=self.trading_mode,
            symbols=symbols,
            contexts=len(self.contexts),
        )
        self.running = True
        for ctx in self.contexts:
            self._refresh_context_snapshot(ctx)
            # write an initial snapshot so dashboards can display something immediately
            self._write_status(ctx)
            ctx.next_poll_time = time.time()
        self._thread = threading.Thread(target=self._training_loop, name="RLTrainerThread", daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        for ctx in self.contexts:
            try:
                self._write_status(ctx)
            except Exception:
                pass
        self.logger.info("Reinforcement learning trainer stopped", contexts=len(self.contexts))

    # --------------------------------------------------------------------- #
    # Setup helpers
    # --------------------------------------------------------------------- #

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _build_contexts(self):
        pairs = self.config.get("pairs") or []
        if not pairs:
            fallback_tf = self.config.get("timeframe", "1h")
            fallback_symbol = self.default_symbol or "BTC-USDT"
            pairs = [{"symbol": fallback_symbol, "timeframes": [fallback_tf]}]
        else:
            # ensure default symbol is included
            if self.default_symbol:
                present = any(
                    (pair.get("symbol") or "").upper().replace("/", "-") == self.default_symbol.upper().replace("/", "-")
                    for pair in pairs
                )
                if not present:
                    fallback_tf = self.config.get("timeframe", "1h")
                    pairs.insert(0, {"symbol": self.default_symbol, "timeframes": [fallback_tf]})

        for pair in pairs:
            symbol = (pair.get("symbol") or self.default_symbol or "BTC-USDT").upper()
            timeframes = pair.get("timeframes") or [self.config.get("timeframe", "1h")]
            timeframes = [tf.strip() for tf in timeframes if tf and tf.strip()]
            if not timeframes:
                timeframes = [self.config.get("timeframe", "1h")]
            primary_tf = timeframes[0]

            model_path = os.path.join(
                self.checkpoint_dir,
                f"{self._normalize_symbol_for_path(symbol)}_model.pkl",
            )
            
            agent = NeuralNetworkRLAgent(
                actions=self.ACTIONS,
                learning_rate=self.learning_rate,
                discount_factor=self.discount_factor,
                initial_epsilon=self.initial_epsilon,
                epsilon_min=self.epsilon_min,
                epsilon_decay=self.epsilon_decay,
                model_path=model_path,
            )

            fetchers: Dict[str, CoinExDataFetcher] = {}
            histories: Dict[str, Deque[Dict]] = {}
            last_timestamps: Dict[str, Optional[str]] = {}
            for tf in timeframes:
                fetcher = CoinExDataFetcher(
                    symbol=normalize_symbol_to_ccxt(symbol),
                    timeframe_internal=tf,
                    max_candles=max(self.max_history, 200),
                    mode=self.mode,
                )
                try:
                    fetcher.update_initial(limit=max(self.max_history, 200))
                except Exception as exc:
                    self.logger.warning(
                        "Initial candle fetch failed",
                        symbol=symbol,
                        timeframe=tf,
                        error=str(exc),
                    )
                history = deque(maxlen=self.max_history)
                for candle in list(fetcher.candles)[-self.max_history :]:
                    history.append(candle)
                if history:
                    last_timestamps[tf] = history[-1].get("timestamp")
                else:
                    last_timestamps[tf] = None
                fetchers[tf] = fetcher
                histories[tf] = history

            data_paths, data_last_written = self._prepare_data_files(symbol, timeframes, histories)

            status_path = os.path.join(
                "sim_results",
                self._normalize_symbol_for_path(symbol),
                "rl_training_status.json",
            )
            os.makedirs(os.path.dirname(status_path), exist_ok=True)
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{self._normalize_symbol_for_path(symbol)}.json",
            )

            model_path = os.path.join(
                self.checkpoint_dir,
                f"{self._normalize_symbol_for_path(symbol)}_model.pkl",
            )
            
            context = TrainingContext(
                symbol=symbol,
                timeframes=timeframes,
                agent=agent,
                fetchers=fetchers,
                histories=histories,
                last_timestamps=last_timestamps,
                data_paths=data_paths,
                data_last_written=data_last_written,
                status_path=status_path,
                checkpoint_path=checkpoint_path,
                model_path=model_path,
                poll_seconds=self._resolve_poll_seconds(primary_tf),
                precision_window=self.precision_window,
            )
            self._load_checkpoint(context)
            self.contexts.append(context)

    # --------------------------------------------------------------------- #
    # Main loop
    # --------------------------------------------------------------------- #

    def _training_loop(self):
        while self.running:
            now = time.time()
            processed_any = False
            for context in self.contexts:
                if now < context.next_poll_time:
                    continue
                try:
                    processed = self._process_context(context)
                except Exception as exc:
                    self.logger.error(
                        "RL training step failed",
                        symbol=context.symbol,
                        timeframe=context.primary_timeframe,
                        error=str(exc),
                    )
                    processed = False
                interval = context.poll_seconds if processed else min(context.poll_seconds, 30)
                context.next_poll_time = time.time() + max(1, interval)
                processed_any = processed_any or processed
            if not processed_any:
                time.sleep(1.0)

    def _process_context(self, context: TrainingContext) -> bool:
        primary_candle = None
        new_primary = False
        for tf, fetcher in context.fetchers.items():
            candle = self._fetch_latest_candle(context, tf, fetcher)
            if not candle:
                continue
            last_ts = context.last_timestamps.get(tf)
            if candle.get("timestamp") == last_ts:
                continue
            context.histories[tf].append(candle)
            context.last_timestamps[tf] = candle.get("timestamp")
            self._append_candle_to_file(context, tf, candle)
            if tf == context.primary_timeframe:
                primary_candle = candle
                new_primary = True

        if not new_primary or primary_candle is None:
            return False

        session_info = self.session_manager.get_session_info()
        state, volatility = self._build_state(context, session_info)
        current_price = float(primary_candle.get("close", 0.0) or 0.0)

        if (
            context.previous_state is not None
            and context.previous_action is not None
            and context.previous_price is not None
        ):
            reward = self._calculate_reward(
                context.previous_action,
                context.previous_price,
                current_price,
                context.previous_volatility,
            )
            # Update neural network agent (learns from mistakes)
            context.agent.update(context.previous_state, context.previous_action, reward, state)
            context.agent.decay_epsilon()
            self._update_metrics(context, reward, context.previous_action)
            self._record_outcome(context, context.previous_action, reward)
            context.metrics["epsilon"] = context.agent.epsilon
            context.metrics["last_reward"] = reward
            context.metrics["total_reward"] += reward
            context.metrics["training_steps"] = context.agent.training_steps
            
            # Check if we should save the model (target accuracy reached)
            if not context.metrics.get("model_saved", False):
                if self._should_save_model(context):
                    self._save_trained_model(context)

        action = context.agent.select_action(state)

        context.previous_state = state
        context.previous_price = current_price
        context.previous_action = action
        context.previous_volatility = volatility

        context.metrics["steps"] += 1
        context.steps_since_log += 1
        context.status_counter += 1
        context.last_session_info = session_info

        self._refresh_context_snapshot(context)
        self._log_action(context, action, primary_candle, session_info)

        if context.steps_since_log >= self.log_interval:
            self._log_progress(context)
            context.steps_since_log = 0

        if context.status_counter >= self.status_write_interval:
            self._write_status(context)
            context.status_counter = 0

        if context.metrics["ready"] and not context.metrics.get("ready_timestamp"):
            context.metrics["ready_timestamp"] = datetime.utcnow().isoformat()
            self.logger.info(
                "Reinforcement trainer met readiness criteria",
                symbol=context.symbol,
                timeframe=context.primary_timeframe,
                win_rate=f"{context.metrics['win_rate']:.2%}",
                window=self.precision_window,
            )

        return True

    # --------------------------------------------------------------------- #
    # State & reward helpers
    # --------------------------------------------------------------------- #

    def _build_state(self, context: TrainingContext, session_info: Dict) -> Tuple[Tuple[str, ...], float]:
        features: List[str] = []
        volatility_samples: List[float] = []

        for tf in context.timeframes:
            candles = context.histories.get(tf)
            if not candles or len(candles) < 3:
                continue
            recent = list(candles)[-max(self.vol_window + 2, 6) :]
            closes = [float(c.get("close", 0.0) or 0.0) for c in recent if c.get("close") is not None]
            if len(closes) < 3:
                continue

            returns = [
                (closes[i] - closes[i - 1]) / max(closes[i - 1], 1e-8)
                for i in range(1, len(closes))
            ]
            last_return = returns[-1]
            trend = "up" if last_return > 0.0015 else "down" if last_return < -0.0015 else "flat"
            volatility = pstdev(returns) if len(returns) > 1 else 0.0
            volatility_samples.append(volatility)
            vol_bucket = self._bucketize(volatility, self.VOL_BUCKET_THRESHOLDS)
            drift = (closes[-1] - closes[0]) / max(abs(closes[0]), 1e-8)
            bias = "bull" if drift > 0.001 else "bear" if drift < -0.001 else "neutral"

            features.append(f"{tf}_trend_{trend}")
            features.append(f"{tf}_vol_{vol_bucket}")
            features.append(f"{tf}_bias_{bias}")

        if not features:
            features.append("insufficient_history")

        session_type = session_info.get("session_type", "unknown")
        features.append(f"session_{session_type}")
        conf_threshold = int(session_info.get("confidence_threshold", 0))
        features.append(f"conf_{conf_threshold // 5 if conf_threshold else 0}")
        features.append(f"trade_mode_{self.trading_mode}")

        primary_hist = context.histories.get(context.primary_timeframe)
        hour_bucket = 0
        if primary_hist and primary_hist[-1].get("timestamp"):
            try:
                dt = datetime.fromisoformat(primary_hist[-1]["timestamp"].replace("Z", "+00:00"))
                hour_bucket = dt.hour // 3
            except Exception:
                hour_bucket = 0
        features.append(f"hour_{hour_bucket}")

        aggregated_vol = (
            sum(volatility_samples) / len(volatility_samples) if volatility_samples else 0.0
        )
        return tuple(features), aggregated_vol

    def _bucketize(self, value: float, thresholds: List[float]) -> int:
        for idx, threshold in enumerate(thresholds):
            if value < threshold:
                return idx
        return len(thresholds)

    def _calculate_reward(
        self,
        action: str,
        entry_price: float,
        exit_price: float,
        volatility: float,
    ) -> float:
        if entry_price <= 0 or exit_price <= 0:
            return 0.0

        change_pct = (exit_price - entry_price) / entry_price
        profit_component = change_pct * self.reward_scaling
        if action == "short":
            profit_component = -profit_component
        elif action == "hold":
            profit_component = -abs(change_pct) * self.hold_penalty

        risk_component = abs(volatility) * self.risk_penalty
        return profit_component - risk_component

    # --------------------------------------------------------------------- #
    # Metrics & logging
    # --------------------------------------------------------------------- #

    def _update_metrics(self, context: TrainingContext, reward: float, action: str):
        if action in ("long", "short"):
            context.metrics["trades"] += 1
            if reward > 0:
                context.metrics["wins"] += 1
            elif reward < 0:
                context.metrics["losses"] += 1

            total = (context.metrics["wins"] or 0) + (context.metrics["losses"] or 0)
            if total > 0:
                context.metrics["win_rate"] = context.metrics["wins"] / total

            if (
                total >= self.precision_window
                and context.metrics["win_rate"] >= self.target_precision
            ):
                context.metrics["ready"] = True

    def _record_outcome(self, context: TrainingContext, action: str, reward: float):
        if action not in ("long", "short"):
            return
        outcome = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "reward": reward,
            "win": reward > 0,
            "timeframe": context.primary_timeframe,
        }
        context.training_history.append(outcome)

        if context.training_history:
            recent = [entry for entry in context.training_history if entry.get("action") in ("long", "short")]
            if recent:
                wins = sum(1 for entry in recent if entry.get("win"))
                context.metrics["win_rate"] = wins / len(recent)
                if len(recent) >= self.precision_window and context.metrics["win_rate"] >= self.target_precision:
                    context.metrics["ready"] = True

    def _refresh_context_snapshot(self, context: TrainingContext):
        context.metrics["epsilon"] = context.agent.epsilon
        context.metrics["accuracy_pct"] = round((context.metrics.get("win_rate") or 0.0) * 100.0, 2)
        context.metrics["stage"] = self._determine_stage(context)
        context.metrics["balance_snapshot"] = self._get_balance_snapshot()
        context.metrics["training_steps"] = context.agent.training_steps

    def _determine_stage(self, context: TrainingContext) -> str:
        if context.metrics.get("ready"):
            return "ready"
        trades = context.metrics.get("trades") or 0
        epsilon = context.agent.epsilon
        win_rate = context.metrics.get("win_rate") or 0.0
        if trades < max(40, self.precision_window // 6):
            return "exploration"
        if epsilon > max(self.epsilon_min * 3, 0.25):
            return "epsilon_decay"
        if win_rate >= self.target_precision * 0.9:
            return "fine_tuning"
        return "optimization"

    def _get_balance_snapshot(self) -> Optional[float]:
        if not self.simulator:
            return None
        ledger = getattr(self.simulator, "ledger", None)
        if ledger and hasattr(ledger, "get_balance"):
            try:
                return float(ledger.get_balance())
            except Exception:
                pass
        if hasattr(self.simulator, "balance"):
            try:
                return float(self.simulator.balance)
            except Exception:
                pass
        return None

    def _log_action(self, context: TrainingContext, action: str, candle: Dict, session_info: Dict):
        if action == "hold":
            return
        side = "BUY" if action == "long" else "SELL"
        entry_price = float(candle.get("close", 0.0) or 0.0)
        trade_payload = {
            "event": "RL_ACTION",
            "symbol": context.symbol,
            "side": side,
            "entry": entry_price,
            "timeframe": context.primary_timeframe,
            "session_type": session_info.get("session_type"),
            "mode": "paper",
            "source": "reinforcement_trainer",
        }
        try:
            ComponentLogger.trading_logger().log_trade(trade_payload)
        except Exception:
            self.logger.debug(
                "Unable to forward RL action to trading logger",
                symbol=context.symbol,
                action=action,
            )

    def _log_progress(self, context: TrainingContext):
        with self.status_lock:
            self.logger.info(
                "RL training progress",
                symbol=context.symbol,
                timeframe=context.primary_timeframe,
                steps=context.metrics.get("steps"),
                trades=context.metrics.get("trades"),
                win_rate=f"{context.metrics.get('win_rate', 0.0):.2%}",
                accuracy=f"{context.metrics.get('accuracy_pct', 0.0):.2f}%",
                epsilon=f"{context.metrics.get('epsilon', context.agent.epsilon):.3f}",
                total_reward=f"{context.metrics.get('total_reward', 0.0):.2f}",
                ready=context.metrics.get("ready"),
                stage=context.metrics.get("stage"),
                balance=context.metrics.get("balance_snapshot"),
            )

    def _write_status(self, context: TrainingContext):
        with self.status_lock:
            status_snapshot = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": context.symbol,
                "mode": self.mode,
                "trading_mode": self.trading_mode,
                "timeframes": context.timeframes,
                "primary_timeframe": context.primary_timeframe,
                "metrics": context.metrics,
                "stage": context.metrics.get("stage"),
                "accuracy_pct": context.metrics.get("accuracy_pct"),
                "balance_snapshot": context.metrics.get("balance_snapshot"),
                "target_precision": self.target_precision,
                "precision_window": self.precision_window,
                "reward_scaling": self.reward_scaling,
                "risk_penalty": self.risk_penalty,
                "recent_trades": list(context.training_history)[-10:],
                "last_session_info": context.last_session_info,
            }
            try:
                with open(context.status_path, "w", encoding="utf-8") as f:
                    json.dump(status_snapshot, f, indent=2)
            except Exception as exc:
                self.logger.error(
                    "Failed to write RL training status",
                    symbol=context.symbol,
                    error=str(exc),
                )
        self._save_checkpoint(context)

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #

    def _load_checkpoint(self, context: TrainingContext):
        # Try to load neural network model first
        if os.path.exists(context.model_path):
            if context.agent.load_model(context.model_path):
                self.logger.info(
                    "Loaded trained RL neural network model",
                    symbol=context.symbol,
                    path=context.model_path,
                )
        
        # Load checkpoint data
        if not os.path.exists(context.checkpoint_path):
            return
        try:
            with open(context.checkpoint_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            self.logger.warning(
                "Failed to load RL checkpoint",
                symbol=context.symbol,
                error=str(exc),
            )
            return

        context.agent.epsilon = float(payload.get("epsilon", context.agent.epsilon))
        context.agent.training_steps = int(payload.get("training_steps", 0))

        # Metrics & history
        stored_metrics = payload.get("metrics") or {}
        context.metrics.update(stored_metrics)
        context.metrics["epsilon"] = context.agent.epsilon
        context.metrics["training_steps"] = context.agent.training_steps
        context.training_history.clear()
        for item in payload.get("training_history", []):
            context.training_history.append(item)

        context.previous_state = tuple(payload.get("previous_state", [])) if payload.get("previous_state") else None
        context.previous_action = payload.get("previous_action")
        context.previous_price = payload.get("previous_price")
        context.previous_volatility = payload.get("previous_volatility", 0.0)
        context.last_timestamps.update(payload.get("last_timestamps") or {})
        self._refresh_context_snapshot(context)

    def _save_checkpoint(self, context: TrainingContext):
        self._refresh_context_snapshot(context)
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": context.symbol,
            "timeframes": context.timeframes,
            "primary_timeframe": context.primary_timeframe,
            "epsilon": context.agent.epsilon,
            "training_steps": context.agent.training_steps,
            "metrics": context.metrics,
            "training_history": list(context.training_history),
            "previous_state": list(context.previous_state) if context.previous_state else None,
            "previous_action": context.previous_action,
            "previous_price": context.previous_price,
            "previous_volatility": context.previous_volatility,
            "last_timestamps": context.last_timestamps,
        }
        try:
            with open(context.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            self.logger.error(
                "Failed to persist RL checkpoint",
                symbol=context.symbol,
                error=str(exc),
            )
    
    def _should_save_model(self, context: TrainingContext) -> bool:
        """Check if model should be saved (target accuracy reached)"""
        if context.metrics.get("model_saved", False):
            return False
        
        accuracy = context.metrics.get("accuracy_pct", 0.0) / 100.0
        trades = context.metrics.get("trades", 0)
        
        # Save model when target accuracy is reached with sufficient trades
        if trades >= self.precision_window and accuracy >= self.target_precision:
            return True
        
        return False
    
    def _save_trained_model(self, context: TrainingContext):
        """Save the trained neural network model when target accuracy is reached"""
        try:
            context.agent.save_model(context.model_path)
            context.metrics["model_saved"] = True
            context.metrics["model_saved_timestamp"] = datetime.utcnow().isoformat()
            
            self.logger.info(
                "✅ Trained RL model saved successfully",
                symbol=context.symbol,
                accuracy=f"{context.metrics.get('accuracy_pct', 0.0):.2f}%",
                target_accuracy=f"{self.target_precision * 100:.2f}%",
                trades=context.metrics.get("trades", 0),
                model_path=context.model_path,
            )
            
            # Also save a production-ready model in a dedicated directory
            production_dir = os.path.join("models", "rl_production")
            os.makedirs(production_dir, exist_ok=True)
            production_path = os.path.join(
                production_dir,
                f"{self._normalize_symbol_for_path(context.symbol)}_rl_model.pkl",
            )
            context.agent.save_model(production_path)
            
            self.logger.info(
                "✅ Production RL model saved",
                symbol=context.symbol,
                production_path=production_path,
            )
        except Exception as exc:
            self.logger.error(
                "Failed to save trained RL model",
                symbol=context.symbol,
                error=str(exc),
            )

    # --------------------------------------------------------------------- #
    # Data persistence utilities
    # --------------------------------------------------------------------- #

    def _prepare_data_files(
        self, symbol: str, timeframes: List[str], histories: Dict[str, Deque[Dict]]
    ) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
        base = self._base_asset_folder(symbol)
        base_dir = os.path.join("data", base)
        os.makedirs(base_dir, exist_ok=True)
        data_paths: Dict[str, str] = {}
        data_last_written: Dict[str, Optional[str]] = {}
        for tf in timeframes:
            path = os.path.join(base_dir, f"{tf}.json")
            history = histories.get(tf)
            last_written = self._initialize_data_file(path, history)
            data_paths[tf] = path
            data_last_written[tf] = last_written
        return data_paths, data_last_written

    def _initialize_data_file(
        self, path: str, history: Optional[Deque[Dict]]
    ) -> Optional[str]:
        last_timestamp = None
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        last_timestamp = json.loads(last_line).get("timestamp")
            except Exception:
                last_timestamp = None
        else:
            try:
                open(path, "w", encoding="utf-8").close()
            except Exception:
                return None
        if last_timestamp is None and history:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    for candle in history:
                        f.write(json.dumps(candle) + "\n")
                last_timestamp = history[-1].get("timestamp")
            except Exception:
                pass
        self._trim_data_file(path)
        return last_timestamp

    def _append_candle_to_file(self, context: TrainingContext, timeframe: str, candle: Dict):
        path = context.data_paths.get(timeframe)
        if not path:
            return
        timestamp = candle.get("timestamp")
        if not timestamp:
            return
        if context.data_last_written.get(timeframe) == timestamp:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(candle) + "\n")
            context.data_last_written[timeframe] = timestamp
            self._trim_data_file(path)
        except Exception:
            pass

    def _trim_data_file(self, path: str):
        if self.max_history <= 0 or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= self.max_history:
                return
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines[-self.max_history :])
        except Exception:
            pass

    @staticmethod
    def _base_asset_folder(symbol: str) -> str:
        s = (symbol or "").strip()
        if not s:
            return "unknown"
        for sep in ["-", "/", "_"]:
            if sep in s:
                parts = s.split(sep)
                if parts:
                    return parts[0].lower()
        common_quotes = ["USDT", "USD", "USDC", "BTC", "ETH", "BUSD", "TUSD", "EUR", "GBP", "JPY"]
        upper = s.upper()
        for quote in common_quotes:
            if upper.endswith(quote) and len(upper) > len(quote):
                base = upper[: -len(quote)]
                return base.lower()
        return s.lower()

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #

    def _fetch_latest_candle(self, context: TrainingContext, timeframe: str, fetcher: CoinExDataFetcher):
        try:
            return fetcher.fetch_latest_closed()
        except Exception as exc:
            self.logger.warning(
                "Failed to fetch latest candle",
                symbol=context.symbol,
                timeframe=timeframe,
                error=str(exc),
            )
            return None

    def _resolve_poll_seconds(self, timeframe: str) -> int:
        tf_seconds = self.TIMEFRAME_SECONDS.get(timeframe, 3600)
        derived = tf_seconds // 6 if tf_seconds >= 60 else tf_seconds
        derived = max(self.poll_min_seconds, derived or self.poll_min_seconds)
        return min(self.poll_max_seconds, derived)

    @staticmethod
    def _normalize_symbol_for_path(symbol: str) -> str:
        return (symbol or "UNKNOWN").upper().replace("/", "-").replace("_", "-")

