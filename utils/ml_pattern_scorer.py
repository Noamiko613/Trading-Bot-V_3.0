"""
Machine Learning-Based Pattern Scoring
=======================================

Uses ML to improve pattern confidence scoring based on historical performance.
Features:
- Train models on historical pattern outcomes
- Predict pattern success probability
- Feature engineering from technical indicators
- Model persistence and retraining
- Performance tracking and model evaluation
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler


class MLPatternScorer:
    """ML-based pattern confidence scoring"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ML pattern scorer
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model if exists"""
        model_path = os.path.join(self.model_dir, "pattern_scorer.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                # Load feature names
                meta_path = os.path.join(self.model_dir, "model_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        self.feature_names = meta.get('feature_names', [])
                
                self.is_trained = True
                print(f"[ML] Loaded trained model from {model_path}")
            except Exception as e:
                print(f"[ML] Error loading model: {e}")
                self.is_trained = False
    
    def _save_model(self):
        """Save trained model"""
        model_path = os.path.join(self.model_dir, "pattern_scorer.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        meta_path = os.path.join(self.model_dir, "model_meta.json")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        meta = {
            'feature_names': self.feature_names,
            'trained_at': datetime.now().isoformat(),
            'model_type': type(self.model).__name__
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"[ML] Model saved to {model_path}")
    
    def extract_features(self, pattern_data: Dict) -> Dict[str, float]:
        """
        Extract features from pattern data for ML model
        
        Args:
            pattern_data: Dictionary containing pattern information and technical indicators
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Pattern-specific features
        features['confidence'] = pattern_data.get('confidence', 0)
        
        # Technical indicator features
        indicators = pattern_data.get('indicators', {})
        features['rsi'] = indicators.get('rsi', 50)
        features['macd'] = indicators.get('macd', 0)
        features['macd_signal'] = indicators.get('macd_signal', 0)
        features['macd_hist'] = indicators.get('macd_hist', 0)
        
        # Volume features
        features['volume_ratio'] = pattern_data.get('volume_ratio', 1.0)
        features['avg_volume'] = pattern_data.get('avg_volume', 0)
        
        # Price features
        features['price'] = pattern_data.get('price', 0)
        features['atr'] = indicators.get('atr', 0)
        features['atr_pct'] = (indicators.get('atr', 0) / pattern_data.get('price', 1)) * 100
        
        # Moving average features
        features['ma50'] = indicators.get('ma50', 0)
        features['ma200'] = indicators.get('ma200', 0)
        features['price_to_ma50'] = (pattern_data.get('price', 0) / indicators.get('ma50', 1)) - 1 if indicators.get('ma50', 0) > 0 else 0
        features['price_to_ma200'] = (pattern_data.get('price', 0) / indicators.get('ma200', 1)) - 1 if indicators.get('ma200', 0) > 0 else 0
        features['ma50_to_ma200'] = (indicators.get('ma50', 0) / indicators.get('ma200', 1)) - 1 if indicators.get('ma200', 0) > 0 else 0
        
        # EMA features
        for ema in [5, 8, 13, 21]:
            ema_key = f'ema{ema}'
            features[ema_key] = indicators.get(ema_key, 0)
        
        # Session features
        session_info = pattern_data.get('session', {})
        features['session_type'] = self._encode_session_type(session_info.get('type', 'low_liquidity'))
        features['position_multiplier'] = session_info.get('position_multiplier', 1.0)
        
        # Multi-timeframe features
        mtf = pattern_data.get('multi_timeframe', {})
        features['higher_tf_aligned'] = 1.0 if mtf.get('aligned', False) else 0.0
        features['num_confirming_tfs'] = mtf.get('confirming_timeframes', 0)
        
        # Pattern type encoding
        pattern_name = pattern_data.get('pattern', 'unknown')
        features['pattern_type'] = self._encode_pattern_type(pattern_name)
        
        # Risk/Reward features
        setup = pattern_data.get('setup', {})
        features['rr_ratio'] = setup.get('rr', 0)
        features['risk_pct'] = setup.get('risk_pct', 0)
        
        return features
    
    def _encode_session_type(self, session_type: str) -> float:
        """Encode session type as numeric value"""
        session_map = {
            'high_liquidity': 3.0,
            'medium_liquidity': 2.0,
            'low_liquidity': 1.0,
            'weekend': 0.0
        }
        return session_map.get(session_type, 1.0)
    
    def _encode_pattern_type(self, pattern_name: str) -> float:
        """Encode pattern type as numeric value"""
        pattern_map = {
            'golden_cross': 10.0,
            'death_cross': 9.0,
            'macd_cross': 8.0,
            'rsi_divergence': 7.0,
            'head_shoulders': 6.0,
            'double_top_bottom': 5.0,
            'bull_flag': 4.0,
            'triangle_breakout': 3.0,
            'bullish_engulfing': 2.0,
            'hammer': 1.0,
            'confluence_engine': 11.0
        }
        
        for key, value in pattern_map.items():
            if key.lower() in pattern_name.lower():
                return value
        
        return 0.0
    
    def train(self, training_data: List[Dict], test_size: float = 0.2):
        """
        Train ML model on historical pattern data
        
        Args:
            training_data: List of dictionaries with pattern data and outcomes
            test_size: Fraction of data to use for testing
        """
        if len(training_data) < 50:
            print(f"[ML] Insufficient training data ({len(training_data)} samples). Need at least 50.")
            return
        
        # Extract features and labels
        X = []
        y = []
        
        for data in training_data:
            features = self.extract_features(data)
            X.append(list(features.values()))
            
            # Label: 1 if trade was profitable, 0 otherwise
            outcome = data.get('outcome', {})
            y.append(1 if outcome.get('pnl', 0) > 0 else 0)
        
        # Store feature names
        self.feature_names = list(features.keys())
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        print(f"[ML] Training model on {len(X_train)} samples...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        print(f"[ML] Model Performance:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"  CV Score:  {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n[ML] Top 10 Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.is_trained = True
        self._save_model()
    
    def predict_confidence(self, pattern_data: Dict) -> float:
        """
        Predict pattern confidence using trained ML model
        
        Args:
            pattern_data: Dictionary containing pattern information
        
        Returns:
            Confidence score (0-100)
        """
        if not self.is_trained:
            # Fallback to original confidence if model not trained
            return pattern_data.get('confidence', 50)
        
        # Extract features
        features = self.extract_features(pattern_data)
        X = np.array([list(features.values())])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probability of success
        prob = self.model.predict_proba(X_scaled)[0][1]  # Probability of class 1 (profitable)
        
        # LAYER 4: Use ML only as a nudge, not full decision maker
        original_confidence = pattern_data.get('confidence', 50)
        ml_confidence = prob * 100
        
        # Calculate ML nudge (difference between ML and original)
        ml_nudge = ml_confidence - original_confidence
        
        # Bound the nudge (±5% initially, configurable)
        ml_nudge_limit = float(os.getenv('ML_NUDGE_LIMIT', '0.05')) * 100  # Convert to percentage points
        ml_nudge = max(-ml_nudge_limit, min(ml_nudge_limit, ml_nudge))
        
        # Apply bounded nudge to original confidence
        adjusted_confidence = original_confidence + ml_nudge
        
        return min(100, max(0, adjusted_confidence))
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from trained model"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


class PatternDataCollector:
    """Collect pattern data for ML training"""
    
    def __init__(self, db_path: str = "sim_results/trades.db"):
        """
        Initialize data collector
        
        Args:
            db_path: Path to trades database
        """
        self.db_path = db_path
        self.training_data = []
    
    def collect_from_trades(self) -> List[Dict]:
        """Collect training data from closed trades"""
        import sqlite3
        
        if not os.path.exists(self.db_path):
            return []
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades_closed")
            columns = [desc[0] for desc in cursor.description]
            
            training_data = []
            for row in cursor.fetchall():
                trade = dict(zip(columns, row))
                
                # Convert to training format
                # Note: This assumes pattern data was stored somewhere
                # In production, you'd need to store full pattern data with each trade
                training_data.append(trade)
            
            return training_data
        finally:
            conn.close()


if __name__ == "__main__":
    # Test ML pattern scorer
    scorer = MLPatternScorer()
    
    # Example pattern data
    pattern_data = {
        'pattern': 'Golden Cross',
        'confidence': 75,
        'price': 45000,
        'volume_ratio': 1.5,
        'indicators': {
            'rsi': 65,
            'macd': 150,
            'macd_signal': 100,
            'macd_hist': 50,
            'atr': 500,
            'ma50': 44000,
            'ma200': 43000,
            'ema5': 45200,
            'ema8': 45100,
            'ema13': 45000,
            'ema21': 44800
        },
        'session': {
            'type': 'high_liquidity',
            'position_multiplier': 1.0
        },
        'multi_timeframe': {
            'aligned': True,
            'confirming_timeframes': 3
        },
        'setup': {
            'rr': 2.5,
            'risk_pct': 1.0
        }
    }
    
    # Extract features
    features = scorer.extract_features(pattern_data)
    print("Extracted Features:")
    for k, v in features.items():
        print(f"  {k}: {v}")
    
    # Note: Training would require historical data
    print("\nML Pattern Scorer initialized. To train, provide historical trade data.")
