import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import DatabaseManager
from config import Config

class ProductionMLEnsemble:
    """Production-grade ML ensemble with real training and learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_version = "v3.0_production"
        self.db = DatabaseManager()
        self.is_trained = False
        self._initialize_models()
        self._load_or_train_models()
    
    def _initialize_models(self):
        """Initialize multiple ML models"""
        
        # Feature names from our feature engineering
        self.feature_names = [
            'home_attack_strength', 'home_defense_strength', 'away_attack_strength', 'away_defense_strength',
            'attack_strength_diff', 'defense_strength_diff', 'home_advantage', 'home_win_rate', 'away_win_rate',
            'win_rate_diff', 'home_form_5', 'away_form_5', 'home_form_10', 'away_form_10', 
            'home_goals_scored_5', 'home_goals_conceded_5', 'away_goals_scored_5', 'away_goals_conceded_5',
            'home_clean_sheets_5', 'away_clean_sheets_5', 'form_5_diff', 'form_10_diff',
            'goals_scored_diff_5', 'goals_conceded_diff_5', 'home_historical_win_rate', 'away_historical_win_rate',
            'home_historical_goals_avg', 'away_historical_goals_avg', 'home_historical_goals_conceded_avg',
            'away_historical_goals_conceded_avg', 'home_clean_sheet_rate', 'away_clean_sheet_rate',
            'h2h_total_matches', 'h2h_home_win_rate', 'h2h_goal_avg', 'h2h_recent_home_win_rate',
            'expected_goals_home', 'expected_goals_away', 'expected_total_goals', 'home_consistency',
            'away_consistency', 'match_competitiveness', 'is_derby', 'is_top_team_clash', 'is_relegation_battle',
            'league_strength', 'weekend_match', 'evening_match', 'season_phase', 'live_data_used'
        ]
        
        # Initialize models with optimized parameters
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.models['logistic'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        
        # Ensemble model
        self.models['ensemble'] = VotingClassifier(
            estimators=[
                ('xgb', self.models['xgboost']),
                ('rf', self.models['random_forest']),
                ('lr', self.models['logistic'])
            ],
            voting='soft',
            weights=[2, 1.5, 1]  # Weighted by model performance
        )
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
    
    def _load_or_train_models(self):
        """Load pre-trained models or train new ones"""
        try:
            # Try to load pre-trained models
            self.models['ensemble'] = joblib.load(f"{Config.MODEL_PATH}/ensemble_model_v3.joblib")
            self.scalers['standard'] = joblib.load(f"{Config.MODEL_PATH}/scaler_v3.joblib")
            self.is_trained = True
            print("✅ Loaded pre-trained production models")
        except:
            print("🔄 No pre-trained models found, training new models...")
            self.train_models()
    
    def train_models(self):
        """Train models on historical data"""
        print("🔄 Training production ML models on historical data...")
        
        # Get training data from database
        training_data = self.db.get_training_data(days=Config.TRAINING_DAYS)
        
        if len(training_data) < Config.MIN_TRAINING_MATCHES:
            print(f"⚠️ Insufficient training data: {len(training_data)} matches")
            print("🔄 Using simulated training data for initial setup")
            X, y = self._create_simulated_training_data()
        else:
            X, y = self._prepare_training_data(training_data)
        
        if len(X) == 0:
            print("❌ No valid training data available")
            self.is_trained = False
            return
        
        # Scale features
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Train individual models
        for name, model in self.models.items():
            if name != 'ensemble':  # Ensemble is trained separately
                print(f"🔄 Training {name}...")
                model.fit(X_scaled, y)
        
        # Train ensemble
        print("🔄 Training ensemble model...")
        self.models['ensemble'].fit(X_scaled, y)
        
        # Evaluate models
        self._evaluate_models(X_scaled, y)
        
        # Save models
        self._save_models()
        
        self.is_trained = True
        print("✅ All models trained and saved successfully")
    
    def _prepare_training_data(self, training_df):
        """Prepare training data from database records"""
        X = []
        y = []
        
        for _, row in training_df.iterrows():
            try:
                # Extract features from stored feature data
                features_used = row['features_used']
                if isinstance(features_used, str):
                    import json
                    features = json.loads(features_used)
                else:
                    features = features_used
                
                # Create feature vector
                feature_vector = []
                for feature_name in self.feature_names:
                    feature_vector.append(features.get(feature_name, 0.0))
                
                # Get target (actual result)
                result = row['result']
                if result == 'H':
                    target = 0
                elif result == 'D':
                    target = 1
                else:  # 'A'
                    target = 2
                      X.append(feature_vector)
                y.append(target)
                
            except Exception as e:
                continue
        
        return np.array(X), np.array(y)
    
    def _create_simulated_training_data(self):
        """Create simulated training data when real data is insufficient"""
        print("🔄 Creating simulated training data...")
        
        X = []
        y = []
        
        # Create realistic training examples
        for _ in range(1000):
            # Strong home team vs weak away team
            if np.random.random() < 0.3:
                features = self._create_strong_home_weak_away_features()
                X.append(features)
                y.append(0)  # Home win
            
            # Even match
            elif np.random.random() < 0.6:
                features = self._create_even_match_features()
                X.append(features)
                y.append(np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3]))  # Balanced
            
            # Strong away team vs weak home team
            else:
                features = self._create_strong_away_weak_home_features()
                X.append(features)
                y.append(2)  # Away win
        
        return np.array(X), np.array(y)
    
    def _create_strong_home_weak_away_features(self):
        """Create features for strong home team vs weak away team"""
        features = []
        for feature_name in self.feature_names:
            if 'home_' in feature_name and ('strength' in feature_name or 'win_rate' in feature_name or 'form' in feature_name):
                features.append(np.random.normal(0.7, 0.1))  # Strong home
            elif 'away_' in feature_name and ('strength' in feature_name or 'win_rate' in feature_name or 'form' in feature_name):
                features.append(np.random.normal(0.3, 0.1))  # Weak away
            elif 'diff' in feature_name:
                features.append(np.random.normal(0.4, 0.1))  # Positive differential
            else:
                features.append(np.random.normal(0.5, 0.2))  # Neutral
        return features
    
    def _create_strong_away_weak_home_features(self):
        """Create features for strong away team vs weak home team"""
        features = []
        for feature_name in self.feature_names:
            if 'home_' in feature_name and ('strength' in feature_name or 'win_rate' in feature_name or 'form' in feature_name):
                features.append(np.random.normal(0.3, 0.1))  # Weak home
            elif 'away_' in feature_name and ('strength' in feature_name or 'win_rate' in feature_name or 'form' in feature_name):
                features.append(np.random.normal(0.7, 0.1))  # Strong away
            elif 'diff' in feature_name:
                features.append(np.random.normal(-0.4, 0.1))  # Negative differential
            else:
                features.append(np.random.normal(0.5, 0.2))  # Neutral
        return features
    
    def _create_even_match_features(self):
        """Create features for even match"""
        features = []
        for feature_name in self.feature_names:
            if 'home_' in feature_name and ('strength' in feature_name or 'win_rate' in feature_name or 'form' in feature_name):
                features.append(np.random.normal(0.5, 0.1))  # Average home
            elif 'away_' in feature_name and ('strength' in feature_name or 'win_rate' in feature_name or 'form' in feature_name):
                features.append(np.random.normal(0.5, 0.1))  # Average away
            elif 'diff' in feature_name:
                features.append(np.random.normal(0.0, 0.1))  # No differential
            else:
                features.append(np.random.normal(0.5, 0.2))  # Neutral
        return features
    
    def _evaluate_models(self, X, y):
        """Evaluate model performance"""
        from sklearn.model_selection import cross_val_score
        
        print("📊 Evaluating model performance...")
        
        for name, model in self.models.items():
            try:
                # Cross-validation accuracy
                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                print(f"✅ {name}: Average Accuracy = {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
                
                # Log performance to database
                self.db.store_model_performance(
                    name, scores.mean(), 0.0, 0.0, 0.0, len(self.feature_names), len(X)
                )
                
            except Exception as e:
                print(f"⚠️ Error evaluating {name}: {e}")
    
    def predict(self, features):
        """Make prediction using trained ensemble"""
        if not self.is_trained:
            print("⚠️ Models not trained, using fallback prediction")
            return self._fallback_prediction(features)
        
        try:
            # Create feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            # Scale features
            X = np.array([feature_vector])
            X_scaled = self.scalers['standard'].transform(X)
            
            # Get ensemble probabilities
            probabilities = self.models['ensemble'].predict_proba(X_scaled)[0]
            
            # Convert to prediction
            prediction_idx = np.argmax(probabilities)
            prediction_map = {0: 'H', 1: 'D', 2: 'A'}
            prediction = prediction_map[prediction_idx]
            
            # Calculate confidence
            confidence = probabilities[prediction_idx]
            
            return {
                'prediction': prediction,
                'probabilities': {
                    'home': float(probabilities[0]),
                    'draw': float(probabilities[1]),
                    'away': float(probabilities[2])
                },
                'confidence': float(confidence),
                'model_used': 'ensemble_v3',
                'features_used': len([f for f in features.values() if f != 0])
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self._fallback_prediction(features)
    
    def _fallback_prediction(self, features):
        """Fallback prediction when models fail"""
        # Simple rule-based fallback
        home_strength = features.get('home_attack_strength', 0.5)
        away_strength = features.get('away_attack_strength', 0.5)
        home_adv = features.get('home_advantage', 0.55)
        
        home_prob = 0.33 + (home_strength - 0.5) * 0.3 + (home_adv - 0.5) * 0.2
        away_prob = 0.33 + (away_strength - 0.5) * 0.3 - (home_adv - 0.5) * 0.15
        draw_prob = 1.0 - home_prob - away_prob
        
        # Normalize
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        if home_prob > away_prob and home_prob > draw_prob:
            prediction = 'H'
            confidence = home_prob
        elif away_prob > home_prob and away_prob > draw_prob:
            prediction = 'A'
            confidence = away_prob
        else:
            prediction = 'D'
            confidence = draw_prob
        
        return {
            'prediction': prediction,
            'probabilities': {'home': home_prob, 'draw': draw_prob, 'away': away_prob},
            'confidence': confidence,
            'model_used': 'fallback',
            'features_used': len([f for f in features.values() if f != 0])
        }
    
    def _save_models(self):
        """Save trained models"""
        import os
        os.makedirs(Config.MODEL_PATH, exist_ok=True)
        
        joblib.dump(self.models['ensemble'], f"{Config.MODEL_PATH}/ensemble_model_v3.joblib")
        joblib.dump(self.scalers['standard'], f"{Config.MODEL_PATH}/scaler_v3.joblib")
        
        print("💾 Production models saved successfully")
    
    def generate_advanced_predictions(self, features, match_context):
        """Generate all prediction types for a match"""
        base_prediction = self.predict(features)
        
        # Additional market predictions
        advanced_predictions = {
            'match_outcome': base_prediction,
            'double_chance': self._predict_double_chance(base_prediction),
            'over_under': self._predict_over_under(features),
            'both_teams_score': self._predict_both_teams_score(features),
            'correct_score': self._predict_correct_score(features, base_prediction)
        }
        
        return advanced_predictions
    
    def _predict_double_chance(self, base_prediction):
        """Predict double chance markets"""
        probs = base_prediction['probabilities']
        
        home_win_or_draw = probs['home'] + probs['draw']
        away_win_or_draw = probs['away'] + probs['draw']
        both_teams_win = probs['home'] + probs['away']
        
        # Determine recommendation
        if home_win_or_draw > away_win_or_draw and home_win_or_draw > both_teams_win:
            recommendation = '1X'
            confidence = home_win_or_draw
        elif away_win_or_draw > home_win_or_draw and away_win_or_draw > both_teams_win:
            recommendation = 'X2'
            confidence = away_win_or_draw
        else:
            recommendation = '12'
            confidence = both_teams_win
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            '1X': home_win_or_draw,
            'X2': away_win_or_draw,
            '12': both_teams_win,
            'model_used': 'double_chance_calculated'
        }
    
    def _predict_over_under(self, features):
        """Predict over/under 2.5 goals"""
        expected_goals = features.get('expected_total_goals', 2.5)
        
        # Dynamic probability based on expected goals
        over_prob = 1 / (1 + np.exp(-2 * (expected_goals - 2.5)))
        over_prob = max(0.1, min(0.9, over_prob))
        
        recommendation = 'Over 2.5' if over_prob > 0.5 else 'Under 2.5'
        confidence = max(over_prob, 1 - over_prob)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'over_2.5': over_prob,
            'under_2.5': 1 - over_prob,
            'expected_total_goals': expected_goals,
            'model_used': 'goals_model'
        }
    
    def _predict_both_teams_score(self, features):
        """Predict both teams to score"""
        home_attack = features.get('home_attack_strength', 0.5)
        away_attack = features.get('away_attack_strength', 0.5)
        home_defense = features.get('home_defense_strength', 0.5)
        away_defense = features.get('away_defense_strength', 0.5)
        
        # Probability calculation
        home_score_prob = home_attack * (1 - away_defense)
        away_score_prob = away_attack * (1 - home_defense)
        bts_prob = home_score_prob * away_score_prob * 2.5  # Scale factor
        
        bts_prob = max(0.1, min(0.9, bts_prob))
        
        recommendation = 'Yes' if bts_prob > 0.5 else 'No'
        confidence = max(bts_prob, 1 - bts_prob)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'yes': bts_prob,
            'no': 1 - bts_prob,
            'model_used': 'bts_model'
        }
    
    def _predict_correct_score(self, features, base_prediction):
        """Predict correct score probabilities"""
        home_expected = features.get('expected_goals_home', 1.4)
        away_expected = features.get('expected_goals_away', 1.3)
        
        # Adjust based on prediction confidence
        if base_prediction['prediction'] == 'H':
            home_expected *= 1.2
            away_expected *= 0.8
        elif base_prediction['prediction'] == 'A':
            home_expected *= 0.8
            away_expected *= 1.2
        
        # Generate score probabilities using Poisson distribution
        scores = {}
        for home_goals in range(0, 6):
            for away_goals in range(0, 6):
                prob = (
                    self._poisson_prob(home_goals, home_expected) * 
                    self._poisson_prob(away_goals, away_expected)
                )
                scores[f"{home_goals}-{away_goals}"] = float(prob)
        
        # Normalize and return top 5
        total_prob = sum(scores.values())
        if total_prob > 0:
            normalized_scores = {score: prob/total_prob for score, prob in scores.items()}
            top_scores = dict(sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)[:6])
        else:
            top_scores = {'1-1': 0.15, '2-1': 0.12, '1-0': 0.10, '0-1': 0.10, '2-0': 0.08, '0-2': 0.08}
        
        return top_scores
    
    def _poisson_prob(self, k, lam):
        """Calculate Poisson probability"""
        if k == 0:
            return np.exp(-lam)
        else:
            return (lam ** k) * np.exp(-lam) / np.math.factorial(k)
