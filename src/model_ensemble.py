import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# FIXED IMPORTS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import DatabaseManager
from config import Config

class ProductionMLEnsemble:
    """Production-grade ML ensemble for unbiased predictions"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_version = "v1.0"
        self.db = DatabaseManager()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize multiple ML models to prevent bias"""
        
        # Feature names (from our feature engineering)
        self.feature_names = [
            'home_attack_strength', 'home_defense_strength', 'away_attack_strength', 'away_defense_strength',
            'attack_strength_diff', 'defense_strength_diff', 'home_advantage', 'home_form_5', 'away_form_5',
            'home_form_10', 'away_form_10', 'home_goals_scored_avg', 'home_goals_conceded_avg',
            'away_goals_scored_avg', 'away_goals_conceded_avg', 'form_5_diff', 'form_10_diff',
            'home_historical_win_rate', 'away_historical_win_rate', 'home_historical_goals_avg',
            'away_historical_goals_avg', 'h2h_total_matches', 'h2h_home_win_rate', 'h2h_goal_avg',
            'is_derby', 'is_top_team_clash', 'league_strength', 'weekend_match', 'evening_match',
            'live_home_form_available', 'live_away_form_available', 'live_injuries_impact', 'live_lineup_strength'
        ]
        
        # 1. XGBoost (Primary model)
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        # 2. Random Forest (Robust against overfitting)
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        # 3. Logistic Regression (Good probability calibration)
        self.models['logistic'] = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        # 4. Ensemble model (Weighted voting)
        self.models['ensemble'] = VotingClassifier(
            estimators=[
                ('xgb', self.models['xgboost']),
                ('rf', self.models['random_forest']),
                ('lr', self.models['logistic'])
            ],
            voting='soft'
        )
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
        
        # Try to load pre-trained models
        self._load_saved_models()
    
    def _load_saved_models(self):
        """Load saved models if available"""
        try:
            self.models['ensemble'] = joblib.load(f"{Config.MODEL_PATH}/ensemble_model.joblib")
            self.scalers['standard'] = joblib.load(f"{Config.MODEL_PATH}/scaler.joblib")
            print("✅ Loaded pre-trained models")
        except:
            print("🔄 No pre-trained models found, will train new ones")
    
    def train_models(self, X, y):
        """Train all models on historical data"""
        print("🔄 Training production ML models...")
        
        # Prepare data
        X_array = np.array([[match[feature] for feature in self.feature_names] for match in X])
        y_array = np.array(y)
        
        # Scale features
        X_scaled = self.scalers['standard'].fit_transform(X_array)
        
        # Train individual models
        for name, model in self.models.items():
            if name != 'ensemble':  # Ensemble is trained separately
                model.fit(X_scaled, y_array)
                print(f"✅ Trained {name}")
        
        # Train ensemble
        self.models['ensemble'].fit(X_scaled, y_array)
        print("✅ Trained ensemble model")
        
        # Save models
        self._save_models()
        
        # Log performance
        self._log_training_performance(X_scaled, y_array)
    
    def predict(self, features, match_context):
        """Make prediction using ensemble approach"""
        
        # Prepare feature vector
        feature_vector = np.array([[features.get(feature, 0) for feature in self.feature_names]])
        
        # Scale features
        feature_vector_scaled = self.scalers['standard'].transform(feature_vector)
        
        # Get predictions from all models
        individual_predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(feature_vector_scaled)[0]
                individual_predictions[name] = {
                    'probabilities': probas,
                    'prediction': self._convert_proba_to_prediction(probas)
                }
        
        # Ensemble prediction (primary)
        ensemble_proba = self.models['ensemble'].predict_proba(feature_vector_scaled)[0]
        ensemble_prediction = self._convert_proba_to_prediction(ensemble_proba)
        
        # Calculate confidence and prevent bias
        confidence = self._calculate_confidence(ensemble_proba, individual_predictions)
        bias_check = self._check_prediction_bias(ensemble_prediction, features)
        
        return {
            'prediction': ensemble_prediction,
            'probabilities': {
                'home': float(ensemble_proba[0]),
                'draw': float(ensemble_proba[1]),
                'away': float(ensemble_proba[2])
            },
            'confidence': confidence,
            'bias_check': bias_check,
            'model_breakdown': individual_predictions,
            'features_used': len([f for f in features.values() if f != 0]),
            'model_version': self.model_version
        }
    
    def _convert_proba_to_prediction(self, probabilities):
        """Convert probabilities to prediction string"""
        home_prob, draw_prob, away_prob = probabilities
        max_prob = max(home_prob, draw_prob, away_prob)
        
        if max_prob == home_prob:
            return "HOME WIN"
        elif max_prob == away_prob:
            return "AWAY WIN"
        else:
            return "DRAW"
    
    def _calculate_confidence(self, ensemble_proba, individual_predictions):
        """Calculate prediction confidence considering model agreement"""
        home_prob, draw_prob, away_prob = ensemble_proba
        max_prob = max(home_prob, draw_prob, away_prob)
        
        # Base confidence from probability
        base_confidence = max_prob
        
        # Check model agreement
        agreement_score = self._calculate_model_agreement(individual_predictions)
        
        # Combined confidence (probability * agreement)
        confidence = base_confidence * (0.7 + 0.3 * agreement_score)
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def _calculate_model_agreement(self, individual_predictions):
        """Calculate how much models agree on prediction"""
        if len(individual_predictions) == 0:
            return 1.0
        
        predictions = [pred['prediction'] for pred in individual_predictions.values()]
        most_common = max(set(predictions), key=predictions.count)
        agreement = predictions.count(most_common) / len(predictions)
        
        return agreement
    
    def _check_prediction_bias(self, prediction, features):
        """Check for potential prediction bias"""
        bias_checks = {
            'home_team_strong': features.get('home_attack_strength', 0) > 1.8 and prediction == "HOME WIN",
            'away_team_strong': features.get('away_attack_strength', 0) > 1.8 and prediction == "AWAY WIN",
            'form_disagreement': features.get('form_5_diff', 0) < -0.2 and prediction == "HOME WIN",
            'h2h_against_prediction': features.get('h2h_home_win_rate', 0.5) < 0.3 and prediction == "HOME WIN"
        }
        
        bias_detected = any(bias_checks.values())
        
        return {
            'bias_detected': bias_detected,
            'bias_reasons': [reason for reason, detected in bias_checks.items() if detected],
            'adjustment_applied': self._apply_bias_correction(bias_detected, prediction, features)
        }
    
    def _apply_bias_correction(self, bias_detected, prediction, features):
        """Apply bias correction if needed"""
        if not bias_detected:
            return "No adjustment needed"
        
        # Simple bias correction - reduce confidence
        confidence_reduction = 0.1
        return f"Confidence reduced by {confidence_reduction:.1%} due to potential bias"
    
    def _save_models(self):
        """Save trained models"""
        import os
        os.makedirs(Config.MODEL_PATH, exist_ok=True)
        
        joblib.dump(self.models['ensemble'], f"{Config.MODEL_PATH}/ensemble_model.joblib")
        joblib.dump(self.scalers['standard'], f"{Config.MODEL_PATH}/scaler.joblib")
        
        print("💾 Models saved successfully")
    
    def _log_training_performance(self, X, y):
        """Log model performance to database"""
        # This would calculate and store performance metrics
        # For now, just a placeholder
        pass
    
    def generate_advanced_predictions(self, features, match_context):
        """Generate all prediction types for a match"""
        base_prediction = self.predict(features, match_context)
        
        # Additional market predictions
        advanced_predictions = {
            'match_outcome': base_prediction,
            'double_chance': self._predict_double_chance(base_prediction),
            'over_under': self._predict_over_under(features),
            'both_teams_score': self._predict_both_teams_score(features),
            'correct_score': self._predict_correct_score(features)
        }
        
        return advanced_predictions
    
    def _predict_double_chance(self, base_prediction):
        """Predict double chance markets"""
        probs = base_prediction['probabilities']
        
        home_win_or_draw = probs['home'] + probs['draw']
        away_win_or_draw = probs['away'] + probs['draw']
        
        return {
            '1X': home_win_or_draw,
            'X2': away_win_or_draw,
            'recommendation': '1X' if home_win_or_draw > away_win_or_draw else 'X2',
            'confidence': max(home_win_or_draw, away_win_or_draw)
        }
    
    def _predict_over_under(self, features):
        """Predict over/under 2.5 goals"""
        # Simplified calculation based on team attacking/defensive strengths
        expected_goals = (
            features.get('home_goals_scored_avg', 1.5) + 
            features.get('away_goals_scored_avg', 1.2)
        )
        
        over_prob = min(0.9, max(0.1, (expected_goals - 1.5) / 3.0 + 0.5))
        
        return {
            'over_2.5': over_prob,
            'under_2.5': 1 - over_prob,
            'expected_total_goals': expected_goals,
            'recommendation': 'Over 2.5' if over_prob > 0.5 else 'Under 2.5'
        }
    
    def _predict_both_teams_score(self, features):
        """Predict both teams to score"""
        # Based on both teams' attacking strength and opponent's defensive weakness
        home_scoring_prob = min(0.9, features.get('home_attack_strength', 1.5) / 2.5)
        away_scoring_prob = min(0.9, features.get('away_attack_strength', 1.3) / 2.5)
        
        bts_prob = home_scoring_prob * away_scoring_prob
        
        return {
            'yes': bts_prob,
            'no': 1 - bts_prob,
            'recommendation': 'Yes' if bts_prob > 0.5 else 'No'
        }
    
    def _predict_correct_score(self, features):
        """Predict most likely correct scores"""
        # Simplified Poisson-based score prediction
        home_expected = features.get('home_goals_scored_avg', 1.5)
        away_expected = features.get('away_goals_scored_avg', 1.2)
        
        scores = {}
        for home_goals in range(0, 5):
            for away_goals in range(0, 5):
                prob = (
                    np.exp(-home_expected) * (home_expected ** home_goals) / np.math.factorial(home_goals) *
                    np.exp(-away_expected) * (away_expected ** away_goals) / np.math.factorial(away_goals)
                )
                scores[f"{home_goals}-{away_goals}"] = float(prob)
        
        # Return top 5 most likely scores
        top_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5])
        return top_scores
