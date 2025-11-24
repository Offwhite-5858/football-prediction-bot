import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils.database import DatabaseManager

class ContinuousLearningSystem:
    """Real-time learning system that improves from mistakes"""
    
    def __init__(self, ml_ensemble):
        self.ml_ensemble = ml_ensemble
        self.db = DatabaseManager()
        self.learning_rate = 0.1
        self.error_threshold = 0.3  # Retrain if error > 30%
        self.last_retraining = datetime.now()
        
    def on_prediction_outcome(self, prediction_id, actual_result):
        """Learn from prediction outcome (called when match result is known)"""
        
        # Get the original prediction
        original_pred = self.db.get_prediction_by_id(prediction_id)
        if not original_pred:
            return
        
        # Calculate prediction error
        error = self._calculate_prediction_error(original_pred, actual_result)
        
        # Store error for analysis
        self._store_prediction_error(prediction_id, error, actual_result)
        
        # Immediate learning for significant errors
        if error > self.error_threshold:
            self._immediate_error_correction(original_pred, actual_result, error)
        
        # Check if it's time for batch retraining
        if self._should_retrain_batch():
            self._batch_retrain_models()
    
    def _calculate_prediction_error(self, prediction, actual_result):
        """Calculate how wrong the prediction was"""
        pred_probs = prediction['predictions']['match_outcome']['probabilities']
        
        if actual_result == 'H':
            actual_prob = 1.0
            predicted_prob = pred_probs['home']
        elif actual_result == 'A':
            actual_prob = 1.0  
            predicted_prob = pred_probs['away']
        else:  # Draw
            actual_prob = 1.0
            predicted_prob = pred_probs['draw']
        
        # Error is the difference between predicted and actual
        error = abs(actual_prob - predicted_prob)
        return error
    
    def _store_prediction_error(self, prediction_id, error, actual_result):
        """Store error analysis in database"""
        conn = self.db._get_connection()
        conn.execute('''
            INSERT OR REPLACE INTO prediction_errors 
            (prediction_id, error_value, actual_result, analyzed_at)
            VALUES (?, ?, ?, ?)
        ''', (prediction_id, error, actual_result, datetime.now()))
        conn.commit()
        conn.close()
    
    def _immediate_error_correction(self, prediction, actual_result, error):
        """Immediate correction for significant errors"""
        print(f"🚨 Significant prediction error detected: {error:.1%}")
        
        # Analyze error pattern
        error_analysis = self._analyze_error_pattern(prediction, actual_result)
        
        # Adjust model weights temporarily
        self._adjust_ensemble_weights(error_analysis)
        
        # Log the correction
        self._log_error_correction(prediction, error, error_analysis)
    
    def _analyze_error_pattern(self, prediction, actual_result):
        """Analyze what went wrong with the prediction"""
        features = prediction['features_used']
        pred_probs = prediction['predictions']['match_outcome']['probabilities']
        
        analysis = {
            'error_type': self._classify_error_type(pred_probs, actual_result),
            'feature_analysis': self._analyze_feature_contribution(features),
            'model_breakdown': prediction['predictions']['match_outcome'].get('model_breakdown', {}),
            'confidence_issue': pred_probs['confidence'] > 0.7 and self._calculate_prediction_error(prediction, actual_result) > 0.4
        }
        
        return analysis
    
    def _classify_error_type(self, pred_probs, actual_result):
        """Classify the type of prediction error"""
        max_pred = max(pred_probs.values())
        max_outcome = [k for k, v in pred_probs.items() if v == max_pred][0]
        
        if actual_result == 'H' and max_outcome != 'home':
            return 'UNDERESTIMATED_HOME'
        elif actual_result == 'A' and max_outcome != 'away':  
            return 'UNDERESTIMATED_AWAY'
        elif actual_result == 'D' and max_outcome != 'draw':
            return 'MISSED_DRAW'
        else:
            return 'CONFIDENCE_ERROR'
    
    def _analyze_feature_contribution(self, features):
        """Analyze which features might have contributed to error"""
        suspicious_features = []
        
        # Check for extreme values that might have misled the model
        for feature, value in features.items():
            if abs(value) > 2.0:  # Extreme feature value
                suspicious_features.append({
                    'feature': feature,
                    'value': value,
                    'issue': 'extreme_value'
                })
            elif value == 0:  # Missing data
                suspicious_features.append({
                    'feature': feature, 
                    'value': value,
                    'issue': 'missing_data'
                })
        
        return suspicious_features
    
    def _adjust_ensemble_weights(self, error_analysis):
        """Temporarily adjust ensemble model weights"""
        # Reduce weight of models that performed poorly
        model_breakdown = error_analysis.get('model_breakdown', {})
        
        for model_name, model_pred in model_breakdown.items():
            model_error = self._calculate_model_error(model_pred, error_analysis['error_type'])
            
            if model_error > 0.4:
                print(f"📉 Reducing weight for {model_name} due to high error")
                # In a full implementation, this would adjust VotingClassifier weights
    
    def _calculate_model_error(self, model_pred, error_type):
        """Calculate error for individual model"""
        # Simplified error calculation
        return abs(model_pred['probabilities'][0] - 0.5)  # Placeholder
    
    def _should_retrain_batch(self):
        """Check if it's time for batch retraining"""
        time_since_retrain = datetime.now() - self.last_retraining
        return time_since_retrain.days >= 1  # Retrain daily
    
    def _batch_retrain_models(self):
        """Retrain models with all available data"""
        print("🔄 Starting daily batch retraining...")
        
        # Get all historical predictions with outcomes
        training_data = self._prepare_training_data()
        
        if len(training_data) > 10:  # Only retrain if we have enough data
            X = [data['features'] for data in training_data]
            y = [data['outcome'] for data in training_data]
            
            # Retrain the ensemble
            self.ml_ensemble.train_models(X, y)
            
            self.last_retraining = datetime.now()
            print("✅ Models retrained successfully")
        
        # Clean old error logs
        self._clean_old_error_logs()
    
    def _prepare_training_data(self):
        """Prepare training data from historical predictions"""
        conn = self.db._get_connection()
        
        # Get predictions with known outcomes
        query = '''
            SELECT p.prediction_data, m.result as actual_result
            FROM predictions p
            JOIN matches m ON p.match_id = m.match_id
            WHERE m.result IS NOT NULL
            AND p.created_at > datetime('now', '-30 days')
        '''
        
        training_data = []
        for row in conn.execute(query):
            pred_data = row[0]
            actual_result = row[1]
            
            if isinstance(pred_data, str):
                import json
                pred_data = json.loads(pred_data)
            
            training_data.append({
                'features': pred_data.get('features_used', {}),
                'outcome': actual_result
            })
        
        conn.close()
        return training_data
    
    def _clean_old_error_logs(self):
        """Clean old error logs to prevent database bloat"""
        conn = self.db._get_connection()
        conn.execute('''
            DELETE FROM prediction_errors 
            WHERE analyzed_at < datetime('now', '-7 days')
        ''')
        conn.commit()
        conn.close()
    
    def _log_error_correction(self, prediction, error, analysis):
        """Log error correction for monitoring"""
        conn = self.db._get_connection()
        conn.execute('''
            INSERT INTO learning_logs 
            (prediction_id, error_value, error_type, correction_applied, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            prediction.get('match_id', 'unknown'),
            error,
            analysis['error_type'],
            'ensemble_weight_adjustment',
            datetime.now()
        ))
        conn.commit()
        conn.close()
    
    def get_learning_metrics(self):
        """Get learning system performance metrics"""
        conn = self.db._get_connection()
        
        # Calculate recent accuracy
        accuracy_query = '''
            SELECT COUNT(*) as total, 
                   SUM(CASE WHEN pe.error_value < 0.3 THEN 1 ELSE 0 END) as correct
            FROM prediction_errors pe
            WHERE pe.analyzed_at > datetime('now', '-7 days')
        '''
        
        result = conn.execute(accuracy_query).fetchone()
        recent_accuracy = result[1] / result[0] if result[0] > 0 else 0.5
        
        # Get error distribution
        error_query = '''
            SELECT error_type, COUNT(*) as count
            FROM learning_logs 
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY error_type
        '''
        
        error_distribution = {}
        for row in conn.execute(error_query):
            error_distribution[row[0]] = row[1]
        
        conn.close()
        
        return {
            'recent_accuracy': recent_accuracy,
            'error_distribution': error_distribution,
            'last_retraining': self.last_retraining,
            'learning_rate': self.learning_rate,
            'total_corrections_applied': sum(error_distribution.values())
        }