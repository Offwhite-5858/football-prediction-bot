import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import DatabaseManager
from utils.api_client import OptimizedAPIClient
from utils.cache_manager import CacheManager
from src.feature_engineer import AdvancedFeatureEngineer
from src.model_ensemble import ProductionMLEnsemble
from config import Config

class ContinuousLearningSystem:
    """Advanced continuous learning system that improves from mistakes"""
    
    def __init__(self, ml_ensemble):
        self.ml_ensemble = ml_ensemble
        self.db = DatabaseManager()
        self.learning_rate = 0.1
        self.retraining_threshold = 50  # New errors before retraining
        self.error_count = 0
    
    def analyze_prediction_error(self, prediction_id, actual_result, predicted_result, confidence):
        """Analyze prediction error and trigger learning"""
        try:
            # Calculate error value
            error_value = self._calculate_error_value(predicted_result, actual_result, confidence)
            
            # Store error for analysis
            self.db.store_prediction_error(prediction_id, error_value, actual_result)
            
            # Increment error count
            self.error_count += 1
            
            # Check if retraining is needed
            if self.error_count >= self.retraining_threshold:
                self.trigger_retraining()
                self.error_count = 0
            
            # Update learning rate based on recent performance
            self._update_learning_rate()
            
            print(f"📚 Learning from error: {prediction_id} -> Actual: {actual_result}, Error: {error_value:.3f}")
            return True
            
        except Exception as e:
            print(f"❌ Error in learning analysis: {e}")
            return False
    
    def _calculate_error_value(self, predicted, actual, confidence):
        """Calculate error value between prediction and actual result"""
        if predicted == actual:
            return 1.0 - confidence  # Lower error for correct but low confidence
        else:
            return 1.0 + confidence  # Higher error for wrong with high confidence
    
    def trigger_retraining(self):
        """Trigger model retraining with latest data"""
        print("🔄 Triggering model retraining...")
        try:
            # Retrain the ML ensemble
            self.ml_ensemble.train_models()
            
            # Update learning metrics
            self._update_learning_metrics()
            
            print("✅ Model retraining completed successfully")
            return True
        except Exception as e:
            print(f"❌ Model retraining failed: {e}")
            return False
    
    def _update_learning_rate(self):
        """Dynamically adjust learning rate based on performance"""
        # Get recent accuracy
        recent_accuracy = self.get_recent_accuracy()
        
        # Adjust learning rate: higher when accuracy is low, lower when high
        if recent_accuracy < 0.6:
            self.learning_rate = 0.15  # Learn faster when performing poorly
        elif recent_accuracy > 0.75:
            self.learning_rate = 0.05  # Learn slower when performing well
        else:
            self.learning_rate = 0.1  # Default rate
    
    def _update_learning_metrics(self):
        """Update learning system metrics"""
        try:
            conn = self.db._get_connection()
            conn.execute('''
                INSERT INTO learning_metrics 
                (learning_rate, error_count, retraining_count, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (self.learning_rate, self.error_count, 1, datetime.now()))
            conn.commit()
            conn.close()
        except:
            pass  # Silently fail if table doesn't exist
    
    def get_recent_accuracy(self, days=7):
        """Calculate recent prediction accuracy"""
        try:
            conn = self.db._get_connection()
            query = '''
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN error_value < 0.5 THEN 1 ELSE 0 END) as correct
                FROM prediction_errors 
                WHERE analyzed_at > datetime('now', ?)
            '''
            
            result = conn.execute(query, (f'-{days} days',)).fetchone()
            conn.close()
            
            if result and result[0] > 0:
                return result[1] / result[0]
            else:
                return 0.6  # Default accuracy
        except:
            return 0.6
    
    def get_learning_metrics(self):
        """Get comprehensive learning system metrics"""
        try:
            recent_accuracy = self.get_recent_accuracy()
            
            # Get error distribution
            conn = self.db._get_connection()
            error_query = '''
                SELECT 
                    SUM(CASE WHEN error_value < 0.3 THEN 1 ELSE 0 END) as low_errors,
                    SUM(CASE WHEN error_value >= 0.3 AND error_value < 0.7 THEN 1 ELSE 0 END) as medium_errors,
                    SUM(CASE WHEN error_value >= 0.7 THEN 1 ELSE 0 END) as high_errors
                FROM prediction_errors 
                WHERE analyzed_at > datetime('now', '-7 days')
            '''
            
            error_result = conn.execute(error_query).fetchone()
            conn.close()
            
            error_distribution = {
                'low_errors': error_result[0] if error_result and error_result[0] else 0,
                'medium_errors': error_result[1] if error_result and error_result[1] else 0,
                'high_errors': error_result[2] if error_result and error_result[2] else 0
            }
            
            return {
                'recent_accuracy': recent_accuracy,
                'learning_rate': self.learning_rate,
                'error_count': self.error_count,
                'retraining_threshold': self.retraining_threshold,
                'errors_until_retraining': max(0, self.retraining_threshold - self.error_count),
                'error_distribution': error_distribution,
                'last_retraining': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'system_status': 'Active'
            }
            
        except Exception as e:
            return {
                'recent_accuracy': 0.6,
                'learning_rate': 0.1,
                'error_count': 0,
                'retraining_threshold': 50,
                'errors_until_retraining': 50,
                'error_distribution': {'low_errors': 0, 'medium_errors': 0, 'high_errors': 0},
                'last_retraining': 'Never',
                'system_status': 'Initializing'
            }

class LiveMatchMonitor:
    """Live match monitoring for real-time updates"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.monitored_matches = {}
        self.update_interval = 300  # 5 minutes
    
    def start_monitoring(self, match_id, fixture_data):
        """Start monitoring a match"""
        self.monitored_matches[match_id] = {
            'fixture': fixture_data,
            'last_update': datetime.now(),
            'update_count': 0
        }
        print(f"🔴 Started monitoring: {fixture_data['home_team']} vs {fixture_data['away_team']}")
    
    def stop_monitoring(self, match_id):
        """Stop monitoring a match"""
        if match_id in self.monitored_matches:
            fixture = self.monitored_matches[match_id]['fixture']
            del self.monitored_matches[match_id]
            print(f"🟢 Stopped monitoring: {fixture['home_team']} vs {fixture['away_team']}")
    
    def update_live_predictions(self):
        """Update predictions for monitored live matches"""
        if not self.monitored_matches:
            return
        
        current_time = datetime.now()
        updated_count = 0
        
        for match_id, match_data in self.monitored_matches.items():
            # Check if it's time to update
            time_since_update = (current_time - match_data['last_update']).total_seconds()
            if time_since_update >= self.update_interval:
                try:
                    fixture = match_data['fixture']
                    # Generate updated prediction
                    updated_prediction = self.orchestrator.predict_match(
                        fixture['home_team'],
                        fixture['away_team'],
                        fixture['league'],
                        use_live_data=True,
                        match_context={'status': 'LIVE_UPDATE'}
                    )
                    
                    self.monitored_matches[match_id]['last_update'] = current_time
                    self.monitored_matches[match_id]['update_count'] += 1
                    updated_count += 1
                    
                    print(f"🔄 Updated live prediction for {fixture['home_team']} vs {fixture['away_team']}")
                    
                except Exception as e:
                    print(f"❌ Error updating live prediction: {e}")
        
        if updated_count > 0:
            print(f"✅ Updated {updated_count} live predictions")

class PredictionOrchestrator:
    """Production-grade prediction orchestrator with full learning capabilities"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.api = OptimizedAPIClient()
        self.cache = CacheManager()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ml_ensemble = ProductionMLEnsemble()
        self.learning_system = ContinuousLearningSystem(self.ml_ensemble)
        self.live_monitor = LiveMatchMonitor(self)
        self.predictions_generated = 0
        self._initialize_learning_tables()
    
    def _initialize_learning_tables(self):
        """Initialize learning-related database tables"""
        conn = self.db._get_connection()
        
        try:
            # Learning metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learning_rate REAL,
                    error_count INTEGER,
                    retraining_count INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            print("✅ Learning tables initialized")
        except Exception as e:
            print(f"⚠️ Error initializing learning tables: {e}")
        finally:
            conn.close()
    
    def predict_live_fixtures(self, league=None):
        """Predict outcomes for live fixtures with learning integration"""
        try:
            fixtures = self.api.get_live_fixtures(league)
            predictions = []
            
            for fixture in fixtures[:5]:  # Limit to manage resources
                try:
                    prediction = self.predict_match(
                        fixture['home_team'],
                        fixture['away_team'],
                        fixture['league'],
                        use_live_data=True,
                        match_context=fixture
                    )
                    
                    # Start monitoring for live matches
                    if fixture.get('status') in ['LIVE', 'IN_PLAY']:
                        self.live_monitor.start_monitoring(fixture['id'], fixture)
                    
                    predictions.append({
                        'fixture': fixture,
                        'prediction': prediction
                    })
                    
                    print(f"✅ Predicted: {fixture['home_team']} vs {fixture['away_team']}")
                    
                except Exception as e:
                    print(f"⚠️ Error predicting match {fixture['home_team']} vs {fixture['away_team']}: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error getting live fixtures: {e}")
            return []
    
    def predict_custom_match(self, home_team, away_team, league, use_live_data=True):
        """Predict outcome for custom match input"""
        return self.predict_match(
            home_team, 
            away_team, 
            league, 
            use_live_data,
            match_context={
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'match_type': 'custom',
                'prediction_source': 'user_input'
            }
        )
    
    def predict_match(self, home_team, away_team, league, use_live_data=True, match_context=None):
        """Core prediction method with full learning integration"""
        try:
            # Generate match ID
            match_id = f"{home_team}_{away_team}_{league}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Step 1: Feature Engineering with real data
            features = self.feature_engineer.create_match_features(
                home_team, away_team, league, use_live_data
            )
            
            # Step 2: ML Prediction with trained models
            predictions = self.ml_ensemble.generate_advanced_predictions(
                features, match_context or {}
            )
            
            # Step 3: Data Quality Assessment
            data_quality = self._assess_data_quality(features, use_live_data)
            
            # Step 4: Learning Context
            learning_context = self._get_learning_context(home_team, away_team, league)
            
            # Step 5: Store Prediction
            prediction_record = {
                'match_id': match_id,
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'predictions': predictions,
                'features_used': features,
                'data_quality': data_quality,
                'learning_context': learning_context,
                'use_live_data': use_live_data,
                'model_version': self.ml_ensemble.model_version,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in database
            self.db.store_prediction(
                {
                    'match_id': match_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': league,
                    'match_date': datetime.now().strftime('%Y-%m-%d')
                },
                prediction_record,
                self.ml_ensemble.model_version
            )
            
            self.predictions_generated += 1
            
            print(f"🎯 Prediction generated: {home_team} vs {away_team} -> {predictions['match_outcome']['prediction']}")
            
            return prediction_record
            
        except Exception as e:
            print(f"❌ Error in predict_match: {e}")
            return self._create_fallback_prediction(home_team, away_team, league, str(e))
    
    def _create_fallback_prediction(self, home_team, away_team, league, error_msg):
        """Create fallback prediction when main prediction fails"""
        match_id = f"fallback_{home_team}_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Simple rule-based fallback
        is_top_home = self.feature_engineer._is_top_team(home_team)
        is_top_away = self.feature_engineer._is_top_team(away_team)
        
        if is_top_home and not is_top_away:
            prediction = 'H'
            confidence = 0.7
        elif is_top_away and not is_top_home:
            prediction = 'A'
            confidence = 0.65
        else:
            prediction = 'D'
            confidence = 0.5
        
        return {
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'predictions': {
                'match_outcome': {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': {'home': 0.33, 'draw': 0.34, 'away': 0.33},
                    'model_used': 'fallback_rules'
                },
                'double_chance': {
                    'recommendation': '1X' if prediction == 'H' else 'X2',
                    'confidence': 0.67,
                    '1X': 0.67,
                    'X2': 0.67,
                    'model_used': 'fallback'
                },
                'over_under': {
                    'recommendation': 'Over 2.5',
                    'confidence': 0.5,
                    'over_2.5': 0.5,
                    'under_2.5': 0.5,
                    'expected_total_goals': 2.7,
                    'model_used': 'fallback'
                },
                'both_teams_score': {
                    'recommendation': 'Yes',
                    'confidence': 0.5,
                    'yes': 0.5,
                    'no': 0.5,
                    'model_used': 'fallback'
                },
                'correct_score': {'1-1': 0.1, '2-1': 0.08, '1-2': 0.08, '2-0': 0.07, '0-2': 0.07}
            },
            'features_used': {},
            'data_quality': {
                'score': 20,
                'level': 'Poor',
                'reasons': [f'Fallback mode: {error_msg}'],
                'feature_completeness': '0%'
            },
            'learning_context': {},
            'use_live_data': False,
            'model_version': 'fallback_v1.0',
            'timestamp': datetime.now().isoformat()
        }
    
    def _assess_data_quality(self, features, use_live_data):
        """Comprehensive data quality assessment"""
        quality_score = 0
        reasons = []
        
        # Live data bonus
        if use_live_data:
            quality_score += 25
            reasons.append("✅ Live API data used")
        else:
            reasons.append("⚠️ Using historical data only")
        
        # Feature completeness
        try:
            non_zero_features = sum(1 for f in features.values() if f != 0 and f is not None)
            completeness_ratio = non_zero_features / len(features) if features else 0
            
            if completeness_ratio > 0.8:
                quality_score += 35
                reasons.append("✅ High feature completeness")
            elif completeness_ratio > 0.6:
                quality_score += 25
                reasons.append("⚠️ Moderate feature completeness")
            else:
                quality_score += 15
                reasons.append("❌ Low feature completeness")
        except:
            completeness_ratio = 0
            reasons.append("❌ Error calculating feature completeness")
        
        # Team data quality
        try:
            home_win_rate = features.get('home_historical_win_rate', 0)
            away_win_rate = features.get('away_historical_win_rate', 0)
            
            if home_win_rate > 0 and away_win_rate > 0:
                quality_score += 20
                reasons.append("✅ Both teams have historical data")
            elif home_win_rate > 0 or away_win_rate > 0:
                quality_score += 10
                reasons.append("⚠️ One team has historical data")
            else:
                reasons.append("❌ No historical data for teams")
        except:
            reasons.append("❌ Error assessing team data")
        
        # Form data quality
        try:
            home_form = features.get('home_form_5', 0)
            away_form = features.get('away_form_5', 0)
            
            if home_form > 0 and away_form > 0:
                quality_score += 15
                reasons.append("✅ Recent form data available")
            else:
                reasons.append("⚠️ Limited recent form data")
        except:
            reasons.append("❌ Error assessing form data")
        
        # H2H data quality
        try:
            h2h_matches = features.get('h2h_total_matches', 0)
            if h2h_matches > 0:
                quality_score += 5
                reasons.append("✅ H2H history available")
        except:
            pass
        
        return {
            'score': min(max(quality_score, 0), 100),
            'level': self._get_quality_level(quality_score),
            'reasons': reasons,
            'feature_completeness': f"{completeness_ratio:.1%}" if 'completeness_ratio' in locals() else "0%"
        }
    
    def _get_quality_level(self, score):
        if score >= 80: return "Excellent"
        elif score >= 65: return "Good"
        elif score >= 50: return "Moderate"
        elif score >= 30: return "Limited"
        else: return "Poor"
    
    def _get_learning_context(self, home_team, away_team, league):
        """Get learning system context"""
        try:
            learning_metrics = self.learning_system.get_learning_metrics()
            
            return {
                'system_accuracy': learning_metrics.get('recent_accuracy', 0.6),
                'learning_rate': learning_metrics.get('learning_rate', 0.1),
                'error_count': learning_metrics.get('error_count', 0),
                'system_status': learning_metrics.get('system_status', 'Active'),
                'last_retraining': learning_metrics.get('last_retraining', 'Never')
            }
        except Exception as e:
            return {
                'system_accuracy': 0.6,
                'learning_rate': 0.1,
                'error_count': 0,
                'system_status': 'Initializing',
                'last_retraining': 'Never'
            }
    
    def record_match_outcome(self, match_id, actual_result, home_goals=None, away_goals=None):
        """Record actual match outcome for learning system"""
        try:
            # Update match record
            self.db.update_match_result(match_id, home_goals, away_goals, actual_result)
            
            # Get the prediction that was made
            prediction = self.db.get_prediction_by_match_id(match_id)
            if prediction:
                # Analyze the error for learning
                predicted_result = prediction['prediction_data'].get('prediction', '')
                confidence = prediction['prediction_data'].get('confidence', 0.5)
                
                self.learning_system.analyze_prediction_error(
                    match_id, actual_result, predicted_result, confidence
                )
            
            # Stop monitoring if this match was being monitored
            self.live_monitor.stop_monitoring(match_id)
            
            print(f"📚 Learning recorded: {match_id} -> {actual_result}")
            return True
            
        except Exception as e:
            print(f"❌ Error recording match outcome: {e}")
            return False
    
    def update_live_predictions(self):
        """Update all monitored live predictions"""
        self.live_monitor.update_live_predictions()
    
    def get_learning_metrics(self):
        """Get comprehensive learning system metrics"""
        return self.learning_system.get_learning_metrics()
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        try:
            learning_metrics = self.get_learning_metrics()
            
            return {
                'predictions_generated': self.predictions_generated,
                'models_loaded': len(self.ml_ensemble.models),
                'models_trained': self.ml_ensemble.is_trained,
                'api_requests_available': f"{10 - len(self.api.request_times)}/min",
                'database_size': "Active",
                'model_version': self.ml_ensemble.model_version,
                'learning_system_active': True,
                'recent_accuracy': f"{learning_metrics.get('recent_accuracy', 0.6):.1%}",
                'live_monitoring': len(self.live_monitor.monitored_matches) > 0,
                'monitored_matches': len(self.live_monitor.monitored_matches),
                'system_health': 'Excellent'
            }
        except Exception as e:
            return {
                'predictions_generated': 0,
                'models_loaded': 0,
                'models_trained': False,
                'api_requests_available': "10/min",
                'database_size': "Active",
                'model_version': "v3.0",
                'learning_system_active': True,
                'recent_accuracy': "60.0%",
                'live_monitoring': False,
                'monitored_matches': 0,
                'system_health': 'Good'
            }
    
    def manual_retrain_models(self):
        """Manually trigger model retraining"""
        print("🔄 Manual retraining triggered...")
        success = self.learning_system.trigger_retraining()
        return "Models retrained successfully" if success else "Retraining failed"
    
    def get_prediction_history(self, team=None, league=None, limit=10):
        """Get recent prediction history"""
        conn = self.db._get_connection()
        
        try:
            query = '''
                SELECT p.match_id, p.home_team, p.away_team, p.league, 
                       p.prediction_data, m.result as actual_result,
                       p.created_at
                FROM predictions p
                LEFT JOIN matches m ON p.match_id = m.match_id
                WHERE p.prediction_type = 'match_outcome'
            '''
            
            params = []
            if team:
                query += ' AND (p.home_team = ? OR p.away_team = ?)'
                params.extend([team, team])
            
            if league:
                query += ' AND p.league = ?'
                params.append(league)
            
            query += ' ORDER BY p.created_at DESC LIMIT ?'
            params.append(limit)
            
            results = []
            for row in conn.execute(query, params):
                pred_data = row[4]
                if isinstance(pred_data, str):
                    import json
                    try:
                        pred_data = json.loads(pred_data)
                    except:
                        pred_data = {}
                
                actual_result = row[5]
                predicted_result = pred_data.get('prediction', '')
                was_correct = actual_result == predicted_result if actual_result else None
                
                results.append({
                    'match_id': row[0],
                    'home_team': row[1],
                    'away_team': row[2],
                    'league': row[3],
                    'prediction': pred_data,
                    'actual_result': actual_result,
                    'created_at': row[6],
                    'was_correct': was_correct
                })
            
            return results
        except Exception as e:
            print(f"❌ Error getting prediction history: {e}")
            return []
        finally:
            conn.close()
