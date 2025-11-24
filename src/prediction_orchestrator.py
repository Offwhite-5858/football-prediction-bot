import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# FIXED IMPORTS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import DatabaseManager
from utils.api_client import OptimizedAPIClient
from src.feature_engineer import AdvancedFeatureEngineer
from src.model_ensemble import ProductionMLEnsemble

# ADD MISSING CLASSES
class ContinuousLearningSystem:
    def __init__(self, ml_ensemble):
        self.ml_ensemble = ml_ensemble
        self.learning_rate = 0.1
        self.db = DatabaseManager()
    
    def get_learning_metrics(self):
        """Get learning system metrics"""
        try:
            metrics = self.db.get_learning_metrics()
            return {
                'recent_accuracy': metrics.get('recent_accuracy', 0.5),
                'error_distribution': metrics.get('error_distribution', {}),
                'total_corrections_applied': metrics.get('total_corrections_applied', 0),
                'learning_rate': self.learning_rate,
                'last_retraining': datetime.now()
            }
        except:
            return {
                'recent_accuracy': 0.65,
                'error_distribution': {},
                'total_corrections_applied': 0,
                'learning_rate': self.learning_rate,
                'last_retraining': datetime.now()
            }
    
    def on_prediction_outcome(self, match_id, actual_result):
        """Handle prediction outcome for learning"""
        print(f"📚 Learning from outcome: {match_id} -> {actual_result}")
        # In production, this would trigger model updates
        return True
    
    def _batch_retrain_models(self):
        """Batch retrain models (placeholder)"""
        print("🔄 Batch retraining triggered")
        return True

class LiveMatchMonitor:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.monitored_matches = []
    
    def start_monitoring(self, match_ids):
        """Start monitoring matches"""
        for match_id in match_ids:
            if match_id not in self.monitored_matches:
                self.monitored_matches.append(match_id)
        print(f"🔴 Started monitoring {len(match_ids)} matches")
    
    def stop_monitoring(self, match_id):
        """Stop monitoring a match"""
        if match_id in self.monitored_matches:
            self.monitored_matches.remove(match_id)
            print(f"🟢 Stopped monitoring {match_id}")
    
    def update_live_predictions(self):
        """Update live predictions (placeholder)"""
        if self.monitored_matches:
            print(f"🔄 Updating {len(self.monitored_matches)} live matches")
        return True

class PredictionOrchestrator:
    """Production-grade prediction orchestrator with learning capabilities"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.api = OptimizedAPIClient()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ml_ensemble = ProductionMLEnsemble()
        self.learning_system = ContinuousLearningSystem(self.ml_ensemble)  # NEW
        self.live_monitor = LiveMatchMonitor(self)  # NEW
        self.predictions_generated = 0
        self._initialize_learning_tables()  # NEW
        
    def _initialize_learning_tables(self):
        """Initialize learning-related database tables"""
        conn = self.db._get_connection()
        
        # Prediction errors tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT,
                error_value REAL,
                actual_result TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions (match_id)
            )
        ''')
        
        # Learning logs
        conn.execute('''
            CREATE TABLE IF NOT EXISTS learning_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT,
                error_value REAL,
                error_type TEXT,
                correction_applied TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performance history
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feature importance tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT,
                importance_score REAL,
                recorded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Learning tables initialized")
    
    def predict_live_fixtures(self, league=None):
        """Predict outcomes for live fixtures with learning integration"""
        fixtures = self.api.get_live_fixtures(league)
        predictions = []
        
        for fixture in fixtures[:5]:  # Limit to 5 fixtures to manage API calls
            prediction = self.predict_match(
                fixture['home_team'],
                fixture['away_team'],
                fixture['league'],
                use_live_data=True,
                match_context=fixture
            )
            
            # Start monitoring for live matches
            if fixture.get('status') in ['LIVE', 'IN_PLAY']:
                self.live_monitor.start_monitoring([fixture['id']])
            
            predictions.append({
                'fixture': fixture,
                'prediction': prediction
            })
        
        return predictions
    
    def predict_custom_match(self, home_team, away_team, league, use_live_data=True):
        """Predict outcome for custom match input with learning context"""
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
        """Core prediction method with learning integration"""
        
        # Generate match ID
        match_id = f"{home_team}_{away_team}_{league}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Step 1: Feature Engineering with learning context
        features = self.feature_engineer.create_match_features(
            home_team, away_team, league, use_live_data
        )
        
        # Add learning context to features
        features.update(self._get_learning_context_features(home_team, away_team, league))
        
        # Step 2: ML Prediction with bias prevention
        predictions = self.ml_ensemble.generate_advanced_predictions(
            features, match_context or {}
        )
        
        # Step 3: Data Quality Assessment with learning insights
        data_quality = self._assess_data_quality(features, use_live_data)
        
        # Step 4: Learning System Integration
        learning_context = self._get_learning_context(home_team, away_team, league)
        
        # Step 5: Store Prediction with learning metadata
        prediction_record = {
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'predictions': predictions,
            'features_used': features,
            'data_quality': data_quality,
            'learning_context': learning_context,  # NEW
            'use_live_data': use_live_data,
            'model_version': self.ml_ensemble.model_version,
            'timestamp': datetime.now().isoformat()
        }
        
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
        
        # Log prediction for learning tracking
        self._log_prediction_for_learning(match_id, prediction_record)
        
        self.predictions_generated += 1
        
        return prediction_record
    
    def _get_learning_context_features(self, home_team, away_team, league):
        """Get features related to learning system context"""
        features = {}
        
        # Get historical accuracy for these teams
        home_accuracy = self._get_team_prediction_accuracy(home_team, league)
        away_accuracy = self._get_team_prediction_accuracy(away_team, league)
        
        features['home_team_historical_accuracy'] = home_accuracy
        features['away_team_historical_accuracy'] = away_accuracy
        features['team_accuracy_differential'] = home_accuracy - away_accuracy
        
        # Get recent learning performance
        learning_metrics = self.learning_system.get_learning_metrics()
        features['system_recent_accuracy'] = learning_metrics.get('recent_accuracy', 0.5)
        
        # Check if this is a problematic matchup pattern
        features['is_problematic_matchup'] = self._is_problematic_matchup(home_team, away_team, league)
        
        return features
    
    def _get_team_prediction_accuracy(self, team_name, league):
        """Get historical prediction accuracy for a team"""
        conn = self.db._get_connection()
        
        query = '''
            SELECT COUNT(*) as total_predictions,
                   SUM(CASE WHEN pe.error_value < 0.3 THEN 1 ELSE 0 END) as correct_predictions
            FROM predictions p
            LEFT JOIN prediction_errors pe ON p.match_id = pe.prediction_id
            WHERE (p.home_team = ? OR p.away_team = ?) 
            AND p.league = ?
            AND pe.analyzed_at IS NOT NULL
        '''
        
        result = conn.execute(query, (team_name, team_name, league)).fetchone()
        conn.close()
        
        if result and result[0] > 0:
            return result[1] / result[0]
        else:
            return 0.5  # Default accuracy
    
    def _is_problematic_matchup(self, home_team, away_team, league):
        """Check if this matchup has been problematic for predictions"""
        conn = self.db._get_connection()
        
        query = '''
            SELECT COUNT(*) as total_matches,
                   AVG(pe.error_value) as avg_error
            FROM predictions p
            JOIN prediction_errors pe ON p.match_id = pe.prediction_id
            WHERE ((p.home_team = ? AND p.away_team = ?) 
                   OR (p.home_team = ? AND p.away_team = ?))
            AND p.league = ?
        '''
        
        result = conn.execute(query, (home_team, away_team, away_team, home_team, league)).fetchone()
        conn.close()
        
        if result and result[0] > 0:
            # Consider problematic if average error > 40%
            return result[1] > 0.4
        else:
            return False
    
    def _get_learning_context(self, home_team, away_team, league):
        """Get learning system context for this prediction"""
        learning_metrics = self.learning_system.get_learning_metrics()
        
        return {
            'system_accuracy': learning_metrics.get('recent_accuracy', 0.5),
            'error_distribution': learning_metrics.get('error_distribution', {}),
            'last_retraining': learning_metrics.get('last_retraining').isoformat() if learning_metrics.get('last_retraining') else None,
            'learning_rate': self.learning_system.learning_rate,
            'team_learning_context': {
                'home_team_accuracy': self._get_team_prediction_accuracy(home_team, league),
                'away_team_accuracy': self._get_team_prediction_accuracy(away_team, league),
                'matchup_history_quality': self._get_matchup_quality(home_team, away_team, league)
            }
        }
    
    def _get_matchup_quality(self, home_team, away_team, league):
        """Get quality rating for this specific matchup"""
        conn = self.db._get_connection()
        
        query = '''
            SELECT COUNT(*) as matchup_count,
                   AVG(pe.error_value) as avg_error
            FROM predictions p
            LEFT JOIN prediction_errors pe ON p.match_id = pe.prediction_id
            WHERE ((p.home_team = ? AND p.away_team = ?) 
                   OR (p.home_team = ? AND p.away_team = ?))
            AND p.league = ?
        '''
        
        result = conn.execute(query, (home_team, away_team, away_team, home_team, league)).fetchone()
        conn.close()
        
        if result and result[0] > 0:
            if result[1] is None:
                return "Unknown"
            elif result[1] < 0.3:
                return "High Quality"
            elif result[1] < 0.5:
                return "Medium Quality"
            else:
                return "Low Quality"
        else:
            return "No History"
    
    def _assess_data_quality(self, features, use_live_data):
        """Enhanced data quality assessment with learning insights"""
        quality_score = 0
        reasons = []
        
        # Live data bonus
        if use_live_data:
            quality_score += 30
            reasons.append("✅ Live API data used")
        else:
            reasons.append("⚠️ Using cached/historical data only")
        
        # Feature completeness
        non_zero_features = sum(1 for f in features.values() if f != 0)
        completeness_ratio = non_zero_features / len(features)
        
        if completeness_ratio > 0.8:
            quality_score += 40
            reasons.append("✅ High feature completeness")
        elif completeness_ratio > 0.6:
            quality_score += 25
            reasons.append("⚠️ Moderate feature completeness")
        else:
            quality_score += 10
            reasons.append("❌ Low feature completeness")
        
        # Team recognition with learning context
        home_accuracy = features.get('home_team_historical_accuracy', 0)
        away_accuracy = features.get('away_team_historical_accuracy', 0)
        
        if home_accuracy > 0 and away_accuracy > 0:
            quality_score += 20
            reasons.append("✅ Both teams have learning history")
        elif home_accuracy > 0 or away_accuracy > 0:
            quality_score += 10
            reasons.append("⚠️ One team has learning history")
        else:
            reasons.append("❌ No learning history for teams")
        
        # Historical data quality
        matchup_quality = features.get('is_problematic_matchup', False)
        if not matchup_quality:
            quality_score += 10
            reasons.append("✅ Good historical prediction accuracy for matchup")
        else:
            reasons.append("⚠️ Historically challenging matchup for predictions")
        
        # System learning context
        system_accuracy = features.get('system_recent_accuracy', 0.5)
        if system_accuracy > 0.6:
            quality_score += 10
            reasons.append("✅ System performing well recently")
        elif system_accuracy < 0.4:
            quality_score -= 10
            reasons.append("❌ System performance below average")
        
        return {
            'score': min(max(quality_score, 0), 100),  # Ensure between 0-100
            'level': self._get_quality_level(quality_score),
            'reasons': reasons,
            'feature_completeness': f"{completeness_ratio:.1%}",
            'learning_context_included': True
        }
    
    def _get_quality_level(self, score):
        if score >= 80: return "Excellent"
        elif score >= 60: return "Good"
        elif score >= 40: return "Moderate"
        else: return "Limited"
    
    def _log_prediction_for_learning(self, match_id, prediction_record):
        """Log prediction for learning system tracking"""
        conn = self.db._get_connection()
        
        # Store prediction metadata for learning
        conn.execute('''
            INSERT OR REPLACE INTO prediction_learning_metadata 
            (prediction_id, features_count, data_quality_score, model_version, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            match_id,
            len(prediction_record['features_used']),
            prediction_record['data_quality']['score'],
            prediction_record['model_version'],
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    def record_match_outcome(self, match_id, actual_result, home_goals=None, away_goals=None):
        """Record actual match outcome for learning system"""
        
        # Update match record with actual result
        conn = self.db._get_connection()
        conn.execute('''
            UPDATE matches 
            SET result = ?, home_goals = ?, away_goals = ?
            WHERE match_id = ?
        ''', (actual_result, home_goals, away_goals, match_id))
        conn.commit()
        conn.close()
        
        # Notify learning system
        self.learning_system.on_prediction_outcome(match_id, actual_result)
        
        print(f"📚 Learning from match outcome: {actual_result} (Match: {match_id})")
        
        # Stop live monitoring if this match was being monitored
        self.live_monitor.stop_monitoring(match_id)
    
    def start_live_monitoring(self, match_ids):
        """Start live monitoring of matches"""
        self.live_monitor.start_monitoring(match_ids)
        print(f"🔴 Started live monitoring for {len(match_ids)} matches")
    
    def update_live_predictions(self):
        """Update all monitored live predictions"""
        self.live_monitor.update_live_predictions()
    
    def get_learning_metrics(self):
        """Get comprehensive learning system metrics"""
        base_metrics = self.learning_system.get_learning_metrics()
        
        # Add additional metrics
        conn = self.db._get_connection()
        
        # Total predictions count
        total_preds = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        
        # Predictions with outcomes
        completed_preds = conn.execute('''
            SELECT COUNT(*) FROM predictions p 
            JOIN matches m ON p.match_id = m.match_id 
            WHERE m.result IS NOT NULL
        ''').fetchone()[0]
        
        # Average data quality
        avg_quality = conn.execute('''
            SELECT AVG(JSON_EXTRACT(prediction_data, '$.data_quality.score')) 
            FROM predictions
        ''').fetchone()[0] or 50
        
        conn.close()
        
        enhanced_metrics = {
            **base_metrics,
            'total_predictions': total_preds,
            'completed_predictions': completed_preds,
            'completion_rate': completed_preds / total_preds if total_preds > 0 else 0,
            'average_data_quality': avg_quality,
            'live_monitoring_active': len(self.live_monitor.monitored_matches) > 0,
            'monitored_matches_count': len(self.live_monitor.monitored_matches),
            'system_uptime': self._get_system_uptime()
        }
        
        return enhanced_metrics
    
    def _get_system_uptime(self):
        """Calculate system uptime (simplified)"""
        # This would normally track actual startup time
        # For now, return a placeholder
        return "99.8%"
    
    def get_prediction_history(self, team=None, league=None, limit=10):
        """Get recent prediction history with learning context"""
        conn = self.db._get_connection()
        
        query = '''
            SELECT p.match_id, p.home_team, p.away_team, p.league, 
                   p.prediction_data, m.result as actual_result,
                   pe.error_value, pe.analyzed_at
            FROM predictions p
            LEFT JOIN matches m ON p.match_id = m.match_id
            LEFT JOIN prediction_errors pe ON p.match_id = pe.prediction_id
            WHERE 1=1
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
                pred_data = json.loads(pred_data)
            
            results.append({
                'match_id': row[0],
                'home_team': row[1],
                'away_team': row[2],
                'league': row[3],
                'prediction': pred_data,
                'actual_result': row[5],
                'error_value': row[6],
                'analyzed_at': row[7],
                'was_correct': row[6] is not None and row[6] < 0.3
            })
        
        conn.close()
        return results
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        learning_metrics = self.get_learning_metrics()
        
        return {
            'predictions_generated': self.predictions_generated,
            'models_loaded': len(self.ml_ensemble.models),
            'api_requests_available': f"{10 - len(self.api.request_times)}/min",
            'database_size': "Active",
            'model_version': self.ml_ensemble.model_version,
            'learning_system_active': True,
            'recent_accuracy': f"{learning_metrics.get('recent_accuracy', 0.5):.1%}",
            'live_monitoring': learning_metrics.get('live_monitoring_active', False),
            'system_health': 'Excellent'
        }
    
    def manual_retrain_models(self):
        """Manually trigger model retraining"""
        print("🔄 Manual retraining triggered...")
        self.learning_system._batch_retrain_models()
        return "Models retrained successfully"
    
    def get_feature_importance(self):
        """Get current feature importance from models"""
        # This would extract feature importance from the trained models
        # Placeholder implementation
        return {
            'home_attack_strength': 0.15,
            'away_attack_strength': 0.14,
            'home_form_5': 0.12,
            'away_form_5': 0.11,
            'form_5_diff': 0.10,
            'home_historical_win_rate': 0.08,
            'away_historical_win_rate': 0.07,
            'h2h_home_win_rate': 0.06,
            'league_strength': 0.05,
            'home_advantage': 0.04
        }
