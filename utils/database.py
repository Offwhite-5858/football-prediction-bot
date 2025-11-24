import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class DatabaseManager:
    """Production-grade SQLite database management with learning capabilities"""
    
    def __init__(self, db_path="database/predictions.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database with all required tables including learning tables"""
        conn = self._get_connection()
        
        # ========== CORE MATCHES TABLE ==========
        conn.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT UNIQUE,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                league TEXT NOT NULL,
                match_date DATE,
                home_goals INTEGER,
                away_goals INTEGER,
                result TEXT CHECK(result IN ('H', 'D', 'A')),
                season INTEGER,
                status TEXT DEFAULT 'SCHEDULED',
                venue TEXT,
                referee TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ========== PREDICTIONS TABLE ==========
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT,
                prediction_type TEXT NOT NULL,
                prediction_data JSON,
                confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
                model_version TEXT,
                features_used JSON,
                data_quality_score INTEGER CHECK(data_quality_score >= 0 AND data_quality_score <= 100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches (match_id),
                UNIQUE(match_id, prediction_type)
            )
        ''')
        
        # ========== LEARNING SYSTEM TABLES ==========
        
        # Prediction errors tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL,
                error_value REAL CHECK(error_value >= 0 AND error_value <= 1),
                actual_result TEXT CHECK(actual_result IN ('H', 'D', 'A')),
                error_type TEXT,
                features_analysis JSON,
                model_breakdown JSON,
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
                correction_details JSON,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions (match_id)
            )
        ''')
        
        # Model performance history
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                features_used_count INTEGER,
                training_samples INTEGER
            )
        ''')
        
        # ========== FEATURE ENGINEERING TABLES ==========
        
        # Feature cache for rapid predictions
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feature_cache (
                team_name TEXT NOT NULL,
                league TEXT NOT NULL,
                feature_type TEXT NOT NULL,
                feature_data JSON,
                last_updated TIMESTAMP,
                expiry_time TIMESTAMP,
                PRIMARY KEY (team_name, league, feature_type)
            )
        ''')
        
        # Team statistics (aggregated for performance)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS team_statistics (
                team_name TEXT NOT NULL,
                league TEXT NOT NULL,
                season INTEGER,
                matches_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                goals_scored INTEGER DEFAULT 0,
                goals_conceded INTEGER DEFAULT 0,
                clean_sheets INTEGER DEFAULT 0,
                avg_goals_scored REAL DEFAULT 0,
                avg_goals_conceded REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                home_win_rate REAL DEFAULT 0,
                away_win_rate REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_name, league, season)
            )
        ''')
        
        # ========== API & SYSTEM MONITORING TABLES ==========
        
        # API request tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS api_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                parameters TEXT,
                response_code INTEGER,
                response_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # System performance metrics
        conn.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ========== USER & APPLICATION TABLES ==========
        
        # Prediction feedback (for future user input)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT,
                user_rating INTEGER CHECK(user_rating >= 1 AND user_rating <= 5),
                feedback_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions (match_id)
            )
        ''')
        
        # Prediction learning metadata - FIXED INDENTATION
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_learning_metadata (
                prediction_id TEXT PRIMARY KEY,
                features_count INTEGER,
                data_quality_score REAL,
                model_version TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions (match_id)
            )
        ''')
        
        # Create indexes for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_matches_team ON matches(home_team, away_team)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_errors_prediction ON prediction_errors(prediction_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_errors_date ON prediction_errors(analyzed_at)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_features_team ON feature_cache(team_name, league)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_api_requests_time ON api_requests(timestamp)')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized with all production tables")
    
    def _get_connection(self):
        """Get database connection with error handling"""
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    # ========== CORE PREDICTION METHODS ==========
    
    def store_prediction(self, match_data, prediction_data, model_version="v1.0"):
        """Store prediction in database with full context"""
        conn = self._get_connection()
        
        try:
            # Store or update match
            conn.execute('''
                INSERT OR REPLACE INTO matches 
                (match_id, home_team, away_team, league, match_date, season, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_data['match_id'],
                match_data['home_team'],
                match_data['away_team'],
                match_data['league'],
                match_data.get('match_date'),
                match_data.get('season', 2025),
                match_data.get('status', 'SCHEDULED')
            ))
            
            # Store prediction for each type
            for pred_type, pred_data in prediction_data.get('predictions', {}).items():
                conn.execute('''
                    INSERT OR REPLACE INTO predictions 
                    (match_id, prediction_type, prediction_data, confidence, model_version, features_used, data_quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    match_data['match_id'],
                    pred_type,
                    json.dumps(pred_data),
                    pred_data.get('confidence', 0.5),
                    model_version,
                    json.dumps(prediction_data.get('features_used', {})),
                    prediction_data.get('data_quality', {}).get('score', 50)
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error storing prediction: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def update_match_result(self, match_id, home_goals, away_goals, result):
        """Update match with final result for learning"""
        conn = self._get_connection()
        
        try:
            conn.execute('''
                UPDATE matches 
                SET home_goals = ?, away_goals = ?, result = ?, updated_at = CURRENT_TIMESTAMP
                WHERE match_id = ?
            ''', (home_goals, away_goals, result, match_id))
            
            conn.commit()
            print(f"✅ Updated match result: {match_id} -> {result}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating match result: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    # ========== LEARNING SYSTEM METHODS ==========
    
    def store_prediction_error(self, prediction_id, error_value, actual_result, error_type, features_analysis, model_breakdown):
        """Store prediction error for learning system"""
        conn = self._get_connection()
        
        try:
            conn.execute('''
                INSERT INTO prediction_errors 
                (prediction_id, error_value, actual_result, error_type, features_analysis, model_breakdown)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                prediction_id,
                error_value,
                actual_result,
                error_type,
                json.dumps(features_analysis),
                json.dumps(model_breakdown)
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error storing prediction error: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def store_learning_log(self, prediction_id, error_value, error_type, correction_applied, correction_details):
        """Store learning system activity"""
        conn = self._get_connection()
        
        try:
            conn.execute('''
                INSERT INTO learning_logs 
                (prediction_id, error_value, error_type, correction_applied, correction_details)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                prediction_id,
                error_value,
                error_type,
                correction_applied,
                json.dumps(correction_details)
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error storing learning log: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def store_model_performance(self, model_name, accuracy, precision, recall, f1_score, features_used_count, training_samples):
        """Store model performance metrics"""
        conn = self._get_connection()
        
        try:
            conn.execute('''
                INSERT INTO model_performance_history 
                (model_name, accuracy, precision, recall, f1_score, features_used_count, training_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name,
                accuracy,
                precision,
                recall,
                f1_score,
                features_used_count,
                training_samples
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error storing model performance: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    # ========== DATA RETRIEVAL METHODS ==========
    
    def get_team_historical_data(self, team_name, league, limit=50):
        """Get historical matches for a team"""
        conn = self._get_connection()
        
        try:
            query = '''
                SELECT * FROM matches 
                WHERE (home_team = ? OR away_team = ?) 
                AND league = ?
                AND result IS NOT NULL
                ORDER BY match_date DESC 
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(team_name, team_name, league, limit))
            return df
            
        except Exception as e:
            print(f"❌ Error getting team historical data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_prediction_by_match_id(self, match_id):
        """Get prediction by match ID"""
        conn = self._get_connection()
        
        try:
            query = '''
                SELECT * FROM predictions 
                WHERE match_id = ? 
                AND prediction_type = 'match_outcome'
                ORDER BY created_at DESC 
                LIMIT 1
            '''
            
            result = conn.execute(query, (match_id,)).fetchone()
            if result:
                return {
                    'id': result[0],
                    'match_id': result[1],
                    'prediction_type': result[2],
                    'prediction_data': json.loads(result[3]),
                    'confidence': result[4],
                    'model_version': result[5],
                    'features_used': json.loads(result[6]) if result[6] else {},
                    'data_quality_score': result[7],
                    'created_at': result[8]
                }
            return None
            
        except Exception as e:
            print(f"❌ Error getting prediction: {e}")
            return None
        finally:
            conn.close()
    
    def get_recent_predictions_with_outcomes(self, days=30):
        """Get recent predictions with known outcomes for training"""
        conn = self._get_connection()
        
        try:
            query = '''
                SELECT p.match_id, p.prediction_data, p.features_used, m.result
                FROM predictions p
                JOIN matches m ON p.match_id = m.match_id
                WHERE p.prediction_type = 'match_outcome'
                AND m.result IS NOT NULL
                AND p.created_at > datetime('now', ?)
                ORDER BY p.created_at DESC
            '''
            
            results = conn.execute(query, (f'-{days} days',)).fetchall()
            
            training_data = []
            for row in results:
                training_data.append({
                    'match_id': row[0],
                    'prediction_data': json.loads(row[1]),
                    'features_used': json.loads(row[2]) if row[2] else {},
                    'actual_result': row[3]
                })
            
            return training_data
            
        except Exception as e:
            print(f"❌ Error getting training data: {e}")
            return []
        finally:
            conn.close()
    
    # ========== FEATURE CACHE METHODS ==========
    
    def cache_team_features(self, team_name, league, feature_type, feature_data, expiry_hours=6):
        """Cache team features for performance"""
        conn = self._get_connection()
        
        try:
            expiry_time = datetime.now() + timedelta(hours=expiry_hours)
            
            conn.execute('''
                INSERT OR REPLACE INTO feature_cache 
                (team_name, league, feature_type, feature_data, last_updated, expiry_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                team_name,
                league,
                feature_type,
                json.dumps(feature_data),
                datetime.now(),
                expiry_time
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error caching features: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_cached_team_features(self, team_name, league, feature_type):
        """Get cached team features if not expired"""
        conn = self._get_connection()
        
        try:
            query = '''
                SELECT feature_data, last_updated, expiry_time 
                FROM feature_cache 
                WHERE team_name = ? AND league = ? AND feature_type = ?
                AND expiry_time > datetime('now')
            '''
            
            result = conn.execute(query, (team_name, league, feature_type)).fetchone()
            if result:
                return json.loads(result[0])
            return None
            
        except Exception as e:
            print(f"❌ Error getting cached features: {e}")
            return None
        finally:
            conn.close()
    
    # ========== ANALYTICS & MONITORING METHODS ==========
    
    def get_learning_metrics(self, days=7):
        """Get learning system performance metrics"""
        conn = self._get_connection()
        
        try:
              # Recent accuracy
            accuracy_query = '''
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN error_value < 0.3 THEN 1 ELSE 0 END) as correct
                FROM prediction_errors 
                WHERE analyzed_at > datetime('now', ?)
            '''
            
            accuracy_result = conn.execute(accuracy_query, (f'-{days} days',)).fetchone()
            recent_accuracy = accuracy_result[1] / accuracy_result[0] if accuracy_result[0] > 0 else 0.5
            
            # Error distribution
            error_query = '''
                SELECT error_type, COUNT(*) as count
                FROM prediction_errors 
                WHERE analyzed_at > datetime('now', ?)
                GROUP BY error_type
            '''
            
            error_distribution = {}
            for row in conn.execute(error_query, (f'-{days} days',)):
                error_distribution[row[0]] = row[1]
            
            # Learning activity
            learning_query = '''
                SELECT COUNT(*) as total_corrections
                FROM learning_logs 
                WHERE timestamp > datetime('now', ?)
            '''
            
            learning_result = conn.execute(learning_query, (f'-{days} days',)).fetchone()
            total_corrections = learning_result[0] if learning_result else 0
            
            return {
                'recent_accuracy': recent_accuracy,
                'error_distribution': error_distribution,
                'total_corrections_applied': total_corrections,
                'total_predictions_analyzed': accuracy_result[0] if accuracy_result else 0
            }
            
        except Exception as e:
            print(f"❌ Error getting learning metrics: {e}")
            return {
                'recent_accuracy': 0.5,
                'error_distribution': {},
                'total_corrections_applied': 0,
                'total_predictions_analyzed': 0
            }
        finally:
            conn.close()
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        conn = self._get_connection()
        
        try:
            stats = {}
            
            # Total predictions
            pred_result = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()
            stats['total_predictions'] = pred_result[0] if pred_result else 0
            
            # Total matches
            matches_result = conn.execute('SELECT COUNT(*) FROM matches').fetchone()
            stats['total_matches'] = matches_result[0] if matches_result else 0
            
            # Recent accuracy
            accuracy_result = conn.execute('''
                SELECT COUNT(*), 
                       SUM(CASE WHEN error_value < 0.3 THEN 1 ELSE 0 END) 
                FROM prediction_errors 
                WHERE analyzed_at > datetime('now', '-7 days')
            ''').fetchone()
            
            if accuracy_result and accuracy_result[0] > 0:
                stats['recent_accuracy'] = accuracy_result[1] / accuracy_result[0]
            else:
                stats['recent_accuracy'] = 0.5
            
            # API usage
            api_result = conn.execute('''
                SELECT COUNT(*) FROM api_requests 
                WHERE timestamp > datetime('now', '-1 day')
            ''').fetchone()
            stats['daily_api_requests'] = api_result[0] if api_result else 0
            
            # Database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            stats['database_size_mb'] = round(db_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting system stats: {e}")
            return {}
        finally:
            conn.close()
    
    def log_api_request(self, endpoint, parameters, response_code, response_time=None):
        """Log API requests for monitoring and rate limiting"""
        conn = self._get_connection()
        
        try:
            success = response_code == 200
            
            conn.execute('''
                INSERT INTO api_requests 
                (endpoint, parameters, response_code, response_time, success)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                endpoint,
                json.dumps(parameters) if parameters else None,
                response_code,
                response_time,
                success
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error logging API request: {e}")
            return False
        finally:
            conn.close()
    
    def cleanup_old_data(self, days_to_keep=90):
        """Clean up old data to prevent database bloat"""
        conn = self._get_connection()
        
        try:
            cutoff_date = f"-{days_to_keep} days"
            
            # Clean old prediction errors
            conn.execute('DELETE FROM prediction_errors WHERE analyzed_at < datetime("now", ?)', (cutoff_date,))
            
            # Clean old learning logs
            conn.execute('DELETE FROM learning_logs WHERE timestamp < datetime("now", ?)', (cutoff_date,))
            
            # Clean old API requests
            conn.execute('DELETE FROM api_requests WHERE timestamp < datetime("now", ?)', (cutoff_date,))
            
            # Clean expired feature cache
            conn.execute('DELETE FROM feature_cache WHERE expiry_time < datetime("now")')
            
            conn.commit()
            print(f"✅ Cleaned up data older than {days_to_keep} days")
            return True
            
        except Exception as e:
            print(f"❌ Error cleaning up old data: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

# Singleton instance for easy access
db_manager = DatabaseManager()
