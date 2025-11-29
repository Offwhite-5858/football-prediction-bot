import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class DatabaseManager:
    """Production-grade SQLite database management with learning capabilities"""
    
    def __init__(self, db_path="database/predictions.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database with all required tables - FIXED VERSION"""
        conn = self._get_connection()
        
        # Core tables - FIXED: Ensure all columns exist
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT,
                prediction_type TEXT NOT NULL,
                prediction_data TEXT,
                confidence REAL,
                model_version TEXT,
                features_used TEXT,
                data_quality_score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches (match_id)
            )
        ''')
        
        # Learning system tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT,
                actual_result TEXT,
                error_value REAL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Team statistics
        conn.execute('''
            CREATE TABLE IF NOT EXISTS team_stats (
                team_name TEXT NOT NULL,
                league TEXT NOT NULL,
                matches_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                goals_scored INTEGER DEFAULT 0,
                goals_conceded INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_name, league)
            )
        ''')
        
        # Additional tables for monitoring
        conn.execute('''
            CREATE TABLE IF NOT EXISTS error_logs (
                error_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                error_type TEXT,
                error_message TEXT,
                context TEXT,
                traceback TEXT,
                recovery_action TEXT,
                recovery_success BOOLEAN,
                recovery_result TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS api_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT,
                params TEXT,
                response_code INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learning_rate REAL,
                error_count INTEGER,
                retraining_count INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT,
                actual_result TEXT,
                confidence REAL,
                features_count INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_matches_team ON matches(home_team, away_team)',
            'CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league)',
            'CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)',
            'CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id)',
            'CREATE INDEX IF NOT EXISTS idx_team_stats ON team_stats(team_name, league)'
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except Exception as e:
                print(f"⚠️ Could not create index: {e}")
        
        conn.commit()
        conn.close()
        print("✅ Database initialized with production tables")
    
    def _get_connection(self):
        """Get database connection with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    def store_prediction(self, match_data, prediction_data, model_version="v2.0"):
        """Store prediction in database - FIXED VERSION"""
        conn = self._get_connection()
        
        try:
            # Store or update match - FIXED: Use correct column names
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
                match_data.get('season', 2024),
                match_data.get('status', 'SCHEDULED')
            ))
            
            # Store prediction - FIXED: Handle missing features_used gracefully
            for pred_type, pred_data in prediction_data.get('predictions', {}).items():
                features_used = prediction_data.get('features_used', {})
                data_quality = prediction_data.get('data_quality', {})
                
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
                    json.dumps(features_used),
                    data_quality.get('score', 50)
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
        """Update match with final result"""
        conn = self._get_connection()
        
        try:
            conn.execute('''
                UPDATE matches 
                SET home_goals = ?, away_goals = ?, result = ?
                WHERE match_id = ?
            ''', (home_goals, away_goals, result, match_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error updating match result: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_training_data(self, days=365):
        """Get training data for model training - FIXED VERSION"""
        conn = self._get_connection()
        
        try:
            # FIXED: Use simpler query that doesn't require joins with predictions table
            query = '''
                SELECT home_team, away_team, league, home_goals, away_goals, result
                FROM matches 
                WHERE result IS NOT NULL
                AND match_date > date('now', ?)
                ORDER BY match_date DESC
                LIMIT 1000
            '''
            
            df = pd.read_sql_query(query, conn, params=(f'-{days} days',))
            return df
            
        except Exception as e:
            print(f"❌ Error getting training data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
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
    
    def store_prediction_error(self, prediction_id, error_value, actual_result):
        """Store prediction error for learning"""
        conn = self._get_connection()
        
        try:
            conn.execute('''
                INSERT INTO prediction_errors 
                (prediction_id, error_value, actual_result)
                VALUES (?, ?, ?)
            ''', (prediction_id, error_value, actual_result))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error storing prediction error: {e}")
            return False
        finally:
            conn.close()
    
    def store_model_performance(self, model_name, accuracy, precision, recall, f1_score):
        """Store model performance metrics"""
        conn = self._get_connection()
        
        try:
            conn.execute('''
                INSERT INTO model_performance 
                (model_name, accuracy, precision, recall, f1_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_name, accuracy, precision, recall, f1_score))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error storing model performance: {e}")
            return False
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
    
    def log_api_request(self, endpoint, params, response_code):
        """Log API request to database"""
        conn = self._get_connection()
        
        try:
            conn.execute('''
                INSERT INTO api_requests 
                (endpoint, params, response_code)
                VALUES (?, ?, ?)
            ''', (endpoint, str(params), response_code))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error logging API request: {e}")
            return False
        finally:
            conn.close()

# Singleton instance
db_manager = DatabaseManager()
