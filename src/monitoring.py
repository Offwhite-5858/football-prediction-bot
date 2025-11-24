import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils.database import DatabaseManager

class PerformanceMonitor:
    """Comprehensive performance monitoring and analytics"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        try:
            conn = self.db._get_connection()
            
            # Basic counts
            predictions_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM predictions", conn
            ).iloc[0]['count']
            
            matches_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM matches", conn
            ).iloc[0]['count']
            
            # Accuracy metrics
            accuracy_query = '''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN m.result = json_extract(p.prediction_data, '$.prediction') THEN 1 ELSE 0 END) as correct
                FROM predictions p
                JOIN matches m ON p.match_id = m.match_id
                WHERE p.prediction_type = 'match_outcome'
                AND m.result IS NOT NULL
            '''
            
            accuracy_result = pd.read_sql_query(accuracy_query, conn).iloc[0]
            accuracy = accuracy_result['correct'] / accuracy_result['total'] if accuracy_result['total'] > 0 else 0.0
            
            # Recent activity
            hour_ago = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            recent_predictions = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM predictions WHERE created_at > ?", 
                conn, params=(hour_ago,)
            ).iloc[0]['count']
            
            conn.close()
            
            return {
                'total_predictions': predictions_count,
                'total_matches': matches_count,
                'overall_accuracy': accuracy,
                'recent_activity': recent_predictions,
                'system_status': 'Healthy' if predictions_count > 0 else 'Initializing',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            
        except Exception as e:
            return {
                'total_predictions': 0,
                'total_matches': 0,
                'overall_accuracy': 0.0,
                'recent_activity': 0,
                'system_status': f'Error: {e}',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
    
    def get_performance_trends(self, days=30):
        """Get performance trends over time"""
        try:
            conn = self.db._get_connection()
            
            query = '''
                SELECT date(p.created_at) as date,
                       COUNT(*) as prediction_count,
                       SUM(CASE WHEN m.result = json_extract(p.prediction_data, '$.prediction') THEN 1 ELSE 0 END) as correct_count
                FROM predictions p
                LEFT JOIN matches m ON p.match_id = m.match_id
                WHERE p.created_at > date('now', ?)
                AND p.prediction_type = 'match_outcome'
                GROUP BY date(p.created_at)
                ORDER BY date(p.created_at)
            '''
            
            df = pd.read_sql_query(query, conn, params=(f'-{days} days',))
            conn.close()
            
            if len(df) > 0:
                df['accuracy'] = df['correct_count'] / df['prediction_count']
                df['date'] = pd.to_datetime(df['date'])
                
                return {
                    'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
                    'accuracy': df['accuracy'].tolist(),
                    'predictions': df['prediction_count'].tolist()
                }
            else:
                return {'dates': [], 'accuracy': [], 'predictions': []}
                
        except Exception as e:
            print(f"Error getting performance trends: {e}")
            return {'dates': [], 'accuracy': [], 'predictions': []}
    
    def get_league_performance(self):
        """Get performance breakdown by league"""
        try:
            conn = self.db._get_connection()
            
            query = '''
                SELECT p.league,
                       COUNT(*) as total_predictions,
                       SUM(CASE WHEN m.result = json_extract(p.prediction_data, '$.prediction') THEN 1 ELSE 0 END) as correct_predictions
                FROM predictions p
                LEFT JOIN matches m ON p.match_id = m.match_id
                WHERE p.prediction_type = 'match_outcome'
                AND m.result IS NOT NULL
                GROUP BY p.league
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) > 0:
                df['accuracy'] = df['correct_predictions'] / df['total_predictions']
                return df.to_dict('records')
            else:
                return []
                
        except Exception as e:
            print(f"Error getting league performance: {e}")
            return []
    
    def create_performance_dashboard(self):
        """Create comprehensive performance dashboard"""
        stats = self.get_system_stats()
        trends = self.get_performance_trends()
        league_performance = self.get_league_performance()
        
        dashboard = {
            'overview': stats,
            'trends': trends,
            'league_breakdown': league_performance,
            'generated_at': datetime.now().isoformat()
        }
        
        return dashboard
    
    def log_prediction_outcome(self, prediction_id, actual_result, confidence, features_used):
        """Log detailed prediction outcome for analysis"""
        try:
            conn = self.db._get_connection()
            
            conn.execute('''
                INSERT INTO prediction_analytics 
                (prediction_id, actual_result, confidence, features_count, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (prediction_id, actual_result, confidence, len(features_used), datetime.now()))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error logging prediction outcome: {e}")
            return False
