# src/monitoring.py - Basic performance monitoring
import pandas as pd
from datetime import datetime, timedelta
from utils.database import DatabaseManager

class PerformanceMonitor:
    """Basic performance monitoring for the prediction system"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def get_system_stats(self):
        """Get basic system statistics"""
        try:
            conn = self.db._get_connection()
            
            # Count predictions
            predictions_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM predictions", conn
            ).iloc[0]['count']
            
            # Count matches
            matches_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM matches", conn
            ).iloc[0]['count']
            
            # API requests in last hour
            hour_ago = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            api_requests = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM api_requests WHERE timestamp > ?", 
                conn, params=(hour_ago,)
            ).iloc[0]['count']
            
            conn.close()
            
            return {
                'total_predictions': predictions_count,
                'total_matches': matches_count,
                'api_requests_last_hour': api_requests,
                'system_status': 'Healthy' if predictions_count > 0 else 'Initializing'
            }
            
        except Exception as e:
            return {
                'total_predictions': 0,
                'total_matches': 0,
                'api_requests_last_hour': 0,
                'system_status': f'Error: {e}'
            }
    
    def log_prediction_outcome(self, prediction_id, actual_result, correct):
        """Log prediction outcomes for learning"""
        # This will be expanded in Phase 3 for continuous learning
        pass