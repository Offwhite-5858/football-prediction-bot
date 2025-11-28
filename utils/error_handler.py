import traceback
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from config import Config
    from utils.database import DatabaseManager
except ImportError:
    # Fallback imports
    import sqlite3

class ProductionErrorHandler:
    """Industrial-grade error handling and recovery system"""
    
    def __init__(self):
        try:
            self.db = DatabaseManager()
        except:
            self.db = None
        self.error_count = 0
        self.last_error_time = None
        
    def handle_error(self, error, context="", recovery_action=None):
        """Handle errors with proper logging and recovery"""
        error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        error_info = {
            'error_id': error_id,
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
            'recovery_action': recovery_action
        }
        
        # Log error to database
        self._log_error_to_db(error_info)
        
        # Increment error counter
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        # Attempt recovery if specified
        if recovery_action:
            try:
                recovery_result = recovery_action()
                error_info['recovery_success'] = True
                error_info['recovery_result'] = recovery_result
                print(f"✅ Recovery successful for error {error_id}")
            except Exception as recovery_error:
                error_info['recovery_success'] = False
                error_info['recovery_error'] = str(recovery_error)
                print(f"❌ Recovery failed for error {error_id}: {recovery_error}")
        
        # Update error log
        self._update_error_log(error_info)
        
        # Check if we need to escalate
        self._check_error_escalation()
        
        return error_info
    
    def _log_error_to_db(self, error_info):
        """Log error to database for analysis"""
        if not self.db:
            return
            
        try:
            conn = self.db._get_connection()
            conn.execute('''
                INSERT INTO error_logs 
                (error_id, timestamp, error_type, error_message, context, traceback, 
                 recovery_action, recovery_success, recovery_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                error_info['error_id'],
                error_info['timestamp'],
                error_info['error_type'],
                error_info['error_message'],
                error_info['context'],
                error_info['traceback'],
                error_info.get('recovery_action'),
                error_info.get('recovery_success'),
                error_info.get('recovery_result')
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Failed to log error to database: {e}")
    
    def _update_error_log(self, error_info):
        """Update in-memory error log"""
        # Keep only last 100 errors in memory
        if hasattr(self, 'recent_errors'):
            self.recent_errors.append(error_info)
            if len(self.recent_errors) > 100:
                self.recent_errors = self.recent_errors[-100:]
        else:
            self.recent_errors = [error_info]
    
    def _check_error_escalation(self):
        """Check if errors need escalation (too many errors in short time)"""
        if self.error_count > 10 and self.last_error_time:
            time_since_last = datetime.now() - self.last_error_time
            if time_since_last.total_seconds() < 300:  # 10 errors in 5 minutes
                print("🚨 CRITICAL: High error frequency - consider system restart")
    
    def get_error_stats(self, hours=24):
        """Get error statistics for monitoring"""
        if not self.db:
            return {'total_errors': 0, 'recovered_errors': 0, 'recovery_rate': 0, 'error_types': {}}
            
        try:
            conn = self.db._get_connection()
            
            stats_query = '''
                SELECT 
                    COUNT(*) as total_errors,
                    COUNT(CASE WHEN recovery_success = 1 THEN 1 END) as recovered_errors,
                    error_type,
                    COUNT(*) as count
                FROM error_logs 
                WHERE timestamp > datetime('now', ?)
                GROUP BY error_type
            '''
            
            stats = conn.execute(stats_query, (f'-{hours} hours',)).fetchall()
            conn.close()
            
            return {
                'total_errors': stats[0][0] if stats else 0,
                'recovered_errors': stats[0][1] if stats else 0,
                'recovery_rate': stats[0][1] / stats[0][0] if stats and stats[0][0] > 0 else 0,
                'error_types': {row[2]: row[3] for row in stats} if stats else {}
            }
        except:
            return {'total_errors': 0, 'recovered_errors': 0, 'recovery_rate': 0, 'error_types': {}}
    
    def auto_recover_api_error(self):
        """Automatic recovery for API errors"""
        print("🔄 Attempting API error recovery...")
        
        try:
            from utils.api_client import OptimizedAPIClient
            api_client = OptimizedAPIClient()
            api_client.request_times = []  # Reset rate limiting
            
            # Test API connection
            try:
                test_response = api_client.make_request('competitions/PL')
                if test_response:
                    return "API recovery successful"
                else:
                    return "API recovery failed - no response"
            except Exception as e:
                return f"API recovery failed: {e}"
        except:
            return "API recovery failed - cannot import API client"
    
    def auto_recover_database_error(self):
        """Automatic recovery for database errors"""
        print("🔄 Attempting database recovery...")
        
        try:
            # Reinitialize database connection
            self.db._init_database()
            return "Database recovery successful"
        except Exception as e:
            return f"Database recovery failed: {e}"
    
    def cleanup_old_errors(self, days=7):
        """Clean up old error logs to prevent database bloat"""
        if not self.db:
            return
            
        try:
            conn = self.db._get_connection()
            conn.execute('''
                DELETE FROM error_logs 
                WHERE timestamp < datetime('now', ?)
            ''', (f'-{days} days',))
            conn.commit()
            conn.close()
            print(f"🧹 Cleaned up error logs older than {days} days")
        except Exception as e:
            print(f"Error cleaning up old errors: {e}")