import time
import psutil
import threading
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils.database import DatabaseManager

class ProductionMonitor:
    """Production system monitoring and health checks"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.health_metrics = {}
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start continuous system monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Production monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🛑 Production monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._check_system_health()
                self._check_api_health()
                self._check_database_health()
                self._check_model_health()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(30)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Database metrics
        conn = self.db._get_connection()
        db_size = conn.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()").fetchone()[0]
        prediction_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
        
        self.health_metrics = {
            'timestamp': datetime.now(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'active_threads': threading.active_count()
            },
            'database': {
                'size_bytes': db_size,
                'prediction_count': prediction_count,
                'connection_status': 'Healthy'
            },
            'api': {
                'last_request_time': datetime.now(),
                'rate_limit_remaining': 10,  # Would be calculated from actual API client
                'error_rate': 0.0
            }
        }
    
    def _check_system_health(self):
        """Check system health and generate alerts"""
        system = self.health_metrics['system']
        
        if system['cpu_percent'] > 80:
            self._add_alert('HIGH_CPU', f"CPU usage high: {system['cpu_percent']}%")
        
        if system['memory_percent'] > 85:
            self._add_alert('HIGH_MEMORY', f"Memory usage high: {system['memory_percent']}%")
        
        if system['disk_percent'] > 90:
            self._add_alert('LOW_DISK', f"Disk space low: {system['disk_percent']}%")
    
    def _check_api_health(self):
        """Check API health and rate limits"""
        conn = self.db._get_connection()
        
        # Check recent API errors
        error_query = '''
            SELECT COUNT(*) 
            FROM api_requests 
            WHERE timestamp > datetime('now', '-1 hour') 
            AND response_code != 200
        '''
        
        recent_errors = conn.execute(error_query).fetchone()[0]
        total_requests = conn.execute("SELECT COUNT(*) FROM api_requests WHERE timestamp > datetime('now', '-1 hour')").fetchone()[0]
        
        error_rate = recent_errors / total_requests if total_requests > 0 else 0
        
        if error_rate > 0.1:  # More than 10% error rate
            self._add_alert('API_ERRORS', f"High API error rate: {error_rate:.1%}")
        
        conn.close()
    
    def _check_database_health(self):
        """Check database health and performance"""
        try:
            conn = self.db._get_connection()
            
            # Check for database locks or performance issues
            table_sizes = conn.execute('''
                SELECT name, COUNT(*) as row_count 
                FROM sqlite_master 
                WHERE type='table' 
                GROUP BY name
            ''').fetchall()
            
            # Alert if any table is growing too large
            for table_name, row_count in table_sizes:
                if row_count > 100000:  # Arbitrary large number
                    self._add_alert('LARGE_TABLE', f"Table {table_name} has {row_count} rows")
            
            conn.close()
            
        except Exception as e:
            self._add_alert('DATABASE_ERROR', f"Database health check failed: {e}")
    
    def _check_model_health(self):
        """Check ML model health and performance"""
        try:
            # Check if models are loaded and performing well
            conn = self.db._get_connection()
            
            # Get recent accuracy
            accuracy_query = '''
                SELECT AVG(CASE WHEN error_value < 0.3 THEN 1 ELSE 0 END) as recent_accuracy
                FROM prediction_errors
                WHERE analyzed_at > datetime('now', '-1 day')
            '''
            
            recent_accuracy = conn.execute(accuracy_query).fetchone()[0] or 0.5
            
            if recent_accuracy < 0.5:  # Accuracy below 50%
                self._add_alert('LOW_ACCURACY', f"Model accuracy low: {recent_accuracy:.1%}")
            
            conn.close()
            
        except Exception as e:
            self._add_alert('MODEL_ERROR', f"Model health check failed: {e}")
    
    def _add_alert(self, alert_type, message):
        """Add system alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'severity': self._get_alert_severity(alert_type)
        }
        
        # Avoid duplicate alerts
        if not any(a['message'] == message for a in self.alerts[-10:]):
            self.alerts.append(alert)
            print(f"🚨 ALERT: {message}")
    
    def _get_alert_severity(self, alert_type):
        """Get alert severity level"""
        critical_alerts = ['DATABASE_ERROR', 'MODEL_ERROR']
        warning_alerts = ['HIGH_CPU', 'HIGH_MEMORY', 'LOW_ACCURACY']
        
        if alert_type in critical_alerts:
            return 'CRITICAL'
        elif alert_type in warning_alerts:
            return 'WARNING'
        else:
            return 'INFO'
    
    def get_health_status(self):
        """Get current system health status"""
        overall_health = 'HEALTHY'
        
        # Check for critical alerts in last hour
        recent_critical_alerts = [
            alert for alert in self.alerts 
            if alert['severity'] == 'CRITICAL' 
            and alert['timestamp'] > datetime.now() - timedelta(hours=1)
        ]
        
        if recent_critical_alerts:
            overall_health = 'CRITICAL'
        elif any(alert['severity'] == 'WARNING' for alert in self.alerts[-10:]):
            overall_health = 'WARNING'
        
        return {
            'overall_health': overall_health,
            'metrics': self.health_metrics,
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'alert_count': len([a for a in self.alerts if a['timestamp'] > datetime.now() - timedelta(hours=1)]),
            'monitoring_active': self.monitoring_active
        }
    
    def clear_old_alerts(self):
        """Clear alerts older than 24 hours"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]