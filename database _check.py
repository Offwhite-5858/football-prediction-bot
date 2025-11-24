import streamlit as st
import sqlite3
import pandas as pd
import os
from utils.database import DatabaseManager

class DatabaseHealthCheck:
    def __init__(self):
        self.db = DatabaseManager()
    
    def run_health_check(self):
        """Run comprehensive database health check"""
        st.title("🔍 Database Health Check")
        st.markdown("Verify database tables and initialize if needed")
        
        # Database Status
        st.subheader("📊 Database Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            db_exists = os.path.exists("database/predictions.db")
            st.metric("Database File", "✅ Found" if db_exists else "❌ Missing")
        
        with col2:
            db_size = os.path.getsize("database/predictions.db") if db_exists else 0
            st.metric("Database Size", f"{db_size / 1024:.1f} KB")
        
        with col3:
            table_count = self._get_table_count()
            st.metric("Tables Found", table_count)
        
        # Table Verification
        st.subheader("📋 Table Verification")
        
        required_tables = [
            'matches', 'predictions', 'prediction_errors', 'learning_logs',
            'model_performance_history', 'feature_importance_history',
            'feature_cache', 'team_statistics', 'api_requests',
            'system_metrics', 'user_feedback', 'prediction_learning_metadata'
        ]
        
        table_status = []
        for table in required_tables:
            exists = self._check_table_exists(table)
            count = self._get_table_row_count(table) if exists else 0
            table_status.append({
                'table': table,
                'exists': exists,
                'row_count': count,
                'status': '✅' if exists else '❌'
            })
        
        # Display table status
        status_df = pd.DataFrame(table_status)
        st.dataframe(status_df, use_container_width=True)
        
        # Initialize Database Section
        st.subheader("🛠️ Database Initialization")
        
        if st.button("🔄 Initialize All Tables", type="primary"):
            with st.spinner("Initializing database tables..."):
                success = self._initialize_all_tables()
                if success:
                    st.success("✅ All tables initialized successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize tables")
        
        # Add Sample Data Section
        st.subheader("📝 Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Add Sample Predictions"):
                self._add_sample_predictions()
                st.success("✅ Sample predictions added!")
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear All Data"):
                if st.checkbox("I'm sure I want to delete all data"):
                    self._clear_all_data()
                    st.success("✅ All data cleared!")
                    st.rerun()
        
        # Database Queries Section
        st.subheader("🔍 Database Queries")
        
        query_tab1, query_tab2, query_tab3 = st.tabs(["Matches", "Predictions", "Errors"])
        
        with query_tab1:
            self._show_matches_data()
        
        with query_tab2:
            self._show_predictions_data()
        
        with query_tab3:
            self._show_errors_data()
    
    def _get_table_count(self):
        """Get number of tables in database"""
        try:
            conn = self.db._get_connection()
            result = conn.execute("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """).fetchone()
            conn.close()
            return result[0] if result else 0
        except:
            return 0
    
    def _check_table_exists(self, table_name):
        """Check if a table exists"""
        try:
            conn = self.db._get_connection()
            result = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,)).fetchone()
            conn.close()
            return result is not None
        except:
            return False
    
    def _get_table_row_count(self, table_name):
        """Get row count for a table"""
        try:
            conn = self.db._get_connection()
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            conn.close()
            return result[0] if result else 0
        except:
            return 0
    
    def _initialize_all_tables(self):
        """Initialize all database tables"""
        try:
            # Re-initialize the database
            self.db._init_database()
            return True
        except Exception as e:
            st.error(f"Error initializing tables: {e}")
            return False
    
    def _add_sample_predictions(self):
        """Add sample prediction data for testing"""
        conn = self.db._get_connection()
        
        try:
            # Add sample matches
            sample_matches = [
                ('match_001', 'Manchester City', 'Liverpool', 'Premier League', '2024-03-10', 'H'),
                ('match_002', 'Arsenal', 'Chelsea', 'Premier League', '2024-03-09', 'D'),
                ('match_003', 'Real Madrid', 'Barcelona', 'La Liga', '2024-03-08', 'A')
            ]
            
            for match in sample_matches:
                conn.execute('''
                    INSERT OR REPLACE INTO matches 
                    (match_id, home_team, away_team, league, match_date, result)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', match)
            
            # Add sample predictions
            sample_predictions = [
                ('match_001', 'match_outcome', '{"prediction": "H", "confidence": 0.75}', 0.75, 'v2.1', '{}', 85),
                ('match_002', 'match_outcome', '{"prediction": "D", "confidence": 0.65}', 0.65, 'v2.1', '{}', 80),
                ('match_003', 'match_outcome', '{"prediction": "A", "confidence": 0.70}', 0.70, 'v2.1', '{}', 78)
            ]
            
            for pred in sample_predictions:
                conn.execute('''
                    INSERT OR REPLACE INTO predictions 
                    (match_id, prediction_type, prediction_data, confidence, model_version, features_used, data_quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', pred)
            
            # Add sample prediction errors
            sample_errors = [
                ('match_001', 0.15, 'H', 'minor_error'),
                ('match_002', 0.25, 'D', 'minor_error'),
                ('match_003', 0.10, 'A', 'minor_error')
            ]
            
            for error in sample_errors:
                conn.execute('''
                    INSERT OR REPLACE INTO prediction_errors 
                    (prediction_id, error_value, actual_result, error_type)
                    VALUES (?, ?, ?, ?)
                ''', error)
            
            conn.commit()
            st.success("✅ Sample data added successfully!")
            
        except Exception as e:
            st.error(f"Error adding sample data: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _clear_all_data(self):
        """Clear all data from tables"""
        conn = self.db._get_connection()
        
        try:
            tables = [
                'matches', 'predictions', 'prediction_errors', 'learning_logs',
                'model_performance_history', 'feature_importance_history',
                'feature_cache', 'team_statistics', 'api_requests',
                'system_metrics', 'user_feedback', 'prediction_learning_metadata'
            ]
            
            for table in tables:
                conn.execute(f'DELETE FROM {table}')
            
            conn.commit()
            st.success("✅ All data cleared!")
            
        except Exception as e:
            st.error(f"Error clearing data: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _show_matches_data(self):
        """Show matches data"""
        try:
            conn = self.db._get_connection()
            df = pd.read_sql_query("SELECT * FROM matches ORDER BY created_at DESC LIMIT 10", conn)
            conn.close()
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No matches data found")
        except Exception as e:
            st.error(f"Error loading matches data: {e}")
    
    def _show_predictions_data(self):
        """Show predictions data"""
        try:
            conn = self.db._get_connection()
            df = pd.read_sql_query("""
                SELECT match_id, prediction_type, confidence, model_version, data_quality_score, created_at 
                FROM predictions 
                ORDER BY created_at DESC LIMIT 10
            """, conn)
            conn.close()
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No predictions data found")
        except Exception as e:
            st.error(f"Error loading predictions data: {e}")
    
    def _show_errors_data(self):
        """Show prediction errors data"""
        try:
            conn = self.db._get_connection()
            df = pd.read_sql_query("""
                SELECT prediction_id, error_value, actual_result, error_type, analyzed_at 
                FROM prediction_errors 
                ORDER BY analyzed_at DESC LIMIT 10
            """, conn)
            conn.close()
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No prediction errors data found")
        except Exception as e:
            st.error(f"Error loading errors data: {e}")

# Run the health check
if __name__ == "__main__":
    checker = DatabaseHealthCheck()
    checker.run_health_check()
