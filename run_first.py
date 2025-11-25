# run_first.py - UPDATED for Streamlit Cloud deployment
import os
import sqlite3
import pandas as pd
import sys
import streamlit as st

def initialize_system():
    st.title("🚀 Football Prediction Bot - Initialization")
    st.info("Running first-time setup...")
    
    # Create all required directories
    directories = [
        "database",
        "models", 
        "data/historical",
        "data/cache",
        "src",
        "utils"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        st.success(f"✅ Created directory: {directory}")
    
    # Create __init__.py files for proper imports
    init_files = [
        "src/__init__.py",
        "utils/__init__.py", 
        "data/__init__.py",
        "data/historical/__init__.py"
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Package initialization\n')
        st.success(f"✅ Created: {init_file}")
    
    # Initialize database with production schema
    try:
        from utils.database import DatabaseManager
        db = DatabaseManager()
        st.success("✅ Database initialized with production schema")
    except Exception as e:
        st.error(f"❌ Database initialization failed: {e}")
        return False
    
    # Load comprehensive historical data
    try:
        from data.initial_historical_data import initialize_historical_data
        st.info("📥 Loading comprehensive historical data...")
        historical_data = initialize_historical_data()
        st.success("✅ Real historical data loaded successfully")
    except Exception as e:
        st.error(f"⚠️ Historical data loading failed: {e}")
        st.info("📋 Using built-in comprehensive fallback data...")
    
    # Initialize ML models
    try:
        from src.model_ensemble import ProductionMLEnsemble
        st.info("🤖 Initializing ML models...")
        ml_ensemble = ProductionMLEnsemble()
        if ml_ensemble.is_trained:
            st.success("✅ ML models trained and ready")
        else:
            st.warning("🔄 ML models initialized (will train on first use)")
    except Exception as e:
        st.error(f"⚠️ ML model initialization failed: {e}")
    
    st.success("🎉 PRODUCTION SYSTEM INITIALIZATION COMPLETE!")
    st.balloons()
    
    st.markdown("---")
    st.subheader("📝 Next Steps:")
    st.info("""
    1. **Add your API keys** in Streamlit Cloud secrets
    2. **Run the main app** by visiting your app URL
    3. **The bot is ready** for predictions!
    """)
    
    return True

if __name__ == "__main__":
    initialize_system()
