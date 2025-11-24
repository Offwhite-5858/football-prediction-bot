import os
from datetime import datetime

class Config:
    # API Configuration
    FOOTBALL_DATA_API = os.getenv('FOOTBALL_DATA_API')
    ODDS_API_KEY = os.getenv('ODDS_API_KEY')
    
    # API Limits - 10 requests per MINUTE!
    REQUESTS_PER_MINUTE = 10
    REQUEST_INTERVAL = 6  # seconds between requests
    
    # Database
    DATABASE_PATH = "database/predictions.db"
    MODEL_PATH = "models/"
    DATA_PATH = "data/"
    
    # Current Season (for 2025 context)
    CURRENT_SEASON = 2025
    CURRENT_YEAR = 2025
    
    # Feature Engineering
    MIN_MATCHES_FOR_FORM = 5
    FORM_WINDOW_DAYS = 90  # 3 months for current form
    
    # Add this to handle file paths
    @staticmethod
    def ensure_directories():
        """Ensure all required directories exist"""
        os.makedirs("database", exist_ok=True)
        os.makedirs("models", exist_ok=True) 
        os.makedirs("data/historical", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
