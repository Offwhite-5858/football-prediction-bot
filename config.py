import os
from datetime import datetime

class Config:
    # API Configuration - NO HARDCODED KEYS!
    FOOTBALL_DATA_API = os.getenv('FOOTBALL_DATA_API', '3292bc6b3ad4459fa739ede03966a02b')
    ODDS_API_KEY = os.getenv('ODDS_API_KEY', '8eebed5664851eb764da554b65c5f171')
    
    # API Limits
    REQUESTS_PER_MINUTE = 10
    REQUEST_INTERVAL = 6
    
    # Database
    DATABASE_PATH = "database/predictions.db"
    MODEL_PATH = "models/"
    DATA_PATH = "data/"
    
    # Current Season
    CURRENT_SEASON = 2024
    CURRENT_YEAR = 2024
    
    # Feature Engineering
    MIN_MATCHES_FOR_FORM = 5
    FORM_WINDOW_DAYS = 90
    
    # Model Training
    TRAINING_DAYS = 365
    MIN_TRAINING_MATCHES = 100
    
    @staticmethod
    def ensure_directories():
        """Ensure all required directories exist"""
        os.makedirs("database", exist_ok=True)
        os.makedirs("models", exist_ok=True) 
        os.makedirs("data/historical", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
