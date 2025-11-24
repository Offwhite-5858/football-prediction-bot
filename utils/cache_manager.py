# utils/cache_manager.py - SMART CACHING SYSTEM
import json
import os
from datetime import datetime, timedelta
import pandas as pd

class CacheManager:
    """Smart caching system for API data and features"""
    
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def cache_data(self, key, data, expiry_hours=6):
        """Cache data with expiry"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        cache_data = {
            'data': data,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=expiry_hours)).isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    
    def get_cached_data(self, key):
        """Get cached data if not expired"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            expires_at = datetime.fromisoformat(cache_data['expires_at'])
            if datetime.now() > expires_at:
                os.remove(cache_file)  # Remove expired cache
                return None
            
            return cache_data['data']
        except:
            return None
    
    def cache_team_features(self, team_name, league, features):
        """Cache team features for performance"""
        key = f"features_{team_name}_{league}".replace(" ", "_").lower()
        self.cache_data(key, features, expiry_hours=12)
    
    def get_cached_team_features(self, team_name, league):
        """Get cached team features"""
        key = f"features_{team_name}_{league}".replace(" ", "_").lower()
        return self.get_cached_data(key)
