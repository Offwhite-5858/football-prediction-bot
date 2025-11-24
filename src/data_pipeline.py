# src/data_pipeline.py - ENHANCED WITH CSV FALLBACKS
import pandas as pd
import os
from utils.cache_manager import CacheManager

class DataPipeline:
    """Enhanced data pipeline with API → Cache → CSV fallbacks"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.historical_data_loaded = False
        
    def get_team_data(self, team_name, league, use_live_data=True):
        """Get team data with fallback hierarchy"""
        # 1. Try live API data
        if use_live_data:
            from utils.api_client import OptimizedAPIClient
            api = OptimizedAPIClient()
            live_data = api.get_team_data(team_name, league)
            if live_data:
                self.cache.cache_team_features(team_name, league, live_data)
                return live_data
        
        # 2. Try cached data
        cached_data = self.cache.get_cached_team_features(team_name, league)
        if cached_data:
            return cached_data
        
        # 3. Try historical CSV data
        historical_data = self.load_team_historical_data(team_name, league)
        if historical_data:
            return historical_data
        
        # 4. Return basic data as last resort
        return self._get_basic_team_data(team_name, league)
    
    def load_team_historical_data(self, team_name, league):
        """Load team data from historical CSV files"""
        csv_file = f"data/historical/{league.lower().replace(' ', '_')}_2024.csv"
        
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                team_matches = df[
                    (df['home_team'] == team_name) | 
                    (df['away_team'] == team_name)
                ].head(10)
                
                if len(team_matches) > 0:
                    return {
                        'team_info': {'name': team_name, 'league': league},
                        'recent_matches': team_matches.to_dict('records'),
                        'data_source': 'historical_csv'
                    }
            except Exception as e:
                print(f"Error loading historical data for {team_name}: {e}")
        
        return None
    
    def _get_basic_team_data(self, team_name, league):
        """Get basic team data as last resort"""
        return {
            'team_info': {'name': team_name, 'league': league},
            'recent_matches': [],
            'data_source': 'basic_fallback'
        }
