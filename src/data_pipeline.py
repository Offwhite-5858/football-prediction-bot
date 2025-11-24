import pandas as pd
import os

class DataPipeline:
    def __init__(self):
        self.historical_data_loaded = False
        
    def load_historical_data(self):
        """Load historical data with fallbacks"""
        try:
            # Try to load from CSV files
            historical_paths = {
                'Premier League': 'data/historical/premier_league.csv',
                'La Liga': 'data/historical/la_liga.csv',
                # Add other leagues...
            }
            
            for league, path in historical_paths.items():
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    print(f"✅ Loaded historical data for {league}: {len(df)} matches")
                else:
                    print(f"⚠️ No historical data found for {league}")
                    
            self.historical_data_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            print("🔄 Using simulated data for initial setup")