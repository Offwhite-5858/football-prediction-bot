import requests
import time
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
import os

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from config import Config
    from utils.database import DatabaseManager
    from utils.cache_manager import CacheManager
    print("✅ All imports successful in api_client")
except ImportError as e:
    print(f"❌ Import error in api_client: {e}")
    import sqlite3
    
class OptimizedAPIClient:
    """High-frequency API client with smart caching"""
    
    def __init__(self):
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': Config.FOOTBALL_DATA_API}
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.request_times = []
        
    def _rate_limit(self):
        """Enforce 10 requests per minute limit"""
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= Config.REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                time.sleep(wait_time)
            self.request_times = self.request_times[1:]
        
        self.request_times.append(current_time)
    
    def make_request(self, endpoint, params=None):
        """Make API request with rate limiting and caching"""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            self.db.log_api_request(endpoint, params, response.status_code)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"API Request failed: {e}")
            self.db.log_api_request(endpoint, params, 0)
            return None
    
    def get_live_fixtures(self, league=None):
        """Get live and upcoming fixtures with caching"""
        # Try cache first
        if league:
            cached_fixtures = self.cache.get_cached_fixtures(league)
            if cached_fixtures:
                print(f"✅ Using cached fixtures for {league}")
                return cached_fixtures
        
        params = {
            'status': 'LIVE,SCHEDULED',
            'dateFrom': datetime.now().strftime('%Y-%m-%d'),
            'dateTo': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'limit': 20
        }
        
        if league:
            league_codes = {
                'Premier League': 'PL', 'La Liga': 'PD', 
                'Bundesliga': 'BL1', 'Serie A': 'SA', 'Ligue 1': 'FL1'
            }
            if league in league_codes:
                params['competitions'] = league_codes[league]
        
        fixtures_data = self.make_request('matches', params)
        
        if fixtures_data:
            fixtures = []
            for match in fixtures_data.get('matches', []):
                fixture = {
                    'id': match['id'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'league': match['competition']['name'],
                    'date': match['utcDate'][:10],
                    'time': match['utcDate'][11:16],
                    'status': match['status'],
                    'matchday': match.get('matchday', 1)
                }
                
                if match['status'] == 'FINISHED':
                    fixture['home_goals'] = match['score']['fullTime']['home']
                    fixture['away_goals'] = match['score']['fullTime']['away']
                
                fixtures.append(fixture)
            
            # Cache the results
            if league and fixtures:
                self.cache.cache_fixtures(league, fixtures)
            
            return fixtures
        
        # Fallback to historical data
        return self._get_fallback_fixtures(league)
    
    def _get_fallback_fixtures(self, league):
        """Generate fallback fixtures when API fails"""
        from data.initial_historical_data import RealHistoricalData
        historical_data = RealHistoricalData()
        
        try:
            # Get recent matches from database as fallback fixtures
            conn = self.db._get_connection()
            query = '''
                SELECT DISTINCT home_team, away_team, league, match_date as date
                FROM matches 
                WHERE league = ? 
                AND match_date > date('now', '-30 days')
                ORDER BY match_date DESC 
                LIMIT 10
            '''
            
            results = conn.execute(query, (league,)).fetchall()
            conn.close()
            
            fixtures = []
            for row in results:
                fixtures.append({
                    'id': f'fallback_{row[0]}_{row[1]}',
                    'home_team': row[0],
                    'away_team': row[1],
                    'league': row[2],
                    'date': row[3],
                    'time': '15:00',
                    'status': 'SCHEDULED',
                    'matchday': 1,
                    'data_source': 'historical_fallback'
                })
            
            return fixtures if fixtures else self._create_basic_fixtures(league)
            
        except Exception as e:
            print(f"Fallback fixtures failed: {e}")
            return self._create_basic_fixtures(league)
    
    def _create_basic_fixtures(self, league):
        """Create basic fixtures as last resort"""
        teams = {
            'Premier League': ['Manchester City', 'Arsenal', 'Liverpool', 'Chelsea', 'Manchester United'],
            'La Liga': ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia'],
            'Bundesliga': ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen'],
            'Serie A': ['Inter Milan', 'AC Milan', 'Juventus', 'Napoli', 'Roma'],
            'Ligue 1': ['PSG', 'Monaco', 'Lyon', 'Marseille', 'Lille']
        }
        
        league_teams = teams.get(league, teams['Premier League'])
        fixtures = []
        
        for i in range(min(5, len(league_teams))):
            fixtures.append({
                'id': f'basic_{i}',
                'home_team': league_teams[i],
                'away_team': league_teams[(i + 1) % len(league_teams)],
                'league': league,
                'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                'time': '15:00',
                'status': 'SCHEDULED',
                'matchday': 1,
                'data_source': 'basic_fallback'
            })
        
        return fixtures
                    
