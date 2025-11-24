import requests
import time
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils.database import DatabaseManager

class OptimizedAPIClient:
    """High-frequency API client (10 requests/minute)"""
    
    def __init__(self):
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': Config.FOOTBALL_DATA_API}
        self.db = DatabaseManager()
        self.request_times = []
        
    def _rate_limit(self):
        """Enforce 10 requests per minute limit"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= Config.REQUESTS_PER_MINUTE:
            # Wait until we can make another request
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
    
    def get_team_data(self, team_name, league):
        """Get comprehensive team data"""
        # First try to find team ID
        teams_data = self.make_request('teams', {'limit': 100})
        
        if teams_data:
            for team in teams_data.get('teams', []):
                if team['name'].lower() == team_name.lower():
                    team_id = team['id']
                    
                    # Get detailed team data
                    team_details = self.make_request(f'teams/{team_id}')
                    team_matches = self.make_request(f'teams/{team_id}/matches', {'limit': 10})
                    
                    return {
                        'team_info': team_details,
                        'recent_matches': team_matches,
                        'cached_at': datetime.now().isoformat()
                    }
        
        return None
    
    def get_live_fixtures(self, league=None):
        """Get live and upcoming fixtures"""
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
                
                # Add score if available
                if match['status'] == 'FINISHED':
                    fixture['home_goals'] = match['score']['fullTime']['home']
                    fixture['away_goals'] = match['score']['fullTime']['away']
                
                fixtures.append(fixture)
            
            return fixtures
        
        return self._get_fallback_fixtures(league)
    
    def _get_fallback_fixtures(self, league):
        """Generate fallback fixtures when API fails"""
        teams = {
            'Premier League': ['Manchester City', 'Arsenal', 'Liverpool', 'Chelsea', 'Manchester United'],
            'La Liga': ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia'],
            'Bundesliga': ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Eintracht Frankfurt'],
            'Serie A': ['Inter Milan', 'AC Milan', 'Juventus', 'Napoli', 'Roma'],
            'Ligue 1': ['PSG', 'Monaco', 'Lyon', 'Marseille', 'Lille']
        }
        
        league_teams = teams.get(league, teams['Premier League'])
        fixtures = []
        
        for i in range(min(5, len(league_teams))):
            fixtures.append({
                'id': f'fallback_{i}',
                'home_team': league_teams[i],
                'away_team': league_teams[(i + 1) % len(league_teams)],
                'league': league,
                'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                'time': '15:00',
                'status': 'SCHEDULED',
                'matchday': 1
            })
        
        return fixtures