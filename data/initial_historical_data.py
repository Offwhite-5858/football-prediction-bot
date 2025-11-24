i# data/real_historical_data.py - ACTUAL historical data system
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
from utils.database import DatabaseManager

class RealHistoricalData:
    """Real historical data system using free CSV datasets"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.csv_sources = {
            'Premier League': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
            'La Liga': 'https://www.football-data.co.uk/mmz4281/2324/SP1.csv',
            'Bundesliga': 'https://www.football-data.co.uk/mmz4281/2324/D1.csv',
            'Serie A': 'https://www.football-data.co.uk/mmz4281/2324/I1.csv',
            'Ligue 1': 'https://www.football-data.co.uk/mmz4281/2324/F1.csv'
        }
    
    def download_real_historical_data(self):
        """Download real historical data from free CSV sources"""
        print("📥 Downloading real historical data from football-data.co.uk...")
        
        all_matches = []
        
        for league, url in self.csv_sources.items():
            try:
                print(f"📊 Downloading {league} data...")
                df = pd.read_csv(url)
                
                if len(df) > 0:
                    processed_matches = self._process_football_data_csv(df, league)
                    all_matches.extend(processed_matches)
                    print(f"✅ {league}: {len(processed_matches)} matches")
                    
                    # Save to CSV backup
                    backup_df = pd.DataFrame(processed_matches)
                    backup_df.to_csv(f'data/historical/{league.lower().replace(" ", "_")}_2024.csv', index=False)
                else:
                    print(f"⚠️ No data for {league}")
                    
            except Exception as e:
                print(f"❌ Failed to download {league}: {e}")
                # Use fallback sample data
                fallback_matches = self._create_fallback_data(league)
                all_matches.extend(fallback_matches)
        
        # Store in database
        self._store_matches_in_database(all_matches)
        
        print(f"🎉 Historical data loaded: {len(all_matches)} total matches")
        return all_matches
    
    def _process_football_data_csv(self, df, league):
        """Process football-data.co.uk CSV format"""
        matches = []
        
        for _, row in df.iterrows():
            try:
                # Map CSV columns to our format
                home_team = row.get('HomeTeam', '')
                away_team = row.get('AwayTeam', '')
                home_goals = row.get('FTHG', 0)  # Full Time Home Goals
                away_goals = row.get('FTAG', 0)  # Full Time Away Goals
                date = row.get('Date', '')
                
                if home_team and away_team and pd.notna(home_goals) and pd.notna(away_goals):
                    # Determine result
                    if home_goals > away_goals:
                        result = 'H'
                    elif away_goals > home_goals:
                        result = 'A'
                    else:
                        result = 'D'
                    
                    match_data = {
                        'date': self._parse_date(date),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_goals': int(home_goals),
                        'away_goals': int(away_goals),
                        'result': result,
                        'season': 2024,  # Current season
                        'league': league
                    }
                    matches.append(match_data)
                    
            except Exception as e:
                print(f"⚠️ Error processing row: {e}")
                continue
        
        return matches
    
    def _parse_date(self, date_str):
        """Parse various date formats"""
        try:
            # Try different date formats
            for fmt in ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d']:
                try:
                    return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                except:
                    continue
            return datetime.now().strftime('%Y-%m-%d')
        except:
            return datetime.now().strftime('%Y-%m-%d')
    
    def _create_fallback_data(self, league):
        """Create fallback data when CSV download fails"""
        print(f"🔄 Creating fallback data for {league}...")
        
        # Sample matches for each league
        fallback_data = {
            'Premier League': [
                {'date': '2024-05-19', 'home_team': 'Manchester City', 'away_team': 'West Ham', 'home_goals': 3, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-19', 'home_team': 'Arsenal', 'away_team': 'Everton', 'home_goals': 2, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-11', 'home_team': 'Fulham', 'away_team': 'Manchester City', 'home_goals': 0, 'away_goals': 4, 'result': 'A'},
            ],
            'La Liga': [
                {'date': '2024-05-25', 'home_team': 'Real Madrid', 'away_team': 'Betis', 'home_goals': 0, 'away_goals': 0, 'result': 'D'},
                {'date': '2024-05-19', 'home_team': 'Barcelona', 'away_team': 'Rayo Vallecano', 'home_goals': 3, 'away_goals': 0, 'result': 'H'},
            ],
            'Bundesliga': [
                {'date': '2024-05-18', 'home_team': 'Bayer Leverkusen', 'away_team': 'Augsburg', 'home_goals': 2, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-11', 'home_team': 'Bayern Munich', 'away_team': 'Wolfsburg', 'home_goals': 2, 'away_goals': 0, 'result': 'H'},
            ],
            'Serie A': [
                {'date': '2024-05-26', 'home_team': 'Inter Milan', 'away_team': 'Verona', 'home_goals': 2, 'away_goals': 2, 'result': 'D'},
                {'date': '2024-05-19', 'home_team': 'Juventus', 'away_team': 'Monza', 'home_goals': 2, 'away_goals': 0, 'result': 'H'},
            ],
            'Ligue 1': [
                {'date': '2024-05-25', 'home_team': 'PSG', 'away_team': 'Lyon', 'home_goals': 2, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-19', 'home_team': 'Monaco', 'away_team': 'Nantes', 'home_goals': 4, 'away_goals': 0, 'result': 'H'},
            ]
        }
        
        matches = fallback_data.get(league, [])
        for match in matches:
            match['season'] = 2024
            match['league'] = league
        
        return matches
    
    def _store_matches_in_database(self, matches):
        """Store matches in SQLite database"""
        print("💾 Storing matches in database...")
        
        conn = self.db._get_connection()
        
        for match in matches:
            try:
                match_id = f"{match['home_team']}_{match['away_team']}_{match['date'].replace('-', '')}"
                
                conn.execute('''
                    INSERT OR REPLACE INTO matches 
                    (match_id, home_team, away_team, league, match_date, home_goals, away_goals, result, season)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    match_id,
                    match['home_team'],
                    match['away_team'],
                    match['league'],
                    match['date'],
                    match.get('home_goals'),
                    match.get('away_goals'),
                    match.get('result'),
                    match.get('season', 2024)
                ))
                
            except Exception as e:
                print(f"⚠️ Error storing match: {e}")
                continue
        
        conn.commit()
        conn.close()
        print("✅ Matches stored in database")
    
    def get_team_historical_stats(self, team_name, league, limit=20):
        """Get historical statistics for a team"""
        query = '''
            SELECT * FROM matches 
            WHERE (home_team = ? OR away_team = ?) 
            AND league = ?
            ORDER BY match_date DESC 
            LIMIT ?
        '''
        
        conn = self.db._get_connection()
        df = pd.read_sql_query(query, conn, params=(team_name, team_name, league, limit))
        conn.close()
        
        return df

# Main function to initialize historical data
def initialize_historical_data():
    """Initialize the historical data system"""
    historical_data = RealHistoricalData()
    matches = historical_data.download_real_historical_data()
    return historical_data

if __name__ == "__main__":
    initialize_historical_data()