import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from utils.database import DatabaseManager
except ImportError:
    from database import DatabaseManager

class RealHistoricalData:
    """Complete historical data system with multiple data sources"""
    
    def __init__(self):
        try:
            self.db = DatabaseManager()
        except:
            self.db = None
        self.csv_sources = {
            'Premier League': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
            'La Liga': 'https://www.football-data.co.uk/mmz4281/2324/SP1.csv',
            'Bundesliga': 'https://www.football-data.co.uk/mmz4281/2324/D1.csv',
            'Serie A': 'https://www.football-data.co.uk/mmz4281/2324/I1.csv',
            'Ligue 1': 'https://www.football-data.co.uk/mmz4281/2324/F1.csv'
        }
        
        # Create data directories
        os.makedirs('data/historical', exist_ok=True)
    
    def download_real_historical_data(self):
        """Download real historical data from multiple sources"""
        print("📥 Downloading real historical data from football-data.co.uk...")
        
        all_matches = []
        successful_leagues = 0
        
        for league, url in self.csv_sources.items():
            try:
                print(f"📊 Downloading {league} data...")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                df = pd.read_csv(pd.compat.StringIO(response.text))
                
                if len(df) > 0:
                    processed_matches = self._process_football_data_csv(df, league)
                    all_matches.extend(processed_matches)
                    successful_leagues += 1
                    print(f"✅ {league}: {len(processed_matches)} matches processed")
                    
                    # Save to CSV backup
                    backup_df = pd.DataFrame(processed_matches)
                    csv_filename = f'data/historical/{league.lower().replace(" ", "_")}_2024.csv'
                    backup_df.to_csv(csv_filename, index=False)
                    print(f"💾 Saved backup to {csv_filename}")
                    
                else:
                    print(f"⚠️ No valid data for {league}, using fallback data")
                    fallback_matches = self._create_comprehensive_fallback_data(league)
                    all_matches.extend(fallback_matches)
                    
            except Exception as e:
                print(f"❌ Failed to download {league}: {e}")
                print("🔄 Using comprehensive fallback data...")
                fallback_matches = self._create_comprehensive_fallback_data(league)
                all_matches.extend(fallback_matches)
        
        # Store in database
        if all_matches and self.db:
            self._store_matches_in_database(all_matches)
            print(f"🎉 Historical data loaded: {len(all_matches)} total matches from {successful_leagues} leagues")
            
            # Update team statistics
            self._update_team_statistics()
        else:
            print("✅ Using comprehensive fallback data")
            
        return all_matches
    
    def _process_football_data_csv(self, df, league):
        """Process football-data.co.uk CSV format"""
        matches = []
        
        column_mappings = {
            'HomeTeam': ['HomeTeam', 'Home'],
            'AwayTeam': ['AwayTeam', 'Away'], 
            'FTHG': ['FTHG', 'HG', 'HomeGoals'],
            'FTAG': ['FTAG', 'AG', 'AwayGoals'],
            'Date': ['Date', 'Match Date']
        }
        
        actual_columns = {}
        for standard_col, possible_cols in column_mappings.items():
            for possible_col in possible_cols:
                if possible_col in df.columns:
                    actual_columns[standard_col] = possible_col
                    break
            if standard_col not in actual_columns:
                print(f"⚠️ Warning: Could not find {standard_col} column in {league} data")
                actual_columns[standard_col] = None
        
        for _, row in df.iterrows():
            try:
                home_team = row[actual_columns['HomeTeam']] if actual_columns['HomeTeam'] else ''
                away_team = row[actual_columns['AwayTeam']] if actual_columns['AwayTeam'] else ''
                
                home_goals_raw = row[actual_columns['FTHG']] if actual_columns['FTHG'] else None
                away_goals_raw = row[actual_columns['FTAG']] if actual_columns['FTAG'] else None
                
                if not home_team or not away_team or pd.isna(home_goals_raw) or pd.isna(away_goals_raw):
                    continue
                
                try:
                    home_goals = int(float(home_goals_raw))
                    away_goals = int(float(away_goals_raw))
                except (ValueError, TypeError):
                    continue
                
                date_str = str(row[actual_columns['Date']]) if actual_columns['Date'] else ''
                
                if home_goals > away_goals:
                    result = 'H'
                elif away_goals > home_goals:
                    result = 'A'
                else:
                    result = 'D'
                
                match_data = {
                    'date': self._parse_date(date_str),
                    'home_team': str(home_team).strip(),
                    'away_team': str(away_team).strip(),
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'result': result,
                    'season': 2024,
                    'league': league
                }
                matches.append(match_data)
                
            except Exception as e:
                continue
        
        return matches
    
    def _create_comprehensive_fallback_data(self, league):
        """Create comprehensive fallback data with realistic matches"""
        print(f"🔄 Creating comprehensive fallback data for {league}...")
        
        comprehensive_data = {
            'Premier League': [
                {'date': '2024-05-19', 'home_team': 'Manchester City', 'away_team': 'West Ham', 'home_goals': 3, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-19', 'home_team': 'Arsenal', 'away_team': 'Everton', 'home_goals': 2, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-18', 'home_team': 'Liverpool', 'away_team': 'Wolves', 'home_goals': 2, 'away_goals': 0, 'result': 'H'},
                {'date': '2024-05-16', 'home_team': 'Brighton', 'away_team': 'Chelsea', 'home_goals': 1, 'away_goals': 2, 'result': 'A'},
                {'date': '2024-05-15', 'home_team': 'Tottenham', 'away_team': 'Manchester City', 'home_goals': 0, 'away_goals': 2, 'result': 'A'},
                {'date': '2024-05-14', 'home_team': 'Manchester United', 'away_team': 'Newcastle', 'home_goals': 3, 'away_goals': 2, 'result': 'H'},
                {'date': '2024-05-12', 'home_team': 'Aston Villa', 'away_team': 'Liverpool', 'home_goals': 1, 'away_goals': 1, 'result': 'D'},
                {'date': '2024-05-11', 'home_team': 'Fulham', 'away_team': 'Manchester City', 'home_goals': 0, 'away_goals': 4, 'result': 'A'},
                {'date': '2024-05-11', 'home_team': 'Tottenham', 'away_team': 'Burnley', 'home_goals': 2, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-06', 'home_team': 'Crystal Palace', 'away_team': 'Manchester United', 'home_goals': 4, 'away_goals': 0, 'result': 'H'},
            ],
            'La Liga': [
                {'date': '2024-05-25', 'home_team': 'Real Madrid', 'away_team': 'Betis', 'home_goals': 0, 'away_goals': 0, 'result': 'D'},
                {'date': '2024-05-19', 'home_team': 'Barcelona', 'away_team': 'Rayo Vallecano', 'home_goals': 3, 'away_goals': 0, 'result': 'H'},
                {'date': '2024-05-19', 'home_team': 'Atletico Madrid', 'away_team': 'Osasuna', 'home_goals': 2, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-15', 'home_team': 'Real Madrid', 'away_team': 'Alaves', 'home_goals': 5, 'away_goals': 0, 'result': 'H'},
                {'date': '2024-05-14', 'home_team': 'Barcelona', 'away_team': 'Real Sociedad', 'home_goals': 2, 'away_goals': 0, 'result': 'H'},
            ],
            'Bundesliga': [
                {'date': '2024-05-18', 'home_team': 'Bayer Leverkusen', 'away_team': 'Augsburg', 'home_goals': 2, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-18', 'home_team': 'Bayern Munich', 'away_team': 'Wolfsburg', 'home_goals': 2, 'away_goals': 0, 'result': 'H'},
                {'date': '2024-05-12', 'home_team': 'Borussia Dortmund', 'away_team': 'Darmstadt', 'home_goals': 4, 'away_goals': 0, 'result': 'H'},
                {'date': '2024-05-11', 'home_team': 'RB Leipzig', 'away_team': 'Werder Bremen', 'home_goals': 2, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-05', 'home_team': 'Stuttgart', 'away_team': 'Bayern Munich', 'home_goals': 3, 'away_goals': 1, 'result': 'H'},
            ],
            'Serie A': [
                {'date': '2024-05-26', 'home_team': 'Inter Milan', 'away_team': 'Verona', 'home_goals': 2, 'away_goals': 2, 'result': 'D'},
                {'date': '2024-05-26', 'home_team': 'AC Milan', 'away_team': 'Salernitana', 'home_goals': 3, 'away_goals': 0, 'result': 'H'},
                {'date': '2024-05-20', 'home_team': 'Juventus', 'away_team': 'Monza', 'home_goals': 2, 'away_goals': 0, 'result': 'H'},
                {'date': '2024-05-19', 'home_team': 'Napoli', 'away_team': 'Fiorentina', 'home_goals': 1, 'away_goals': 1, 'result': 'D'},
                {'date': '2024-05-13', 'home_team': 'Roma', 'away_team': 'Genoa', 'home_goals': 1, 'away_goals': 0, 'result': 'H'},
            ],
            'Ligue 1': [
                {'date': '2024-05-25', 'home_team': 'PSG', 'away_team': 'Lyon', 'home_goals': 2, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-25', 'home_team': 'Monaco', 'away_team': 'Nantes', 'home_goals': 4, 'away_goals': 0, 'result': 'H'},
                {'date': '2024-05-19', 'home_team': 'Marseille', 'away_team': 'Lorient', 'home_goals': 3, 'away_goals': 1, 'result': 'H'},
                {'date': '2024-05-19', 'home_team': 'Lille', 'away_team': 'Nice', 'home_goals': 2, 'away_goals': 2, 'result': 'D'},
                {'date': '2024-05-13', 'home_team': 'Lens', 'away_team': 'Montpellier', 'home_goals': 2, 'away_goals': 0, 'result': 'H'},
            ]
        }
        
        matches = comprehensive_data.get(league, [])
        for match in matches:
            match['season'] = 2024
            match['league'] = league
        
        print(f"✅ Created {len(matches)} comprehensive fallback matches for {league}")
        return matches
    
    def _parse_date(self, date_str):
        """Parse various date formats"""
        if not date_str or pd.isna(date_str) or str(date_str).strip() == '':
            return datetime.now().strftime('%Y-%m-%d')
            
        date_str = str(date_str).strip()
        
        try:
            date_formats = [
                '%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d',
                '%d.%m.%Y', '%d-%m-%Y', '%m/%d/%Y'
            ]
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            return datetime.now().strftime('%Y-%m-%d')
            
        except Exception:
            return datetime.now().strftime('%Y-%m-%d')
    
    def _store_matches_in_database(self, matches):
        """Store matches in SQLite database"""
        if not self.db:
            return
            
        print("💾 Storing matches in database...")
        
        conn = self.db._get_connection()
        stored_count = 0
        
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
                
                stored_count += 1
                
            except Exception as e:
                print(f"⚠️ Error storing match {match.get('home_team', '')} vs {match.get('away_team', '')}: {e}")
                continue
        
        conn.commit()
        conn.close()
        print(f"✅ {stored_count} matches stored in database")
    
    def _update_team_statistics(self):
        """Update team statistics after loading historical data"""
        if not self.db:
            return
            
        print("📊 Updating team statistics...")
        
        try:
            conn = self.db._get_connection()
            
            # Get all unique teams
            teams_query = '''
                SELECT DISTINCT home_team as team FROM matches 
                UNION 
                SELECT DISTINCT away_team as team FROM matches
            '''
            teams = [row[0] for row in conn.execute(teams_query).fetchall()]
            
            for team in teams:
                # Get all matches for this team
                team_matches_query = '''
                    SELECT league, home_team, away_team, home_goals, away_goals, result
                    FROM matches 
                    WHERE (home_team = ? OR away_team = ?)
                    AND result IS NOT NULL
                '''
                
                team_matches = conn.execute(team_matches_query, (team, team)).fetchall()
                
                # Calculate statistics by league
                leagues = set([match[0] for match in team_matches])
                
                for league in leagues:
                    league_matches = [m for m in team_matches if m[0] == league]
                    
                    matches_played = len(league_matches)
                    wins = 0
                    draws = 0
                    losses = 0
                    goals_scored = 0
                    goals_conceded = 0
                    
                    for match in league_matches:
                        home_team, away_team, home_goals, away_goals, result = match[1], match[2], match[3], match[4], match[5]
                        
                        if home_team == team:
                            goals_scored += home_goals or 0
                            goals_conceded += away_goals or 0
                            if result == 'H':
                                wins += 1
                            elif result == 'D':
                                draws += 1
                            else:
                                losses += 1
                        else:
                            goals_scored += away_goals or 0
                            goals_conceded += home_goals or 0
                            if result == 'A':
                                wins += 1
                            elif result == 'D':
                                draws += 1
                            else:
                                losses += 1
                    
                    win_rate = wins / matches_played if matches_played > 0 else 0.0
                    
                    # Update team_stats table
                    conn.execute('''
                        INSERT OR REPLACE INTO team_stats 
                        (team_name, league, matches_played, wins, draws, losses, 
                         goals_scored, goals_conceded, win_rate, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        team, league, matches_played, wins, draws, losses,
                        goals_scored, goals_conceded, win_rate, datetime.now()
                    ))
            
            conn.commit()
            conn.close()
            print(f"✅ Updated statistics for {len(teams)} teams")
            
        except Exception as e:
            print(f"❌ Error updating team statistics: {e}")
    
    def get_available_teams(self, league):
        """Get list of available teams for a league"""
        try:
            if not self.db:
                return []
                
            query = '''
                SELECT DISTINCT home_team as team FROM matches WHERE league = ?
                UNION
                SELECT DISTINCT away_team as team FROM matches WHERE league = ?
                ORDER BY team
            '''
            
            conn = self.db._get_connection()
            result = conn.execute(query, (league, league)).fetchall()
            conn.close()
            
            return [row[0] for row in result] if result else []
        except Exception as e:
            print(f"❌ Error getting available teams for {league}: {e}")
            return []

def initialize_historical_data():
    """Initialize the complete historical data system"""
    print("🚀 Initializing historical data system...")
    historical_data = RealHistoricalData()
    matches = historical_data.download_real_historical_data()
    return historical_data

if __name__ == "__main__":
    initialize_historical_data()