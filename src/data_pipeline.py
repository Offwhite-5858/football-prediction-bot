import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from utils.cache_manager import CacheManager
    from utils.database import DatabaseManager
except ImportError:
    # Fallback imports
    import sqlite3

class DataPipeline:
    """Complete data pipeline with API → Cache → CSV fallbacks"""
    
    def __init__(self):
        try:
            self.cache = CacheManager()
            self.db = DatabaseManager()
        except:
            self.cache = None
            self.db = None
        self.historical_data_loaded = False
    
    def load_historical_data(self):
        """Load historical data from multiple sources"""
        print("🔄 Loading historical data from multiple sources...")
        
        try:
            # Try to load from CSV files first
            csv_data = self._load_csv_historical_data()
            
            # Load from database
            db_data = self._load_database_historical_data()
            
            # Combine data sources
            all_data = csv_data + db_data
            
            if len(all_data) > 0:
                self.historical_data_loaded = True
                print(f"✅ Loaded {len(all_data)} historical matches from multiple sources")
            else:
                print("⚠️ No historical data found, system will use simulated data")
                
            return all_data
            
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return []
    
    def _load_csv_historical_data(self):
        """Load historical data from CSV files"""
        historical_paths = {
            'Premier League': 'data/historical/premier_league_2024.csv',
            'La Liga': 'data/historical/la_liga_2024.csv',
            'Bundesliga': 'data/historical/bundesliga_2024.csv',
            'Serie A': 'data/historical/serie_a_2024.csv',
            'Ligue 1': 'data/historical/ligue_1_2024.csv'
        }
        
        all_matches = []
        
        for league, path in historical_paths.items():
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    print(f"✅ Loaded {len(df)} matches from {league} CSV")
                    
                    # Convert to standard format
                    for _, row in df.iterrows():
                        match_data = {
                            'home_team': row.get('home_team', ''),
                            'away_team': row.get('away_team', ''),
                            'league': league,
                            'home_goals': row.get('home_goals', 0),
                            'away_goals': row.get('away_goals', 0),
                            'result': self._determine_result(row.get('home_goals', 0), row.get('away_goals', 0)),
                            'date': row.get('date', datetime.now().strftime('%Y-%m-%d')),
                            'season': 2024,
                            'data_source': 'csv'
                        }
                        all_matches.append(match_data)
                        
                except Exception as e:
                    print(f"❌ Error loading {league} CSV: {e}")
        
        return all_matches
    
    def _load_database_historical_data(self):
        """Load historical data from database"""
        if not self.db:
            return []
            
        try:
            conn = self.db._get_connection()
            query = '''
                SELECT home_team, away_team, league, home_goals, away_goals, result, match_date as date
                FROM matches 
                WHERE result IS NOT NULL
                ORDER BY match_date DESC
                LIMIT 1000
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            matches = []
            for _, row in df.iterrows():
                matches.append({
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'league': row['league'],
                    'home_goals': row['home_goals'],
                    'away_goals': row['away_goals'],
                    'result': row['result'],
                    'date': row['date'],
                    'season': 2024,
                    'data_source': 'database'
                })
            
            print(f"✅ Loaded {len(matches)} matches from database")
            return matches
            
        except Exception as e:
            print(f"❌ Error loading database historical data: {e}")
            return []
    
    def _determine_result(self, home_goals, away_goals):
        """Determine match result from goals"""
        if home_goals > away_goals:
            return 'H'
        elif away_goals > home_goals:
            return 'A'
        else:
            return 'D'
    
    def get_team_form_data(self, team_name, league, days=90):
        """Get recent form data for a team"""
        if not self.db:
            return pd.DataFrame()
            
        try:
            conn = self.db._get_connection()
            
            query = '''
                SELECT home_team, away_team, home_goals, away_goals, result, match_date
                FROM matches 
                WHERE (home_team = ? OR away_team = ?)
                AND league = ?
                AND match_date > date('now', ?)
                AND result IS NOT NULL
                ORDER BY match_date DESC
                LIMIT 10
            '''
            
            df = pd.read_sql_query(query, conn, params=(
                team_name, team_name, league, f'-{days} days'
            ))
            conn.close()
            
            return df
            
        except Exception as e:
            print(f"Error getting team form data for {team_name}: {e}")
            return pd.DataFrame()
    
    def update_team_statistics(self):
        """Update team statistics in database"""
        if not self.db:
            return
            
        print("🔄 Updating team statistics...")
        
        try:
            # Get all teams and their matches
            conn = self.db._get_connection()
            
            teams_query = '''
                SELECT DISTINCT home_team as team FROM matches 
                UNION 
                SELECT DISTINCT away_team as team FROM matches
            '''
            teams = [row[0] for row in conn.execute(teams_query).fetchall()]
            
            for team in teams:
                # Calculate statistics for each team
                stats = self._calculate_team_statistics(team)
                
                # Update database
                for league, league_stats in stats.items():
                    conn.execute('''
                        INSERT OR REPLACE INTO team_stats 
                        (team_name, league, matches_played, wins, draws, losses, 
                         goals_scored, goals_conceded, win_rate, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        team, league, league_stats['matches_played'], league_stats['wins'],
                        league_stats['draws'], league_stats['losses'], league_stats['goals_scored'],
                        league_stats['goals_conceded'], league_stats['win_rate'], datetime.now()
                    ))
            
            conn.commit()
            conn.close()
            print("✅ Team statistics updated successfully")
            
        except Exception as e:
            print(f"❌ Error updating team statistics: {e}")
    
    def _calculate_team_statistics(self, team_name):
        """Calculate comprehensive statistics for a team"""
        if not self.db:
            return {}
            
        conn = self.db._get_connection()
        
        try:
            # Get all matches for the team
            query = '''
                SELECT league, home_team, away_team, home_goals, away_goals, result
                FROM matches 
                WHERE (home_team = ? OR away_team = ?)
                AND result IS NOT NULL
            '''
            
            df = pd.read_sql_query(query, conn, params=(team_name, team_name))
            
            stats_by_league = {}
            
            for league in df['league'].unique():
                league_matches = df[df['league'] == league]
                
                matches_played = len(league_matches)
                wins = 0
                draws = 0
                losses = 0
                goals_scored = 0
                goals_conceded = 0
                
                for _, match in league_matches.iterrows():
                    if match['home_team'] == team_name:
                        goals_scored += match['home_goals'] or 0
                        goals_conceded += match['away_goals'] or 0
                        if match['result'] == 'H':
                            wins += 1
                        elif match['result'] == 'D':
                            draws += 1
                        else:
                            losses += 1
                    else:
                        goals_scored += match['away_goals'] or 0
                        goals_conceded += match['home_goals'] or 0
                        if match['result'] == 'A':
                            wins += 1
                        elif match['result'] == 'D':
                            draws += 1
                        else:
                            losses += 1
                
                win_rate = wins / matches_played if matches_played > 0 else 0.0
                
                stats_by_league[league] = {
                    'matches_played': matches_played,
                    'wins': wins,
                    'draws': draws,
                    'losses': losses,
                    'goals_scored': goals_scored,
                    'goals_conceded': goals_conceded,
                    'win_rate': win_rate
                }
            
            return stats_by_league
            
        except Exception as e:
            print(f"Error calculating statistics for {team_name}: {e}")
            return {}
        finally:
            conn.close()