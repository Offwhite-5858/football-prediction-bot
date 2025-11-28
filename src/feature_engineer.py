import pandas as pd
import numpy as np
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
    from utils.database import DatabaseManager
    from utils.cache_manager import CacheManager
    from config import Config
    print("✅ All imports successful in feature_engineer")
except ImportError as e:
    print(f"❌ Import error in feature_engineer: {e}")
    import sqlite3

class AdvancedFeatureEngineer:
    """Production-grade feature engineering with real features"""
    
    def __init__(self):
        try:
            self.db = DatabaseManager()
            self.cache = CacheManager()
        except:
            self.db = None
            self.cache = None
        self.team_strengths = self._load_team_strengths()
    
    def create_match_features(self, home_team, away_team, league, use_live_data=True):
        """Create comprehensive features for ML prediction"""
        
        features = {}
        
        # 1. Basic team information
        features.update(self._get_team_basic_features(home_team, away_team, league))
        
        # 2. Current form features
        features.update(self._get_form_features(home_team, away_team, league))
        
        # 3. Historical performance features
        features.update(self._get_historical_features(home_team, away_team, league))
        
        # 4. Head-to-head features
        features.update(self._get_h2h_features(home_team, away_team, league))
        
        # 5. Advanced statistical features
        features.update(self._get_advanced_stats(home_team, away_team, league))
        
        # 6. Contextual features
        features.update(self._get_contextual_features(home_team, away_team, league))
        
        # 7. Live data features (if available)
        if use_live_data:
            live_features = self._get_live_features(home_team, away_team, league)
            features.update(live_features)
            features['live_data_used'] = 1.0
        else:
            features['live_data_used'] = 0.0
        
        # Ensure no NaN values
        for key, value in features.items():
            if pd.isna(value):
                features[key] = 0.0
        
        return features
    
    def _get_team_basic_features(self, home_team, away_team, league):
        """Basic team strength features using real data"""
        features = {}
        
        # Get team strengths from historical data
        home_stats = self._get_team_stats(home_team, league)
        away_stats = self._get_team_stats(away_team, league)
        
        # Attack and defense strengths based on actual performance
        features['home_attack_strength'] = home_stats.get('goals_scored_per_match', 1.5)
        features['home_defense_strength'] = home_stats.get('goals_conceded_per_match', 1.3)
        features['away_attack_strength'] = away_stats.get('goals_scored_per_match', 1.2)
        features['away_defense_strength'] = away_stats.get('goals_conceded_per_match', 1.5)
        
        # Strength differentials
        features['attack_strength_diff'] = features['home_attack_strength'] - features['away_attack_strength']
        features['defense_strength_diff'] = features['home_defense_strength'] - features['away_defense_strength']
        
        # Home advantage based on historical performance
        home_adv = self._calculate_home_advantage(home_team, league)
        features['home_advantage'] = home_adv
        
        # Team quality indicators
        features['home_win_rate'] = home_stats.get('win_rate', 0.5)
        features['away_win_rate'] = away_stats.get('win_rate', 0.4)
        features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
        
        return features
    
    def _get_form_features(self, home_team, away_team, league):
        """Current form features based on recent matches"""
        features = {}
        
        # Get recent matches
        home_recent = self._get_team_recent_matches(home_team, league, 10)
        away_recent = self._get_team_recent_matches(away_team, league, 10)
        
        # Home team form
        if len(home_recent) >= 3:
            features['home_form_5'] = self._calculate_recent_form(home_recent.head(5), home_team)
            features['home_form_10'] = self._calculate_recent_form(home_recent, home_team)
            features['home_goals_scored_5'] = self._calculate_goals_scored(home_recent.head(5), home_team)
            features['home_goals_conceded_5'] = self._calculate_goals_conceded(home_recent.head(5), home_team)
            features['home_clean_sheets_5'] = self._calculate_clean_sheets(home_recent.head(5), home_team)
        else:
            # Use historical averages if insufficient recent data
            home_stats = self._get_team_stats(home_team, league)
            features.update({
                'home_form_5': home_stats.get('win_rate', 0.5),
                'home_form_10': home_stats.get('win_rate', 0.5),
                'home_goals_scored_5': home_stats.get('goals_scored_per_match', 1.5),
                'home_goals_conceded_5': home_stats.get('goals_conceded_per_match', 1.3),
                'home_clean_sheets_5': home_stats.get('clean_sheet_rate', 0.2)
            })
        
        # Away team form
        if len(away_recent) >= 3:
            features['away_form_5'] = self._calculate_recent_form(away_recent.head(5), away_team)
            features['away_form_10'] = self._calculate_recent_form(away_recent, away_team)
            features['away_goals_scored_5'] = self._calculate_goals_scored(away_recent.head(5), away_team)
            features['away_goals_conceded_5'] = self._calculate_goals_conceded(away_recent.head(5), away_team)
            features['away_clean_sheets_5'] = self._calculate_clean_sheets(away_recent.head(5), away_team)
        else:
            away_stats = self._get_team_stats(away_team, league)
            features.update({
                'away_form_5': away_stats.get('win_rate', 0.4),
                'away_form_10': away_stats.get('win_rate', 0.4),
                'away_goals_scored_5': away_stats.get('goals_scored_per_match', 1.2),
                'away_goals_conceded_5': away_stats.get('goals_conceded_per_match', 1.5),
                'away_clean_sheets_5': away_stats.get('clean_sheet_rate', 0.15)
            })
        
        # Form differentials and momentum
        features['form_5_diff'] = features['home_form_5'] - features['away_form_5']
        features['form_10_diff'] = features['home_form_10'] - features['away_form_10']
        features['goals_scored_diff_5'] = features['home_goals_scored_5'] - features['away_goals_scored_5']
        features['goals_conceded_diff_5'] = features['home_goals_conceded_5'] - features['away_goals_conceded_5']
        
        return features
    
    def _get_historical_features(self, home_team, away_team, league):
        """Historical performance patterns"""
        features = {}
        
        home_stats = self._get_team_stats(home_team, league)
        away_stats = self._get_team_stats(away_team, league)
        
        # Historical performance metrics
        features.update({
            'home_historical_win_rate': home_stats.get('win_rate', 0.5),
            'away_historical_win_rate': away_stats.get('win_rate', 0.4),
            'home_historical_goals_avg': home_stats.get('goals_scored_per_match', 1.5),
            'away_historical_goals_avg': away_stats.get('goals_scored_per_match', 1.2),
            'home_historical_goals_conceded_avg': home_stats.get('goals_conceded_per_match', 1.3),
            'away_historical_goals_conceded_avg': away_stats.get('goals_conceded_per_match', 1.5),
            'home_clean_sheet_rate': home_stats.get('clean_sheet_rate', 0.2),
            'away_clean_sheet_rate': away_stats.get('clean_sheet_rate', 0.15)
        })
        
        return features
    
    def _get_h2h_features(self, home_team, away_team, league):
        """Head-to-head historical features"""
        features = {}
        
        # Get H2H matches
        h2h_matches = self._get_h2h_matches(home_team, away_team, league)
        
        if len(h2h_matches) > 0:
            features['h2h_total_matches'] = len(h2h_matches)
            features['h2h_home_wins'] = len([m for m in h2h_matches if (
                (m['home_team'] == home_team and m['result'] == 'H') or
                (m['away_team'] == home_team and m['result'] == 'A')
            )])
            features['h2h_away_wins'] = len([m for m in h2h_matches if (
                (m['home_team'] == away_team and m['result'] == 'H') or
                (m['away_team'] == away_team and m['result'] == 'A')
            )])
            features['h2h_draws'] = len([m for m in h2h_matches if m['result'] == 'D'])
            
            features['h2h_home_win_rate'] = features['h2h_home_wins'] / len(h2h_matches)
            features['h2h_goal_avg'] = np.mean([m.get('home_goals', 0) + m.get('away_goals', 0) 
                                              for m in h2h_matches if m.get('home_goals') is not None])
            
            # Recent H2H form (last 3 matches)
            recent_h2h = h2h_matches[:3]
            if recent_h2h:
                features['h2h_recent_home_wins'] = len([m for m in recent_h2h if (
                    (m['home_team'] == home_team and m['result'] == 'H') or
                    (m['away_team'] == home_team and m['result'] == 'A')
                )])
                features['h2h_recent_home_win_rate'] = features['h2h_recent_home_wins'] / len(recent_h2h)
        else:
            # Default values for no H2H history
            features.update({
                'h2h_total_matches': 0, 'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0,
                'h2h_home_win_rate': 0.5, 'h2h_goal_avg': 2.5,
                'h2h_recent_home_wins': 0, 'h2h_recent_home_win_rate': 0.5
            })
        
        return features
    
    def _get_advanced_stats(self, home_team, away_team, league):
        """Advanced statistical features"""
        features = {}
        
        home_stats = self._get_team_stats(home_team, league)
        away_stats = self._get_team_stats(away_team, league)
        
        # Expected goals calculations
        features['home_xg_offensive'] = home_stats.get('goals_scored_per_match', 1.5) * 1.1
        features['away_xg_offensive'] = away_stats.get('goals_scored_per_match', 1.2) * 0.9
        features['home_xg_defensive'] = home_stats.get('goals_conceded_per_match', 1.3) * 0.9
        features['away_xg_defensive'] = away_stats.get('goals_conceded_per_match', 1.5) * 1.1
        
        # Combined expected goals
        features['expected_goals_home'] = (features['home_xg_offensive'] + features['away_xg_defensive']) / 2
        features['expected_goals_away'] = (features['away_xg_offensive'] + features['home_xg_defensive']) / 2
        features['expected_total_goals'] = features['expected_goals_home'] + features['expected_goals_away']
        
        # Performance consistency
        features['home_consistency'] = home_stats.get('consistency', 0.7)
        features['away_consistency'] = away_stats.get('consistency', 0.6)
        
        # Match competitiveness
        home_win_rate = features.get('home_win_rate', 0.5)
        away_win_rate = features.get('away_win_rate', 0.4)
        strength_diff = abs(home_win_rate - away_win_rate)
        features['match_competitiveness'] = 1.0 - strength_diff
        
        return features
    
    def _get_contextual_features(self, home_team, away_team, league):
        """Contextual match features"""
        features = {}
        
        # Match importance
        features['is_derby'] = 1.0 if self._is_derby(home_team, away_team) else 0.0
        features['is_top_team_clash'] = 1.0 if (self._is_top_team(home_team) and self._is_top_team(away_team)) else 0.0
        features['is_relegation_battle'] = 1.0 if (self._is_bottom_team(home_team) and self._is_bottom_team(away_team)) else 0.0
        
        # League context
        features['league_strength'] = self._get_league_strength(league)
        
        # Time factors
        current_date = datetime.now()
        features['weekend_match'] = 1.0 if current_date.weekday() >= 5 else 0.0
        features['evening_match'] = 1.0 if current_date.hour >= 17 else 0.0
        
        # Season phase
        features['season_phase'] = self._get_season_phase()
        
        return features
    
    def _get_live_features(self, home_team, away_team, league):
        """Live data features from API"""
        features = {}
        
        # These would be populated from live API data
        # For now, using enhanced placeholders
        
        features['live_injuries_impact'] = 0.0  # Would be calculated from injury data
        features['live_suspensions_impact'] = 0.0
        features['live_lineup_strength'] = 1.0
        
        # Recent form from API
        features['live_home_form'] = 0.0  # Would be updated from API
        features['live_away_form'] = 0.0
        
        return features
    
    # Helper methods with real data integration
    def _get_team_stats(self, team_name, league):
        """Get comprehensive team statistics from database"""
        if not self.db:
            return self._get_default_team_stats(team_name)
            
        conn = self.db._get_connection()
        
        try:
            # Get all matches for the team
            query = '''
                SELECT home_team, away_team, home_goals, away_goals, result 
                FROM matches 
                WHERE (home_team = ? OR away_team = ?) 
                AND league = ?
                AND result IS NOT NULL
            '''
            
            df = pd.read_sql_query(query, conn, params=(team_name, team_name, league))
            
            if len(df) == 0:
                return self._get_default_team_stats(team_name)
            
            # Calculate statistics
            stats = {}
            stats['matches_played'] = len(df)
            
            # Wins, draws, losses
            wins = 0
            goals_scored = 0
            goals_conceded = 0
            clean_sheets = 0
            
            for _, match in df.iterrows():
                if match['home_team'] == team_name:
                    goals_scored += match['home_goals'] or 0
                    goals_conceded += match['away_goals'] or 0
                    if match['result'] == 'H':
                        wins += 1
                    if match['away_goals'] == 0:
                        clean_sheets += 1
                else:
                    goals_scored += match['away_goals'] or 0
                    goals_conceded += match['home_goals'] or 0
                    if match['result'] == 'A':
                        wins += 1
                    if match['home_goals'] == 0:
                        clean_sheets += 1
            
            stats['wins'] = wins
            stats['draws'] = len(df) - wins - (len(df) - wins - (len(df) - wins))  # Simplified
            stats['losses'] = len(df) - wins - stats['draws']
            stats['win_rate'] = wins / len(df) if len(df) > 0 else 0.5
            stats['goals_scored_per_match'] = goals_scored / len(df) if len(df) > 0 else 1.5
            stats['goals_conceded_per_match'] = goals_conceded / len(df) if len(df) > 0 else 1.3
            stats['clean_sheet_rate'] = clean_sheets / len(df) if len(df) > 0 else 0.2
            
            # Consistency (win rate in last 10 vs overall)
            if len(df) >= 10:
                recent_df = df.head(10)
                recent_wins = 0
                for _, match in recent_df.iterrows():
                    if (match['home_team'] == team_name and match['result'] == 'H') or \
                       (match['away_team'] == team_name and match['result'] == 'A'):
                        recent_wins += 1
                recent_win_rate = recent_wins / 10
                stats['consistency'] = 1.0 - abs(stats['win_rate'] - recent_win_rate)
            else:
                stats['consistency'] = 0.7
            
            return stats
            
        except Exception as e:
            print(f"Error getting team stats for {team_name}: {e}")
            return self._get_default_team_stats(team_name)
        finally:
            conn.close()
    
    def _get_team_recent_matches(self, team_name, league, limit=10):
        """Get recent matches for a team"""
        if not self.db:
            return pd.DataFrame()
            
        try:
            conn = self.db._get_connection()
            query = '''
                SELECT home_team, away_team, home_goals, away_goals, result, match_date
                FROM matches 
                WHERE (home_team = ? OR away_team = ?)
                AND league = ?
                AND result IS NOT NULL
                ORDER BY match_date DESC
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(team_name, team_name, league, limit))
            conn.close()
            return df
        except:
            return pd.DataFrame()
    
    def _get_default_team_stats(self, team_name):
        """Get default stats for teams with no history"""
        # Basic stats based on team reputation
        if self._is_top_team(team_name):
            return {
                'win_rate': 0.65, 'goals_scored_per_match': 2.0, 
                'goals_conceded_per_match': 1.0, 'clean_sheet_rate': 0.3,
                'consistency': 0.8
            }
        elif self._is_bottom_team(team_name):
            return {
                'win_rate': 0.25, 'goals_scored_per_match': 1.0, 
                'goals_conceded_per_match': 2.0, 'clean_sheet_rate': 0.1,
                'consistency': 0.5
            }
        else:
            return {
                'win_rate': 0.45, 'goals_scored_per_match': 1.5, 
                'goals_conceded_per_match': 1.5, 'clean_sheet_rate': 0.2,
                'consistency': 0.7
            }
    
    def _calculate_recent_form(self, matches, team):
        """Calculate recent form (points per game)"""
        if len(matches) == 0:
            return 0.5
        
        points = 0
        for _, match in matches.iterrows():
            if match['home_team'] == team:
                if match['result'] == 'H': points += 3
                elif match['result'] == 'D': points += 1
            else:
                if match['result'] == 'A': points += 3
                elif match['result'] == 'D': points += 1
        
        max_points = 3 * len(matches)
        return points / max_points if max_points > 0 else 0.5
    
    def _calculate_goals_scored(self, matches, team):
        """Calculate average goals scored in recent matches"""
        goals = []
        for _, match in matches.iterrows():
            if match['home_team'] == team:
                goals.append(match.get('home_goals', 0))
            else:
                goals.append(match.get('away_goals', 0))
        return np.mean(goals) if goals else 1.5
    
    def _calculate_goals_conceded(self, matches, team):
        """Calculate average goals conceded in recent matches"""
        goals = []
        for _, match in matches.iterrows():
            if match['home_team'] == team:
                goals.append(match.get('away_goals', 0))
            else:
                goals.append(match.get('home_goals', 0))
        return np.mean(goals) if goals else 1.3
    
    def _calculate_clean_sheets(self, matches, team):
        """Calculate clean sheet rate in recent matches"""
        if len(matches) == 0:
            return 0.2
        
        clean_sheets = 0
        for _, match in matches.iterrows():
            if match['home_team'] == team and match.get('away_goals', 1) == 0:
                clean_sheets += 1
            elif match['away_team'] == team and match.get('home_goals', 1) == 0:
                clean_sheets += 1
        
        return clean_sheets / len(matches)
    
    def _calculate_home_advantage(self, team, league):
        """Calculate home advantage for a team"""
        if not self.db:
            return 0.55
            
        conn = self.db._get_connection()
        
        try:
            query = '''
                SELECT COUNT(*) as total_home,
                       SUM(CASE WHEN result = 'H' THEN 1 ELSE 0 END) as home_wins
                FROM matches 
                WHERE home_team = ? AND league = ? AND result IS NOT NULL
            '''
            
            result = conn.execute(query, (team, league)).fetchone()
            
            if result and result[0] > 0:
                home_win_rate = result[1] / result[0]
                # Normalize to 0-1 scale where 0.5 is average
                return min(max((home_win_rate - 0.45) * 2, 0.3), 0.7)
            else:
                return 0.55  # Default home advantage
                
        except Exception as e:
            return 0.55
        finally:
            conn.close()
    
    def _get_h2h_matches(self, home_team, away_team, league):
        """Get head-to-head matches from database"""
        if not self.db:
            return []
            
        conn = self.db._get_connection()
        
        try:
            query = '''
                SELECT home_team, away_team, home_goals, away_goals, result 
                FROM matches 
                WHERE ((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?))
                AND league = ?
                AND result IS NOT NULL
                ORDER BY match_date DESC
            '''
            
            results = conn.execute(query, (home_team, away_team, away_team, home_team, league)).fetchall()
            
            matches = []
            for row in results:
                matches.append({
                    'home_team': row[0],
                    'away_team': row[1],
                    'home_goals': row[2],
                    'away_goals': row[3],
                    'result': row[4]
                })
            
            return matches
            
        except Exception as e:
            print(f"Error getting H2H matches: {e}")
            return []
        finally:
            conn.close()
    
    def _load_team_strengths(self):
        """Load team strength ratings"""
        return {
            'Premier League': {
                'Manchester City': {'attack': 2.4, 'defense': 0.8, 'overall': 2.1},
                'Arsenal': {'attack': 2.2, 'defense': 0.9, 'overall': 2.0},
                'Liverpool': {'attack': 2.1, 'defense': 1.0, 'overall': 1.9},
                'Aston Villa': {'attack': 1.9, 'defense': 1.1, 'overall': 1.7},
                'Tottenham': {'attack': 1.8, 'defense': 1.3, 'overall': 1.6},
            },
            'La Liga': {
                'Real Madrid': {'attack': 2.3, 'defense': 0.8, 'overall': 2.0},
                'Barcelona': {'attack': 2.2, 'defense': 0.9, 'overall': 1.9},
                'Atletico Madrid': {'attack': 1.9, 'defense': 0.8, 'overall': 1.7},
                'Girona': {'attack': 1.8, 'defense': 1.2, 'overall': 1.6},
            }
        }
    
    def _is_top_team(self, team):
        """Check if team is considered top team"""
        top_teams = {
            'Manchester City', 'Arsenal', 'Liverpool', 'Chelsea', 'Manchester United',
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Bayern Munich', 
            'Borussia Dortmund', 'PSG', 'Inter Milan', 'AC Milan', 'Juventus'
        }
        return team in top_teams
    
    def _is_bottom_team(self, team):
        """Check if team is considered bottom team"""
        bottom_teams = {
            'Burnley', 'Sheffield United', 'Luton Town', 'Nottingham Forest',
            'Almeria', 'Granada', 'Mainz', 'Darmstadt'
        }
        return team in bottom_teams
    
    def _is_derby(self, home_team, away_team):
        """Check if match is a derby"""
        derbies = [
            ('Manchester United', 'Manchester City'),
            ('Arsenal', 'Tottenham'),
            ('Liverpool', 'Everton'),
            ('Real Madrid', 'Barcelona'),
            ('AC Milan', 'Inter Milan')
        ]
        return (home_team, away_team) in derbies or (away_team, home_team) in derbies
    
    def _get_league_strength(self, league):
        """Get league strength rating"""
        strengths = {
            'Premier League': 1.0,
            'La Liga': 0.95,
            'Bundesliga': 0.9,
            'Serie A': 0.88,
            'Ligue 1': 0.85
        }
        return strengths.get(league, 0.8)
    
    def _get_season_phase(self):
        """Get current season phase"""
        month = datetime.now().month
        if month in [8, 9]: return 0.2  # Early season
        elif month in [10, 11, 12]: return 0.5  # Mid-season
        elif month in [1, 2]: return 0.7  # Transfer window
        elif month in [3, 4]: return 0.9  # Run-in
        else: return 1.0  # End of season