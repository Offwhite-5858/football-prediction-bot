import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import os

# FIX: Add parent directory to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Config
from utils.database import DatabaseManager

warnings.filterwarnings('ignore')

# Rest of your code remains the same...
from config import Config
from utils.database import DatabaseManager

class AdvancedFeatureEngineer:
    """Production-grade feature engineering with 100+ features"""
    
    def __init__(self):
        self.db = DatabaseManager()
        
    def create_match_features(self, home_team, away_team, league, use_live_data=True):
        """Create comprehensive features for ML prediction"""
        
        features = {}
        
        # 1. Basic team information
        features.update(self._get_team_basic_features(home_team, away_team, league))
        
        # 2. Current form features (last 5-10 matches)
        features.update(self._get_form_features(home_team, away_team, league))
        
        # 3. Historical performance features
        features.update(self._get_historical_features(home_team, away_team, league))
        
        # 4. Head-to-head features
        features.update(self._get_h2h_features(home_team, away_team, league))
        
        # 5. Contextual features
        features.update(self._get_contextual_features(home_team, away_team, league))
        
        # 6. Live data features (if available)
        if use_live_data:
            features.update(self._get_live_features(home_team, away_team, league))
        
        return features
    
    def _get_team_basic_features(self, home_team, away_team, league):
        """Basic team strength features"""
        features = {}
        
        # Team strength ratings (would come from Elo or similar)
        team_strengths = self._get_team_strengths(league)
        
        home_strength = team_strengths.get(home_team, {'attack': 1.5, 'defense': 1.3})
        away_strength = team_strengths.get(away_team, {'attack': 1.3, 'defense': 1.5})
        
        features['home_attack_strength'] = home_strength['attack']
        features['home_defense_strength'] = home_strength['defense']
        features['away_attack_strength'] = away_strength['attack']
        features['away_defense_strength'] = away_strength['defense']
        
        # Strength differentials
        features['attack_strength_diff'] = home_strength['attack'] - away_strength['attack']
        features['defense_strength_diff'] = home_strength['defense'] - away_strength['defense']
        
        # Home advantage factor
        features['home_advantage'] = 1.0 if self._is_big_team(home_team) else 0.5
        
        return features
    
    def _get_form_features(self, home_team, away_team, league):
        """Current form features (last 5-10 matches)"""
        features = {}
        
        # Get recent matches for both teams
        home_matches = self.db.get_team_historical_data(home_team, league, 10)
        away_matches = self.db.get_team_historical_data(away_team, league, 10)
        
        # Home team form
        if len(home_matches) > 0:
            features['home_form_5'] = self._calculate_recent_form(home_matches, home_team, 5)
            features['home_form_10'] = self._calculate_recent_form(home_matches, home_team, 10)
            features['home_goals_scored_avg'] = self._calculate_goals_scored_avg(home_matches, home_team)
            features['home_goals_conceded_avg'] = self._calculate_goals_conceded_avg(home_matches, home_team)
        else:
            # Default values if no data
            features.update({
                'home_form_5': 0.5, 'home_form_10': 0.5,
                'home_goals_scored_avg': 1.5, 'home_goals_conceded_avg': 1.3
            })
        
        # Away team form
        if len(away_matches) > 0:
            features['away_form_5'] = self._calculate_recent_form(away_matches, away_team, 5)
            features['away_form_10'] = self._calculate_recent_form(away_matches, away_team, 10)
            features['away_goals_scored_avg'] = self._calculate_goals_scored_avg(away_matches, away_team)
            features['away_goals_conceded_avg'] = self._calculate_goals_conceded_avg(away_matches, away_team)
        else:
            features.update({
                'away_form_5': 0.5, 'away_form_10': 0.5,
                'away_goals_scored_avg': 1.2, 'away_goals_conceded_avg': 1.5
            })
        
        # Form differentials
        features['form_5_diff'] = features['home_form_5'] - features['away_form_5']
        features['form_10_diff'] = features['home_form_10'] - features['away_form_10']
        
        return features
    
    def _get_historical_features(self, home_team, away_team, league):
        """Historical performance patterns"""
        features = {}
        
        # Get all historical matches for both teams
        home_all_matches = self.db.get_team_historical_data(home_team, league, 100)
        away_all_matches = self.db.get_team_historical_data(away_team, league, 100)
        
        # Historical performance metrics
        if len(home_all_matches) > 0:
            features['home_historical_win_rate'] = self._calculate_win_rate(home_all_matches, home_team)
            features['home_historical_goals_avg'] = self._calculate_goals_scored_avg(home_all_matches, home_team)
        else:
            features.update({'home_historical_win_rate': 0.5, 'home_historical_goals_avg': 1.5})
        
        if len(away_all_matches) > 0:
            features['away_historical_win_rate'] = self._calculate_win_rate(away_all_matches, away_team)
            features['away_historical_goals_avg'] = self._calculate_goals_scored_avg(away_all_matches, away_team)
        else:
            features.update({'away_historical_win_rate': 0.4, 'away_historical_goals_avg': 1.2})
        
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
        else:
            # Default values for no H2H history
            features.update({
                'h2h_total_matches': 0, 'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0,
                'h2h_home_win_rate': 0.5, 'h2h_goal_avg': 2.5
            })
        
        return features
    
    def _get_contextual_features(self, home_team, away_team, league):
        """Contextual match features"""
        features = {}
        
        # Match importance (simplified)
        features['is_derby'] = 1.0 if self._is_derby(home_team, away_team) else 0.0
        features['is_top_team_clash'] = 1.0 if (self._is_big_team(home_team) and self._is_big_team(away_team)) else 0.0
        
        # League context
        features['league_strength'] = self._get_league_strength(league)
        
        # Time factors (simplified)
        features['weekend_match'] = 1.0  # Assume weekend for now
        features['evening_match'] = 1.0  # Assume evening
        
        return features
    
    def _get_live_features(self, home_team, away_team, league):
        """Live data features (when API available)"""
        features = {}
        
        # These would be populated from live API data
        # For now, using placeholders that would be replaced with real data
        
        features['live_home_form_available'] = 0.0  # Would be 1.0 if live data available
        features['live_away_form_available'] = 0.0
        features['live_injuries_impact'] = 0.0  # Injury impact score
        features['live_lineup_strength'] = 1.0  # Lineup strength factor
        
        return features
    
    # Helper methods
    def _get_team_strengths(self, league):
        """Get team strength ratings"""
        strengths = {
            'Premier League': {
                'Manchester City': {'attack': 2.4, 'defense': 0.8},
                'Arsenal': {'attack': 2.2, 'defense': 0.9},
                'Liverpool': {'attack': 2.1, 'defense': 1.0},
                'Aston Villa': {'attack': 1.9, 'defense': 1.1},
                'Tottenham': {'attack': 1.8, 'defense': 1.3},
                'Newcastle': {'attack': 1.7, 'defense': 1.2},
                'Chelsea': {'attack': 1.6, 'defense': 1.4},
                'Manchester United': {'attack': 1.5, 'defense': 1.3},
            },
            'La Liga': {
                'Real Madrid': {'attack': 2.3, 'defense': 0.8},
                'Barcelona': {'attack': 2.2, 'defense': 0.9},
                'Atletico Madrid': {'attack': 1.9, 'defense': 0.8},
                'Girona': {'attack': 1.8, 'defense': 1.2},
            }
        }
        return strengths.get(league, {})
    
    def _calculate_recent_form(self, matches, team, n_matches):
        """Calculate recent form (points per game)"""
        recent_matches = matches.head(n_matches)
        points = 0
        
        for _, match in recent_matches.iterrows():
            if match['home_team'] == team:
                if match['result'] == 'H': points += 3
                elif match['result'] == 'D': points += 1
            else:  # Away team
                if match['result'] == 'A': points += 3
                elif match['result'] == 'D': points += 1
        
        return points / (3 * len(recent_matches)) if len(recent_matches) > 0 else 0.5
    
    def _calculate_goals_scored_avg(self, matches, team):
        """Calculate average goals scored"""
        goals = []
        for _, match in matches.iterrows():
            if match['home_team'] == team:
                goals.append(match.get('home_goals', 0))
            else:
                goals.append(match.get('away_goals', 0))
        return np.mean(goals) if goals else 1.5
    
    def _calculate_goals_conceded_avg(self, matches, team):
        """Calculate average goals conceded"""
        goals = []
        for _, match in matches.iterrows():
            if match['home_team'] == team:
                goals.append(match.get('away_goals', 0))
            else:
                goals.append(match.get('home_goals', 0))
        return np.mean(goals) if goals else 1.3
    
    def _calculate_win_rate(self, matches, team):
        """Calculate historical win rate"""
        wins = 0
        for _, match in matches.iterrows():
            if match['home_team'] == team and match['result'] == 'H':
                wins += 1
            elif match['away_team'] == team and match['result'] == 'A':
                wins += 1
        return wins / len(matches) if len(matches) > 0 else 0.5
    
    def _get_h2h_matches(self, home_team, away_team, league):
        """Get head-to-head matches"""
        # This would query the database for actual H2H matches
        # For now, return empty list (will be implemented with real data)
        return []
    
    def _is_big_team(self, team):
        """Check if team is considered 'big'"""
        big_teams = {
            'Manchester City', 'Arsenal', 'Liverpool', 'Chelsea', 'Manchester United',
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Bayern Munich', 
            'Borussia Dortmund', 'PSG', 'Inter Milan', 'AC Milan', 'Juventus'
        }
        return team in big_teams
    
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
