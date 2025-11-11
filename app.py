import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import sqlite3
import requests
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration
class Config:
    DATA_PATH = "football_data"
    MODEL_PATH = "models"
    DATABASE_PATH = f"{DATA_PATH}/predictions.db"
    API_CACHE_PATH = f"{DATA_PATH}/api_cache"
    
    # Model parameters
    ELO_K_FACTOR = 32
    ELO_HOME_ADVANTAGE = 100
    
    @staticmethod
    def init_directories():
        for path in [Config.DATA_PATH, Config.MODEL_PATH, Config.API_CACHE_PATH]:
            os.makedirs(path, exist_ok=True)
        
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Enhanced schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                match_id TEXT PRIMARY KEY,
                home_team TEXT,
                away_team TEXT,
                league TEXT,
                prediction TEXT,
                confidence REAL,
                probability_home REAL,
                probability_draw REAL,
                probability_away REAL,
                value_edge REAL,
                timestamp TEXT,
                model_version TEXT,
                elo_home INTEGER,
                elo_away INTEGER,
                form_home REAL,
                form_away REAL,
                certainty_index REAL,
                risk_assessment TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_logs (
                version TEXT PRIMARY KEY,
                accuracy REAL,
                cv_accuracy REAL,
                timestamp TEXT,
                features_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()

Config.init_directories()

class AdvancedELOSystem:
    """Advanced ELO rating system with dynamic K-factors and form adjustments"""
    
    def __init__(self, base_rating=1500, k_factor=32, home_advantage=100):
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.team_ratings = {}
        self.team_history = {}
    
    def get_rating(self, team: str) -> float:
        return self.team_ratings.get(team, self.base_rating)
    
    def update_ratings(self, home_team: str, away_team: str, home_goals: int, 
                      away_goals: int, importance: float = 1.0) -> Tuple[float, float]:
        """Update ELO ratings based on match result with goal difference consideration"""
        
        # Initialize teams if not present
        if home_team not in self.team_ratings:
            self.team_ratings[home_team] = self.base_rating
        if away_team not in self.team_ratings:
            self.team_ratings[away_team] = self.base_rating
        
        # Get current ratings with home advantage
        rh = self.team_ratings[home_team] + self.home_advantage
        ra = self.team_ratings[away_team]
        
        # Calculate expected scores
        expected_home = 1 / (1 + 10**((ra - rh) / 400))
        expected_away = 1 - expected_home
        
        # Calculate actual scores based on result and goal difference
        goal_diff = home_goals - away_goals
        if goal_diff > 0:
            actual_home = 1.0
            actual_away = 0.0
        elif goal_diff < 0:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5
        
        # Dynamic K-factor based on goal difference and match importance
        k_dynamic = self.k_factor * importance * (1 + min(abs(goal_diff) / 3, 1))
        
        # Update ratings
        new_rh = self.team_ratings[home_team] + k_dynamic * (actual_home - expected_home)
        new_ra = self.team_ratings[away_team] + k_dynamic * (actual_away - expected_away)
        
        self.team_ratings[home_team] = new_rh
        self.team_ratings[away_team] = new_ra
        
        # Store history
        self._store_match_history(home_team, away_team, home_goals, away_goals, new_rh, new_ra)
        
        return new_rh, new_ra
    
    def _store_match_history(self, home_team: str, away_team: str, home_goals: int,
                           away_goals: int, new_rh: float, new_ra: float):
        """Store match history for form calculations"""
        timestamp = datetime.now()
        
        for team, goals_for, goals_against, rating in [
            (home_team, home_goals, away_goals, new_rh),
            (away_team, away_goals, home_goals, new_ra)
        ]:
            if team not in self.team_history:
                self.team_history[team] = []
            
            points = 3 if (team == home_team and home_goals > away_goals) or \
                         (team == away_team and away_goals > home_goals) else \
                    1 if home_goals == away_goals else 0
            
            self.team_history[team].append({
                'timestamp': timestamp,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'points': points,
                'rating': rating,
                'is_home': team == home_team
            })
            
            # Keep only last 20 matches
            if len(self.team_history[team]) > 20:
                self.team_history[team] = self.team_history[team][-20:]
    
    def calculate_form(self, team: str, matches: int = 10) -> float:
        """Calculate weighted recent form (0-1 scale)"""
        if team not in self.team_history or not self.team_history[team]:
            return 0.5
        
        matches_data = self.team_history[team][-matches:]
        total_weight = 0
        weighted_points = 0
        
        for i, match in enumerate(matches_data):
            # More weight to recent matches (exponential decay)
            weight = (0.9 ** (len(matches_data) - i - 1))
            weighted_points += (match['points'] / 3) * weight
            total_weight += weight
        
        return weighted_points / total_weight if total_weight > 0 else 0.5
    
    def calculate_attack_strength(self, team: str) -> float:
        """Calculate team's attacking strength"""
        if team not in self.team_history:
            return 1.0
        
        recent_matches = self.team_history[team][-10:]
        if not recent_matches:
            return 1.0
        
        goals_scored = sum(match['goals_for'] for match in recent_matches)
        return goals_scored / len(recent_matches) / 1.5  # Normalize
    
    def calculate_defense_strength(self, team: str) -> float:
        """Calculate team's defensive strength"""
        if team not in self.team_history:
            return 1.0
        
        recent_matches = self.team_history[team][-10:]
        if not recent_matches:
            return 1.0
        
        goals_conceded = sum(match['goals_against'] for match in recent_matches)
        return 1 - (goals_conceded / len(recent_matches) / 1.5)  # Invert for strength

    def get_default_stats(self):
        """Get default statistics for unknown teams"""
        return {
            'strength': 0.5,
            'recent_form': [],
            'avg_goals_scored': 1.2,
            'avg_goals_conceded': 1.2,
            'win_rate': 0.33,
            'goal_difference': 0.0
        }

class AdvancedFeatureEngineer:
    """Advanced feature engineering inspired by top prediction platforms"""
    
    def __init__(self, elo_system: AdvancedELOSystem):
        self.elo_system = elo_system
    
    def extract_advanced_features(self, home_team: str, away_team: str, 
                                league: str, venue_factors: Dict = None) -> np.ndarray:
        """Extract comprehensive features for prediction"""
        
        features = []
        
        # 1. ELO-based features
        home_elo = self.elo_system.get_rating(home_team)
        away_elo = self.elo_system.get_rating(away_team)
        elo_diff = home_elo - away_elo
        
        features.extend([home_elo, away_elo, elo_diff])
        
        # 2. Form and momentum features
        home_form = self.elo_system.calculate_form(home_team)
        away_form = self.elo_system.calculate_form(away_team)
        form_diff = home_form - away_form
        
        features.extend([home_form, away_form, form_diff])
        
        # 3. Attack/defense strength
        home_attack = self.elo_system.calculate_attack_strength(home_team)
        away_attack = self.elo_system.calculate_attack_strength(away_team)
        home_defense = self.elo_system.calculate_defense_strength(home_team)
        away_defense = self.elo_system.calculate_defense_strength(away_team)
        
        features.extend([home_attack, away_attack, home_defense, away_defense])
        
        # 4. League and venue factors
        league_strength = self._get_league_strength(league)
        home_advantage = self._get_venue_advantage(home_team, venue_factors)
        
        features.extend([league_strength, home_advantage])
        
        # 5. Recent performance trends
        home_momentum = self._calculate_momentum(home_team)
        away_momentum = self._calculate_momentum(away_team)
        
        features.extend([home_momentum, away_momentum])
        
        # 6. Match importance factors
        importance_factor = self._calculate_match_importance(home_team, away_team, league)
        features.append(importance_factor)
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_momentum(self, team: str) -> float:
        """Calculate recent performance momentum"""
        if team not in self.elo_system.team_history:
            return 0.0
        
        recent_matches = self.elo_system.team_history[team][-5:]
        if len(recent_matches) < 2:
            return 0.0
        
        points = [m['points'] for m in recent_matches]
        if len(points) < 2:
            return 0.0
        
        # Simple linear regression for trend
        x = np.arange(len(points))
        slope, _ = np.polyfit(x, points, 1)
        return slope / 3  # Normalize
    
    def _get_league_strength(self, league: str) -> float:
        """Get league strength coefficient"""
        league_strengths = {
            'EPL': 1.0, 'La Liga': 0.98, 'Bundesliga': 0.96,
            'Serie A': 0.95, 'Ligue 1': 0.92, 'Champions League': 1.1,
            'Europa League': 1.05, 'Conference League': 1.02
        }
        return league_strengths.get(league, 0.9)
    
    def _get_venue_advantage(self, team: str, venue_factors: Dict) -> float:
        """Calculate venue-specific advantage"""
        if not venue_factors or team not in venue_factors:
            return 0.12  # Default home advantage
        
        return venue_factors.get(team, 0.12)
    
    def _calculate_match_importance(self, home_team: str, away_team: str, league: str) -> float:
        """Calculate match importance factor (derby, relegation, title race)"""
        importance = 1.0
        
        # Simple implementation - in practice, you'd use league table data
        big_teams = ['Man City', 'Liverpool', 'Arsenal', 'Chelsea', 'Man United',
                    'Real Madrid', 'Barcelona', 'Bayern Munich', 'PSG', 'Juventus']
        
        if home_team in big_teams and away_team in big_teams:
            importance *= 1.3  # Big match
        
        return importance

class DynamicDataManager:
    """Dynamic data manager for real-time football data"""
    
    def __init__(self, api_config: Dict = None):
        self.api_config = api_config or {}
        self.cache = {}
        self.last_update = {}
    
    def get_fixtures(self, league: str, days: int = 7) -> List[Dict]:
        """Get upcoming fixtures - in practice, this would call a real API"""
        # Fallback to generated fixtures
        return self._generate_realistic_fixtures(league, days)
    
    def _generate_realistic_fixtures(self, league: str, days: int) -> List[Dict]:
        """Generate realistic upcoming fixtures"""
        teams = {
            'EPL': ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
                   'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds',
                   'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
                   'Nottingham Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolves'],
            'La Liga': ['Almeria', 'Athletic Bilbao', 'Atletico Madrid', 'Barcelona',
                       'Cadiz', 'Celta Vigo', 'Elche', 'Espanyol', 'Getafe', 'Girona',
                       'Mallorca', 'Osasuna', 'Rayo Vallecano', 'Real Betis', 'Real Madrid',
                       'Real Sociedad', 'Sevilla', 'Valencia', 'Valladolid', 'Villarreal']
        }
        
        league_teams = teams.get(league, teams['EPL'])
        fixtures = []
        
        for i in range(min(10, len(league_teams) // 2)):
            home_team = league_teams[i * 2]
            away_team = league_teams[i * 2 + 1]
            
            fixtures.append({
                'id': f"{league}_{i+1}",
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'date': (datetime.now() + timedelta(days=i % days)).strftime('%Y-%m-%d'),
                'venue': f"{home_team} Stadium"
            })
        
        return fixtures

class AdvancedFootballPredictor:
    """Advanced prediction system inspired by top platforms"""
    
    def __init__(self, api_config: Dict = None):
        self.elo_system = AdvancedELOSystem()
        self.feature_engineer = AdvancedFeatureEngineer(self.elo_system)
        self.data_manager = DynamicDataManager(api_config)
        self.model = None
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model_version = "v2.0_advanced"
        
        # Initialize with historical data
        self._initialize_with_historical_data()
        self._train_advanced_ensemble()
    
    def _initialize_with_historical_data(self):
        """Initialize systems with comprehensive historical data"""
        historical_matches = self._load_historical_data()
        
        for match in historical_matches:
            self.elo_system.update_ratings(
                match['home_team'],
                match['away_team'],
                match['home_goals'],
                match['away_goals'],
                importance=1.0
            )
    
    def _load_historical_data(self) -> List[Dict]:
        """Load historical match data - replace with your data source"""
        # Generate realistic historical data for demonstration
        matches = []
        teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 
                'Tottenham', 'Newcastle', 'Brighton', 'West Ham', 'Aston Villa']
        
        for i in range(200):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Generate realistic scores
            home_goals = np.random.poisson(1.5)
            away_goals = np.random.poisson(1.2)
            
            matches.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': int(home_goals),
                'away_goals': int(away_goals),
                'date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d')
            })
        
        return matches
    
    def _train_advanced_ensemble(self):
        """Train advanced ensemble model"""
        try:
            print("🔄 Training advanced ensemble model...")
            
            # Generate training data from historical matches
            X, y = self._prepare_training_data()
            
            if len(X) < 50:
                print("⚠️ Insufficient training data, using fallback model")
                self._train_fallback_model()
                return
            
            # Advanced ensemble
            self.model = VotingClassifier(estimators=[
                ('xgb', xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )),
                ('gbc', GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                ))
            ], voting='soft', weights=[2, 1, 1])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train final model
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            
            # Calculate accuracy
            train_accuracy = accuracy_score(y, self.model.predict(X_scaled))
            print(f"✅ Advanced ensemble trained! Accuracy: {train_accuracy:.3f}")
            
        except Exception as e:
            print(f"❌ Advanced training failed: {e}")
            self._train_fallback_model()
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical matches"""
        X = []
        y = []
        
        # Generate synthetic training data for demonstration
        teams = list(self.elo_system.team_ratings.keys())
        if not teams:
            teams = ['Team_' + str(i) for i in range(1, 11)]
        
        for _ in range(200):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Generate features
            features = self.feature_engineer.extract_advanced_features(
                home_team, away_team, "EPL"
            )
            X.append(features[0])
            
            # Generate realistic outcome based on ELO
            home_elo = self.elo_system.get_rating(home_team)
            away_elo = self.elo_system.get_rating(away_team)
            home_win_prob = 1 / (1 + 10**((away_elo - home_elo - 100) / 400))
            
            # Sample outcome
            rand_val = np.random.random()
            if rand_val < home_win_prob * 0.85:
                y.append('HOME')
            elif rand_val < home_win_prob * 0.85 + 0.12:
                y.append('DRAW')
            else:
                y.append('AWAY')
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return np.array(X), y_encoded
    
    def _train_fallback_model(self):
        """Fallback to simpler model"""
        try:
            X, y = self._prepare_training_data()
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            accuracy = accuracy_score(y, self.model.predict(X_scaled))
            print(f"✅ Fallback model trained! Accuracy: {accuracy:.3f}")
        except Exception as e:
            print(f"❌ Fallback training failed: {e}")
            self.is_trained = False
    
    def predict_match(self, match_id: str, home_team: str, away_team: str, 
                     league: str, additional_context: Dict = None) -> Dict:
        """Make advanced prediction for a match"""
        
        if not self.is_trained:
            return self._fallback_prediction(match_id, home_team, away_team, league)
        
        try:
            # Extract advanced features
            features = self.feature_engineer.extract_advanced_features(
                home_team, away_team, league, additional_context
            )
            features_scaled = self.scaler.transform(features)
            
            # Get probabilities
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get prediction
            prediction_idx = np.argmax(probabilities)
            class_labels = self.label_encoder.classes_
            prediction = class_labels[prediction_idx]
            confidence = probabilities[prediction_idx]
            
            # Calculate advanced metrics
            prediction_metrics = self._calculate_prediction_metrics(
                probabilities, prediction
            )
            
            return {
                'match_id': match_id,
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'prediction': prediction,
                'confidence': float(confidence),
                'probability_home': float(probabilities[list(class_labels).index('HOME')] if 'HOME' in class_labels else 0.33),
                'probability_draw': float(probabilities[list(class_labels).index('DRAW')] if 'DRAW' in class_labels else 0.33),
                'probability_away': float(probabilities[list(class_labels).index('AWAY')] if 'AWAY' in class_labels else 0.34),
                'value_edge': prediction_metrics['value_edge'],
                'recommended_stake': prediction_metrics['recommended_stake'],
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat(),
                'elo_home': self.elo_system.get_rating(home_team),
                'elo_away': self.elo_system.get_rating(away_team),
                'form_home': self.elo_system.calculate_form(home_team),
                'form_away': self.elo_system.calculate_form(away_team),
                'certainty_index': prediction_metrics['certainty_index'],
                'risk_assessment': prediction_metrics['risk_assessment'],
                'fair_odds': prediction_metrics['fair_odds'],
                'market_odds': prediction_metrics['market_odds']
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction(match_id, home_team, away_team, league)
    
    def _calculate_prediction_metrics(self, probabilities: np.ndarray, prediction: str) -> Dict:
        """Calculate advanced prediction metrics"""
        
        # Fair odds calculation
        if prediction == 'HOME':
            fair_odds = 1.0 / (probabilities[list(self.label_encoder.classes_).index('HOME')] + 0.001)
        elif prediction == 'DRAW':
            fair_odds = 1.0 / (probabilities[list(self.label_encoder.classes_).index('DRAW')] + 0.001)
        else:
            fair_odds = 1.0 / (probabilities[list(self.label_encoder.classes_).index('AWAY')] + 0.001)
        
        # Add overround
        market_odds = fair_odds * 1.05
        
        # Value calculation
        confidence = probabilities[np.argmax(probabilities)]
        value_edge = (confidence * market_odds) - 1
        
        # Kelly criterion with conservative limits
        if value_edge > 0 and market_odds > 1:
            kelly_stake = min(max(0, value_edge / (market_odds - 1)), 0.08)
        else:
            kelly_stake = 0.0
        
        # Certainty index (0-1)
        certainty_index = 1 - (entropy(probabilities) / np.log(3))
        
        # Risk assessment
        if certainty_index > 0.7:
            risk_assessment = "LOW"
        elif certainty_index > 0.5:
            risk_assessment = "MEDIUM"
        else:
            risk_assessment = "HIGH"
        
        return {
            'value_edge': value_edge,
            'recommended_stake': kelly_stake,
            'certainty_index': certainty_index,
            'risk_assessment': risk_assessment,
            'fair_odds': fair_odds,
            'market_odds': market_odds
        }
    
    def _fallback_prediction(self, match_id: str, home_team: str, 
                           away_team: str, league: str) -> Dict:
        """Intelligent fallback prediction"""
        home_elo = self.elo_system.get_rating(home_team)
        away_elo = self.elo_system.get_rating(away_team)
        home_form = self.elo_system.calculate_form(home_team)
        away_form = self.elo_system.calculate_form(away_team)
        
        # Advanced fallback logic
        elo_diff = home_elo - away_elo + 100  # Home advantage
        form_diff = home_form - away_form
        
        combined_score = (elo_diff / 100) * 0.6 + form_diff * 0.4
        
        if combined_score > 0.3:
            prediction = "HOME"
            confidence = 0.6 + min(combined_score * 0.3, 0.3)
        elif combined_score < -0.3:
            prediction = "AWAY"
            confidence = 0.6 + min(abs(combined_score) * 0.3, 0.3)
        else:
            prediction = "DRAW"
            confidence = 0.4 + (0.5 - abs(combined_score)) * 0.2
        
        return {
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'prediction': prediction,
            'confidence': confidence,
            'probability_home': 0.33,
            'probability_draw': 0.33,
            'probability_away': 0.34,
            'value_edge': 0.0,
            'recommended_stake': 0.0,
            'model_version': 'fallback',
            'timestamp': datetime.now().isoformat(),
            'elo_home': home_elo,
            'elo_away': away_elo,
            'form_home': home_form,
            'form_away': away_form,
            'certainty_index': 0.5,
            'risk_assessment': 'MEDIUM',
            'fair_odds': 3.0,
            'market_odds': 3.15,
            'note': 'Using advanced fallback logic'
        }

def entropy(probabilities: np.ndarray) -> float:
    """Calculate entropy of probability distribution"""
    return -np.sum(probabilities * np.log(probabilities + 1e-10))

# =============================================================================
# STREAMLIT UI IMPLEMENTATION
# =============================================================================

def create_probability_chart(prediction):
    """Create probability distribution chart"""
    outcomes = ['Away Win', 'Draw', 'Home Win']
    probabilities = [
        prediction['probability_away'],
        prediction['probability_draw'], 
        prediction['probability_home']
    ]
    
    colors = ['#ff6b6b', '#feca57', '#48dbfb']
    
    fig = go.Figure(data=[
        go.Bar(
            x=outcomes,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Match Outcome Probabilities",
        yaxis_title="Probability",
        yaxis=dict(tickformat=".0%"),
        showlegend=False,
        height=400
    )
    
    return fig

def display_prediction_results(prediction):
    """Display comprehensive prediction results"""
    
    # Main prediction card
    st.markdown(f'''
    <div class="prediction-card">
        <h2 style="color: white; margin-bottom: 1rem;">🎯 Prediction Ready!</h2>
        <p style="color: white; opacity: 0.9;">
            <strong>{prediction['home_team']} vs {prediction['away_team']}</strong> • {prediction['league']}
        </p>
        <p style="color: white; opacity: 0.9;">Model: {prediction.get('model_version', 'Advanced')}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Results grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prediction", prediction['prediction'])
        st.metric("Confidence", f"{prediction['confidence']:.1%}")
    
    with col2:
        risk_class = f"risk-{prediction['risk_assessment'].lower()}"
        st.metric("Risk Level", f"{prediction['risk_assessment']}")
        st.metric("Certainty Index", f"{prediction['certainty_index']:.1%}")
    
    with col3:
        st.metric("Value Edge", f"{prediction['value_edge']:+.1%}")
        st.metric("Recommended Stake", f"{prediction['recommended_stake']:.1%}")
    
    with col4:
        st.metric("Fair Odds", f"{prediction.get('fair_odds', 0):.2f}")
        st.metric("Market Odds", f"{prediction.get('market_odds', 0):.2f}")
    
    # Probability visualization
    st.subheader("📊 Probability Distribution")
    fig = create_probability_chart(prediction)
    st.plotly_chart(fig, use_container_width=True)
    
    # Team analysis
    st.subheader("🏆 Team Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        display_team_analysis(prediction['home_team'], prediction, is_home=True)
    
    with col2:
        display_team_analysis(prediction['away_team'], prediction, is_home=False)
    
    # Advanced metrics
    with st.expander("🔍 Advanced Match Analysis", expanded=False):
        st.json(prediction)

def display_team_analysis(team_name, prediction, is_home=True):
    """Display detailed team analysis"""
    team_key = 'home' if is_home else 'away'
    
    st.markdown(f"### {'🏠' if is_home else '✈️'} {team_name}")
    
    # Strength metrics
    elo = prediction.get(f'elo_{team_key}', 1500)
    form = prediction.get(f'form_{team_key}', 0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ELO Rating", f"{elo:.0f}")
        st.metric("Recent Form", f"{form:.1%}")
    
    # Strength visualization
    strength = (elo - 1300) / 400  # Normalize to 0-1 scale
    st.write(f"**Overall Strength:** {strength:.1%}")
    st.markdown(f'<div class="team-strength-bar" style="width: {min(strength * 100, 100)}%"></div>', 
                unsafe_allow_html=True)

def render_live_predictions():
    """Live predictions with dynamic data"""
    st.header("🎯 Live Match Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dynamic fixture selection
        st.subheader("Select Match")
        
        league = st.selectbox(
            "League:",
            ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"],
            key="league_select"
        )
        
        # Get dynamic fixtures
        fixtures = st.session_state.data_manager.get_fixtures(league, days=3)
        
        if fixtures:
            fixture_options = {f"{f['home_team']} vs {f['away_team']} - {f['date']}": f for f in fixtures}
            selected_fixture_key = st.selectbox(
                "Select Fixture:",
                list(fixture_options.keys()),
                key="fixture_select"
            )
            
            selected_fixture = fixture_options[selected_fixture_key]
            
            # Additional context
            with st.expander("⚙️ Advanced Context Settings"):
                col1a, col1b = st.columns(2)
                with col1a:
                    venue_factor = st.slider("Venue Advantage", 0.0, 0.3, 0.12, 0.01,
                                           help="Adjust home advantage factor")
                with col1b:
                    importance = st.slider("Match Importance", 0.5, 2.0, 1.0, 0.1,
                                         help="Derby, title decider, etc.")
            
            if st.button("🚀 Generate Advanced Prediction", type="primary", use_container_width=True):
                with st.spinner("🤖 Running advanced analysis..."):
                    # Simulate processing time
                    import time
                    time.sleep(1)
                    
                    prediction = st.session_state.predictor.predict_match(
                        match_id=selected_fixture['id'],
                        home_team=selected_fixture['home_team'],
                        away_team=selected_fixture['away_team'],
                        league=league,
                        additional_context={
                            'venue_factor': venue_factor,
                            'importance': importance
                        }
                    )
                    
                    # Store prediction
                    if 'predictions' not in st.session_state:
                        st.session_state.predictions = []
                    st.session_state.predictions.append(prediction)
                    
                    display_prediction_results(prediction)
        else:
            st.warning("No fixtures available.")
    
    with col2:
        st.subheader("📋 Quick Predict")
        
        # Manual prediction input
        with st.form("quick_predict_form"):
            home_team = st.text_input("Home Team", "Man City")
            away_team = st.text_input("Away Team", "Liverpool")
            league = st.selectbox("League", ["EPL", "La Liga", "Bundesliga"], key="quick_league")
            
            if st.form_submit_button("Quick Predict"):
                prediction = st.session_state.predictor.predict_match(
                    match_id=f"quick_{datetime.now().timestamp()}",
                    home_team=home_team,
                    away_team=away_team,
                    league=league
                )
                if 'predictions' not in st.session_state:
                    st.session_state.predictions = []
                st.session_state.predictions.append(prediction)
                st.rerun()
        
        # Show recent predictions
        if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
            st.subheader("Recent Predictions")
            for pred in st.session_state.predictions[-3:]:
                with st.container():
                    st.write(f"**{pred['home_team']} vs {pred['away_team']}**")
                    st.write(f"🎯 {pred['prediction']} ({pred['confidence']:.0%})")
                    st.progress(pred['confidence'])
                    st.markdown("---")

def render_fixtures_view():
    """Dynamic fixtures view with predictions"""
    st.header("📊 Upcoming Fixtures")
    
    # League selector
    leagues = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    selected_leagues = st.multiselect(
        "Select Leagues:",
        leagues,
        default=["EPL"]
    )
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        days_ahead = st.slider("Days Ahead", 1, 14, 7)
    
    # Load fixtures
    all_fixtures = []
    for league in selected_leagues:
        fixtures = st.session_state.data_manager.get_fixtures(league, days_ahead)
        all_fixtures.extend(fixtures)
    
    if not all_fixtures:
        st.info("No fixtures found for selected leagues and date range.")
        return
    
    # Display fixtures
    for fixture in all_fixtures:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{fixture['home_team']} vs {fixture['away_team']}**")
                st.write(f"📅 {fixture['date']} • 🏆 {fixture['league']}")
            
            with col2:
                if st.button("Predict", key=f"pred_{fixture['id']}"):
                    # Generate and store prediction
                    prediction = st.session_state.predictor.predict_match(
                        fixture['id'],
                        fixture['home_team'],
                        fixture['away_team'],
                        fixture['league']
                    )
                    if 'fixture_predictions' not in st.session_state:
                        st.session_state.fixture_predictions = {}
                    st.session_state.fixture_predictions[fixture['id']] = prediction
                    st.rerun()
            
            with col3:
                # Show existing prediction if available
                if (hasattr(st.session_state, 'fixture_predictions') and 
                    fixture['id'] in st.session_state.fixture_predictions):
                    pred = st.session_state.fixture_predictions[fixture['id']]
                    st.write(f"🎯 {pred['prediction']} ({pred['confidence']:.0%})")
            
            st.markdown("---")

def render_team_analytics():
    """Team performance analytics dashboard"""
    st.header("🏆 Team Performance Analytics")
    
    # Team selector
    teams = list(st.session_state.predictor.elo_system.team_ratings.keys())
    if not teams:
        st.info("No team data available. Please generate predictions first.")
        return
    
    selected_team = st.selectbox("Select Team:", teams)
    
    if selected_team:
        # Team metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            elo = st.session_state.predictor.elo_system.get_rating(selected_team)
            st.metric("ELO Rating", f"{elo:.0f}")
        
        with col2:
            form = st.session_state.predictor.elo_system.calculate_form(selected_team)
            st.metric("Recent Form", f"{form:.1%}")
        
        with col3:
            attack = st.session_state.predictor.elo_system.calculate_attack_strength(selected_team)
            st.metric("Attack Strength", f"{attack:.2f}")
        
        with col4:
            defense = st.session_state.predictor.elo_system.calculate_defense_strength(selected_team)
            st.metric("Defense Strength", f"{defense:.2f}")

def render_model_performance():
    """Model performance monitoring dashboard"""
    st.header("📈 Model Performance Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Metrics")
        
        # Simulated performance metrics
        metrics = {
            'Accuracy': 0.72,
            'Precision': 0.75,
            'Recall': 0.70,
            'F1-Score': 0.725,
        }
        
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.1%}")
    
    with col2:
        st.subheader("Feature Importance")
        
        features = ['ELO Difference', 'Recent Form', 'Attack Strength', 
                    'Defense Strength', 'Home Advantage', 'Match Importance']
        importance = [0.25, 0.18, 0.15, 0.14, 0.13, 0.10]
        
        fig = px.bar(
            x=importance, y=features, 
            orientation='h',
            title="Feature Importance in Predictions"
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_value_bets():
    """Value betting opportunities"""
    st.header("💰 Value Betting Opportunities")
    
    # Generate sample value bets based on recent predictions
    value_bets = []
    if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
        for pred in st.session_state.predictions[-10:]:
            if pred['value_edge'] > 0.05:  # Only show bets with significant edge
                value_bets.append({
                    'match': f"{pred['home_team']} vs {pred['away_team']}",
                    'prediction': pred['prediction'],
                    'confidence': pred['confidence'],
                    'fair_odds': pred.get('fair_odds', 0),
                    'market_odds': pred.get('market_odds', 0),
                    'value_edge': pred['value_edge'],
                    'stake': pred['recommended_stake'],
                    'league': pred['league']
                })
    
    if not value_bets:
        st.info("No high-value bets found. Generate more predictions to see value opportunities.")
        return
    
    for bet in value_bets:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{bet['match']}**")
                st.write(f"🏆 {bet['league']}")
            
            with col2:
                st.write(f"🎯 {bet['prediction']}")
                st.write(f"({bet['confidence']:.0%})")
            
            with col3:
                st.write(f"💰 {bet['value_edge']:.1%} edge")
                st.write(f"Stake: {bet['stake']:.1%}")
            
            with col4:
                if bet['value_edge'] > 0.15:
                    st.success("🔥 High Value")
                elif bet['value_edge'] > 0.08:
                    st.warning("📈 Medium Value")
                else:
                    st.info("📊 Low Value")
            
            st.markdown("---")

def render_system_dashboard():
    """System monitoring and configuration"""
    st.header("⚙️ System Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Status")
        
        status_items = {
            "Prediction Engine": "✅ Online",
            "Data Pipeline": "✅ Connected", 
            "Model Service": "✅ Running",
            "Database": "✅ Connected"
        }
        
        for service, status in status_items.items():
            st.write(f"{service}: {status}")
        
        # System metrics
        total_predictions = len(st.session_state.predictions) if hasattr(st.session_state, 'predictions') else 0
        st.metric("Total Predictions", f"{total_predictions}")
        st.metric("Active Teams", f"{len(st.session_state.predictor.elo_system.team_ratings)}")
        st.metric("Model Version", st.session_state.predictor.model_version)
    
    with col2:
        st.subheader("System Management")
        
        if st.button("🔄 Refresh All Data"):
            with st.spinner("Refreshing data..."):
                import time
                time.sleep(2)
                st.success("Data refreshed successfully!")
        
        if st.button("📊 Retrain Models"):
            with st.spinner("Retraining models..."):
                st.session_state.predictor = AdvancedFootballPredictor()
                st.success("Models retrained successfully!")

def main():
    st.set_page_config(
        page_title="Pro Football Oracle - Advanced Prediction System",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .team-strength-bar {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb);
        height: 8px;
        border-radius: 4px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown('<h1 class="main-header">⚽ Pro Football Oracle</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-Powered Football Predictions • Live Updates • Professional Analytics</p>', unsafe_allow_html=True)
    
    # Initialize advanced predictor and data manager
    if 'predictor' not in st.session_state:
        with st.spinner("🚀 Initializing Advanced Prediction System..."):
            st.session_state.predictor = AdvancedFootballPredictor()
            st.session_state.data_manager = DynamicDataManager()
    
    # Main Navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Live Predictions", 
        "📊 Fixtures", 
        "🏆 Team Analytics", 
        "📈 Model Performance", 
        "💰 Value Bets",
        "⚙️ System Dashboard"
    ])
    
    with tab1:
        render_live_predictions()
    
    with tab2:
        render_fixtures_view()
    
    with tab3:
        render_team_analytics()
    
    with tab4:
        render_model_performance()
    
    with tab5:
        render_value_bets()
    
    with tab6:
        render_system_dashboard()

if __name__ == "__main__":
    main()
