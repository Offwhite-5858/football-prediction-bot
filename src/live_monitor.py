import time
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils.api_client import OptimizedAPIClient
from utils.database import DatabaseManager

class LiveMatchMonitor:
    """Real-time match monitoring for in-play predictions"""
    
    def __init__(self, prediction_orchestrator):
        self.predictor = prediction_orchestrator
        self.api = OptimizedAPIClient()
        self.db = DatabaseManager()
        self.monitored_matches = {}
        
    def start_monitoring(self, match_ids):
        """Start monitoring specific matches"""
        for match_id in match_ids:
            self.monitored_matches[match_id] = {
                'last_update': datetime.now(),
                'prediction_history': [],
                'events': []
            }
        
        print(f"🔴 Started monitoring {len(match_ids)} matches")
        
    def update_live_predictions(self):
        """Update predictions for monitored matches based on live events"""
        current_time = datetime.now()
        
        for match_id, match_data in self.monitored_matches.items():
            # Only update every 5 minutes to manage API calls
            if (current_time - match_data['last_update']).seconds < 300:
                continue
            
            try:
                # Get live match data
                live_data = self.api.get_match_live_data(match_id)
                if live_data:
                    # Update prediction based on live events
                    updated_prediction = self._update_prediction_with_live_data(
                        match_id, live_data
                    )
                    
                    # Store updated prediction
                    match_data['prediction_history'].append({
                        'timestamp': current_time,
                        'prediction': updated_prediction,
                        'match_state': live_data.get('match_state', {})
                    })
                    
                    match_data['last_update'] = current_time
                    
                    # Check for significant changes
                    if self._prediction_changed_significantly(match_data):
                        self._notify_prediction_update(match_id, updated_prediction)
                        
            except Exception as e:
                print(f"❌ Error updating match {match_id}: {e}")
    
    def _update_prediction_with_live_data(self, match_id, live_data):
        """Update prediction based on live match events"""
        # Get original prediction
        original_pred = self.db.get_prediction_by_match_id(match_id)
        if not original_pred:
            return None
        
        # Extract live features
        live_features = self._extract_live_features(live_data, original_pred)
        
        # Generate updated prediction (simplified - would use specialized live model)
        updated_pred = self._adjust_prediction_with_live_data(
            original_pred, live_features
        )
        
        return updated_pred
    
    def _extract_live_features(self, live_data, original_pred):
        """Extract features from live match data"""
        features = {}
        
        match_state = live_data.get('match_state', {})
        
        # Current score impact
        home_goals = match_state.get('home_goals', 0)
        away_goals = match_state.get('away_goals', 0)
        features['goal_difference'] = home_goals - away_goals
        
        # Match minute
        features['match_minute'] = match_state.get('minute', 0)
        
        # Recent events
        features['recent_goals'] = len([
            e for e in live_data.get('events', []) 
            if e['type'] == 'GOAL' and e['minute'] > features['match_minute'] - 10
        ])
        
        # Cards and substitutions
        features['home_red_cards'] = len([
            e for e in live_data.get('events', [])
            if e['type'] == 'RED_CARD' and e['team'] == 'home'
        ])
        features['away_red_cards'] = len([
            e for e in live_data.get('events', [])
            if e['type'] == 'RED_CARD' and e['team'] == 'away'
        ])
        
        # Momentum indicators
        features['home_shots_recent'] = len([
            e for e in live_data.get('events', [])
            if e['type'] == 'SHOT' and e['team'] == 'home' and e['minute'] > features['match_minute'] - 15
        ])
        
        return features
    
    def _adjust_prediction_with_live_data(self, original_pred, live_features):
        """Adjust prediction based on live match context"""
        original_probs = original_pred['predictions']['match_outcome']['probabilities']
        
        # Simplified adjustment logic
        goal_diff = live_features.get('goal_difference', 0)
        minute = live_features.get('match_minute', 0)
        
        # Adjust probabilities based on current score and time
        if minute > 0:
            time_factor = minute / 90.0  # Match progress
            
            if goal_diff > 0:  # Home team leading
                home_boost = goal_diff * 0.1 * time_factor
                original_probs['home'] = min(0.95, original_probs['home'] + home_boost)
                original_probs['away'] = max(0.05, original_probs['away'] - home_boost * 0.5)
                
            elif goal_diff < 0:  # Away team leading  
                away_boost = abs(goal_diff) * 0.1 * time_factor
                original_probs['away'] = min(0.95, original_probs['away'] + away_boost)
                original_probs['home'] = max(0.05, original_probs['home'] - away_boost * 0.5)
        
        # Red card impact
        home_red_cards = live_features.get('home_red_cards', 0)
        away_red_cards = live_features.get('away_red_cards', 0)
        
        if home_red_cards > 0:
            original_probs['home'] = max(0.05, original_probs['home'] * 0.7)
            original_probs['away'] = min(0.95, original_probs['away'] * 1.3)
        
        if away_red_cards > 0:
            original_probs['away'] = max(0.05, original_probs['away'] * 0.7)
            original_probs['home'] = min(0.95, original_probs['home'] * 1.3)
        
        # Normalize probabilities
        total = sum(original_probs.values())
        for key in original_probs:
            original_probs[key] /= total
        
        # Update prediction
        updated_pred = original_pred.copy()
        updated_pred['predictions']['match_outcome']['probabilities'] = original_probs
        updated_pred['predictions']['match_outcome']['prediction'] = self._get_prediction_from_probs(original_probs)
        updated_pred['live_features'] = live_features
        updated_pred['last_updated'] = datetime.now().isoformat()
        
        return updated_pred
    
    def _get_prediction_from_probs(self, probabilities):
        """Get prediction string from probabilities"""
        max_prob = max(probabilities.values())
        for outcome, prob in probabilities.items():
            if prob == max_prob:
                if outcome == 'home':
                    return "HOME WIN"
                elif outcome == 'away':
                    return "AWAY WIN"
                else:
                    return "DRAW"
        return "HOME WIN"  # Default
    
    def _prediction_changed_significantly(self, match_data):
        """Check if prediction changed significantly"""
        if len(match_data['prediction_history']) < 2:
            return False
        
        current_pred = match_data['prediction_history'][-1]
        previous_pred = match_data['prediction_history'][-2]
        
        current_probs = current_pred['prediction']['predictions']['match_outcome']['probabilities']
        previous_probs = previous_pred['prediction']['predictions']['match_outcome']['probabilities']
        
        # Check if any probability changed by more than 15%
        for outcome in ['home', 'draw', 'away']:
            if abs(current_probs[outcome] - previous_probs[outcome]) > 0.15:
                return True
        
        return False
    
    def _notify_prediction_update(self, match_id, updated_prediction):
        """Notify about significant prediction changes"""
        print(f"🔄 Prediction updated for match {match_id}")
        # In a full implementation, this would send notifications to users
        
    def stop_monitoring(self, match_id):
        """Stop monitoring a specific match"""
        if match_id in self.monitored_matches:
            del self.monitored_matches[match_id]
            print(f"🟢 Stopped monitoring match {match_id}")
    
    def get_monitoring_status(self):
        """Get status of monitored matches"""
        status = {}
        for match_id, data in self.monitored_matches.items():
            status[match_id] = {
                'last_update': data['last_update'],
                'prediction_count': len(data['prediction_history']),
                'event_count': len(data['events'])
            }
        return status