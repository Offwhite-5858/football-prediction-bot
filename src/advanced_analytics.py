import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils.database import DatabaseManager

class AdvancedAnalytics:
    """Industrial-grade analytics and visualization system"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def get_performance_metrics(self, days=30):
        """Get comprehensive performance metrics"""
        conn = self.db._get_connection()
        
        # Overall accuracy
        accuracy_query = '''
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN pe.error_value < 0.3 THEN 1 ELSE 0 END) as correct_predictions,
                AVG(pe.error_value) as avg_error,
                MIN(pe.error_value) as min_error,
                MAX(pe.error_value) as max_error
            FROM prediction_errors pe
            WHERE pe.analyzed_at > datetime('now', ?)
        '''
        
        accuracy_data = conn.execute(accuracy_query, (f'-{days} days',)).fetchone()
        
        # Accuracy by league
        league_accuracy_query = '''
            SELECT 
                m.league,
                COUNT(*) as total,
                SUM(CASE WHEN pe.error_value < 0.3 THEN 1 ELSE 0 END) as correct,
                AVG(pe.error_value) as avg_error
            FROM prediction_errors pe
            JOIN matches m ON pe.prediction_id = m.match_id
            WHERE pe.analyzed_at > datetime('now', ?)
            GROUP BY m.league
        '''
        
        league_accuracy = pd.read_sql_query(
            league_accuracy_query, conn, params=(f'-{days} days',)
        )
        
        # Prediction type performance
        type_accuracy_query = '''
            SELECT 
                p.prediction_type,
                COUNT(*) as total,
                AVG(pe.error_value) as avg_error
            FROM prediction_errors pe
            JOIN predictions p ON pe.prediction_id = p.match_id
            WHERE pe.analyzed_at > datetime('now', ?)
            GROUP BY p.prediction_type
        '''
        
        type_accuracy = pd.read_sql_query(
            type_accuracy_query, conn, params=(f'-{days} days',)
        )
        
        conn.close()
        
        return {
            'overall': {
                'total_predictions': accuracy_data[0],
                'correct_predictions': accuracy_data[1],
                'accuracy_rate': accuracy_data[1] / accuracy_data[0] if accuracy_data[0] > 0 else 0,
                'avg_error': accuracy_data[2],
                'min_error': accuracy_data[3],
                'max_error': accuracy_data[4]
            },
            'by_league': league_accuracy.to_dict('records'),
            'by_type': type_accuracy.to_dict('records')
        }
    
    def create_performance_trend_chart(self, days=30):
        """Create performance trend over time"""
        conn = self.db._get_connection()
        
        trend_query = '''
            SELECT 
                DATE(pe.analyzed_at) as date,
                COUNT(*) as predictions,
                AVG(CASE WHEN pe.error_value < 0.3 THEN 1 ELSE 0 END) as daily_accuracy
            FROM prediction_errors pe
            WHERE pe.analyzed_at > datetime('now', ?)
            GROUP BY DATE(pe.analyzed_at)
            ORDER BY date
        '''
        
        trend_data = pd.read_sql_query(
            trend_query, conn, params=(f'-{days} days',)
        )
        
        conn.close()
        
        if len(trend_data) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trend_data['date'],
                y=trend_data['daily_accuracy'],
                mode='lines+markers',
                name='Daily Accuracy',
                line=dict(color='#00CC96', width=3)
            ))
            
            fig.update_layout(
                title='📈 Prediction Accuracy Trend',
                xaxis_title='Date',
                yaxis_title='Accuracy Rate',
                yaxis_tickformat='.0%',
                height=400,
                showlegend=True
            )
            
            return fig
        else:
            return self._create_empty_chart("No trend data available yet")
    
    def create_league_comparison_chart(self, days=30):
        """Create league performance comparison"""
        metrics = self.get_performance_metrics(days)
        league_data = metrics['by_league']
        
        if league_data:
            df = pd.DataFrame(league_data)
            df['accuracy_rate'] = df['correct'] / df['total']
            
            fig = px.bar(
                df, 
                x='league', 
                y='accuracy_rate',
                title='🏆 Accuracy by League',
                color='accuracy_rate',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                yaxis_tickformat='.0%',
                xaxis_tickangle=45,
                height=400
            )
            
            return fig
        else:
            return self._create_empty_chart("No league comparison data available")
    
    def create_prediction_type_chart(self, days=30):
        """Create prediction type performance chart"""
        metrics = self.get_performance_metrics(days)
        type_data = metrics['by_type']
        
        if type_data:
            df = pd.DataFrame(type_data)
            
            fig = px.pie(
                df,
                values='total',
                names='prediction_type',
                title='🎯 Prediction Type Distribution',
                hole=0.4
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            return fig
        else:
            return self._create_empty_chart("No prediction type data available")
    
    def create_error_distribution_chart(self, days=30):
        """Create error distribution analysis"""
        conn = self.db._get_connection()
        
        error_query = '''
            SELECT error_value
            FROM prediction_errors
            WHERE analyzed_at > datetime('now', ?)
        '''
        
        error_data = pd.read_sql_query(
            error_query, conn, params=(f'-{days} days',)
        )
        
        conn.close()
        
        if len(error_data) > 0:
            fig = px.histogram(
                error_data,
                x='error_value',
                nbins=20,
                title='📊 Prediction Error Distribution',
                labels={'error_value': 'Prediction Error'}
            )
            
            fig.update_layout(
                xaxis_title='Prediction Error',
                yaxis_title='Frequency',
                height=400
            )
            
            # Add mean line
            mean_error = error_data['error_value'].mean()
            fig.add_vline(
                x=mean_error, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f'Mean: {mean_error:.3f}'
            )
            
            return fig
        else:
            return self._create_empty_chart("No error distribution data available")
    
    def create_bias_analysis_chart(self, days=30):
        """Create bias analysis visualization"""
        conn = self.db._get_connection()
        
        bias_query = '''
            SELECT 
                m.home_team,
                m.away_team,
                p.prediction_data,
                pe.actual_result,
                pe.error_value
            FROM prediction_errors pe
            JOIN matches m ON pe.prediction_id = m.match_id
            JOIN predictions p ON pe.prediction_id = p.match_id
            WHERE pe.analyzed_at > datetime('now', ?)
        '''
        
        bias_data = pd.read_sql_query(
            bias_query, conn, params=(f'-{days} days',)
        )
        
        conn.close()
        
        if len(bias_data) > 0:
            # Analyze home vs away bias
            bias_data['prediction'] = bias_data['prediction_data'].apply(
                lambda x: self._extract_prediction_from_json(x)
            )
            
            bias_analysis = bias_data.groupby('prediction').agg({
                'error_value': 'mean',
                'home_team': 'count'
            }).reset_index()
            
            fig = px.bar(
                bias_analysis,
                x='prediction',
                y='error_value',
                title='⚖️ Prediction Bias Analysis',
                color='error_value',
                color_continuous_scale='RdBu_r'
            )
            
            fig.update_layout(
                yaxis_title='Average Error',
                xaxis_title='Prediction Type',
                height=400
            )
            
            return fig
        else:
            return self._create_empty_chart("No bias analysis data available")
    
    def create_confidence_calibration_chart(self, days=30):
        """Create confidence calibration analysis"""
        conn = self.db._get_connection()
        
        calibration_query = '''
            SELECT 
                p.prediction_data,
                pe.actual_result
            FROM prediction_errors pe
            JOIN predictions p ON pe.prediction_id = p.match_id
            WHERE pe.analyzed_at > datetime('now', ?)
        '''
        
        calibration_data = pd.read_sql_query(
            calibration_query, conn, params=(f'-{days} days',)
        )
        
        conn.close()
        
        if len(calibration_data) > 0:
            # Extract confidence scores and outcomes
            confidence_data = []
            for _, row in calibration_data.iterrows():
                pred_data = row['prediction_data']
                if isinstance(pred_data, str):
                    import json
                    pred_data = json.loads(pred_data)
                
                confidence = pred_data.get('predictions', {}).get('match_outcome', {}).get('confidence', 0.5)
                actual = row['actual_result']
                
                confidence_data.append({
                    'confidence': confidence,
                    'correct': self._is_prediction_correct(pred_data, actual)
                })
            
            df = pd.DataFrame(confidence_data)
            
            # Create calibration buckets
            df['confidence_bucket'] = pd.cut(df['confidence'], bins=10, labels=False)
            calibration = df.groupby('confidence_bucket').agg({
                'confidence': 'mean',
                'correct': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=calibration['confidence'],
                y=calibration['correct'],
                mode='markers+lines',
                name='Actual vs Predicted',
                line=dict(color='#636EFA', width=3)
            ))
            
            # Perfect calibration line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='🎯 Confidence Calibration',
                xaxis_title='Predicted Confidence',
                yaxis_title='Actual Accuracy',
                yaxis_tickformat='.0%',
                xaxis_tickformat='.0%',
                height=400
            )
            
            return fig
        else:
            return self._create_empty_chart("No calibration data available")
    
    def _extract_prediction_from_json(self, pred_data):
        """Extract prediction from JSON data"""
        if isinstance(pred_data, str):
            import json
            pred_data = json.loads(pred_data)
        
        return pred_data.get('predictions', {}).get('match_outcome', {}).get('prediction', 'UNKNOWN')
    
    def _is_prediction_correct(self, pred_data, actual_result):
        """Check if prediction was correct"""
        prediction = self._extract_prediction_from_json(pred_data)
        
        if actual_result == 'H' and prediction == 'HOME WIN':
            return True
        elif actual_result == 'A' and prediction == 'AWAY WIN':
            return True
        elif actual_result == 'D' and prediction == 'DRAW':
            return True
        else:
            return False
    
    def _create_empty_chart(self, message):
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    def get_analytics_summary(self):
        """Get comprehensive analytics summary"""
        metrics = self.get_performance_metrics(30)
        
        summary = {
            'performance_metrics': metrics['overall'],
            'top_performing_league': max(metrics['by_league'], key=lambda x: x.get('correct', 0) / x.get('total', 1)) if metrics['by_league'] else None,
            'worst_performing_league': min(metrics['by_league'], key=lambda x: x.get('correct', 0) / x.get('total', 1)) if metrics['by_league'] else None,
            'total_analysis_period': '30 days',
            'data_quality': 'Good' if metrics['overall']['total_predictions'] > 10 else 'Limited'
        }
        
        return summary