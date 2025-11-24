# utils/ui_enhancements.py
import streamlit as st
import base64
from datetime import datetime

class UIEnhancer:
    """Professional UI enhancements and mobile responsiveness"""
    
    def __init__(self):
        self.set_custom_theme()
    
    def set_custom_theme(self):
        """Set professional custom theme"""
        st.markdown("""
        <style>
            /* Main theme colors */
            :root {
                --primary: #1f77b4;
                --secondary: #ff7f0e;
                --success: #2ca02c;
                --warning: #ffbb78;
                --danger: #d62728;
                --dark: #2c3e50;
                --light: #ecf0f1;
            }
            
            /* Main container */
            .main {
                background-color: #f8f9fa;
            }
            
            /* Headers */
            h1, h2, h3 {
                color: var(--dark);
                font-weight: 600;
            }
            
            /* Cards */
            .prediction-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s ease;
            }
            
            .prediction-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            
            /* Metrics */
            .metric-card {
                background: white;
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid var(--primary);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin: 0.5rem 0;
            }
            
            /* Buttons */
            .stButton button {
                border-radius: 8px;
                font-weight: 500;
                transition: all 0.2s ease;
            }
            
            .stButton button:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: #e8f4f8;
                border-radius: 8px 8px 0px 0px;
                padding: 10px 16px;
                font-weight: 500;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: var(--primary);
                color: white;
            }
            
            /* Mobile responsiveness */
            @media (max-width: 768px) {
                .prediction-card {
                    padding: 1rem;
                    margin: 0.5rem 0;
                }
                
                .metric-card {
                    padding: 0.75rem;
                    margin: 0.25rem 0;
                }
                
                /* Stack columns on mobile */
                [data-testid="column"] {
                    width: 100% !important;
                }
            }
            
            /* Loading spinner */
            .stSpinner > div > div {
                border-color: var(--primary) transparent transparent transparent;
            }
            
            /* Success/Error messages */
            .stAlert {
                border-radius: 8px;
                border-left: 4px solid;
            }
            
            .stAlert [data-testid="stMarkdownContainer"] {
                font-weight: 500;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def create_hero_section(self):
        """Create professional hero section"""
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        '>
            <h1 style='color: white; margin: 0; font-size: 3rem;'>🤖 PRODUCTION FOOTBALL AI</h1>
            <p style='color: white; font-size: 1.2rem; margin: 1rem 0 0 0; opacity: 0.9;'>
                Industrial-Grade Predictions • Real-Time Learning • 24/7 Monitoring
            </p>
            <div style='margin-top: 1.5rem;'>
                <span style='background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;'>
                    🎯 70%+ Accuracy
                </span>
                <span style='background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;'>
                    🔄 Live Learning
                </span>
                <span style='background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;'>
                    📊 Advanced Analytics
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_status_indicator(self, system_status):
        """Create system status indicator"""
        status_config = {
            'HEALTHY': {'color': '#00CC96', 'icon': '🟢', 'text': 'All Systems Operational'},
            'WARNING': {'color': '#FFA726', 'icon': '🟡', 'text': 'Minor Issues Detected'},
            'CRITICAL': {'color': '#EF5350', 'icon': '🔴', 'text': 'Attention Required'}
        }
        
        status = status_config.get(system_status, status_config['HEALTHY'])
        
        st.markdown(f"""
        <div style='
            background: {status['color']}15;
            border: 1px solid {status['color']}30;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
        '>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{status['icon']}</div>
            <div style='font-weight: 600; color: {status['color']};'>{status['text']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_quick_stats(self, stats):
        """Create quick stats bar"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🤖 Predictions", f"{stats.get('total_predictions', 0):,}")
        
        with col2:
            st.metric("🎯 Accuracy", f"{stats.get('accuracy', 0):.1%}")
        
        with col3:
            st.metric("🔄 Models", stats.get('active_models', 4))
        
        with col4:
            st.metric("📈 Learning", f"{stats.get('learning_progress', 0):.0%}")
    
    def create_mobile_friendly_prediction(self, prediction):
        """Create mobile-friendly prediction display"""
        with st.container():
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style='color: white; margin: 0 0 0.5rem 0;'>{prediction['home_team']} vs {prediction['away_team']}</h3>
                <p style='color: white; margin: 0; opacity: 0.9;'>{prediction['league']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mobile-optimized columns
            col1, col2 = st.columns(2)
            
            with col1:
                main_pred = prediction['predictions']['match_outcome']
                st.metric("Prediction", main_pred['prediction'])
                st.metric("Confidence", f"{main_pred['confidence']:.1%}")
            
            with col2:
                dc_pred = prediction['predictions']['double_chance']
                st.metric("Safe Bet", dc_pred['recommendation'])
                
                quality = prediction['data_quality']
                st.metric("Data Quality", quality['level'])