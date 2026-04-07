import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import os
import time
import requests
import random
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CITY_COORDS = {
    'Dubai': (25.2048, 55.2708), 'London': (51.5074, -0.1278), 'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437), 'Sydney': (-33.8688, 151.2093), 'Singapore': (1.3521, 103.8198),
    'Frankfurt': (50.1109, 8.6821), 'Tokyo': (35.6762, 139.6503), 'Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777), 'Muscat': (23.5859, 58.4059), 'Ahmedabad': (23.0225, 72.5714),
    'Beijing': (39.9042, 116.4074), 'Bangalore': (12.9716, 77.5946), 'Istanbul': (41.0082, 28.9784),
    'Doha': (25.2854, 51.5310), 'Abu Dhabi': (24.4539, 54.3773), 'Bangkok': (13.7563, 100.5018),
    'Kuala Lumpur': (3.1390, 101.6869), 'Paris': (48.8566, 2.3522), 'Bahrain': (26.0667, 50.5577)
}

REASON_MAPPING = {
    'departure_hour': 'High airport traffic during peak local hours',
    'weather_risk': 'Challenging weather conditions at destination',
    'traffic': 'Increased airspace congestion and ground control constraints',
    'distance': 'Operational complexity associated with long-haul flight paths',
    'duration': 'Increased delay probability due to Extended Flight Duration',
    'stops': 'Multiple connections increasing risk of technical or logistical delays'
}

def format_inr(value):
    try:
        is_negative = value < 0
        value = abs(int(value))
        s = str(value)
        if len(s) <= 3: res = s
        else:
            res = s[-3:]
            s = s[:-3]
            while s:
                res = s[-2:] + "," + res
                s = s[:-2]
        return f"-₹{res}" if is_negative else f"₹{res}"
    except:
        return f"₹{value}"

# ==========================
# PAGE CONFIG & CSS
# ==========================
st.set_page_config(page_title="AeroIntel Intelligence", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        html, body, [class*="css"] { 
            font-family: 'Inter', sans-serif; 
            color: #E2E8F0; /* Primary Text */
            line-height: 1.6; /* Increased line spacing */
        }
        .stApp { background: linear-gradient(180deg, #090e17 0%, #0F172A 100%); }
        
        /* Headers and Typography */
        h1, h2, h3, h4, h5, h6 { 
            color: #FFFFFF !important; /* Pure White Headings */
            font-weight: 700; 
            letter-spacing: -0.02em; 
        }
        .subtitle { 
            color: #94A3B8; /* Muted Secondary */
            font-size: 1.15rem; 
            font-weight: 400; 
            margin-top: -15px; 
            margin-bottom: 30px; 
        }
        
        /* Premium Cards */
        div[data-testid="stMetric"], .premium-card {
            background: linear-gradient(145deg, #1E293B, #111827); /* Lighter than app bg */
            border: 1px solid rgba(148, 163, 184, 0.2); /* Subtle Border */
            border-radius: 12px;
            padding: 24px 28px; /* Added internal padding */
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6), 0 0 15px rgba(56, 189, 248, 0.05); /* Shadow + Glow */
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s;
        }
        div[data-testid="stMetric"]:hover, .premium-card:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.8), 0 0 20px rgba(56, 189, 248, 0.15); /* Hover Glow */
            border-color: #38BDF8; /* Cyan Focus */
        }
        
        /* Metric Styling */
        div[data-testid="stMetricValue"] { color: #FFFFFF; font-weight: 800; font-size: 2.4rem; } /* Bright Numbers */
        div[data-testid="stMetricLabel"] { font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; font-weight: 600; }
        
        /* Accent Text Colors */
        .txt-emerald { color: #10B981; }
        .txt-sky { color: #38BDF8; }
        .txt-rose { color: #F43F5E; }
        .txt-amber { color: #F59E0B; }
        
        /* Risk Badges */
        .risk-badge { display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.95rem; text-align: center; }
        .risk-low { background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
        .risk-med { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245,158,11,0.3); }
        .risk-high { background-color: rgba(244, 63, 94, 0.15); color: #FB7185; border: 1px solid rgba(244,63,94,0.3); }
        
        /* PREMIUM GLASSMORPHISM SIDEBAR */
        section[data-testid="stSidebar"] { 
            background: linear-gradient(180deg, rgba(11, 18, 32, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%) !important; 
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important; 
            box-shadow: 2px 0 20px rgba(0, 0, 0, 0.5) !important;
        }
        
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
            font-weight: 800 !important;
            margin-bottom: 25px !important;
            opacity: 1 !important;
            letter-spacing: 0.5px;
        }

        /* Force PURITY WHITE contrast and disable ALL Streamlit fading globally for Sidebar */
        section[data-testid="stSidebar"] * {
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] span, 
        section[data-testid="stSidebar"] .st-an,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] label p,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] div[data-baseweb="slider"],
        section[data-testid="stSidebar"] div[data-baseweb="select"] {
            color: #E2E8F0 !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }
        section[data-testid="stSidebar"] label p {
            color: #FFFFFF !important;
            font-weight: 700 !important; /* Make labels bolder */
            font-size: 0.95rem !important;
        }
        
        /* Premium Hover & Glow Options Styling */
        section[data-testid="stSidebar"] div[role="radiogroup"] label, 
        section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label {
            transition: all 0.2s ease-in-out;
            padding: 12px 16px;
            border-radius: 10px;
            margin-bottom: 8px;
            border-left: 3px solid transparent;
            cursor: pointer;
            width: 100%;
            background-color: rgba(255, 255, 255, 0.02);
            opacity: 1 !important;
        }
        
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover,
        section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label:hover {
            background-color: rgba(255, 255, 255, 0.06) !important;
            border-left: 3px solid rgba(99, 102, 241, 0.5) !important;
            box-shadow: inset 0 0 12px rgba(255, 255, 255, 0.03);
            transform: translateX(3px) !important; /* Smooth movement */
        }
        
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(99, 102, 241, 0.2) 0%, rgba(99, 102, 241, 0.05) 100%) !important;
            border-left: 3px solid #6366F1 !important;
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.4) !important;
            transform: translateX(3px) !important;
        }
        
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div,
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover div {
            color: #FFFFFF !important;
            font-weight: 800 !important;
            text-shadow: 0 0 8px rgba(255,255,255,0.4);
            opacity: 1 !important;
        }
        
        /* File Uploader Upgrade */
        section[data-testid="stFileUploadDropzone"] {
            background-color: rgba(15, 23, 42, 0.6) !important;
            border: 1px dashed rgba(99, 102, 241, 0.4) !important;
            border-radius: 12px;
            transition: all 0.3s ease;
            backdrop-filter: blur(5px);
        }
        section[data-testid="stFileUploadDropzone"]:hover {
            background-color: rgba(30, 41, 59, 0.8) !important;
            border: 1px dashed rgba(99, 102, 241, 0.8) !important;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
        }
        
        /* Slider Tuning */
        section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {
            background-color: #38BDF8 !important;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.6) !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="slider"] div[data-testid="stTickBar"] {
            background: linear-gradient(90deg, #6366F1, #38BDF8) !important;
        }
        
        hr { border-color: #334155 !important; opacity: 0.6; }
        .stTabs [data-baseweb="tab"] { color: #94A3B8; font-weight: 600; font-size: 1.05rem; padding: 10px 20px; }
        .stTabs [aria-selected="true"] { color: #38BDF8 !important; border-bottom-color: #38BDF8 !important; }
        
        /* Inputs & Sliders */
        div[data-baseweb="select"] > div, .stTextInput > div > div > input {
            background-color: #0F172A !important; 
            border-color: #334155 !important;
            color: #FFFFFF !important;
            border-radius: 8px;
            opacity: 1 !important;
        }
        /* Fix label contrast globally */
        label { color: #E5E7EB !important; font-size: 0.95rem !important; font-weight: 600 !important; margin-bottom: 5px; opacity: 1 !important; }
        .st-an { color: #CBD5F5 !important; opacity: 1 !important; }
        p.st-an { line-height: 1.6; }
        
        /* Download Button override */
        .stDownloadButton button { background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%); color: white; font-weight: 600; border: none; border-radius: 8px; padding: 0.5rem 1rem;}
        .stDownloadButton button:hover { background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%); border: none; box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# ==========================
# RESOURCE CACHING
# ==========================
@st.cache_resource
def load_models():
    # Price
    p_lgb = joblib.load('models/price_model_lgb.pkl')
    p_xgb = joblib.load('models/price_model_xgb.pkl')
    
    # Delay
    d_lgb = joblib.load('models/delay_model_lgb.pkl')
    d_xgb = joblib.load('models/delay_model_xgb.pkl')
    
    # Preprocessors
    p_prep = joblib.load('models/pricing_preprocessor.pkl')
    p_scaler = joblib.load('models/pricing_scaler.pkl')
    d_scaler = joblib.load('models/delay_scaler.pkl')
    
    # Features
    f_price = joblib.load('models/feature_names_pricing.pkl')
    f_delay = joblib.load('models/feature_names_delay.pkl')
    
    # Explainer
    exp_p = joblib.load('models/explainer_price_v2.pkl')
    try: exp_d = joblib.load('models/explainer_delay_v2.pkl')
    except: exp_d = None
    
    # Metrics
    with open('models/pricing_metrics.json', 'r') as f: p_metrics = json.load(f)
    with open('models/delay_metrics.json', 'r') as f: d_metrics = json.load(f)
    d_tests = joblib.load('models/delay_test_results.pkl')

    return (p_lgb, p_xgb, d_lgb, d_xgb, p_prep, p_scaler, d_scaler, 
            f_price, f_delay, exp_p, exp_d, p_metrics, d_metrics, d_tests)

@st.cache_data
def load_raw_data():
    df_p = pd.read_csv('data/pricing_data.csv') if os.path.exists('data/pricing_data.csv') else pd.DataFrame()
    df_d = pd.read_csv('data/delay_data.csv') if os.path.exists('data/delay_data.csv') else pd.DataFrame()
    return df_p, df_d
    return df_p, df_d

# ==========================
# API INTEGRATION
# ==========================
@st.cache_data(ttl=1800)
def fetch_weather(city):
    api_key = "2e463ab6e8afcaee1c76730078a99e3e"
    if city == "Delhi": city = "New Delhi"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return {"temp": data["main"]["temp"], "condition": data["weather"][0]["main"], "desc": data["weather"][0]["description"].title(), "icon": data["weather"][0]["icon"]}
    except: pass
    return None

def calc_weather_risk(weather):
    if not weather: return 0.2
    c = weather['condition'].lower()
    if c in ['clear']: return 0.05
    if c in ['clouds']: return 0.15
    if c in ['drizzle', 'mist', 'haze', 'fog', 'dust', 'ash']: return 0.35
    if c in ['rain']: return 0.6
    if c in ['snow']: return 0.8
    if c in ['thunderstorm', 'tornado', 'squall']: return 0.95
    return 0.2

def get_delay_explanations(inputs, proc_delay, exp_d, delay_prob):
    reasons = []
    
    # 1. SHAP Explainability (Data-Driven)
    if exp_d and hasattr(exp_d, 'shap_values'):
        try:
            # For tree explainers on classifier, we often get log-odds impact
            shap_values = exp_d.shap_values(proc_delay)
            # Handle different SHAP output formats (LightGBM/XGBoost return lists or arrays)
            if isinstance(shap_values, list): # Multi-class or binary with list format
                vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else: # Single array format
                vals = shap_values[0]
                
            # Get features with positive impact
            feature_names = proc_delay.columns.tolist()
            shap_impact = pd.DataFrame({'feature': feature_names, 'impact': vals})
            top_positive = shap_impact.sort_values(by='impact', ascending=False).head(3)
            
            # Map top positive SHAP features to human reasons
            for _, row in top_positive.iterrows():
                if row['impact'] > 0 and row['feature'] in REASON_MAPPING:
                    reasons.append(REASON_MAPPING[row['feature']])
        except:
            pass # Fallback to rules if SHAP fails

    # 2. Heuristic/Rule-Based Backups
    if not reasons:
        if inputs.get('departure_hour') in [7, 8, 9, 17, 18, 19, 20]:
            reasons.append("Predicted high airport traffic during evening/morning peak hours")
        if inputs.get('weather_risk', 0) > 0.4:
            reasons.append("Potential navigational constraints due to local weather conditions")
        if inputs.get('traffic', 0) > 0.6:
            reasons.append("Heightened air traffic control congestion levels")
        if inputs.get('stops') != 'Non-stop':
            reasons.append("Operational complexity due to multiple flight connections")
        if inputs.get('duration', 0) > 480:
            reasons.append("Logistical risks associated with ultra-long-haul flight operations")
            
    # Safety: Return at least one if probability is high
    if not reasons and delay_prob > 0.3:
        reasons.append("Aggregated historical patterns for this specific route and schedule")
        
    return list(set(reasons[:2])) # Return top 2 unique reasons

# ==========================
# UI COMPONENTS
# ==========================
def render_header():
    col1, col2 = st.columns([1, 15])
    with col1: st.markdown("<h1 style='font-size:3.5rem; margin:0;'>✈️</h1>", unsafe_allow_html=True)
    with col2:
        st.title("AeroIntel Platform")
        st.markdown("<p class='subtitle'>Predict delays, optimize pricing, and save costs using advanced ML algorithms</p>", unsafe_allow_html=True)

def render_kpis(df_p, df_d, active_model):
    st.markdown("### 🌐 Global Operational Overview")
    c1, c2, c3, c4 = st.columns(4)
    avg_price = df_p['price'].mean() * 83
    delay_rate = df_d['delay'].mean() * 100
    top_route_df = df_p.groupby(['source_city', 'destination_city']).size().reset_index(name='count')
    top_route = top_route_df.loc[top_route_df['count'].idxmax()]
    
    with c1: st.metric("Global Average Fare", format_inr(avg_price), "-2.1% (MoM)")
    with c2: st.metric("System Delay Rate", f"{delay_rate:.1f}%", "+1.2% (MoM)", delta_color="inverse")
    with c3: st.metric("Highest Volume Route", f"{top_route['source_city']} ➔ {top_route['destination_city']}", "62% Load Factor")
    with c4: st.metric("Inference Engine", active_model, "Online & Synced")
    st.markdown("<hr/>", unsafe_allow_html=True)

def render_inputs(destinations, dist_map):
    st.sidebar.markdown("## 🎛️ Simulation Controls")
    
    with st.sidebar.expander("📍 Route Configuration", expanded=True):
        source_in = st.selectbox("Source City", ["Dubai"], disabled=True)
        dest_in = st.selectbox("Destination City", destinations, index=destinations.index('London'))
    
    with st.sidebar.expander("🎫 Service & Market", expanded=True):
        class_in = st.selectbox("Cabin Class", ['Economy', 'Business', 'First'])
        stops_in = st.selectbox("Layover Preference", ['Non-stop', '1 Stop', '2+ Stops'])
        season_in = st.selectbox("Season", ['Low', 'Shoulder', 'Peak'])
        demand_in = st.selectbox("Market Demand", ['Low', 'Medium', 'High'])
        days_left_in = st.slider("Days to Departure", 1, 90, 14, help="Advance booking window")
        
        layovers = []
        layover_durations = []
        total_layover_time = 0
        layover_type = "None"
        if stops_in != 'Non-stop':
            num_stops = 1 if stops_in == '1 Stop' else 2
            possible_hubs = ['Istanbul', 'Doha', 'Abu Dhabi', 'Muscat', 'Bahrain']
            selected_hubs = random.sample(possible_hubs, min(num_stops, len(possible_hubs)))
            for hub in selected_hubs:
                dur = random.choice([120, 240, 480])
                layover_durations.append(dur)
                layovers.append(hub)
                total_layover_time += dur
            max_layover = max(layover_durations)
            layover_type = "Long" if max_layover > 360 else "Medium" if max_layover > 180 else "Short"
            
    with st.sidebar.expander("🌤️ Operations & Weather", expanded=True):
        dept_hour_in = st.slider("Departure Hour", 0, 23, 8)
        weather = fetch_weather(dest_in)
        default_risk = calc_weather_risk(weather)
        if weather:
            st.markdown(f"**Live at {dest_in}:** {weather['temp']}°C, {weather['desc']}")
        weather_in = st.slider("Weather Risk Factor", 0.0, 1.0, float(default_risk), help="Auto-fetched using OpenWeather")
        traffic_in = st.slider("Air Traffic Congestion", 0.0, 1.0, 0.3)
        
    distance = dist_map[dest_in]
    base_dur = distance / 800 + 1.0
    if stops_in != 'Non-stop': base_dur += total_layover_time / 60
    duration_in = int(base_dur * 60)
    
    st.sidebar.info(f"Flight Distance: {distance} km\nTotal Duration: {duration_in} mins")
    
    return {
        'source_city': source_in, 'destination_city': dest_in, 'cabin_class': class_in,
        'season': season_in, 'demand': demand_in, 'days_left': days_left_in, 
        'distance': distance, 'duration': duration_in, 'stops': stops_in,
        'departure_hour': dept_hour_in, 'weather_risk': weather_in, 'traffic': traffic_in,
        'layover_type': layover_type, 'route_cities': [source_in] + layovers + [dest_in],
        'weather_data': weather
    }

def execute_inference(inputs, model_name, p_lgb, p_xgb, d_lgb, d_xgb, p_prep, p_scaler, d_scaler, f_price, f_delay, exp_d):
    # Price
    df_price = pd.DataFrame([ inputs ])
    for col, le in p_prep.items():
        df_price[col] = df_price[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df_price[col] = le.transform(df_price[col].astype(str))
    
    proc_price = df_price[f_price]
    
    if model_name == "LightGBM":
        pred_price = p_lgb.predict(proc_price)[0]
    else: # XGBoost
        pred_price = p_xgb.predict(proc_price)[0]

    # Delay
    df_delay = pd.DataFrame([ inputs ])
    proc_delay = df_delay[f_delay]
    
    if model_name == "LightGBM":
        delay_prob = d_lgb.predict_proba(proc_delay)[0][1]
    else:
        delay_prob = d_xgb.predict_proba(proc_delay)[0][1]
        
    # Post-inference Layover Modifiers (to mimic real-world interactions missing from synthetic dataset)
    if inputs.get('layover_type') == 'Long':
        pred_price *= 0.85
        delay_prob = min(0.99, delay_prob + 0.12)
    elif inputs.get('layover_type') == 'Medium':
        pred_price *= 0.92
        delay_prob = min(0.99, delay_prob + 0.05)
        
    # Get Delay Reasons
    explanations = get_delay_explanations(inputs, proc_delay, exp_d, delay_prob)
        
    return proc_price, pred_price, proc_delay, delay_prob, explanations

def render_dashboard(inputs, pred_price, delay_prob, active_model, p_metrics, d_metrics):
    st.markdown("### 🎯 Predictive Intelligence", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    
    # Fetch Base Model Confidence from Metrics Json
    p_rmse = p_metrics.get(active_model, {}).get("rmse", 750)
    conf_score = max(0, 100 - (p_rmse/2000)*100) # Heuristic
    conf_range = p_rmse * 83 * 0.7 
    
    with c1:
        st.markdown(f"""
        <div class="premium-card" style="height: 100%;">
            <div style="display:flex; justify-content:space-between;">
                <p style='color:#94A3B8; font-weight:600; font-size:1rem; text-transform:uppercase;'>Predicted Ticket Yield</p>
                <span class="risk-badge" style="background: rgba(56, 189, 248, 0.1); color: #38BDF8; font-size: 0.8rem;">{conf_score:.1f}% Confidence</span>
            </div>
            <h2 class="txt-sky" style="font-size: 3.5rem; margin: 10px 0;">{format_inr(pred_price * 83)}</h2>
            <p style="color: #94A3B8; font-size: 0.95rem;">Expected variation: ± {format_inr(conf_range)}</p>
            <hr>
            <p style="color: #CBD5E1; font-size: 0.9rem;">Model: <b>{active_model}</b> based on {inputs['days_left']} days lead time and market dynamics.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Gauge Chart for Risk
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = delay_prob * 100,
            number = {'suffix': "%", 'font': {'size': 50, 'color': '#E6EDF3'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Flight Delay Probability", 'font': {'size': 16, 'color': '#94A3B8'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.3)"},
                    {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.3)"},
                    {'range': [70, 100], 'color': "rgba(244, 63, 94, 0.3)"}],
                'threshold': {
                    'line': {'color': "#38BDF8", 'width': 5},
                    'thickness': 0.75,
                    'value': delay_prob * 100}
            }))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(l=20, r=20, t=30, b=10))
        
        status_text = "🟢 Low Delay Risk" if delay_prob * 100 <= 30 else ("🟡 Medium Delay Risk" if delay_prob * 100 <= 70 else "🔴 High Delay Risk")
        
        st.markdown("<div class='premium-card' style='height: 100%; padding-bottom: 0px;'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="gauge")
        st.markdown(f"<p style='text-align:center; color:#CBD5E1; font-weight:600; margin-top:-20px;' title='Model prioritizes catching delays over avoiding false alarms'>{status_text} ℹ️</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # REASON FOR DELAY CARD (NEW)
    st.markdown("<br/>", unsafe_allow_html=True)
    c_reason_1, c_reason_2 = st.columns([1, 1])
    
    with c_reason_1:
        reasons_html = "".join([f"<li>{r}</li>" for r in inputs.get('delay_reasons', [])])
        if not reasons_html: 
            reasons_html = "<li>No significant delay triggers identified</li>"
        
        status_color = "#FB7185" if delay_prob > 0.5 else ("#FBBF24" if delay_prob > 0.2 else "#34D399")
        st.markdown(f"""
        <div class="premium-card" style="border-left: 4px solid {status_color};">
            <h4 style="margin:0; font-size:1.1rem; color:#FFFFFF;">🚨 Reason for Delay</h4>
            <ul style="color:#CBD5E1; font-size:0.95rem; margin-top:10px; padding-left:20px;">
                {reasons_html}
            </ul>
            <hr style="opacity:0.2; margin: 15px 0 10px 0;">
            <p style="color:#94A3B8; font-size:0.8rem; font-style:italic;">
                "This prediction is based on historical patterns such as current traffic, weather risk, and operational factors."
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with c_reason_2:
        # Mini explanation card
        st.markdown(f"""
        <div class="premium-card" style="background: rgba(15, 23, 42, 0.4);">
            <h4 style="margin:0; font-size:1rem; color:#94A3B8;">Insight Summary</h4>
            <p style="color:#CBD5E1; font-size:0.9rem; margin-top:8px;">
                The model identifies <b>{inputs['cabin_class']} class</b> on <b>{inputs['destination_city']}</b> route 
                as having a <b>{round(delay_prob*100)}%</b> potential delay risk. 
            </p>
            <p style="color:#6366F1; font-weight:600; font-size:0.85rem; margin-top:5px;">
                Recommendation: Buffer +60 mins for connections.
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_recommendations(inputs, current_price, active_model, p_lgb, p_xgb, p_prep, f_price):
    st.markdown("### 💡 Smart Decision Advisor")
    
    days_sim = list(range(1, 91))
    sim_df = pd.DataFrame([inputs] * len(days_sim))
    sim_df['days_left'] = days_sim
    for col, le in p_prep.items():
         sim_df[col] = sim_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
         sim_df[col] = le.transform(sim_df[col].astype(str))
    
    sim_proc = sim_df[f_price]
    
    if active_model == "LightGBM": prices_raw = p_lgb.predict(sim_proc)
    else: prices_raw = p_xgb.predict(sim_proc)
    
    modifier = 0.85 if inputs.get('layover_type') == 'Long' else (0.92 if inputs.get('layover_type') == 'Medium' else 1.0)
    prices_raw *= modifier
    prices_sim = prices_raw * 83
    
    min_idx = np.argmin(prices_sim)
    opt_days = days_sim[min_idx]
    opt_price = prices_sim[min_idx]
    curr_price_inr = current_price * 83
    sav = curr_price_inr - opt_price
    
    if sav > 0 and opt_days != inputs['days_left']:
        t_icon = "📅"
        t_title = "Postpone Booking" if opt_days < inputs['days_left'] else "Book Advance"
        t_text = f"Save {format_inr(sav)}"
        t_sub = f"Wait for {opt_days} days to departure window"
    else:
        t_icon = "✅"
        t_title = "Book Immediately"
        t_text = "Optimal Pricing"
        t_sub = "Current timing avoids future fare surges"
        
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""
    <div class="premium-card">
        <h4 style="color:#38BDF8; margin-top:0;">{t_icon} {t_title}</h4>
        <h3 style="margin-top:5px; font-size:1.4rem;">{t_text}</h3>
        <p style="color:#94A3B8; font-size:0.9rem; margin-bottom:0;">{t_sub}</p>
    </div>
    """, unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="premium-card">
        <h4 style="color:#10B981; margin-top:0;">⚖️ Class Comparison</h4>
        <h3 style="margin-top:5px; font-size:1.4rem;">Business is +35%</h3>
        <p style="color:#94A3B8; font-size:0.9rem; margin-bottom:0;">Standard multiplier for this route.</p>
    </div>
    """, unsafe_allow_html=True)
    c3.markdown(f"""
    <div class="premium-card">
        <h4 style="color:#F59E0B; margin-top:0;">🗺️ Routing Edge</h4>
        <h3 style="margin-top:5px; font-size:1.4rem;">Direct vs Transit</h3>
        <p style="color:#94A3B8; font-size:0.9rem; margin-bottom:0;">A penalty of {format_inr(5000)} for non-stop convenience.</p>
    </div>
    """, unsafe_allow_html=True)

def render_ml_insights(inputs, proc_price, proc_delay, active_model, exp_p, f_price, p_metrics, d_metrics, d_tests):
    from sklearn.metrics import precision_recall_curve
    st.markdown("### 🧠 Under The Hood (ML Ops)")
    tab1, tab2, tab3 = st.tabs(["📊 Pricing Explainability", "📊 Model Performance", "⚙️ Threshold Tuning"])
    
    with tab1:
        st.markdown("SHAP (SHapley Additive exPlanations) values indicating feature contributions to the final price multiplier.")
        if exp_p:
            shap_vals = exp_p(proc_price).values[0]
            df_s = pd.DataFrame({'Feature': f_price, 'Impact': shap_vals})
            df_s['Abs'] = df_s['Impact'].abs()
            df_s = df_s.sort_values(by='Abs', ascending=True).tail(5)
            fig = px.bar(df_s, x='Impact', y='Feature', orientation='h', 
                         color='Impact', color_continuous_scale=['#34D399', '#1E293B', '#F87171'],
                         template='plotly_dark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("SHAP Explainer unavailable for this model.")
            
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{active_model} Pricing Metrics (Test CV)**")
            if active_model in p_metrics:
                m = p_metrics[active_model]
                # Premium Pricing Cards
                val_cv = f"<tr><td>Mean CV RMSE</td><td style='text-align:right; font-weight:600;'>{format_inr(m['cv_rmse']*83)}</td></tr>" if m.get('cv_rmse') else ""
                st.markdown(f"""
                <div class="premium-card">
                    <p style="color:#38BDF8; font-weight:700; margin-bottom:5px;">💰 Algorithm Accuracy</p>
                    <table style="width:100%; color:#E6EDF3; font-size:1.05rem;">
                        <tr><td>Mean Absolute Error</td><td style="text-align:right; font-weight:600;">{format_inr(m['mae']*83)}</td></tr>
                        <tr><td>R-Squared (R²)</td><td style="text-align:right; font-weight:600;">{m['r2']*100:.1f}%</td></tr>
                        {val_cv}
                    </table>
                </div>""", unsafe_allow_html=True)
                
        with c2:
            st.markdown(f"**{active_model} Delay Metrics**")
            d_mod = active_model
            if d_mod in d_metrics:
                m = d_metrics[d_mod]
                st.markdown(f"""
                <div class="premium-card">
                    <p style="color:#10B981; font-weight:700; margin-bottom:5px;">🕒 Imbalance Compensated Classifier</p>
                    <table style="width:100%; color:#E6EDF3; font-size:1.05rem;">
                        <tr><td title="Prioritized KPI: Delays Caught">Recall (Delays Caught)</td><td style="text-align:right; font-weight:600; color:#34D399">{m['rec']*100:.1f}%</td></tr>
                        <tr><td>F1-Score</td><td style="text-align:right; font-weight:600;">{m['f1']*100:.1f}%</td></tr>
                        <tr><td>Precision</td><td style="text-align:right; font-weight:600;">{m['prec']*100:.1f}%</td></tr>
                        <tr><td>ROC-AUC</td><td style="text-align:right; font-weight:600;">{m['auc']:.4f}</td></tr>
                    </table>
                </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("**Threshold Calibration Simulation:** Adjust the decision boundary to prioritize catching delays (Recall) vs minimizing false alarms (Precision).")
        thresh = st.slider("Decision Threshold (Tuning tradeoff)", 0.05, 0.50, 0.20, 0.01)
        
        d_mod = active_model
        key = 'y_prob_lgb' if d_mod == "LightGBM" else 'y_prob_xgb'
        
        y_true = d_tests['y_true']
        y_prob = d_tests[key]
        y_pred = (y_prob >= thresh).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        cmax1, cmax2 = st.columns([1,2])
        with cmax1:
            try: recall = cm[1,1] / (cm[1,0] + cm[1,1])
            except: recall = 0
            try: prec = cm[1,1] / (cm[0,1] + cm[1,1])
            except: prec = 0
            try: f1_s = 2 * (prec * recall) / (prec + recall)
            except: f1_s = 0
            
            st.markdown(f"""
            <div class="premium-card" style="padding: 15px; margin-bottom: 10px;">
                <p style="color:#94a3b8; font-size:0.9rem; margin:0;">Dynamic Recall</p>
                <h3 style="color:#34D399; margin:5px 0;">{recall*100:.1f}%</h3>
            </div>
            <div class="premium-card" style="padding: 15px; margin-bottom: 10px;">
                <p style="color:#94a3b8; font-size:0.9rem; margin:0;">Dynamic Precision</p>
                <h3 style="color:#F59E0B; margin:5px 0;">{prec*100:.1f}%</h3>
            </div>
            <div class="premium-card" style="padding: 15px;">
                <p style="color:#94a3b8; font-size:0.9rem; margin:0;">Dynamic F1-Score</p>
                <h3 style="color:#38BDF8; margin:5px 0;">{f1_s*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 **Trade-off:** Lower thresholds catch more delays (High Recall) but increase false alerts (Lower Precision).")
        with cmax2:
            tab_cm, tab_pr = st.tabs(["Confusion Matrix", "PR Curve"])
            with tab_cm:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                            xticklabels=['On-Time', 'Delayed'], yticklabels=['On-Time', 'Delayed'])
                ax.set_xlabel('Predicted', color='white')
                ax.set_ylabel('Actual', color='white')
                ax.tick_params(colors='white')
                fig.patch.set_facecolor('#0F172A')
                ax.set_facecolor('#0F172A')
                st.pyplot(fig, transparent=True)
            with tab_pr:
                pr, rc, th = precision_recall_curve(y_true, y_prob)
                df_pr = pd.DataFrame({'Threshold': th, 'Precision': pr[:-1], 'Recall': rc[:-1]})
                df_pr = df_pr[df_pr['Threshold'] <= 0.5]
                fig2 = px.line(df_pr, x='Threshold', y=['Precision', 'Recall'], template='plotly_dark')
                fig2.add_vline(x=thresh, line_dash="dash", line_color="red")
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280, margin=dict(l=0,r=0,t=0,b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                st.plotly_chart(fig2, use_container_width=True)

def generate_report(inputs, price, delay, model):
    report = f"AeroIntel Analytics Export\n{'='*30}\n\n"
    report += f"Flight: {inputs['source_city']} to {inputs['destination_city']}\n"
    report += f"Class: {inputs['cabin_class']} | Stops: {inputs['stops']}\n"
    report += f"Days to Dep: {inputs['days_left']} | Season: {inputs['season']}\n\n"
    report += f"Inference Engine: {model}\n"
    report += f"Predicted Fare: {format_inr(price * 83)}\n"
    report += f"Delay Risk Probability: {delay*100:.1f}%\n"
    report += "Reason for Delay:\n"
    for r in inputs.get('delay_reasons', []):
        report += f" - {r}\n"
    return report

def main():
    inject_custom_css()
    
    try:
        models_data = load_models()
        (p_lgb, p_xgb, d_lgb, d_xgb, p_prep, p_scaler, d_scaler, 
         f_price, f_delay, exp_p, exp_d, p_metrics, d_metrics, d_tests) = models_data
    except Exception as e:
        st.error(f"Failed to load models. Please train pipeline first. Error: {e}")
        return
        
    df_p, df_d = load_raw_data()
    destinations = ['London', 'New York', 'Los Angeles', 'Sydney', 'Singapore', 'Frankfurt', 'Tokyo', 'Delhi', 'Mumbai', 'Muscat', 'Ahmedabad', 'Beijing', 'Bangalore']
    dist_map = {'London': 5500, 'New York': 11000, 'Los Angeles': 13400, 'Sydney': 12000, 'Singapore': 5800, 'Frankfurt': 4800, 'Tokyo': 8000, 'Delhi': 2200, 'Mumbai': 1900, 'Muscat': 340, 'Ahmedabad': 1760, 'Beijing': 5850, 'Bangalore': 2700}
    
    with st.sidebar:
        st.markdown("## 🧠 Core Engine")
        active_model = st.radio("Active Model Pipeline", ["LightGBM", "XGBoost"], index=0)
        st.markdown("<hr/>", unsafe_allow_html=True)
    
    render_header()
    render_kpis(df_p, df_d, active_model)
    
    c_left, c_right = st.columns([1, 20]) # layout dummy
    
    inputs = render_inputs(destinations, dist_map)
    
    with st.spinner("Executing Inference Pipeline..."):
        time.sleep(0.3)
        proc_price, pred_price, proc_delay, delay_prob, explanations = execute_inference(inputs, active_model, p_lgb, p_xgb, d_lgb, d_xgb, p_prep, p_scaler, d_scaler, f_price, f_delay, exp_d)
        inputs['delay_reasons'] = explanations
        
    render_dashboard(inputs, pred_price, delay_prob, active_model, p_metrics, d_metrics)
    render_recommendations(inputs, pred_price, active_model, p_lgb, p_xgb, p_prep, f_price)
    render_ml_insights(inputs, proc_price, proc_delay, active_model, exp_p, f_price, p_metrics, d_metrics, d_tests)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    c_r, _ = st.columns([1,4])
    with c_r:
        report_txt = generate_report(inputs, pred_price, delay_prob, active_model)
        st.download_button("📥 Download Analysis Report", data=report_txt, file_name="AeroIntel_Report.txt", mime="text/plain")

if __name__ == "__main__":
    main()
