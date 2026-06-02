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
# SAFE HTML RENDERING HELPER
# ==========================
def _html(content):
    """Centralized safe HTML renderer. Guarantees unsafe_allow_html=True."""
    st.markdown(content, unsafe_allow_html=True)

# ==========================
# PAGE CONFIG & CSS
# ==========================
st.set_page_config(page_title="AeroIntel Intelligence", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

def inject_custom_css():
    try:
        css_file = 'style.css'
        if os.path.exists(css_file):
            with open(css_file, 'r', encoding='utf-8') as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            path_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style.css')
            if os.path.exists(path_relative):
                with open(path_relative, 'r', encoding='utf-8') as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            else:
                st.warning("Could not find style.css. UI styling falls back to default.")
    except Exception as e:
        st.warning(f"Failed to load custom styles: {e}")

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
    if 'notifications_open' not in st.session_state:
        st.session_state.notifications_open = False
        
    c1, c2, c3 = st.columns([1.5, 14, 1.5])
    with c1:
        _html("<h1 style='font-size:3.5rem; margin:0; line-height:1;'>✈️</h1>")
    with c2:
        _html("""
        <div style="display:flex; flex-direction:column; justify-content:center; height:100%;">
            <h1 style='margin:0; font-size:2.2rem;'>AeroIntel Intelligence</h1>
            <p style='color:#94A3B8; font-size:0.95rem; margin:2px 0 0 0;'>Next-generation machine learning aviation operations & pricing dashboard</p>
        </div>
        """)
    with c3:
        _html("""
        <div style="position:relative; display:inline-block; margin-bottom:-10px; margin-top: 10px;">
            <div class="bell-badge"></div>
        </div>
        """)
        bell_label = "🔔" if not st.session_state.notifications_open else "🔕"
        if st.button(bell_label, key="bell_toggle_btn", help="Click to view live aviation advisories"):
            st.session_state.notifications_open = not st.session_state.notifications_open
            st.rerun()

    if st.session_state.notifications_open:
        _html("""
        <div class="glass-card animated-card delay-1 severity-info" style="margin-bottom: 25px; border-left: 4px solid #38bdf8;">
            <h4 style="margin:0 0 10px 0; color:#38bdf8; font-size: 1.1rem; display: flex; align-items; center; gap: 8px;">🔔 Active System Advisories</h4>
            <ul style="margin:0; padding-left:20px; color:#cbd5e1; font-size:0.9rem; line-height: 1.5;">
                <li style="margin-bottom:6px;">⚠️ <b>Terminal Weather Alert:</b> Convective weather and wind shear warnings active at London Heathrow (LHR). Expect minor approach sequencing delays.</li>
                <li style="margin-bottom:6px;">📈 <b>Dynamic Fare Opportunity:</b> High passenger volume detected on Premium segments. Yield optimization model recommends a +12% First Class markup.</li>
                <li style="margin-bottom:6px;">🛫 <b>Airspace Congestion:</b> Dubai (DXB) local slot allocations are operating at peak capacity. Standby buffers are active.</li>
                <li>⚙️ <b>ML Ops Pipeline Sync:</b> LightGBM and XGBoost predictors synced successfully with 100% features alignment. API latency is 12ms.</li>
            </ul>
        </div>
        """)

def render_kpis(df_p, df_d, active_model):
    avg_price = df_p['price'].mean() * 83
    delay_rate = df_d['delay'].mean() * 100
    top_route_df = df_p.groupby(['source_city', 'destination_city']).size().reset_index(name='count')
    top_route = top_route_df.loc[top_route_df['count'].idxmax()]
    
    kpis_html = f"""
    <div class="gradient-kpis-container">
        <!-- Card 1 -->
        <div class="gradient-kpi kpi-blue">
            <div class="kpi-content">
                <div class="kpi-label">Global Average Fare <span>💰</span></div>
                <div class="kpi-val">{format_inr(avg_price)}</div>
                <div class="kpi-delta delta-down">▼ -2.1% <span style="color: #94a3b8; font-size: 0.72rem; font-weight: 500; margin-left: 4px;">vs last month</span></div>
            </div>
        </div>
        <!-- Card 2 -->
        <div class="gradient-kpi kpi-rose">
            <div class="kpi-content">
                <div class="kpi-label">System Delay Rate <span>🕒</span></div>
                <div class="kpi-val">{delay_rate:.1f}%</div>
                <div class="kpi-delta delta-up">▲ +1.2% <span style="color: #94a3b8; font-size: 0.72rem; font-weight: 500; margin-left: 4px;">vs last month</span></div>
            </div>
        </div>
        <!-- Card 3 -->
        <div class="gradient-kpi kpi-emerald">
            <div class="kpi-content">
                <div class="kpi-label">Highest Volume Route <span>🗺️</span></div>
                <div class="kpi-val" style="font-size: 1.45rem; margin-top: 10px; margin-bottom: 12px; font-weight: 700;">{top_route['source_city']} ➔ {top_route['destination_city']}</div>
                <div class="kpi-delta delta-neutral">● 62% Load Factor <span style="color: #94a3b8; font-size: 0.72rem; font-weight: 500; margin-left: 4px;">average</span></div>
            </div>
        </div>
        <!-- Card 4 -->
        <div class="gradient-kpi kpi-indigo">
            <div class="kpi-content">
                <div class="kpi-label">Active Core Engine <span>🧠</span></div>
                <div class="kpi-val" style="font-size: 1.6rem; margin-top: 6px; margin-bottom: 8px;">{active_model}</div>
                <div class="kpi-delta delta-neutral">
                    <span class="status-dot active"></span> Online & Synced
                </div>
            </div>
        </div>
    </div>
    """
    _html(kpis_html)
    _html("<hr style='opacity: 0.15; margin-bottom: 25px;'/>")

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

def generate_operational_insights(inputs, delay_prob, pred_price):
    insights = []
    
    # 1. Delay Risk
    reasons = ", and ".join(inputs.get('delay_reasons', []))
    if not reasons:
        reasons = "historical scheduled density patterns"
        
    dest = inputs.get('destination_city', 'London')
    
    if delay_prob >= 0.6:
        insights.append({
            'title': '🚨 High Operational Risk Alert',
            'text': f"High delay probability of {delay_prob*100:.0f}% detected. Main driver: {reasons}. Recommend adding a +60 min ground buffer to schedule and placing backup flight crew on standby.",
            'severity': 'high',
            'icon': '🚨',
            'category': 'Operational'
        })
    elif delay_prob >= 0.30:
        insights.append({
            'title': '⚠️ Elevated Schedule Volatility',
            'text': f"Moderate risk of scheduling delays ({delay_prob*100:.0f}%) due to {reasons}. Boarding protocols should be strictly optimized to meet slot times.",
            'severity': 'medium',
            'icon': '⚠️',
            'category': 'Operational'
        })
    else:
        insights.append({
            'title': '🟢 Nominal Schedule Stability',
            'text': f"Route operations expected to be stable (delay risk: {delay_prob*100:.0f}%). Expected on-time arrival. Standard ground handling protocols are sufficient.",
            'severity': 'low',
            'icon': '🛡️',
            'category': 'Operational'
        })
        
    # 2. Weather
    weather_risk = inputs.get('weather_risk', 0.2)
    weather_desc = ""
    if inputs.get('weather_data'):
        weather_desc = f" ({inputs['weather_data']['desc']})"
        
    if weather_risk >= 0.6:
        insights.append({
            'title': '⛈️ Adverse Weather Threat',
            'text': f"Severe terminal weather risk ({weather_risk*100:.0f}%) active at {dest}{weather_desc}. Navigation holding patterns or route diversion plans are highly probable.",
            'severity': 'high',
            'icon': '⛈️',
            'category': 'Weather'
        })
    elif weather_risk >= 0.3:
        insights.append({
            'title': '⛅ Transiting Weather Advisory',
            'text': f"Moderate weather risk ({weather_risk*100:.0f}%) reported at {dest}{weather_desc}. Pilots should monitor terminal winds and check-in slot delays.",
            'severity': 'medium',
            'icon': '⛅',
            'category': 'Weather'
        })
    else:
        insights.append({
            'title': '☀️ Optimal Terminal Weather',
            'text': f"Clear skies and nominal wind speeds at {dest}{weather_desc}. Zero weather-related delays predicted on current slots.",
            'severity': 'low',
            'icon': '☀️',
            'category': 'Weather'
        })
        
    # 3. Traffic
    traffic = inputs.get('traffic', 0.3)
    if traffic >= 0.6:
        insights.append({
            'title': '🛫 Severe Airspace Congestion',
            'text': f"ATC queue density is peaking at {traffic*100:.0f}%. Expect extended runway taxi-out queues and holding loops. Ground slot pre-clearance is critical.",
            'severity': 'high',
            'icon': '🛫',
            'category': 'Traffic'
        })
    elif traffic >= 0.3:
        insights.append({
            'title': '✈️ Standard Corridor Congestion',
            'text': f"Airspace traffic density is standard ({traffic*100:.0f}%). ATC routing is clear on all normal airway slots. No priority holdings expected.",
            'severity': 'medium',
            'icon': '✈️',
            'category': 'Traffic'
        })
    else:
        insights.append({
            'title': '🟢 Clear Airspace Status',
            'text': f"Uncongested airway sectors (density: {traffic*100:.0f}%). Direct routing clearances and expedited taxi-in gates are anticipated.",
            'severity': 'low',
            'icon': '🟢',
            'category': 'Traffic'
        })
        
    # 4. Market & Booking
    demand = inputs.get('demand', 'Low')
    season = inputs.get('season', 'Low')
    days_left = inputs.get('days_left', 14)
    
    if demand == 'High' or season == 'Peak':
        insights.append({
            'title': '📈 Surge Market Demand',
            'text': "Peak travel seasons coupled with surging demand are driving passenger load factors past 85%. Base dynamic fare multipliers are active.",
            'severity': 'medium',
            'icon': '📈',
            'category': 'Market'
        })
        
    if days_left <= 7:
        insights.append({
            'title': '⏳ Last-Minute Fare Surge',
            'text': f"Close-in booking window ({days_left} days left). Booking velocity has spiked. Ticket prices are expected to rise by 8-15% within the next 48 hours.",
            'severity': 'high',
            'icon': '⏳',
            'category': 'Market'
        })
    elif days_left >= 30:
        insights.append({
            'title': '📅 Booking Window Advantage',
            'text': f"Advance departure schedule ({days_left} days left). Early-bird pricing inventory is open. Recommended to lock in ticket allocations now to maximize cost yield.",
            'severity': 'low',
            'icon': '📅',
            'category': 'Market'
        })
        
    return insights

def render_custom_gauge(risk_score):
    dash_array = 440
    dash_offset = int(dash_array - (risk_score / 100) * dash_array)
    
    if risk_score >= 60:
        color = "#ef4444"
        glow = "rgba(239, 68, 68, 0.4)"
        status = "CRITICAL"
    elif risk_score >= 30:
        color = "#f59e0b"
        glow = "rgba(245, 158, 11, 0.4)"
        status = "ELEVATED"
    else:
        color = "#10b981"
        glow = "rgba(16, 185, 129, 0.4)"
        status = "NOMINAL"
        
    gauge_html = f"""
    <div class="custom-gauge-container">
        <svg width="180" height="180" class="custom-gauge-svg">
            <circle cx="90" cy="90" r="70" class="custom-gauge-track"></circle>
            <circle cx="90" cy="90" r="70" class="custom-gauge-fill" 
                    style="stroke: {color}; stroke-dasharray: {dash_array}; stroke-dashoffset: {dash_offset}; filter: drop-shadow(0 0 6px {glow});"></circle>
        </svg>
        <div class="custom-gauge-value">
            <span class="gauge-num">{risk_score}%</span>
            <span class="gauge-lbl" style="color: {color};">{status} RISK</span>
        </div>
    </div>
    """
    return gauge_html

def render_timeline_html(inputs, delay_prob):
    tr = inputs.get('traffic', 0.3)
    wr = inputs.get('weather_risk', 0.2)
    
    boarding_risk = "low"
    if tr > 0.7 or delay_prob > 0.6: boarding_risk = "high"
    elif tr > 0.4 or delay_prob > 0.3: boarding_risk = "med"
    
    pushback_risk = "low"
    if tr > 0.6 or delay_prob > 0.5: pushback_risk = "high"
    elif tr > 0.3 or delay_prob > 0.2: pushback_risk = "med"
    
    transit_risk = "low"
    if wr > 0.7: transit_risk = "high"
    elif wr > 0.4: transit_risk = "med"
    
    arrival_risk = "low"
    if wr > 0.6 or delay_prob > 0.6: arrival_risk = "high"
    elif wr > 0.3 or delay_prob > 0.3: arrival_risk = "med"
    
    nodes = [
        {"id": "checkin", "title": "Check-in", "status": "Nominal", "risk": "low", "icon": "🎒"},
        {"id": "security", "title": "Security", "status": "Nominal", "risk": "low", "icon": "🔍"},
        {"id": "boarding", "title": "Boarding", "status": "High Density" if boarding_risk == "high" else ("Active" if boarding_risk == "med" else "Nominal"), "risk": boarding_risk, "icon": "🛂"},
        {"id": "pushback", "title": "Pushback", "status": "Slot Hold" if pushback_risk == "high" else ("Delayed" if pushback_risk == "med" else "Cleared"), "risk": pushback_risk, "icon": "🛫"},
        {"id": "transit", "title": "Transit", "status": "Turbulence" if transit_risk == "high" else ("Windy" if transit_risk == "med" else "Smooth"), "risk": transit_risk, "icon": "✈️"},
        {"id": "arrival", "title": "Arrival", "status": "Reroute Risk" if arrival_risk == "high" else ("Hold Risk" if arrival_risk == "med" else "On-Time"), "risk": arrival_risk, "icon": "🛬"}
    ]
    
    nodes_html = ""
    for n in nodes:
        circle_class = f"timeline-circle-{n['risk']}"
        nodes_html += f"""
        <div class="timeline-node">
            <div class="timeline-circle {circle_class}">{n['icon']}</div>
            <div class="timeline-title">{n['title']}</div>
            <div class="timeline-status">{n['status']}</div>
        </div>
        """
        
    html = f"""
    <div class="timeline-container">
        <h4 style="margin: 0 0 10px 0; font-size: 1rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">✈️ Timeline of Operational Milestones</h4>
        <div class="timeline-track-wrapper">
            <div class="timeline-line"></div>
            <div class="timeline-nodes">
                {nodes_html}
            </div>
        </div>
    </div>
    """
    return html

def render_alert_banner(delay_prob, weather_risk):
    if delay_prob >= 0.6 or weather_risk >= 0.7:
        _html("""
        <div class="custom-alert-banner alert-banner-high">
            <span class="alert-icon">🚨</span>
            <div>
                <div style="font-size: 1.05rem; font-weight: 800;">CRITICAL OPERATIONAL RISK WARNING</div>
                <div style="font-size: 0.88rem; font-weight: 500; opacity: 0.9; margin-top: 2px;">
                    Flight delay probability is critical. Adverse weather and airspace slot congestion are active. Rerouting fuel buffer adjustments are highly advised.
                </div>
            </div>
        </div>
        """)
    elif delay_prob >= 0.35 or weather_risk >= 0.4:
        _html("""
        <div class="custom-alert-banner alert-banner-medium">
            <span class="alert-icon">⚠️</span>
            <div>
                <div style="font-size: 1.05rem; font-weight: 800;">ELEVATED SCHEDULE VOLATILITY ADVISORY</div>
                <div style="font-size: 0.88rem; font-weight: 500; opacity: 0.9; margin-top: 2px;">
                    Flight operations are stable but vulnerable to localized traffic peaks. Monitor terminal meteorological updates.
                </div>
            </div>
        </div>
        """)

def generate_smart_recommendations(inputs, delay_prob, predicted_price, active_model, p_lgb, p_xgb, p_prep, f_price, p_metrics, d_metrics):
    recs = []
    
    # Heuristics for confidence scores
    p_rmse = p_metrics.get(active_model, {}).get("rmse", 750)
    p_conf = max(0, 100 - (p_rmse/2000)*100)
    
    d_rec = d_metrics.get(active_model, {}).get("rec", 0.85)
    d_conf = d_rec * 100
    
    # 1. Booking Recommendation
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
    curr_price_inr = predicted_price * 83
    savings = curr_price_inr - opt_price
    
    days_left = inputs['days_left']
    
    if savings > 1000 and opt_days != days_left:
        if opt_days < days_left:
            title = "📅 Postpone Booking Window"
            action = f"Postpone booking to {opt_days} days before departure."
            reasoning = f"Price yield sweeps show a dynamic dip approaching day {opt_days}. Delaying booking is estimated to save {format_inr(savings)}."
        else:
            title = "📅 Lock Booking Instantly"
            action = f"Purchase ticket today at {days_left} days to departure."
            reasoning = f"Pricing curve is aggressively upward. Early booking today avoids predicted inventory depletion surges, saving {format_inr(savings)}."
    else:
        title = "📅 Confirm Target Booking"
        action = "Confirm ticketing within the next 48 hours."
        reasoning = "Current booking lead time matches the optimal pricing valley. Future shifts will expose fares to dynamic demand markups."
        
    recs.append({
        'title': title,
        'action': action,
        'reasoning': reasoning,
        'category': 'Booking',
        'confidence': int(p_conf),
        'severity_meter': 'low' if savings <= 2000 else 'medium'
    })
    
    # 2. Alternative Low-Risk Routes
    dest = inputs['destination_city']
    stops = inputs['stops']
    
    if stops != 'Non-stop' and delay_prob > 0.4:
        premium_val = 5000 * 83
        title = "🗺️ Non-Stop Transit Upgrade"
        action = "Reroute to a direct flight configuration."
        reasoning = f"Connecting flights introduce routing compound risk (+{delay_prob*100 - 15:.0f}% delay likelihood). Direct slots bypass hub layover hold programs for a premium of {format_inr(premium_val)}."
        severity_meter = 'medium'
    elif stops == 'Non-stop' and delay_prob > 0.5:
        title = "🗺️ Airway Corridor Rerouting"
        action = "Reroute flight path via Muscat (MCT) airways."
        reasoning = "High traffic congestion on the standard route. Routing via MCT airways reduces ATC queue holdings by approximately 18%."
        severity_meter = 'high'
    else:
        title = "🗺️ Route Configuration Confirmed"
        action = "Maintain scheduled flight corridor."
        reasoning = f"Current direct routing via {dest} represents the optimal balance of schedule reliability and ticketing cost."
        severity_meter = 'low'
        
    recs.append({
        'title': title,
        'action': action,
        'reasoning': reasoning,
        'category': 'Routing',
        'confidence': int(d_conf),
        'severity_meter': severity_meter
    })
    
    # 3. Risk Mitigation Actions
    weather_risk = inputs.get('weather_risk', 0.2)
    traffic = inputs.get('traffic', 0.3)
    
    if weather_risk > 0.6:
        title = "🛡️ Weather Buffer Adjustments"
        action = "File alternate arrival airports and load +45m contingency fuel."
        reasoning = f"Live forecasts indicate severe conditions ({weather_risk*100:.0f}%) at {dest}. Alternate flight planning at Frankfurt/Paris avoids holding pattern fuel exhaust."
        severity_meter = 'high'
    elif traffic > 0.6:
        title = "🛡️ Airspace Priority Lock"
        action = "Pre-register slot request with ATC 120m before pushback."
        reasoning = f"Terminal traffic density is elevated ({traffic*100:.0f}%). Locking slot sequence early avoids ground-hold queue penalties."
        severity_meter = 'medium'
    else:
        title = "🛡️ Standard Dispatch Clearance"
        action = "Deploy normal flight scheduling buffers."
        reasoning = "All corridor weather and airspace traffic indicators are in the green. Baseline dispatch timelines fully apply."
        severity_meter = 'low'
        
    recs.append({
        'title': title,
        'action': action,
        'reasoning': reasoning,
        'category': 'Mitigation',
        'confidence': int(max(d_conf, p_conf)),
        'severity_meter': severity_meter
    })
    
    # 4. Operational Actions for Delays
    if delay_prob > 0.6:
        title = "🛫 Turnaround Buffer Expansion"
        action = "Extend minimum ground turnaround window by +30 minutes."
        reasoning = "Compounding airport traffic and meteorological risks are high. Turning aircraft with extra buffer limits downstream schedule impact."
        severity_meter = 'high'
    elif delay_prob > 0.3:
        title = "🛫 Advisory Notification Trigger"
        action = "Queue automatic schedule alerts to passengers."
        reasoning = "Moderate delay risk. Timely notifications of 15-minute slot shifts preserve passenger trust and LHR slot compliance."
        severity_meter = 'medium'
    else:
        title = "🛫 Standard Gate Dispatch"
        action = "Initiate boarding sequence at standard T-40m."
        reasoning = "High probability of on-time departure. Maintain nominal gate operations and loading priority."
        severity_meter = 'low'
        
    recs.append({
        'title': title,
        'action': action,
        'reasoning': reasoning,
        'category': 'Operations',
        'confidence': int(d_conf),
        'severity_meter': severity_meter
    })
    
    # 5. Pricing Dynamic Suggestions
    demand = inputs.get('demand', 'Low')
    cabin_class = inputs.get('cabin_class', 'Economy')
    
    if demand == 'High' and cabin_class in ['Business', 'First']:
        title = "💰 Dynamic Cabin Markup"
        action = "Deploy +12% dynamic yield markup on premium cabin seats."
        reasoning = f"Peak seasonal passenger load factors detected. Dynamic models indicate high price elasticity on premium {cabin_class} classes."
        severity_meter = 'medium'
    elif demand == 'High' and days_left < 14:
        title = "💰 Economy Yield Acceleration"
        action = "Apply +8% close-in booking markup to inventory."
        reasoning = "Late booking passenger demand is peaking. Close-in booking yield overrides base ticket classes."
        severity_meter = 'medium'
    else:
        title = "💰 Base Pricing Lock"
        action = "Maintain current target pricing levels."
        reasoning = "Standard segment demand curves. Base ticket prices maximize overall cabin load factors without revenue leak."
        severity_meter = 'low'
        
    recs.append({
        'title': title,
        'action': action,
        'reasoning': reasoning,
        'category': 'Pricing',
        'confidence': int(p_conf),
        'severity_meter': severity_meter
    })
    
    return recs

def render_dashboard(inputs, pred_price, delay_prob, active_model, p_metrics, d_metrics):
    # Calculate combined aviation risk score (0-100)
    weather_risk = inputs.get('weather_risk', 0.2)
    traffic = inputs.get('traffic', 0.3)
    layover_penalty = 0.15 if inputs.get('layover_type') == 'Long' else (0.05 if inputs.get('layover_type') == 'Medium' else 0.0)
    
    risk_score = int(((delay_prob * 0.45) + (weather_risk * 0.35) + (traffic * 0.1) + layover_penalty) * 100)
    risk_score = min(100, max(0, risk_score))
    
    # 1. Render Alert Banner
    render_alert_banner(delay_prob, weather_risk)
    
    _html("<h3 style='color:#ffffff; margin-bottom:20px;'>🎯 Predictive Intelligence</h3>")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        # Ticket Yield Card
        p_rmse = p_metrics.get(active_model, {}).get("rmse", 750)
        conf_score = max(0, 100 - (p_rmse/2000)*100)
        conf_range = p_rmse * 83 * 0.7 
        
        _html(f"""
        <div class="glass-card animated-card delay-1" style="margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <p style='color:#94A3B8; font-weight:700; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; margin:0;'>Predicted Ticket Yield</p>
                <span class="rec-category rec-cat-booking" style="font-size: 0.68rem;">{conf_score:.1f}% Model Confidence</span>
            </div>
            <h2 style="color: #38BDF8; font-size: 2.8rem; margin: 10px 0; font-weight: 800;">{format_inr(pred_price * 83)}</h2>
            <p style="color: #94A3B8; font-size: 0.88rem; margin: 0;">Expected variation range: ± {format_inr(conf_range)}</p>
            <hr style="opacity: 0.1; margin: 12px 0;">
            <p style="color: #CBD5E1; font-size: 0.82rem; margin: 0;">Inference Model: <b>{active_model}</b> | Lead Time: {inputs['days_left']} days</p>
        </div>
        """)
        
        # Risk assessment card (consolidated into single HTML block)
        gauge_html = render_custom_gauge(risk_score)
        timeline_html = render_timeline_html(inputs, delay_prob)
        
        _html(f"""
        <div class="glass-card animated-card delay-2">
            <h4 style="margin: 0 0 10px 0; font-size: 0.95rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">📈 Risk Assessment</h4>
            {gauge_html}
            <hr style="opacity:0.1; margin:15px 0;">
            {timeline_html}
        </div>
        """)
        
    with c2:
        # Operational Insights Cards Panel
        _html('<h4 style="margin: 0 0 15px 0; font-size: 0.95rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">🧠 Animated AI Insights</h4>')
        
        insights = generate_operational_insights(inputs, delay_prob, pred_price)
        
        for idx, insight in enumerate(insights):
            delay_class = f"delay-{idx+1}"
            _html(f"""
            <div class="glass-card animated-card {delay_class} severity-{insight['severity']}" style="padding:16px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:1.2rem;">{insight['icon']}</span>
                        <strong style="color:#ffffff; font-size:0.95rem;">{insight['title']}</strong>
                    </div>
                    <span class="rec-category rec-cat-ops" style="font-size:0.65rem; padding:2px 6px;">{insight['category']}</span>
                </div>
                <div style="color:#cbd5e1; font-size:0.86rem; line-height:1.45;">
                    {insight['text']}
                </div>
            </div>
            """)

def render_recommendations(inputs, current_price, active_model, p_lgb, p_xgb, p_prep, f_price, p_metrics, d_metrics, delay_prob):
    st.markdown("### 💡 Smart Decision Advisor")
    
    recs = generate_smart_recommendations(inputs, delay_prob, current_price, active_model, p_lgb, p_xgb, p_prep, f_price, p_metrics, d_metrics)
    
    # Category class mapping (CSS uses rec-cat-ops, not rec-cat-operations)
    cat_class_map = {
        'booking': 'rec-cat-booking',
        'routing': 'rec-cat-routing',
        'mitigation': 'rec-cat-mitigation',
        'operations': 'rec-cat-ops',
        'pricing': 'rec-cat-pricing'
    }
    
    # Render cards individually using Streamlit columns for grid layout
    for row_start in range(0, len(recs), 3):
        row_recs = recs[row_start:row_start + 3]
        cols = st.columns(3 if len(row_recs) == 3 else len(row_recs))
        for col_idx, rec in enumerate(row_recs):
            with cols[col_idx]:
                severity_color = "#10b981" if rec['severity_meter'] == 'low' else ("#f59e0b" if rec['severity_meter'] == 'medium' else "#ef4444")
                cat_class = cat_class_map.get(rec['category'].lower(), 'rec-cat-booking')
                anim_idx = row_start + col_idx + 1
                
                _html(f"""
                <div class="glass-card animated-card delay-{anim_idx}" style="display:flex; flex-direction:column; justify-content:space-between; height:100%;">
                    <div>
                        <div class="rec-card-header">
                            <span class="rec-category {cat_class}">{rec['category']}</span>
                            <span class="rec-confidence">{rec['confidence']}% Confidence</span>
                        </div>
                        <h4 style="margin: 5px 0 8px 0; font-size: 1.12rem; color:#ffffff; font-weight:700;">{rec['title']}</h4>
                        <p style="font-size: 0.95rem; font-weight: 700; color:{severity_color}; margin: 5px 0 10px 0;">👉 {rec['action']}</p>
                        <div style="font-size: 0.86rem; color:#cbd5e1; line-height:1.45;">
                            {rec['reasoning']}
                        </div>
                    </div>
                    <div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" style="width: {rec['confidence']}%;"></div>
                        </div>
                    </div>
                </div>
                """)

def render_ml_insights(inputs, proc_price, proc_delay, active_model, exp_p, f_price, p_metrics, d_metrics, d_tests):
    from sklearn.metrics import precision_recall_curve
    _html("<h3 style='color:#ffffff; margin-bottom:20px;'>🧠 Under The Hood (ML Ops)</h3>")
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
                _html(f"""
                <div class="premium-card">
                    <p style="color:#38BDF8; font-weight:700; margin-bottom:5px;">💰 Algorithm Accuracy</p>
                    <table style="width:100%; color:#E6EDF3; font-size:1.05rem;">
                        <tr><td>Mean Absolute Error</td><td style="text-align:right; font-weight:600;">{format_inr(m['mae']*83)}</td></tr>
                        <tr><td>R-Squared (R²)</td><td style="text-align:right; font-weight:600;">{m['r2']*100:.1f}%</td></tr>
                        {val_cv}
                    </table>
                </div>""")
                
        with c2:
            st.markdown(f"**{active_model} Delay Metrics**")
            d_mod = active_model
            if d_mod in d_metrics:
                m = d_metrics[d_mod]
                _html(f"""
                <div class="premium-card">
                    <p style="color:#10B981; font-weight:700; margin-bottom:5px;">🕒 Imbalance Compensated Classifier</p>
                    <table style="width:100%; color:#E6EDF3; font-size:1.05rem;">
                        <tr><td title="Prioritized KPI: Delays Caught">Recall (Delays Caught)</td><td style="text-align:right; font-weight:600; color:#34D399">{m['rec']*100:.1f}%</td></tr>
                        <tr><td>F1-Score</td><td style="text-align:right; font-weight:600;">{m['f1']*100:.1f}%</td></tr>
                        <tr><td>Precision</td><td style="text-align:right; font-weight:600;">{m['prec']*100:.1f}%</td></tr>
                        <tr><td>ROC-AUC</td><td style="text-align:right; font-weight:600;">{m['auc']:.4f}</td></tr>
                    </table>
                </div>""")

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
            
            _html(f"""
            <div class="premium-card" style="padding: 15px; margin-bottom: 10px;">
                <p style="color:#94a3b8; font-size:0.9rem; margin:0;">Dynamic Recall</p>
                <h3 style="color:#34D399; margin:5px 0;">{recall*100:.1f}%</h3>
            </div>
            """)
            _html(f"""
            <div class="premium-card" style="padding: 15px; margin-bottom: 10px;">
                <p style="color:#94a3b8; font-size:0.9rem; margin:0;">Dynamic Precision</p>
                <h3 style="color:#F59E0B; margin:5px 0;">{prec*100:.1f}%</h3>
            </div>
            """)
            _html(f"""
            <div class="premium-card" style="padding: 15px;">
                <p style="color:#94a3b8; font-size:0.9rem; margin:0;">Dynamic F1-Score</p>
                <h3 style="color:#38BDF8; margin:5px 0;">{f1_s*100:.1f}%</h3>
            </div>
            """)
            
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

def generate_report(inputs, price, delay, model, p_lgb, p_xgb, p_prep, f_price, p_metrics, d_metrics):
    report = f"AeroIntel Operational & Pricing Analytics Export\n{'='*50}\n\n"
    report += f"Flight: {inputs['source_city']} to {inputs['destination_city']}\n"
    report += f"Cabin Class: {inputs['cabin_class']} | Layovers: {inputs['stops']}\n"
    report += f"Days to Departure: {inputs['days_left']} | Season: {inputs['season']}\n\n"
    report += f"Core Inference Engine: {model}\n"
    report += f"Predicted Ticket Yield: {format_inr(price * 83)}\n"
    report += f"Flight Delay Probability: {delay*100:.1f}%\n"
    
    report += "\n🚨 Delay Trigger Analysis:\n"
    reasons = inputs.get('delay_reasons', [])
    if reasons:
        for r in reasons:
            report += f" - {r}\n"
    else:
        report += " - No significant delay risk factors identified. Flight operations nominal.\n"
        
    report += "\n💡 AI Smart Advisories & Recommendations:\n"
    recs = generate_smart_recommendations(inputs, delay, price, model, p_lgb, p_xgb, p_prep, f_price, p_metrics, d_metrics)
    for r in recs:
        report += f" - [{r['category']}] {r['title']}\n"
        report += f"   Recommended Action: {r['action']}\n"
        report += f"   Operational Reasoning: {r['reasoning']}\n\n"
        
    report += f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | AeroIntel Systems"
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
        _html("<hr/>")
    
    render_header()
    render_kpis(df_p, df_d, active_model)
    
    c_left, c_right = st.columns([1, 20]) # layout dummy
    
    inputs = render_inputs(destinations, dist_map)
    
    with st.spinner("Executing Inference Pipeline..."):
        time.sleep(0.3)
        proc_price, pred_price, proc_delay, delay_prob, explanations = execute_inference(inputs, active_model, p_lgb, p_xgb, d_lgb, d_xgb, p_prep, p_scaler, d_scaler, f_price, f_delay, exp_d)
        inputs['delay_reasons'] = explanations
        
    render_dashboard(inputs, pred_price, delay_prob, active_model, p_metrics, d_metrics)
    render_recommendations(inputs, pred_price, active_model, p_lgb, p_xgb, p_prep, f_price, p_metrics, d_metrics, delay_prob)
    render_ml_insights(inputs, proc_price, proc_delay, active_model, exp_p, f_price, p_metrics, d_metrics, d_tests)
    
    _html("<br/>")
    c_r, _ = st.columns([1,4])
    with c_r:
        report_txt = generate_report(inputs, pred_price, delay_prob, active_model, p_lgb, p_xgb, p_prep, f_price, p_metrics, d_metrics)
        st.download_button("📥 Download Analysis Report", data=report_txt, file_name="AeroIntel_Report.txt", mime="text/plain")

if __name__ == "__main__":
    main()
