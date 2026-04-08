import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             r2_score, mean_absolute_error, mean_squared_error)
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
import lightgbm as lgb
import time
import io

# ==========================
# PAGE CONFIG & CSS
# ==========================
st.set_page_config(page_title="Model Comparison Panel", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stApp { background-color: #0A0F1C; color: #E6EDF3; }
        h1, h2, h3 { color: #FFFFFF !important; font-weight: 700; }
        div[data-testid="stMetric"], .metric-card {
            background-color: #111827; border: 1px solid #2A3441;
            border-radius: 12px; padding: 22px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }
        div[data-testid="stMetricValue"] { color: #FFFFFF; font-weight: 800; font-size: 2.0rem; }
        div[data-testid="stMetricLabel"] { color: #94A3B8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
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
        hr { border-color: #334155 !important; opacity: 0.8; }
        .stTabs [data-baseweb="tab"] { color: #94A3B8; font-weight: 600; font-size: 1.1rem; }
        .stTabs [aria-selected="true"] { color: #38BDF8 !important; border-bottom-color: #38BDF8 !important; }
        
        .winner-card {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 78, 59, 0.4));
            border: 1px solid #10B981;
            border-radius: 12px; padding: 20px;
        }
        .insight-box {
            background-color: #1A2332; border-left: 4px solid #38BDF8;
            padding: 15px; margin-bottom: 15px; border-radius: 0 8px 8px 0;
        }
    </style>
    """, unsafe_allow_html=True)

# ==========================
# DATA LOADING & BASIC PREP
# ==========================
@st.cache_data
def load_data(task_type, uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    try:
        if task_type == 'Classification (Delay)':
            return pd.read_csv('data/delay_data.csv')
        else:
            return pd.read_csv('data/pricing_data.csv')
    except Exception as e:
        st.error(f"Cannot load default data. Please ensure 'data/' directory exists relative to execution folder. {e}")
        return pd.DataFrame()

def prepare_xy(df, task_type):
    df = df.copy()
    if df.empty: return None, None
    
    target_col = 'delay' if 'Classification' in task_type else 'price'
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset!")
        return None, None
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def build_preprocessor(X):
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create numerical preprocessor (Impute + Scale)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Create categorical preprocessor (Impute + OrdinalEncode for tree model compatibility)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough')
        
    return preprocessor

# ==========================
# HYPERPARAMETER TUNING HELPER
# ==========================
def tune_and_evaluate(models, param_grids, X_train, y_train, X_test, y_test, scoring_metrics, refit_metric):
    results = {}
    fitted_models = {}
    
    for name, pipeline in models.items():
        if name in param_grids and param_grids[name]:
            # Basic hyperparameter tuning with RandomizedSearchCV using 5-fold CV
            search = RandomizedSearchCV(
                pipeline, 
                param_distributions=param_grids[name], 
                n_iter=4, # Keep iterations low for UI responsiveness
                cv=5, 
                scoring=scoring_metrics, 
                refit=refit_metric,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            # Extract 5-Fold CV scores for the best model
            best_idx = search.best_index_
            cv_metrics = {m: search.cv_results_[f'mean_test_{m}'][best_idx] for m in scoring_metrics}
        else:
            # No tuning params provided (e.g. LinearRegression), standard 5-fold CV
            cv_scores = cross_validate(pipeline, X_train, y_train, cv=5, scoring=scoring_metrics, n_jobs=-1)
            cv_metrics = {m: cv_scores[f'test_{m}'].mean() for m in scoring_metrics}
            
            best_model = pipeline
            best_model.fit(X_train, y_train)

        # Generate Test Predictions
        y_pred = best_model.predict(X_test)
        if hasattr(best_model.named_steps['model'], "predict_proba"):
            y_prob_test = best_model.predict_proba(X_test)[:, 1]
        else:
            y_prob_test = pd.Series([0]*len(y_test))
            
        # Store results dynamically
        results[name] = {
            'cv_metrics': cv_metrics,
            'y_pred_test': y_pred,
            'y_prob_test': y_prob_test,
            'y_test_true': y_test
        }
        fitted_models[name] = best_model
        
    return results, fitted_models

# ==========================
# MODEL TRAINING CLASSIFICATION
# ==========================
@st.cache_resource(show_spinner=False)
def train_classification_models(_X, _y):
    # Perform train-test split FIRST to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, random_state=42, stratify=_y)
    
    preprocessor = build_preprocessor(X_train)
    
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    models = {
        'Logistic Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                               ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))]),
        'XGBoost': Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', xgb.XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'))]),
        'LightGBM': Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42))])
    }
    
    # Define basic hyperparameter tuning grids
    param_grids = {
        'Logistic Regression': {
            'model__C': [0.1, 1.0, 10.0]
        },
        'XGBoost': {
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.05, 0.1, 0.2]
        },
        'LightGBM': {
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.05, 0.1, 0.2]
        }
    }
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    raw_results, fitted_models = tune_and_evaluate(models, param_grids, X_train, y_train, X_test, y_test, scoring, 'f1')
    return raw_results, fitted_models, X_train, X_test

def get_class_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob),
        'cm': confusion_matrix(y_true, y_pred)
    }

# ==========================
# MODEL TRAINING REGRESSION
# ==========================
@st.cache_resource(show_spinner=False)
def train_regression_models(_X, _y):
    # Perform train-test split FIRST to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, random_state=42)
    
    preprocessor = build_preprocessor(X_train)
    
    models = {
        'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                             ('model', LinearRegression())]),
        'XGBoost': Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', xgb.XGBRegressor(n_estimators=100, random_state=42))]),
        'LightGBM': Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', lgb.LGBMRegressor(n_estimators=100, random_state=42))])
    }
    
    param_grids = {
        'Linear Regression': {}, # No basic HP tuning for plain OLS
        'XGBoost': {
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.05, 0.1, 0.2]
        },
        'LightGBM': {
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.05, 0.1, 0.2]
        }
    }
    
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
    
    raw_results, fitted_models = tune_and_evaluate(models, param_grids, X_train, y_train, X_test, y_test, scoring, 'r2')
    
    # Enrich with test set evaluations 
    for name in raw_results:
        y_t, y_p = raw_results[name]['y_test_true'], raw_results[name]['y_pred_test']
        raw_results[name]['test_r2'] = r2_score(y_t, y_p)
        raw_results[name]['test_mae'] = mean_absolute_error(y_t, y_p)
        raw_results[name]['test_rmse'] = np.sqrt(mean_squared_error(y_t, y_p))
        
    return raw_results, fitted_models, X_train, X_test

# ==========================
# VISUALIZATIONS
# ==========================
def plot_roc_curves(results):
    fig = go.Figure()
    colors = ['#38BDF8', '#10B981', '#F59E0B']
    for i, (name, metrics) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(metrics['y_test_true'], metrics['y_prob_test'])
        cv_auc = metrics['cv_metrics']['roc_auc']
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                 name=f"{name} (AUC={cv_auc:.3f})",
                                 line=dict(width=2, color=colors[i % len(colors)])))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
    fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_residuals(results):
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(results.keys()))
    colors = ['#38BDF8', '#10B981', '#F59E0B']
    for i, (name, metrics) in enumerate(results.items()):
        y_true, y_pred = metrics['y_test_true'], metrics['y_pred_test']
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers', 
                       marker=dict(size=4, color=colors[i], opacity=0.5), name=name),
            row=1, col=i+1
        )
        # Add y=x line
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                       line=dict(dash='dash', color='gray'), showlegend=False),
            row=1, col=i+1
        )
    fig.update_layout(title="Actual vs Predicted (Residuals)", template="plotly_dark", 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, showlegend=False)
    # Update axes titles
    for i in range(1, 4):
        fig.update_xaxes(title_text="Actual", row=1, col=i)
        fig.update_yaxes(title_text="Predicted", row=1, col=i)
    return fig

def plot_feature_importance(models, original_features):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["LightGBM", "XGBoost"])
    
    for i, model_name in enumerate(["LightGBM", "XGBoost"]):
        if model_name in models:
            pipeline = models[model_name]
            classifier = pipeline.named_steps['model']
            
            try:
                feat_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                feat_names = [f.split('__')[-1] for f in feat_names]
            except Exception:
                feat_names = original_features
                
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                if len(importances) == len(feat_names):
                    idx = np.argsort(importances)[-10:] # Top 10
                    fig.add_trace(
                        go.Bar(x=importances[idx], y=[feat_names[j] for j in idx], orientation='h',
                               name=model_name, marker_color='#38BDF8' if i==0 else '#10B981'),
                        row=1, col=i+1
                    )
    fig.update_layout(title="Top 10 Feature Importances", template="plotly_dark", 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500, showlegend=False)
    return fig

# ==========================
# INSIGHTS GENERATOR
# ==========================
def generate_classification_insights(results, test_metrics):
    best_auc_model = max(results.keys(), key=lambda k: results[k]['cv_metrics']['roc_auc'])
    best_f1_model = max(test_metrics.keys(), key=lambda k: test_metrics[k]['f1'])
    
    insights = []
    insights.append(f"🏆 **{best_auc_model}** exhibits the strongest overall discriminatory power (highest ROC-AUC).")
    
    if test_metrics['Logistic Regression']['auc'] < results['XGBoost']['cv_metrics']['roc_auc'] - 0.05:
        insights.append(f"🔍 **Logistic Regression** significantly underperforms tree-based models, suggesting strong **non-linear relationships** in the delay data (e.g. compounding effects of weather and late departures).")
        
    if test_metrics[best_f1_model]['rec'] > 0.8:
        insights.append(f"✅ The current threshold tuning allows the **{best_f1_model}** model to capture >80% of actual delays (High Recall).")
    else:
        insights.append("⚠️ Consider lowering the decision threshold if avoiding delays is critical (increasing Recall at the cost of Precision).")
        
    return insights

def generate_regression_insights(results):
    best_r2_model = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_rmse_model = min(results.keys(), key=lambda k: results[k]['test_rmse'])
    
    insights = []
    insights.append(f"🏆 **{best_r2_model}** explains the most variance in ticket pricing (Highest R²).")
    
    lin_r2 = results['Linear Regression']['test_r2']
    tree_r2 = results['XGBoost']['test_r2']
    if tree_r2 - lin_r2 > 0.1:
         insights.append(f"🔍 The massive leap from **Linear Regression (R²={lin_r2:.2f})** to **XGBoost (R²={tree_r2:.2f})** proves that airline pricing is highly dynamic and non-linear, likely interacting across seasonality, demand shifts, and days-to-departure.")
         
    if best_rmse_model == "LightGBM":
        insights.append("⚡ **LightGBM** provides the lowest RMSE, making it highly suitable for high-frequency pricing updates in a production environment due to its speed and fast tree splits.")
        
    return insights

# ==========================
# MAIN APP EXECUTION
# ==========================
def main():
    inject_custom_css()
    
    st.markdown("<h1 style='font-size: 2.5rem;'>⚖️ AeroIntel Model Comparison</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Professional sandbox to evaluate, tune, and dissect machine learning algorithms for airline analytics.</p><hr/>", unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### ⚙️ Evaluation Settings")
        task_type = st.radio("Target Problem", ["Classification (Delay)", "Regression (Pricing)"])
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### 📁 Custom Dataset (Optional)")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], help="Required columns: 'price' for regression or 'delay' for classification.")
        
        threshold = 0.5
        if "Classification" in task_type:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("### 🎛️ Classification Threshold")
            threshold = st.slider("Decision Boundary", 0.05, 0.95, 0.50, step=0.05, 
                                  help="Lower threshold = Higher Recall (Flag more delays). Higher threshold = Higher Precision (Fewer false alarms).")
            
        st.markdown("<br><br><br><div style='text-align: center; color: #94A3B8; font-size: 0.8rem;'>AeroIntel Analytics Engine v2.0</div>", unsafe_allow_html=True)

    # DATA PROCESSING & MODELING
    with st.spinner("Initializing ML Pipeline & Hyperparameter Tuning... (This takes 10-20 seconds for CV)"):
        df = load_data(task_type, uploaded_file)
        if df.empty: return
        
        X, y = prepare_xy(df, task_type)
        if X is None: return
        original_features = X.columns.tolist()
        
        if "Classification" in task_type:
            raw_results, models, X_train, X_test = train_classification_models(X, y)
            
            # Apply dynamic threshold for classification
            test_metrics = {}
            for name, res in raw_results.items():
                test_metrics[name] = get_class_metrics(res['y_test_true'], res['y_prob_test'], threshold)
                
        else:
            raw_results, models, X_train, X_test = train_regression_models(X, y)
            test_metrics = raw_results # For regression, test metrics are already in raw_results

    # DETERMINE WINNER AND DISPLAY
    if "Classification" in task_type:
        best_model_name = max(test_metrics.keys(), key=lambda k: test_metrics[k]['f1'])
        primary_metric_label = "F1-Score (Test)"
        primary_metric_val = test_metrics[best_model_name]['f1']
    else:
        best_model_name = max(test_metrics.keys(), key=lambda k: raw_results[k]['test_r2'])
        primary_metric_label = "R² Score (Test)"
        primary_metric_val = raw_results[best_model_name]['test_r2']

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "📋 Detailed Metrics", "📈 Visual Insights", "💡 Business Explanation"])
    
    with tab1:
        c1, c2, c3 = st.columns([1, 1, 1])
        c1.metric("Rows Analyzed", f"{len(X):,}")
        c2.metric("Features Utilized", len(original_features))
        c3.metric(primary_metric_label, f"{primary_metric_val:.4f}", delta="Optimal", delta_color="normal")
        
        st.markdown(f"""
        <div class="winner-card" style="margin-top: 20px; margin-bottom: 20px;">
            <h3 style="margin:0; color:#10B981;">🏆 Recommended Model: {best_model_name}</h3>
            <p style="margin:5px 0 0 0; color:#E6EDF3;">Based on current inputs and comprehensive holdout testing, <strong>{best_model_name}</strong> yields the best balanced performance after hyperparameter tuning.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if "Classification" in task_type:
            plot_df = []
            for m in test_metrics:
                plot_df.extend([
                    {'Model': m, 'Metric': 'Accuracy', 'Score': test_metrics[m]['acc']},
                    {'Model': m, 'Metric': 'F1-Score', 'Score': test_metrics[m]['f1']},
                    {'Model': m, 'Metric': 'ROC-AUC', 'Score': test_metrics[m]['auc']}
                ])
            plot_df = pd.DataFrame(plot_df)
            fig = px.bar(plot_df, x="Metric", y="Score", color="Model", barmode="group", template="plotly_dark",
                         color_discrete_sequence=['#38BDF8', '#10B981', '#F59E0B'],
                         title=f"Classification Performance at Threshold = {threshold}")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            plot_df = []
            for m in raw_results:
                plot_df.extend([
                    {'Model': m, 'Metric': 'R² Score', 'Score': raw_results[m]['test_r2']}
                ])
            plot_df = pd.DataFrame(plot_df)
            fig = px.bar(plot_df, x="Metric", y="Score", color="Model", barmode="group", template="plotly_dark",
                         color_discrete_sequence=['#38BDF8', '#10B981', '#F59E0B'],
                         title="Regression R² Performance (Higher is Better)")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Cross-Validation & Test Metrics Matrix *(After HP Tuning)*")
        
        if "Classification" in task_type:
            metrics_data = []
            for m in test_metrics:
                metrics_data.append({
                    'Model Name': m,
                    'CV Accuracy': f"{raw_results[m]['cv_metrics']['accuracy']:.3f}",
                    'Test Accuracy': f"{test_metrics[m]['acc']:.3f}",
                    'Test Precision': f"{test_metrics[m]['prec']:.3f}",
                    'Test Recall': f"{test_metrics[m]['rec']:.3f}",
                    'Test F1-Score': f"{test_metrics[m]['f1']:.3f}",
                    'CV ROC-AUC': f"{raw_results[m]['cv_metrics']['roc_auc']:.3f}"
                })
            df_res = pd.DataFrame(metrics_data)
            st.dataframe(df_res, use_container_width=True)
            
            st.markdown("### Confusion Matrices")
            c1, c2, c3 = st.columns(3)
            for idx, (m, col) in enumerate(zip(test_metrics.keys(), [c1, c2, c3])):
                with col:
                    st.markdown(f"**{m}**")
                    cm = test_metrics[m]['cm']
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                    labels=dict(x="Predicted", y="Actual"),
                                    x=['On-Time (0)', 'Delayed (1)'], y=['On-Time (0)', 'Delayed (1)'])
                    fig.update_layout(coloraxis_showscale=False, margin=dict(l=20, r=20, t=20, b=20),
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True, key=f"cm_{idx}")
        else:
            metrics_data = []
            for m in raw_results:
                metrics_data.append({
                    'Model Name': m,
                    'CV R²': float(f"{raw_results[m]['cv_metrics']['r2']:.3f}"),
                    'Test R²': float(f"{raw_results[m]['test_r2']:.3f}"),
                    'CV MAE': float(f"{-raw_results[m]['cv_metrics']['neg_mean_absolute_error']:.1f}"),
                    'Test MAE': float(f"{raw_results[m]['test_mae']:.1f}"),
                    'CV RMSE': float(f"{-raw_results[m]['cv_metrics']['neg_root_mean_squared_error']:.1f}"),
                    'Test RMSE': float(f"{raw_results[m]['test_rmse']:.1f}")
                })
            df_res = pd.DataFrame(metrics_data)
            st.dataframe(df_res.style.highlight_max(subset=['Test R²'], color='#10B981', axis=0)\
                                   .highlight_min(subset=['Test RMSE'], color='#10B981', axis=0), 
                         use_container_width=True)

    with tab3:
        if "Classification" in task_type:
            st.plotly_chart(plot_roc_curves(raw_results), use_container_width=True)
            st.plotly_chart(plot_feature_importance(models, original_features), use_container_width=True)
        else:
            st.plotly_chart(plot_residuals(raw_results), use_container_width=True)
            st.plotly_chart(plot_feature_importance(models, original_features), use_container_width=True)

    with tab4:
        st.markdown("### 🧠 Automated Insights")
        if "Classification" in task_type:
            insights = generate_classification_insights(raw_results, test_metrics)
        else:
            insights = generate_regression_insights(raw_results)
            
        for insight in insights:
            st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
            
        st.markdown("### 💼 ML Engineering Exegesis")
        st.markdown(f"""
        **Why {best_model_name}?**
        Tree-based Gradient Boosting frameworks natively handle the complex, non-linear feature spaces prevalent in aviation operations (e.g., the compounding effect of time-of-day and specific flight routes). 
        
        **Model Trade-offs:**
        - **LightGBM**: Highly memory efficient due to histogram-based decision tree splitting. Best for low latency (production inference) and scales optimally with massive airline datasets.
        - **XGBoost**: Employs deep regularisation, preventing overfitting on noisy datasets like pricing, but typically requires more exhaustive hyperparameter tuning and computation time.
        - **Linear Models (Baselines)**: Highly interpretable and fast, but severely under-fit complex feature interactions. Primarily utilized as an algorithmic benchmark to measure the non-linearity "premium" provided by gradient boosters.
        """)
        
        # Download logic
        import json
        metrics_dict = {
            "task": task_type,
            "best_model": best_model_name,
            "metrics": metrics_data
        }
        json_str = json.dumps(metrics_dict, indent=4)
        
        st.download_button(
            label="⬇️ Download Evaluation Report (JSON)",
            file_name="model_comparison_report.json",
            mime="application/json",
            data=json_str
        )

if __name__ == "__main__":
    main()
