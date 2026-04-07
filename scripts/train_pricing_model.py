import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_regression(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {model_name} Regression Metrics ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

def plot_feature_importance(model, feature_names, title, filename):
    try:
        importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            
        if importance is not None:
            data = pd.DataFrame({'feature': feature_names, 'importance': importance})
            data = data.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=data, palette='magma')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            print(f"Saved feature importance plot to {filename}")
    except Exception as e:
        print(f"Failed to plot feature importance: {e}")

def train_pricing_model():
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("Loading pricing data...")
    df = pd.read_csv('data/emirates_pricing.csv')
    
    categorical_cols = ['source_city', 'destination_city', 'cabin_class', 'season', 'demand', 'stops']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    X = df.drop('price', axis=1)
    y = df['price']
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    metrics_summary = {}

    # K-Fold CV specifically for XGBoost
    print("\nEvaluating K-Fold Cross Validation for XGBoost...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = []
    for train_idx, val_idx in kf.split(X):
        cv_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
        cv_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = cv_model.predict(X.iloc[val_idx])
        cv_rmse.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
    mean_cv_rmse = np.mean(cv_rmse)
    print(f"XGBoost Mean CV RMSE: {mean_cv_rmse:.2f}")

    # 1. LightGBM
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.08, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42)
    lgb_model.fit(X_train, y_train)
    metrics_summary['LightGBM'] = evaluate_regression(y_test, lgb_model.predict(X_test), "LightGBM")
    joblib.dump(lgb_model, 'models/price_model_lgb.pkl')
    
    # 2. XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(X_train, y_train)
    metrics_summary['XGBoost'] = evaluate_regression(y_test, xgb_model.predict(X_test), "XGBoost")
    metrics_summary['XGBoost']['cv_rmse'] = mean_cv_rmse
    joblib.dump(xgb_model, 'models/price_model_xgb.pkl')
    
    # Preprocessor & Metadata
    joblib.dump(label_encoders, 'models/pricing_preprocessor.pkl')
    joblib.dump(scaler, 'models/pricing_scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names_pricing.pkl')
    with open('models/pricing_metrics.json', 'w') as f:
        json.dump(metrics_summary, f)
    
    plot_feature_importance(xgb_model, feature_names, 'Pricing Predictor Feature Importance (XGBoost)', 'visualizations/price_feature_importance.png')
    
    print("\nSaving SHAP Explainer (Sampled for speed)...")
    try:
        background_data = shap.sample(X_train, 100)
        explainer_price = shap.TreeExplainer(lgb_model, background_data)
        joblib.dump(explainer_price, 'models/explainer_price_v2.pkl')
    except Exception as e:
        print(f"Failed SHAP explainer computation: {e}")

if __name__ == "__main__":
    train_pricing_model()
