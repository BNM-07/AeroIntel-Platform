import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import shap
import os
from preprocessing import AirlineDataPreprocessor, load_and_split_data

os.makedirs('models', exist_ok=True)

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print("\n--- Regression Metrics (Ticket Price) ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def evaluate_classification(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print("\n--- Classification Metrics (Flight Delay) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc}

def train_models():
    print("Loading and splitting data...")
    X_train, X_test, yp_train, yp_test, yd_train, yd_test = load_and_split_data()
    
    print("Preprocessing data...")
    preprocessor = AirlineDataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.pkl')
    
    # Ensure ordered columns are saved
    feature_names = X_train_proc.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print(f"Training shapes -> X: {X_train_proc.shape}, y_price: {yp_train.shape}, y_delay: {yd_train.shape}")
    
    # ==========================
    # 1. Price Prediction Model (Regressor)
    # ==========================
    print("\nTraining XGBoost Regressor (Price)...")
    price_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    price_model.fit(X_train_proc, yp_train)
    
    yp_pred = price_model.predict(X_test_proc)
    evaluate_regression(yp_test, yp_pred)
    joblib.dump(price_model, 'models/price_model.pkl')
    
    # ==========================
    # 2. Delay Prediction Model (Classifier)
    # ==========================
    print("\nTraining XGBoost Classifier (Delay)...")
    delay_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    delay_model.fit(X_train_proc, yd_train)
    
    yd_pred = delay_model.predict(X_test_proc)
    yd_prob = delay_model.predict_proba(X_test_proc)[:, 1]
    evaluate_classification(yd_test, yd_pred, yd_prob)
    joblib.dump(delay_model, 'models/delay_model.pkl')
    
    # ==========================
    # 3. SHAP Feature Importance Initialization
    # ==========================
    print("\nSaving SHAP Explainer (Sampled for speed)...")
    try:
        # Sample background data for TreeExplainer
        background_data = shap.sample(X_train_proc, 100)
        explainer_price = shap.TreeExplainer(price_model, background_data)
        joblib.dump(explainer_price, 'models/explainer_price.pkl')
        print("SHAP explainers saved successfully.")
    except Exception as e:
        print(f"Warning: Could not save SHAP explainer natively. It can be computed on the fly. Error: {e}")
        
    print("\nModel training workflow complete!")

if __name__ == "__main__":
    train_models()
