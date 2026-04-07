import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def validate_model():
    print("Loading data...")
    df = pd.read_csv('data/emirates_delay.csv')
    X = df.drop('delay', axis=1)
    y = df['delay']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        'SMOTE_XGB': [],
        'Weighted_XGB': [],
        'SMOTE_LGB': [],
        'Weighted_LGB': []
    }
    
    # Target recall range: 93-95%
    # We will search for a threshold after each fold or use a fixed threshold of 0.3 for comparison
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 1. SMOTE XGBoost
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        xgb_smote = xgb.XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=5, random_state=42)
        xgb_smote.fit(X_resampled, y_resampled)
        y_prob = xgb_smote.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results['SMOTE_XGB'].append(auc)
        
        # 2. Weighted XGBoost
        # scale_pos_weight = count(negative) / count(positive)
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_weighted = xgb.XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=5, scale_pos_weight=ratio, random_state=42)
        xgb_weighted.fit(X_train, y_train)
        y_prob_w = xgb_weighted.predict_proba(X_test)[:, 1]
        auc_w = roc_auc_score(y_test, y_prob_w)
        results['Weighted_XGB'].append(auc_w)
        
        print(f"Fold {fold+1}: SMOTE XGB AUC: {auc:.4f}, Weighted XGB AUC: {auc_w:.4f}")

    print("\n--- Cross-Validation Results (ROC-AUC) ---")
    for key, values in results.items():
        if values:
            print(f"{key}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}")

if __name__ == "__main__":
    validate_model()
