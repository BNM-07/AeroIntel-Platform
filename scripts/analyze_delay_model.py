import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

def analyze_thresholds():
    # Load test results
    try:
        results = joblib.load('models/delay_test_results.pkl')
        y_true = results['y_true']
        y_prob = results['y_prob_xgb'] # Prioritize XGBoost for analysis
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    thresholds = np.linspace(0.01, 0.99, 100)
    metrics = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics.append({'threshold': thresh, 'precision': prec, 'recall': rec, 'f1': f1})

    df_metrics = pd.DataFrame(metrics)

    # Plot 1: Precision vs Recall
    plt.figure(figsize=(10, 6))
    plt.plot(df_metrics['recall'], df_metrics['precision'], marker='.', label='XGBoost')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig('visualizations/optimized_pr_curve.png')
    plt.close()

    # Plot 2: Threshold vs Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(df_metrics['threshold'], df_metrics['precision'], label='Precision')
    plt.plot(df_metrics['threshold'], df_metrics['recall'], label='Recall')
    plt.plot(df_metrics['threshold'], df_metrics['f1'], label='F1-score')
    plt.axvline(x=0.2, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Decision Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/threshold_metrics_plot.png')
    plt.close()

    # Identify best threshold for 93-95% recall
    target_recall = df_metrics[(df_metrics['recall'] >= 0.93) & (df_metrics['recall'] <= 0.95)]
    if not target_recall.empty:
        best_row = target_recall.sort_values(by='f1', ascending=False).iloc[0]
        print(f"\n--- Optimal Threshold Findings ---")
        print(f"Threshold: {best_row['threshold']:.4f}")
        print(f"Precision: {best_row['precision']:.4f}")
        print(f"Recall:    {best_row['recall']:.4f}")
        print(f"F1-score:  {best_row['f1']:.4f}")
    else:
        print("\nCould not find a threshold in the exact 93-95% recall range. Please check the plots.")

if __name__ == "__main__":
    analyze_thresholds()
