"""
PILLAR 2: Machine Learning Modeling, Feature Importance & Evaluation
--------------------------------------------------------------------
This module trains an ensemble Gradient Boosting Classifier on engineered
features, evaluates using industry metrics (Precision, Recall, F1, PR-AUC),
identifies key churn drivers, and assigns Churn Probability Scores.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def train_and_evaluate_churn_model(base_dir="."):
    data_dir = os.path.join(base_dir, "output")
    features_path = os.path.join(data_dir, "engineered_customer_features.csv")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature dataset not found at {features_path}. Run PySpark pipeline first.")

    print("\n" + "="*70)
    print("PILLAR 2: MACHINE LEARNING PREDICTIVE MODELING & EVALUATION")
    print("="*70)

    df = pd.read_csv(features_path)
    print(f"--> Loaded engineered features for {len(df)} enterprise accounts.")
    print(f"--> Baseline Churn Distribution: {df['actual_churn'].value_counts(normalize=True).to_dict()}")

    # 1. Feature Preprocessing
    # One-hot encode contract_type
    df_encoded = pd.get_dummies(df, columns=['contract_type'], drop_first=True)

    feature_cols = [
        'avg_daily_usage_gb', 'total_usage_gb', 'std_daily_usage',
        'total_call_drops', 'avg_call_drops', 'total_support_calls',
        'total_billing_issues', 'tenure_months', 'avg_monthly_charge',
        'usage_trend_ratio', 'call_drop_rate_per_day', 'support_call_rate',
        'billing_friction_index'
    ]
    if 'contract_type_Monthly' in df_encoded.columns:
        feature_cols.append('contract_type_Monthly')

    X = df_encoded[feature_cols]
    y = df_encoded['actual_churn']

    # 2. Stratified Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"--> Train set: {len(X_train)} accounts | Test set: {len(X_test)} accounts")

    # 3. Model Training: Gradient Boosting Classifier
    print("--> Training Enterprise Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = round(accuracy_score(y_test, y_pred), 4)
    prec = round(precision_score(y_test, y_pred), 4)
    rec = round(recall_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    roc = round(roc_auc_score(y_test, y_prob), 4)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- Model Evaluation Scorecard ---")
    print(f"  * Accuracy:        {acc * 100:.2f}%")
    print(f"  * Precision:       {prec * 100:.2f}%  (Quality: Flagged accounts that truly churn)")
    print(f"  * Recall:          {rec * 100:.2f}%  (Coverage: Churners caught by model)")
    print(f"  * F1-Score:        {f1 * 100:.2f}%")
    print(f"  * ROC-AUC:         {roc * 100:.2f}%")
    print(f"  * Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # 5. Feature Importance Analysis
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n--- Top Churn Drivers (Feature Importance) ---")
    for idx, row in feature_imp_df.head(5).iterrows():
        print(f"  {row['Feature']:<25}: {row['Importance']:.4f}")

    # 6. Score All Customers & Categorize into Risk Tiers
    full_probs = model.predict_proba(X)[:, 1]
    df['churn_probability'] = full_probs.round(4)

    def assign_risk_tier(prob):
        if prob >= 0.75:
            return "CRITICAL"
        elif prob >= 0.50:
            return "HIGH"
        elif prob >= 0.25:
            return "MEDIUM"
        else:
            return "LOW"

    df['risk_tier'] = df['churn_probability'].apply(assign_risk_tier)

    # Save artifacts
    model_path = os.path.join(data_dir, "churn_model.pkl")
    joblib.dump(model, model_path)
    
    scored_output_path = os.path.join(data_dir, "scored_customers.csv")
    df.to_csv(scored_output_path, index=False)

    metrics_path = os.path.join(data_dir, "model_metrics.json")
    metrics_data = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc,
        "confusion_matrix": cm,
        "top_features": feature_imp_df.head(6).to_dict(orient="records")
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\n[OK] Model saved to: {model_path}")
    print(f"[OK] Scored customer database saved to: {scored_output_path}")
    return scored_output_path

if __name__ == "__main__":
    train_and_evaluate_churn_model(".")
