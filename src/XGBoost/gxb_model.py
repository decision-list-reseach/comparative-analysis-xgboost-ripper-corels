import pandas as pd
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import argparse

# ---------------------------------------------------------------------------
# Paths & shared utilities
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from utils import save_scores

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ecommerce", choices=["ecommerce", "telco"])
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Data Loading & Minimal Preprocessing
# ---------------------------------------------------------------------------
if args.dataset == "ecommerce":
    csv_path = os.path.join(project_root, 'data', 'ecommerce', 'data_ecommerce_customer_churn.csv')
    df = pd.read_csv(csv_path)
    df = pd.get_dummies(df, columns=["PreferedOrderCat", "MaritalStatus"])
else:
    csv_path = os.path.join(project_root, 'data', 'telco', 'TEST_telco_customer_churn.csv')
    df = pd.read_csv(csv_path)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(r'^\s*$', 'NaN', regex=True), errors='coerce')
    if 'Churn' in df.columns and set(df['Churn'].dropna().unique()).issubset({'Yes', 'No'}):
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    nominal_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", 
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
                    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
    df = pd.get_dummies(df, columns=[c for c in nominal_cols if c in df.columns])

y = df['Churn'].values
X = df.drop('Churn', axis=1).values

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def make_xgb(scale_pos_weight):
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.2,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=1,
        random_state=42
    )

# ---------------------------------------------------------------------------
# 5-fold stratified cross-validation
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

br_values       = []
br_platt_values = []
br_iso_values   = []

# Accumulate out-of-fold predictions for the pooled reliability diagram
oof_y_true     = []
oof_prob_orig  = []
oof_prob_platt = []
oof_prob_iso   = []

print("=" * 60)
print(f"XGBoost  —  5-Fold Stratified Cross-Validation ({args.dataset})")
print("=" * 60)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    dynamic_scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    xgb = make_xgb(dynamic_scale_pos_weight)

    start = time.perf_counter()
    xgb.fit(X_train, y_train)
    end   = time.perf_counter()
    time_needed = end - start

    preds = xgb.predict(X_test)

    # --- Classification metrics (logged to CSV) ---
    utils.save_scores("scores_from_xgboost", current_dir, y_test, preds, time_needed, dataset=args.dataset)

    # --- Brier score (uncalibrated) ---
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    br_values.append(brier_score_loss(y_test, y_prob_xgb))

    # --- Calibrators fitted on the held-out fold ---
    platt_calibrator = CalibratedClassifierCV(xgb, method='sigmoid',  cv='prefit')
    iso_calibrator   = CalibratedClassifierCV(xgb, method='isotonic', cv='prefit')
    platt_calibrator.fit(X_test, y_test)
    iso_calibrator.fit(X_test, y_test)

    prob_platt = platt_calibrator.predict_proba(X_test)[:, 1]
    prob_iso   = iso_calibrator.predict_proba(X_test)[:, 1]

    br_platt_values.append(brier_score_loss(y_test, prob_platt))
    br_iso_values.append(brier_score_loss(y_test, prob_iso))

    # --- Accumulate for pooled reliability diagram ---
    oof_y_true.extend(y_test)
    oof_prob_orig.extend(y_prob_xgb)
    oof_prob_platt.extend(prob_platt)
    oof_prob_iso.extend(prob_iso)

    print(f"  Fold {fold}/5 | time: {time_needed:.2f}s"
          f" | Brier: {br_values[-1]:.4f}"
          f" | Platt: {br_platt_values[-1]:.4f}"
          f" | Isotonic: {br_iso_values[-1]:.4f}")

# ---------------------------------------------------------------------------
# Summary statistics across folds
# ---------------------------------------------------------------------------
brier_mean       = np.mean(br_values)
brier_std        = np.std(br_values)
brier_platt_mean = np.mean(br_platt_values)
brier_platt_std  = np.std(br_platt_values)
brier_iso_mean   = np.mean(br_iso_values)
brier_iso_std    = np.std(br_iso_values)

print()
print("=" * 60)
print("CV Summary (mean ± std over 5 folds)")
print("=" * 60)
print(f"  Original Brier : {brier_mean:.4f} ± {brier_std:.4f}")
print(f"  Platt Brier    : {brier_platt_mean:.4f} ± {brier_platt_std:.4f}")
print(f"  Isotonic Brier : {brier_iso_mean:.4f} ± {brier_iso_std:.4f}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Reliability diagram — built from pooled OOF predictions
# ---------------------------------------------------------------------------
oof_y_true     = np.array(oof_y_true)
oof_prob_orig  = np.array(oof_prob_orig)
oof_prob_platt = np.array(oof_prob_platt)
oof_prob_iso   = np.array(oof_prob_iso)

prob_true_orig,  prob_pred_orig  = calibration_curve(oof_y_true, oof_prob_orig,  n_bins=10)
prob_true_platt, prob_pred_platt = calibration_curve(oof_y_true, oof_prob_platt, n_bins=10)
prob_true_iso,   prob_pred_iso   = calibration_curve(oof_y_true, oof_prob_iso,   n_bins=10)

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated (Ideal)")
plt.plot(prob_pred_orig,  prob_true_orig,  "s-", color="red",   alpha=0.8,
         label=f"Original XGBoost (Brier: {brier_mean:.4f} ± {brier_std:.4f})")
plt.plot(prob_pred_platt, prob_true_platt, "^-", color="blue",  alpha=0.8,
         label=f"Platt Scaling (Brier: {brier_platt_mean:.4f} ± {brier_platt_std:.4f})")
plt.plot(prob_pred_iso,   prob_true_iso,   "o-", color="green", alpha=0.8,
         label=f"Isotonic Regression (Brier: {brier_iso_mean:.4f} ± {brier_iso_std:.4f})")

plt.ylabel("Actual Fraction of Churners (True Probability)", fontsize=12)
plt.xlabel("Mean Predicted Probability (Model Confidence)", fontsize=12)
plt.title(
    "Reliability Diagram: Original vs. Calibrated XGBoost\n"
    "(5-Fold Stratified CV, Pooled OOF Predictions)",
    fontsize=14, pad=15
)
plt.legend(loc="upper left", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig("calibration_comparison.png", dpi=300, bbox_inches='tight')
plt.show()