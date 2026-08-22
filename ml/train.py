"""
RazorPay Sentinel -- ML Training Pipeline
==========================================
Trains and compares two models for contest-success prediction:
  1. Logistic Regression (baseline) -- interpretable coefficients
  2. XGBoost -- non-linear interactions + SHAP

Target: contest_success (binary)
  1 = dispute FULLY REVERSED in merchant's favor
  0 = otherwise (lost, partial recovery, or accepted)

Architecture separation (CRITICAL):
  - THIS MODULE: trains ML models that output P(success) + reason codes
  - SEPARATE MODULE (decision_policy.py): uses P(success) + evidence + economics
    to decide ACCEPT / ESCALATE / CONTEST
  - The ML model NEVER decides the action. The policy does.

Workflow:
  1. Load data/disputes.csv
  2. Feature engineering
  3. Stratified Train/Val/Test split (60/20/20)
  4. Train both models on Train set
  5. Measure calibration on Validation set; recalibrate if needed
  6. Tune thresholds on Validation set (cost-sensitive)
  7. Multi-criteria model selection on Validation set
  8. Final evaluation on Held-out Test set (one-time, never used for tuning)
  9. Save winning model + metadata + reports

Model selection criteria (in order of priority):
  1. Calibration quality (Brier score, reliability diagram ECE)
  2. Cost-sensitive decision performance (total expected cost on validation set)
  3. Discrimination (ROC-AUC, PR-AUC)
  NOT simply "higher AUC wins."

ALL METRICS ARE SYNTHETIC DEVELOPMENT METRICS.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "disputes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# --- Cost parameters (configurable, used ONLY for threshold tuning, not model training) ---
DEFAULT_COSTS = {
    "contest_operational_cost": 500,       # cost to run contest process
    "false_contest_cost": 1000,            # wasted effort + reputation risk
    "missed_recovery_cost_multiplier": 1.0,  # multiplied by transaction_amount
    "escalation_cost": 200,                # cost of human review
}

# Feature columns used by the model
# NOTE: the model outputs P(success). The decision policy (separate module)
# uses this probability along with evidence and economics to decide actions.
NUMERIC_FEATURES = [
    "transaction_amount",
    "customer_order_count",
    "previous_refunds",
    "previous_disputes",
    "delivery_confirmed",
    "delivery_delay_days",
    "dispute_delay_days",
    "customer_avg_order_value",
    "communication_count",
    "refund_amount_ratio",
    "payment_failures",
    "evidence_items_available",
    "evidence_items_missing",
    "evidence_completeness",
]

CATEGORICAL_FEATURES = ["dispute_reason"]

TARGET = "contest_success"

# Calibration threshold: if ECE exceeds this, recalibrate
ECE_RECALIBRATION_THRESHOLD = 0.05


def load_and_prepare_data(path: str) -> pd.DataFrame:
    """Load CSV and engineer features."""
    df = pd.read_csv(path)

    # Feature engineering
    total_evidence = df["evidence_items_available"] + df["evidence_items_missing"]
    df["evidence_completeness"] = (
        df["evidence_items_available"] / total_evidence
    ).round(4)

    # One-hot encode dispute_reason
    dummies = pd.get_dummies(df["dispute_reason"], prefix="reason", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get the full list of feature columns (numeric + one-hot encoded)."""
    reason_cols = sorted([c for c in df.columns if c.startswith("reason_")])
    return NUMERIC_FEATURES + reason_cols


def split_data(df: pd.DataFrame, feature_cols: list):
    """Stratified 60/20/20 split."""
    X = df[feature_cols].values
    y = df[TARGET].values

    # First split: 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    # Second split: 50% of temp -> 20% val, 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"  Train: {X_train.shape[0]} rows")
    print(f"  Val:   {X_val.shape[0]} rows")
    print(f"  Test:  {X_test.shape[0]} rows")
    print(f"  Train success rate: {y_train.mean():.2%}")
    print(f"  Val success rate:   {y_val.mean():.2%}")
    print(f"  Test success rate:  {y_test.mean():.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_logistic_regression(X_train, y_train, scaler=None):
    """Train L2-regularized Logistic Regression."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
    else:
        X_scaled = scaler.transform(X_train)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def train_xgboost(X_train, y_train):
    """Train XGBoost classifier."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, verbose=False)
    return model


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum over bins of (bin_weight * |bin_accuracy - bin_confidence|)

    Lower is better. A perfectly calibrated model has ECE = 0.
    Do NOT assume any model is calibrated without measuring this.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge in last bin
            mask = mask | (y_prob == bin_edges[i + 1])

        bin_count = mask.sum()
        if bin_count == 0:
            continue

        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_prob[mask].mean()
        bin_weight = bin_count / len(y_true)
        bin_error = abs(bin_accuracy - bin_confidence)
        ece += bin_weight * bin_error

        bin_details.append({
            "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
            "count": int(bin_count),
            "accuracy": round(float(bin_accuracy), 4),
            "confidence": round(float(bin_confidence), 4),
            "error": round(float(bin_error), 4),
        })

    return round(float(ece), 4), bin_details


def measure_calibration(y_true, y_prob, label=""):
    """
    Full calibration analysis for a model.

    Returns Brier score, ECE, and reliability diagram data.
    """
    brier = brier_score_loss(y_true, y_prob)
    ece, bin_details = compute_ece(y_true, y_prob)

    # Reliability diagram data (for plotting)
    try:
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="uniform"
        )
        reliability = {
            "fraction_of_positives": [round(float(x), 4) for x in fraction_pos],
            "mean_predicted_value": [round(float(x), 4) for x in mean_pred],
        }
    except ValueError:
        reliability = {"fraction_of_positives": [], "mean_predicted_value": []}

    result = {
        "label": label,
        "brier_score": round(float(brier), 4),
        "ece": ece,
        "needs_recalibration": ece > ECE_RECALIBRATION_THRESHOLD,
        "recalibration_threshold": ECE_RECALIBRATION_THRESHOLD,
        "bin_details": bin_details,
        "reliability_diagram": reliability,
    }

    print(f"  {label}:")
    print(f"    Brier score: {result['brier_score']:.4f}")
    print(f"    ECE: {result['ece']:.4f} "
          f"({'NEEDS RECALIBRATION' if result['needs_recalibration'] else 'OK'})")

    return result


def recalibrate_model(model, X_val, y_val, method="isotonic", is_lr=False, scaler=None):
    """
    Apply post-hoc calibration (Platt scaling or isotonic regression).

    We use the VALIDATION set for calibration fitting, which is the correct
    approach -- calibration should never be fit on training data.
    """
    if is_lr and scaler is not None:
        X_cal = scaler.transform(X_val)
    else:
        X_cal = X_val

    calibrated = CalibratedClassifierCV(
        model, method=method, cv="prefit"
    )
    calibrated.fit(X_cal, y_val)
    return calibrated


# ---------------------------------------------------------------------------
# Cost-sensitive threshold tuning
# ---------------------------------------------------------------------------

def cost_sensitive_threshold_sweep(
    y_true, y_prob, transaction_amounts=None, costs=None
):
    """
    Sweep thresholds and select operating threshold on validation data.

    Evaluates both classification metrics (precision, recall, F1) and
    expected financial outcome.

    Selection strategy:
      - Maximize F1 with precision >= 0.70 and recall >= 0.75
      - Avoids overly aggressive low thresholds (e.g. 0.06) that yield 38% wasted contests.
    """
    if costs is None:
        costs = DEFAULT_COSTS

    thresholds = np.arange(0.05, 0.96, 0.05)
    results = []

    for thresh in thresholds:
        thresh_val = round(float(thresh), 2)
        y_pred = (y_prob >= thresh_val).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        fp_cost = fp * costs["false_contest_cost"]

        if transaction_amounts is not None:
            fn_mask = (y_true == 1) & (y_pred == 0)
            fn_cost = transaction_amounts[fn_mask].sum() * costs["missed_recovery_cost_multiplier"]
            tp_mask = (y_true == 1) & (y_pred == 1)
            gross_recovery = transaction_amounts[tp_mask].sum()
        else:
            fn_cost = fn * costs["missed_recovery_cost_multiplier"] * 5000
            gross_recovery = tp * 5000

        total_cost = fp_cost + fn_cost
        contest_ops = tp * costs["contest_operational_cost"]
        net_recovery = gross_recovery - contest_ops - fp_cost

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "threshold": thresh_val,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "fp_cost": round(float(fp_cost), 2),
            "fn_cost": round(float(fn_cost), 2),
            "total_cost": round(float(total_cost), 2),
            "gross_recovery": round(float(gross_recovery), 2),
            "net_recovery": round(float(net_recovery), 2),
        })

    results_df = pd.DataFrame(results)

    # Filter by operational constraints (precision >= 0.70, recall >= 0.75)
    candidates = results_df[
        (results_df["precision"] >= 0.70) & (results_df["recall"] >= 0.75)
    ]
    if len(candidates) == 0:
        candidates = results_df[
            (results_df["precision"] >= 0.65) & (results_df["recall"] >= 0.70)
        ]
    if len(candidates) == 0:
        candidates = results_df

    best_idx = candidates["f1"].idxmax()
    best = results_df.iloc[best_idx]

    return results_df, best


def evaluate_model(y_true, y_prob, threshold, label=""):
    """Compute full metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "label": label,
        "data_type": "SYNTHETIC DEVELOPMENT METRICS",
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "total_samples": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
    }
    return metrics


# ---------------------------------------------------------------------------
# Multi-criteria model selection
# ---------------------------------------------------------------------------

def select_winner(lr_report, xgb_report):
    """
    Multi-criteria model selection. NOT simply "higher AUC wins."

    Criteria (evaluated in order):
      1. Calibration: if one model needs recalibration and the other doesn't,
         the calibrated one has an advantage (but recalibration can fix this)
      2. Cost-sensitive performance: lower total_cost on validation set is better
      3. Discrimination: ROC-AUC as tiebreaker

    Both models are evaluated AFTER any recalibration has been applied.
    """
    print("\n  --- Multi-criteria model selection ---")

    scores = {"Logistic Regression": 0, "XGBoost": 0}
    reasons = []

    # Criterion 1: Calibration (Brier score -- lower is better)
    lr_brier = lr_report["calibration"]["brier_score"]
    xgb_brier = xgb_report["calibration"]["brier_score"]
    if lr_brier < xgb_brier:
        scores["Logistic Regression"] += 1
        reasons.append(f"  Calibration (Brier): LR {lr_brier:.4f} < XGB {xgb_brier:.4f} -> +1 LR")
    elif xgb_brier < lr_brier:
        scores["XGBoost"] += 1
        reasons.append(f"  Calibration (Brier): XGB {xgb_brier:.4f} < LR {lr_brier:.4f} -> +1 XGB")
    else:
        reasons.append(f"  Calibration (Brier): tied at {lr_brier:.4f}")

    # Criterion 2: ECE (lower is better)
    lr_ece = lr_report["calibration"]["ece"]
    xgb_ece = xgb_report["calibration"]["ece"]
    if lr_ece < xgb_ece:
        scores["Logistic Regression"] += 1
        reasons.append(f"  Calibration (ECE): LR {lr_ece:.4f} < XGB {xgb_ece:.4f} -> +1 LR")
    elif xgb_ece < lr_ece:
        scores["XGBoost"] += 1
        reasons.append(f"  Calibration (ECE): XGB {xgb_ece:.4f} < LR {lr_ece:.4f} -> +1 XGB")
    else:
        reasons.append(f"  Calibration (ECE): tied at {lr_ece:.4f}")

    # Criterion 3: Cost-sensitive decision performance (lower is better)
    lr_cost = lr_report["best_threshold_cost"]
    xgb_cost = xgb_report["best_threshold_cost"]
    if lr_cost < xgb_cost:
        scores["Logistic Regression"] += 1
        reasons.append(f"  Cost-sensitive: LR {lr_cost:,.0f} < XGB {xgb_cost:,.0f} -> +1 LR")
    elif xgb_cost < lr_cost:
        scores["XGBoost"] += 1
        reasons.append(f"  Cost-sensitive: XGB {xgb_cost:,.0f} < LR {lr_cost:,.0f} -> +1 XGB")
    else:
        reasons.append(f"  Cost-sensitive: tied at {lr_cost:,.0f}")

    # Criterion 4: ROC-AUC (higher is better) -- tiebreaker
    lr_auc = lr_report["val_metrics"]["roc_auc"]
    xgb_auc = xgb_report["val_metrics"]["roc_auc"]
    if lr_auc > xgb_auc:
        scores["Logistic Regression"] += 1
        reasons.append(f"  ROC-AUC: LR {lr_auc:.4f} > XGB {xgb_auc:.4f} -> +1 LR")
    elif xgb_auc > lr_auc:
        scores["XGBoost"] += 1
        reasons.append(f"  ROC-AUC: XGB {xgb_auc:.4f} > LR {lr_auc:.4f} -> +1 XGB")
    else:
        reasons.append(f"  ROC-AUC: tied at {lr_auc:.4f}")

    # Criterion 5: PR-AUC (higher is better)
    lr_prauc = lr_report["val_metrics"]["pr_auc"]
    xgb_prauc = xgb_report["val_metrics"]["pr_auc"]
    if lr_prauc > xgb_prauc:
        scores["Logistic Regression"] += 1
        reasons.append(f"  PR-AUC: LR {lr_prauc:.4f} > XGB {xgb_prauc:.4f} -> +1 LR")
    elif xgb_prauc > lr_prauc:
        scores["XGBoost"] += 1
        reasons.append(f"  PR-AUC: XGB {xgb_prauc:.4f} > LR {lr_prauc:.4f} -> +1 XGB")
    else:
        reasons.append(f"  PR-AUC: tied at {lr_prauc:.4f}")

    for r in reasons:
        print(r)

    print(f"\n  Scores: LR={scores['Logistic Regression']}, XGB={scores['XGBoost']}")

    if scores["Logistic Regression"] >= scores["XGBoost"]:
        winner = "Logistic Regression"
    else:
        winner = "XGBoost"

    print(f"  Winner: {winner}")

    return winner, scores, reasons


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RazorPay Sentinel -- ML Training Pipeline")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # --- 1. Load data ---
    print("[1/8] Loading data...")
    df = load_and_prepare_data(DATA_PATH)
    feature_cols = get_feature_columns(df)
    print(f"  Loaded {len(df)} rows, {len(feature_cols)} features")
    print(f"  Features: {feature_cols}")
    print(f"  Target: {TARGET}")
    print(f"    1 = dispute FULLY REVERSED in merchant's favor")
    print(f"    0 = otherwise (lost, partial recovery, or accepted)")
    print()

    # --- 2. Split ---
    print("[2/8] Splitting data (60/20/20 stratified)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, feature_cols)

    # Keep transaction amounts for cost-sensitive analysis
    amount_idx = feature_cols.index("transaction_amount")
    val_amounts = X_val[:, amount_idx]
    test_amounts = X_test[:, amount_idx]
    print()

    # --- 3. Train models ---
    print("[3/8] Training Logistic Regression (baseline)...")
    lr_model, scaler = train_logistic_regression(X_train, y_train)
    lr_val_prob = lr_model.predict_proba(scaler.transform(X_val))[:, 1]
    print("  Done.")
    print()

    print("[4/8] Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    print("  Done.")
    print()

    # --- 4. Calibration analysis (BEFORE threshold tuning) ---
    print("[5/8] Calibration analysis on VALIDATION set...")
    print("  (Do NOT assume any model is calibrated without measuring.)")
    lr_cal = measure_calibration(y_val, lr_val_prob, "Logistic Regression")
    xgb_cal = measure_calibration(y_val, xgb_val_prob, "XGBoost")

    # Recalibrate if needed
    lr_recalibrated = False
    xgb_recalibrated = False
    lr_calibrator = None
    xgb_calibrator = None

    if lr_cal["needs_recalibration"]:
        print(f"\n  Recalibrating LR (ECE {lr_cal['ece']:.4f} > {ECE_RECALIBRATION_THRESHOLD})...")
        lr_calibrator = recalibrate_model(
            lr_model, X_val, y_val, method="isotonic", is_lr=True, scaler=scaler
        )
        lr_val_prob_cal = lr_calibrator.predict_proba(scaler.transform(X_val))[:, 1]
        lr_cal_after = measure_calibration(y_val, lr_val_prob_cal, "LR (after recalibration)")
        if lr_cal_after["ece"] < lr_cal["ece"]:
            print(f"  Recalibration improved ECE: {lr_cal['ece']:.4f} -> {lr_cal_after['ece']:.4f}")
            lr_val_prob = lr_val_prob_cal
            lr_cal = lr_cal_after
            lr_recalibrated = True
        else:
            print(f"  Recalibration did not improve. Keeping original.")
            lr_calibrator = None

    if xgb_cal["needs_recalibration"]:
        print(f"\n  Recalibrating XGBoost (ECE {xgb_cal['ece']:.4f} > {ECE_RECALIBRATION_THRESHOLD})...")
        xgb_calibrator = recalibrate_model(
            xgb_model, X_val, y_val, method="isotonic", is_lr=False
        )
        xgb_val_prob_cal = xgb_calibrator.predict_proba(X_val)[:, 1]
        xgb_cal_after = measure_calibration(y_val, xgb_val_prob_cal, "XGB (after recalibration)")
        if xgb_cal_after["ece"] < xgb_cal["ece"]:
            print(f"  Recalibration improved ECE: {xgb_cal['ece']:.4f} -> {xgb_cal_after['ece']:.4f}")
            xgb_val_prob = xgb_val_prob_cal
            xgb_cal = xgb_cal_after
            xgb_recalibrated = True
        else:
            print(f"  Recalibration did not improve. Keeping original.")
            xgb_calibrator = None
    print()

    # --- 5. Threshold tuning on validation set (AFTER calibration) ---
    print("[6/8] Cost-sensitive threshold tuning on validation set...")
    print("  (These thresholds are for the DECISION POLICY, not the model.)")
    lr_sweep, lr_best = cost_sensitive_threshold_sweep(y_val, lr_val_prob, val_amounts)
    xgb_sweep, xgb_best = cost_sensitive_threshold_sweep(y_val, xgb_val_prob, val_amounts)

    lr_threshold = float(lr_best["threshold"])
    xgb_threshold = float(xgb_best["threshold"])

    print(f"  LR  optimal threshold: {lr_threshold:.2f} "
          f"(F1={lr_best['f1']:.4f}, cost={lr_best['total_cost']:.0f})")
    print(f"  XGB optimal threshold: {xgb_threshold:.2f} "
          f"(F1={xgb_best['f1']:.4f}, cost={xgb_best['total_cost']:.0f})")
    print()

    # Validation set metrics (for model selection -- NOT final evaluation)
    lr_val_metrics = evaluate_model(y_val, lr_val_prob, lr_threshold, "LR (validation)")
    xgb_val_metrics = evaluate_model(y_val, xgb_val_prob, xgb_threshold, "XGB (validation)")

    # --- 6. Multi-criteria model selection ---
    print("[7/8] Multi-criteria model selection...")
    print("  (NOT simply 'higher AUC wins'.)")

    lr_report = {
        "calibration": lr_cal,
        "best_threshold_cost": float(lr_best["total_cost"]),
        "val_metrics": lr_val_metrics,
        "recalibrated": lr_recalibrated,
    }
    xgb_report = {
        "calibration": xgb_cal,
        "best_threshold_cost": float(xgb_best["total_cost"]),
        "val_metrics": xgb_val_metrics,
        "recalibrated": xgb_recalibrated,
    }

    winner_name, selection_scores, selection_reasons = select_winner(lr_report, xgb_report)

    # Set up winner artifacts
    if winner_name == "XGBoost":
        winner_model = xgb_calibrator if xgb_recalibrated else xgb_model
        winner_threshold = xgb_threshold
        winner_needs_scaler = False
        winner_recalibrated = xgb_recalibrated
    else:
        winner_model = lr_calibrator if lr_recalibrated else lr_model
        winner_threshold = lr_threshold
        winner_needs_scaler = not lr_recalibrated  # calibrated model wraps the scaler
        winner_recalibrated = lr_recalibrated

    print()

    # --- 7. Final evaluation on HELD-OUT TEST SET ---
    print("[8/8] Final evaluation on HELD-OUT TEST SET (one-time, never tuned on)...")

    # Get test probabilities using the actual winning model pipeline
    if winner_name == "Logistic Regression":
        if lr_recalibrated and lr_calibrator is not None:
            lr_test_prob = lr_calibrator.predict_proba(scaler.transform(X_test))[:, 1]
        else:
            lr_test_prob = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
    else:
        lr_test_prob = lr_model.predict_proba(scaler.transform(X_test))[:, 1]

    if xgb_recalibrated and xgb_calibrator is not None:
        xgb_test_prob = xgb_calibrator.predict_proba(X_test)[:, 1]
    else:
        xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]

    lr_test_metrics = evaluate_model(y_test, lr_test_prob, lr_threshold, "Logistic Regression")
    xgb_test_metrics = evaluate_model(y_test, xgb_test_prob, xgb_threshold, "XGBoost")

    # Calibration on test set too (for reporting, not selection)
    lr_test_cal = measure_calibration(y_test, lr_test_prob, "LR (test)")
    xgb_test_cal = measure_calibration(y_test, xgb_test_prob, "XGB (test)")

    print(f"\n  --- Logistic Regression (Held-out Test) ---")
    print(f"  Threshold:  {lr_test_metrics['threshold']}")
    print(f"  Precision:  {lr_test_metrics['precision']}")
    print(f"  Recall:     {lr_test_metrics['recall']}")
    print(f"  F1:         {lr_test_metrics['f1']}")
    print(f"  ROC-AUC:    {lr_test_metrics['roc_auc']}")
    print(f"  PR-AUC:     {lr_test_metrics['pr_auc']}")
    print(f"  Brier:      {lr_test_metrics['brier_score']}")
    print(f"  ECE:        {lr_test_cal['ece']}")
    print(f"  Confusion:  {lr_test_metrics['confusion_matrix']}")
    print(f"  Recalibrated: {lr_recalibrated}")

    print(f"\n  --- XGBoost (Held-out Test) ---")
    print(f"  Threshold:  {xgb_test_metrics['threshold']}")
    print(f"  Precision:  {xgb_test_metrics['precision']}")
    print(f"  Recall:     {xgb_test_metrics['recall']}")
    print(f"  F1:         {xgb_test_metrics['f1']}")
    print(f"  ROC-AUC:    {xgb_test_metrics['roc_auc']}")
    print(f"  PR-AUC:     {xgb_test_metrics['pr_auc']}")
    print(f"  Brier:      {xgb_test_metrics['brier_score']}")
    print(f"  ECE:        {xgb_test_cal['ece']}")
    print(f"  Confusion:  {xgb_test_metrics['confusion_matrix']}")
    print(f"  Recalibrated: {xgb_recalibrated}")
    print()

    # --- 8. Save artifacts ---
    print("Saving artifacts...")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Save winning model
    model_path = os.path.join(MODELS_DIR, "risk_model.joblib")
    joblib.dump(winner_model, model_path)
    print(f"  Model saved -> {model_path}")

    # Always save scaler (needed for LR and for predict.py)
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Save both raw models + calibrators for comparison access
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "lr_model.joblib"))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.joblib"))
    if lr_calibrator is not None:
        joblib.dump(lr_calibrator, os.path.join(MODELS_DIR, "lr_calibrator.joblib"))
    if xgb_calibrator is not None:
        joblib.dump(xgb_calibrator, os.path.join(MODELS_DIR, "xgb_calibrator.joblib"))

    # Model metadata
    metadata = {
        "project": "RazorPay Sentinel",
        "data_type": "SYNTHETIC DEVELOPMENT DATA",
        "target_definition": "1 = dispute FULLY REVERSED in merchant's favor, 0 = otherwise",
        "architecture_note": (
            "The ML model outputs P(success) and reason codes ONLY. "
            "The ACCEPT/ESCALATE/CONTEST decision is made by the separate "
            "decision_policy.py module. The model NEVER decides the action."
        ),
        "winner": winner_name,
        "model_type": winner_name,
        "features": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "optimal_threshold": winner_threshold,
        "needs_scaler": winner_needs_scaler,
        "recalibrated": winner_recalibrated,
        "selection_method": "multi-criteria (calibration + cost-sensitive + discrimination)",
        "selection_scores": selection_scores,
        "selection_reasons": selection_reasons,
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "training_date": datetime.now().isoformat(),
        "cost_parameters": DEFAULT_COSTS,
        "calibration_ece_threshold": ECE_RECALIBRATION_THRESHOLD,
    }
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved -> {metadata_path}")

    # Main metrics report
    metrics_report = {
        "project": "RazorPay Sentinel",
        "data_type": "SYNTHETIC DEVELOPMENT METRICS",
        "warning": "These metrics are derived from synthetic data and do NOT represent production performance.",
        "winner": winner_name,
        "winner_metrics": lr_test_metrics if winner_name == "Logistic Regression" else xgb_test_metrics,
        "winner_calibration": lr_test_cal if winner_name == "Logistic Regression" else xgb_test_cal,
        "evaluation_date": datetime.now().isoformat(),
    }
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_report, f, indent=2)

    # Model comparison report
    comparison = {
        "project": "RazorPay Sentinel",
        "data_type": "SYNTHETIC DEVELOPMENT METRICS",
        "warning": "These metrics are derived from synthetic data and do NOT represent production performance.",
        "selection_method": "multi-criteria (NOT simply higher AUC wins)",
        "selection_criteria": [
            "1. Calibration quality (Brier score)",
            "2. Calibration quality (ECE)",
            "3. Cost-sensitive decision performance (total expected cost)",
            "4. Discrimination (ROC-AUC)",
            "5. Discrimination (PR-AUC)",
        ],
        "models": {
            "logistic_regression": {
                "test_metrics": lr_test_metrics,
                "test_calibration": lr_test_cal,
                "validation_report": lr_report,
                "optimal_threshold": lr_threshold,
                "recalibrated": lr_recalibrated,
            },
            "xgboost": {
                "test_metrics": xgb_test_metrics,
                "test_calibration": xgb_test_cal,
                "validation_report": xgb_report,
                "optimal_threshold": xgb_threshold,
                "recalibrated": xgb_recalibrated,
            },
        },
        "winner": winner_name,
        "selection_scores": selection_scores,
        "selection_reasons": selection_reasons,
        "comparison_date": datetime.now().isoformat(),
    }
    with open(os.path.join(REPORTS_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    # Threshold sweep data
    lr_sweep.to_csv(os.path.join(REPORTS_DIR, "lr_threshold_sweep.csv"), index=False)
    xgb_sweep.to_csv(os.path.join(REPORTS_DIR, "xgb_threshold_sweep.csv"), index=False)

    print(f"  Reports saved -> {REPORTS_DIR}")
    print()
    print("=" * 60)
    print("Phase 1 Training Complete")
    print(f"Winner: {winner_name}")
    print(f"Selection: multi-criteria (calibration + cost + discrimination)")
    print(f"  Scores: LR={selection_scores['Logistic Regression']}, XGB={selection_scores['XGBoost']}")
    print(f"Threshold: {winner_threshold}")
    print(f"Recalibrated: {winner_recalibrated}")
    print("NOTE: ALL METRICS ARE SYNTHETIC DEVELOPMENT METRICS")
    print("NOTE: The model outputs P(success). The decision policy decides actions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
