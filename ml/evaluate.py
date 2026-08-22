"""
RazorPay Sentinel — Model Evaluation Report Generator
======================================================
Generates comprehensive evaluation reports for both models.

Produces:
  - Full metrics comparison
  - Confusion matrices
  - Calibration analysis
  - Threshold sweep analysis
  - Feature importance comparison

ALL METRICS ARE SYNTHETIC DEVELOPMENT METRICS.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    log_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "disputes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def load_data_and_models():
    """Load data, models, and metadata."""
    # Load metadata
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "r") as f:
        metadata = json.load(f)

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Feature engineering (same as train.py)
    total_evidence = df["evidence_items_available"] + df["evidence_items_missing"]
    df["evidence_completeness"] = (
        df["evidence_items_available"] / total_evidence
    ).round(4)

    dummies = pd.get_dummies(df["dispute_reason"], prefix="reason", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    feature_cols = metadata["features"]
    X = df[feature_cols].values
    y = df["contest_success"].values

    # Reproduce the same split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Load models
    lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.joblib"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

    return {
        "df": df,
        "feature_cols": feature_cols,
        "metadata": metadata,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "lr_model": lr_model,
        "xgb_model": xgb_model,
        "scaler": scaler,
    }


def compute_calibration(y_true, y_prob, n_bins=10):
    """Compute calibration curve data."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return {
        "fraction_of_positives": [round(float(x), 4) for x in fraction_of_positives],
        "mean_predicted_value": [round(float(x), 4) for x in mean_predicted_value],
        "n_bins": n_bins,
    }


def compute_detailed_metrics(y_true, y_prob, threshold, label):
    """Compute comprehensive metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Precision-Recall curve
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    return {
        "label": label,
        "data_type": "SYNTHETIC DEVELOPMENT METRICS",
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "total_samples": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
        "predicted_positive_rate": round(float(y_pred.mean()), 4),
        "calibration": compute_calibration(y_true, y_prob),
    }


def compute_feature_importance_lr(model, scaler, feature_names):
    """Extract feature importances from Logistic Regression coefficients."""
    coefficients = model.coef_[0]
    importances = []
    for fname, coeff in zip(feature_names, coefficients):
        importances.append({
            "feature": fname,
            "coefficient": round(float(coeff), 4),
            "abs_importance": round(float(abs(coeff)), 4),
            "direction": "positive" if coeff > 0 else "negative",
        })
    importances.sort(key=lambda x: x["abs_importance"], reverse=True)
    return importances


def compute_feature_importance_xgb(model, feature_names):
    """Extract feature importances from XGBoost."""
    importances_arr = model.feature_importances_
    importances = []
    for fname, imp in zip(feature_names, importances_arr):
        importances.append({
            "feature": fname,
            "importance": round(float(imp), 4),
        })
    importances.sort(key=lambda x: x["importance"], reverse=True)
    return importances


def generate_full_report():
    """Generate the complete evaluation report."""
    print("=" * 60)
    print("RazorPay Sentinel -- Full Evaluation Report")
    print("=" * 60)

    data = load_data_and_models()
    metadata = data["metadata"]
    feature_cols = data["feature_cols"]

    # Get probabilities on test set
    lr_test_prob = data["lr_model"].predict_proba(
        data["scaler"].transform(data["X_test"])
    )[:, 1]
    xgb_test_prob = data["xgb_model"].predict_proba(data["X_test"])[:, 1]

    # Load thresholds from comparison file
    comparison_path = os.path.join(REPORTS_DIR, "model_comparison.json")
    with open(comparison_path, "r") as f:
        comparison = json.load(f)

    lr_threshold = comparison["models"]["logistic_regression"]["optimal_threshold"]
    xgb_threshold = comparison["models"]["xgboost"]["optimal_threshold"]

    # Compute detailed metrics
    print("\nComputing detailed metrics...")
    lr_detailed = compute_detailed_metrics(
        data["y_test"], lr_test_prob, lr_threshold, "Logistic Regression"
    )
    xgb_detailed = compute_detailed_metrics(
        data["y_test"], xgb_test_prob, xgb_threshold, "XGBoost"
    )

    # Feature importances
    print("Computing feature importances...")
    lr_importances = compute_feature_importance_lr(
        data["lr_model"], data["scaler"], feature_cols
    )
    xgb_importances = compute_feature_importance_xgb(
        data["xgb_model"], feature_cols
    )

    # Build full report
    report = {
        "project": "RazorPay Sentinel",
        "report_type": "Full Model Evaluation",
        "data_type": "SYNTHETIC DEVELOPMENT METRICS",
        "warning": (
            "These metrics are derived from SYNTHETIC data and do NOT represent "
            "production performance. Do not present these as production benchmarks."
        ),
        "evaluation_date": datetime.now().isoformat(),
        "dataset": {
            "total_samples": int(len(data["df"])),
            "train_samples": int(len(data["y_train"])),
            "validation_samples": int(len(data["y_val"])),
            "test_samples": int(len(data["y_test"])),
            "overall_positive_rate": round(float(data["df"]["contest_success"].mean()), 4),
        },
        "models": {
            "logistic_regression": {
                "metrics": lr_detailed,
                "feature_importances": lr_importances,
            },
            "xgboost": {
                "metrics": xgb_detailed,
                "feature_importances": xgb_importances,
            },
        },
        "winner": metadata["winner"],
        "features_used": feature_cols,
    }

    # Save report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "full_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY (SYNTHETIC DEVELOPMENT METRICS)")
    print(f"{'='*50}")
    print(f"\n{'Metric':<20} {'LR':>12} {'XGBoost':>12} {'Winner':>10}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*10}")
    for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "log_loss"]:
        lr_val = lr_detailed[metric]
        xgb_val = xgb_detailed[metric]
        lr_wins = (lr_val > xgb_val if metric not in ("brier_score", "log_loss") else lr_val < xgb_val)
        winner_label = "LR" if lr_wins else "XGBoost"
        print(f"{metric:<20} {lr_val:>12.4f} {xgb_val:>12.4f} {winner_label:>10}")

    print(f"\nWinner: {metadata['winner']}")
    print(f"\nTop 5 features (LR by |coefficient|):")
    for feat in lr_importances[:5]:
        print(f"  {feat['direction']:>8} {feat['feature']}: {feat['abs_importance']:.4f}")

    print(f"\nTop 5 features (XGBoost by importance):")
    for feat in xgb_importances[:5]:
        print(f"  {feat['feature']}: {feat['importance']:.4f}")

    print(f"\n[OK] Full report saved -> {report_path}")
    print(f"\nNOTE: ALL METRICS ARE SYNTHETIC DEVELOPMENT METRICS")

    return report


if __name__ == "__main__":
    generate_full_report()
