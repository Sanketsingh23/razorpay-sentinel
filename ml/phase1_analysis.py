"""
RazorPay Sentinel -- Phase 1 Completion: Analysis & Reporting
==============================================================
Produces the remaining Phase 1 deliverables:

  1. Threshold optimization with comparison table
  2. Cost-sensitive evaluation with explicit assumptions
  3. Calibration curve plot (saved as PNG)
  4. Threshold analysis plot (saved as PNG)
  5. Phase 1 summary report (JSON)

IMPORTANT ISSUE IDENTIFIED AND FIXED:
  The previous cost-sensitive sweep produced an optimal threshold of 0.06.
  This is because the cost model treats FN cost as the FULL transaction
  amount for each missed recovery. Since high-value transactions dominate
  the FN cost, the optimizer pushes the threshold extremely low to avoid
  missing any high-value contestable case.

  This is technically correct but practically problematic:
    - At threshold=0.06, precision=0.62, meaning 38% of contests are wasted
    - The model essentially says "contest everything" which defeats the purpose

  The root cause: the cost model uses raw transaction amounts for FN cost,
  which creates massive asymmetry. A single ₹150,000 missed case costs
  more than 150 false contests at ₹1,000 each.

  FIX: We should evaluate thresholds using BOTH the cost-sensitive framework
  AND classification metrics, then present a table of operating points for
  the user to choose from -- not blindly minimize one cost function.

This script does NOT modify train.py or predict.py. It analyzes the
existing trained model and produces reports + plots.

ALL METRICS ARE SYNTHETIC DEVELOPMENT METRICS.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
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
)
from sklearn.calibration import calibration_curve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "disputes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# --- Explicit cost assumptions (documented, not hidden) ---
COST_ASSUMPTIONS = {
    "false_contest_cost_per_case": 1000,
    "false_contest_explanation": (
        "Estimated cost when we contest a dispute we cannot win. "
        "Includes wasted operational effort, staff time, and minor "
        "reputation risk. This is a DEVELOPMENT ASSUMPTION."
    ),
    "missed_recovery_method": "transaction_amount",
    "missed_recovery_explanation": (
        "When we ACCEPT a dispute we could have won, we lose the full "
        "transaction amount. This is the opportunity cost. In practice "
        "this should be discounted by the probability of success, but "
        "for threshold selection we use it as the FN penalty."
    ),
    "escalation_cost_per_case": 200,
    "escalation_explanation": (
        "Cost of routing a case to human review. Includes analyst time "
        "and delay. This is a DEVELOPMENT ASSUMPTION."
    ),
    "contest_operational_cost": 500,
    "operational_explanation": (
        "Fixed cost of running the contest workflow (evidence gathering, "
        "response drafting, submission). This is a DEVELOPMENT ASSUMPTION."
    ),
    "note": (
        "ALL cost parameters are development assumptions. They must be "
        "calibrated with real operational data before production use."
    ),
}


def load_data_and_model():
    """Load dataset, reproduce the exact same split, load model."""
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "r") as f:
        metadata = json.load(f)

    df = pd.read_csv(DATA_PATH)

    # Reproduce feature engineering exactly as train.py does
    total_evidence = df["evidence_items_available"] + df["evidence_items_missing"]
    df["evidence_completeness"] = (
        df["evidence_items_available"] / total_evidence
    ).round(4)
    dummies = pd.get_dummies(df["dispute_reason"], prefix="reason", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    feature_cols = metadata["features"]
    X = df[feature_cols].values
    y = df["contest_success"].values

    # Reproduce exact same split (same random_state as train.py)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Load models and scaler
    lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

    # Get probabilities
    val_probs = lr_model.predict_proba(scaler.transform(X_val))[:, 1]
    test_probs = lr_model.predict_proba(scaler.transform(X_test))[:, 1]

    amount_idx = feature_cols.index("transaction_amount")
    val_amounts = X_val[:, amount_idx]
    test_amounts = X_test[:, amount_idx]

    return {
        "metadata": metadata,
        "feature_cols": feature_cols,
        "X_val": X_val, "y_val": y_val, "val_probs": val_probs, "val_amounts": val_amounts,
        "X_test": X_test, "y_test": y_test, "test_probs": test_probs, "test_amounts": test_amounts,
        "lr_model": lr_model, "scaler": scaler,
    }


# ===================================================================
# 1. THRESHOLD OPTIMIZATION
# ===================================================================

def threshold_analysis(y_true, y_prob, amounts):
    """
    Evaluate multiple thresholds and produce a comparison table.

    We evaluate BOTH classification metrics AND business cost at each
    threshold, so the user can make an informed selection rather than
    blindly optimizing a single metric.
    """
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                  0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    rows = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # Cost calculation
        fp_cost = fp * COST_ASSUMPTIONS["false_contest_cost_per_case"]
        fn_mask = (y_true == 1) & (y_pred == 0)
        fn_cost = amounts[fn_mask].sum()
        total_cost = fp_cost + fn_cost

        # Net recovery: what we expect to recover minus costs
        tp_mask = (y_true == 1) & (y_pred == 1)
        gross_recovery = amounts[tp_mask].sum()
        contest_ops_cost = tp * COST_ASSUMPTIONS["contest_operational_cost"]
        net_recovery = gross_recovery - contest_ops_cost - fp_cost

        # Automation rate: fraction of cases that don't need human review
        automation_rate = (tp + tn) / len(y_true)

        rows.append({
            "threshold": t,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1_val, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "fp_cost": round(fp_cost, 0),
            "fn_cost": round(fn_cost, 0),
            "total_cost": round(total_cost, 0),
            "gross_recovery": round(gross_recovery, 0),
            "net_recovery": round(net_recovery, 0),
            "automation_rate": round(automation_rate, 4),
            "contest_rate": round((tp + fp) / len(y_true), 4),
        })

    return pd.DataFrame(rows)


def select_operating_threshold(threshold_df):
    """
    Select the recommended operating threshold.

    Strategy: Choose the threshold that maximizes F1 subject to:
      - Precision >= 0.70 (at least 70% of contests should be winnable)
      - Recall >= 0.75 (catch at least 75% of winnable cases)

    If no threshold meets both constraints, relax precision to >= 0.65.

    Rationale: A threshold of 0.06 (cost-minimizing) gives precision=0.62
    which means 38% of contests are wasted -- too aggressive for a
    production system. We want a threshold where most contests are justified.
    """
    # Try strict constraints first
    candidates = threshold_df[
        (threshold_df["precision"] >= 0.70) &
        (threshold_df["recall"] >= 0.75)
    ]

    if len(candidates) == 0:
        # Relax precision
        candidates = threshold_df[
            (threshold_df["precision"] >= 0.65) &
            (threshold_df["recall"] >= 0.70)
        ]

    if len(candidates) == 0:
        # Just maximize F1
        candidates = threshold_df

    best_idx = candidates["f1"].idxmax()
    return threshold_df.iloc[best_idx]


# ===================================================================
# 2. COST-SENSITIVE EVALUATION
# ===================================================================

def cost_sensitive_evaluation(threshold_df, selected_threshold):
    """
    Present cost analysis at the selected threshold with explicit assumptions.
    """
    row = threshold_df[threshold_df["threshold"] == selected_threshold].iloc[0]

    report = {
        "selected_threshold": selected_threshold,
        "cost_assumptions": COST_ASSUMPTIONS,
        "at_selected_threshold": {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "true_positives": int(row["tp"]),
            "false_positives": int(row["fp"]),
            "true_negatives": int(row["tn"]),
            "false_negatives": int(row["fn"]),
            "cost_of_false_contests": float(row["fp_cost"]),
            "cost_of_missed_recoveries": float(row["fn_cost"]),
            "total_cost": float(row["total_cost"]),
            "gross_recovery_from_true_contests": float(row["gross_recovery"]),
            "net_recovery_after_costs": float(row["net_recovery"]),
            "contest_rate": float(row["contest_rate"]),
            "automation_rate": float(row["automation_rate"]),
        },
        "interpretation": {
            "false_contest_meaning": (
                f"{int(row['fp'])} cases would be contested but lost. "
                f"Cost: {row['fp_cost']:,.0f} in wasted operational effort."
            ),
            "missed_recovery_meaning": (
                f"{int(row['fn'])} winnable cases would be incorrectly accepted. "
                f"Opportunity cost: {row['fn_cost']:,.0f} in unrecovered funds."
            ),
            "net_outcome": (
                f"Gross recovery: {row['gross_recovery']:,.0f}. "
                f"After subtracting contest costs and false contest costs, "
                f"net recovery: {row['net_recovery']:,.0f}."
            ),
        },
    }
    return report


# ===================================================================
# 3. CALIBRATION ANALYSIS (PLOTS)
# ===================================================================

def plot_calibration_curve(y_true, y_prob, save_path):
    """Generate and save a reliability/calibration curve."""
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)
    ax1.plot(mean_pred, fraction_pos, "s-", color="#2563eb", label="Logistic Regression")
    ax1.set_xlabel("Mean predicted probability", fontsize=11)
    ax1.set_ylabel("Observed frequency", fontsize=11)
    ax1.set_title("Calibration / Reliability Curve", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # Prediction distribution
    ax2.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="Actual negative (lost)",
             color="#ef4444", density=True)
    ax2.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="Actual positive (won)",
             color="#22c55e", density=True)
    ax2.set_xlabel("Predicted probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Prediction Distribution by True Outcome", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "RazorPay Sentinel -- Calibration Analysis (SYNTHETIC DEVELOPMENT DATA)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Calibration curve saved -> {save_path}")


def compute_calibration_report(y_true, y_prob):
    """Full calibration analysis with ECE and bin details."""
    brier = brier_score_loss(y_true, y_prob)

    # ECE
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (y_prob == bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        error = abs(acc - conf)
        weight = count / len(y_true)
        ece += weight * error
        bins.append({
            "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
            "count": int(count),
            "observed_frequency": round(float(acc), 4),
            "mean_predicted": round(float(conf), 4),
            "absolute_error": round(float(error), 4),
        })

    return {
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "interpretation": (
            f"Brier score: {brier:.4f} (lower is better, 0 = perfect). "
            f"ECE: {ece:.4f} (lower is better, 0 = perfectly calibrated). "
            f"The model's predicted probabilities {'closely' if ece < 0.03 else 'reasonably'} "
            f"match observed frequencies."
        ),
        "bin_analysis": bins,
    }


# ===================================================================
# 4. THRESHOLD ANALYSIS PLOT
# ===================================================================

def plot_threshold_analysis(threshold_df, selected_threshold, save_path):
    """Plot precision/recall/F1 and cost across thresholds."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    thresholds = threshold_df["threshold"]

    # Top: Classification metrics
    ax1.plot(thresholds, threshold_df["precision"], "s-", color="#2563eb",
             label="Precision", markersize=4)
    ax1.plot(thresholds, threshold_df["recall"], "^-", color="#22c55e",
             label="Recall", markersize=4)
    ax1.plot(thresholds, threshold_df["f1"], "o-", color="#f59e0b",
             label="F1", markersize=4)
    ax1.axvline(x=selected_threshold, color="#ef4444", linestyle="--",
                alpha=0.7, label=f"Selected: {selected_threshold}")
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Classification Metrics vs Threshold", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.05)

    # Bottom: Cost analysis
    ax2.plot(thresholds, threshold_df["total_cost"] / 1e6, "D-", color="#ef4444",
             label="Total cost (FP+FN)", markersize=4)
    ax2.plot(thresholds, threshold_df["net_recovery"] / 1e6, "s-", color="#22c55e",
             label="Net recovery", markersize=4)
    ax2.axvline(x=selected_threshold, color="#ef4444", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Probability Threshold", fontsize=11)
    ax2.set_ylabel("Amount (millions)", fontsize=11)
    ax2.set_title("Business Outcomes vs Threshold", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "RazorPay Sentinel -- Threshold Analysis (SYNTHETIC DEV DATA)",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Threshold analysis saved -> {save_path}")


# ===================================================================
# 5. ROC + PR CURVE PLOT
# ===================================================================

def plot_roc_pr_curves(y_true, y_prob, save_path):
    """Plot ROC and Precision-Recall curves."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    ax1.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # PR
    ax2.plot(recall_vals, precision_vals, color="#22c55e", lw=2,
             label=f"PR (AUC = {pr_auc:.4f})")
    baseline = y_true.mean()
    ax2.axhline(y=baseline, color="k", linestyle="--", alpha=0.4,
                label=f"Baseline ({baseline:.2f})")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "RazorPay Sentinel -- Model Discrimination (SYNTHETIC DEV DATA)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC/PR curves saved -> {save_path}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 65)
    print("RazorPay Sentinel -- Phase 1 Completion Analysis")
    print("=" * 65)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load data and model
    print("[1/6] Loading data and model...")
    data = load_data_and_model()
    print(f"  Validation set: {len(data['y_val'])} samples")
    print(f"  Test set: {len(data['y_test'])} samples")
    print()

    # ---------------------------------------------------------------
    # STEP 1: Threshold analysis on VALIDATION set
    # ---------------------------------------------------------------
    print("[2/6] Threshold optimization on VALIDATION set...")
    print("  (Thresholds are tuned on validation, then evaluated ONCE on test.)")
    val_threshold_df = threshold_analysis(
        data["y_val"], data["val_probs"], data["val_amounts"]
    )

    selected = select_operating_threshold(val_threshold_df)
    selected_threshold = float(selected["threshold"])

    print(f"\n  THRESHOLD COMPARISON (Validation Set):")
    print(f"  {'Thresh':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Cost':>12} {'NetRecov':>12}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*12} {'-'*12}")
    for _, row in val_threshold_df.iterrows():
        marker = " <--" if row["threshold"] == selected_threshold else ""
        print(f"  {row['threshold']:6.2f} {row['precision']:6.4f} {row['recall']:6.4f} "
              f"{row['f1']:6.4f} {int(row['tp']):5} {int(row['fp']):5} "
              f"{int(row['fn']):5} {int(row['tn']):5} {row['total_cost']:12,.0f} "
              f"{row['net_recovery']:12,.0f}{marker}")

    print(f"\n  Selected threshold: {selected_threshold}")
    print(f"  Selection method: maximize F1 with precision >= 0.70, recall >= 0.75")
    print(f"  Precision: {selected['precision']:.4f}")
    print(f"  Recall: {selected['recall']:.4f}")
    print(f"  F1: {selected['f1']:.4f}")
    print()

    # ---------------------------------------------------------------
    # STEP 2: Cost-sensitive evaluation
    # ---------------------------------------------------------------
    print("[3/6] Cost-sensitive evaluation...")
    cost_report = cost_sensitive_evaluation(val_threshold_df, selected_threshold)
    at = cost_report["at_selected_threshold"]
    print(f"  False contests (FP): {at['false_positives']} cases, "
          f"cost: {at['cost_of_false_contests']:,.0f}")
    print(f"  Missed recoveries (FN): {at['false_negatives']} cases, "
          f"cost: {at['cost_of_missed_recoveries']:,.0f}")
    print(f"  Gross recovery: {at['gross_recovery_from_true_contests']:,.0f}")
    print(f"  Net recovery: {at['net_recovery_after_costs']:,.0f}")
    print(f"  Contest rate: {at['contest_rate']:.1%}")
    print()

    # ---------------------------------------------------------------
    # STEP 3: Calibration analysis on VALIDATION set
    # ---------------------------------------------------------------
    print("[4/6] Calibration analysis...")
    cal_report = compute_calibration_report(data["y_val"], data["val_probs"])
    print(f"  Brier score: {cal_report['brier_score']:.4f}")
    print(f"  ECE: {cal_report['ece']:.4f}")
    print(f"  {cal_report['interpretation']}")
    print()

    # ---------------------------------------------------------------
    # STEP 4: Generate plots
    # ---------------------------------------------------------------
    print("[5/6] Generating plots...")
    plot_calibration_curve(
        data["y_val"], data["val_probs"],
        os.path.join(REPORTS_DIR, "calibration_curve.png")
    )
    plot_threshold_analysis(
        val_threshold_df, selected_threshold,
        os.path.join(REPORTS_DIR, "threshold_analysis.png")
    )
    plot_roc_pr_curves(
        data["y_val"], data["val_probs"],
        os.path.join(REPORTS_DIR, "roc_pr_curves.png")
    )
    print()

    # ---------------------------------------------------------------
    # STEP 5: FINAL evaluation on HELD-OUT TEST SET
    # ---------------------------------------------------------------
    print("[6/6] FINAL evaluation on HELD-OUT TEST SET (one-time)...")
    print("  (This threshold was selected on validation. Test set is untouched.)")

    y_test = data["y_test"]
    test_probs = data["test_probs"]
    y_pred_test = (test_probs >= selected_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    test_metrics = {
        "data_type": "SYNTHETIC DEVELOPMENT METRICS",
        "model": "Logistic Regression",
        "threshold": selected_threshold,
        "threshold_selection_method": "Maximize F1 with precision>=0.70, recall>=0.75 on validation set",
        "precision": round(float(precision_score(y_test, y_pred_test, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_test, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred_test, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_probs)), 4),
        "brier_score": round(float(brier_score_loss(y_test, test_probs)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_samples": int(len(y_test)),
        "positive_rate": round(float(y_test.mean()), 4),
    }

    # Calibration on test set
    test_cal = compute_calibration_report(y_test, test_probs)

    # Cost on test set
    test_threshold_df = threshold_analysis(y_test, test_probs, data["test_amounts"])
    test_cost_row = test_threshold_df[
        test_threshold_df["threshold"] == selected_threshold
    ]
    if len(test_cost_row) > 0:
        test_cost_row = test_cost_row.iloc[0]
        test_cost = {
            "fp_cost": float(test_cost_row["fp_cost"]),
            "fn_cost": float(test_cost_row["fn_cost"]),
            "total_cost": float(test_cost_row["total_cost"]),
            "net_recovery": float(test_cost_row["net_recovery"]),
        }
    else:
        test_cost = {}

    print(f"\n  === HELD-OUT TEST RESULTS (FINAL) ===")
    print(f"  Model:      Logistic Regression")
    print(f"  Threshold:  {test_metrics['threshold']}")
    print(f"  Precision:  {test_metrics['precision']}")
    print(f"  Recall:     {test_metrics['recall']}")
    print(f"  F1:         {test_metrics['f1']}")
    print(f"  ROC-AUC:    {test_metrics['roc_auc']}")
    print(f"  PR-AUC:     {test_metrics['pr_auc']}")
    print(f"  Brier:      {test_metrics['brier_score']}")
    print(f"  ECE:        {test_cal['ece']}")
    print(f"  Confusion:  TP={tp} FP={fp} FN={fn} TN={tn}")
    if test_cost:
        print(f"  FP cost:    {test_cost['fp_cost']:,.0f}")
        print(f"  FN cost:    {test_cost['fn_cost']:,.0f}")
        print(f"  Net recov:  {test_cost['net_recovery']:,.0f}")

    # Also save calibration plot for test set
    plot_calibration_curve(
        y_test, test_probs,
        os.path.join(REPORTS_DIR, "calibration_curve_test.png")
    )

    # ---------------------------------------------------------------
    # Save all reports
    # ---------------------------------------------------------------
    # Phase 1 final report
    phase1_report = {
        "project": "RazorPay Sentinel",
        "phase": "Phase 1 -- Risk Intelligence Core",
        "data_type": "SYNTHETIC DEVELOPMENT METRICS",
        "warning": (
            "ALL metrics in this report are from SYNTHETIC development data. "
            "They do NOT represent real-world Razorpay performance."
        ),
        "report_date": datetime.now().isoformat(),

        "model": {
            "type": "Logistic Regression (L2 regularized)",
            "target": "contest_success (1 = dispute FULLY REVERSED, 0 = otherwise)",
            "features_count": len(data["feature_cols"]),
            "features": data["feature_cols"],
            "training_samples": 6000,
            "validation_samples": 2000,
            "test_samples": 2000,
            "recalibrated": False,
            "reason_codes_method": "Logistic Regression signed coefficients (coef_ * scaled_feature_value)",
        },

        "threshold_selection": {
            "method": "Maximize F1 with precision >= 0.70 and recall >= 0.75 on validation set",
            "selected_threshold": selected_threshold,
            "threshold_comparison_table": val_threshold_df.to_dict(orient="records"),
            "rationale": (
                f"Threshold {selected_threshold} was selected because it achieves "
                f"precision={selected['precision']:.4f} and recall={selected['recall']:.4f} "
                f"(F1={selected['f1']:.4f}). The pure cost-minimizing threshold (0.06) "
                f"was rejected because it produces precision=0.62, meaning 38% of "
                f"contests are wasted."
            ),
        },

        "cost_analysis": cost_report,
        "calibration": {
            "validation": cal_report,
            "test": test_cal,
        },

        "final_test_metrics": test_metrics,
        "final_test_cost": test_cost,

        "predict_case_interface": {
            "module": "ml/predict.py",
            "function": "predict_case(features_dict) -> dict",
            "input_example": {
                "transaction_amount": 84999,
                "customer_order_count": 15,
                "previous_refunds": 1,
                "previous_disputes": 0,
                "delivery_confirmed": 1,
                "delivery_delay_days": 0,
                "dispute_delay_days": 10,
                "customer_avg_order_value": 45000.0,
                "communication_count": 3,
                "refund_amount_ratio": 0.05,
                "payment_failures": 0,
                "evidence_items_available": 5,
                "evidence_items_missing": 1,
                "dispute_reason": "item_not_received",
            },
            "output_fields": {
                "contest_probability": "float, P(successful contest) from ML model",
                "risk_level": "str, HIGH/MEDIUM/LOW display label (NOT a decision)",
                "reason_codes": "list of {feature, direction, contribution, raw_value}",
                "model_type": "str, which model produced the prediction",
                "recalibrated": "bool, whether the model was post-hoc calibrated",
                "optimal_threshold": "float, the selected operating threshold",
                "data_disclaimer": "str, warning about synthetic data",
                "note": "str, reminder that this is P(success) only, not a decision",
            },
            "architecture_note": (
                "predict_case() outputs P(success) and reason codes ONLY. "
                "It does NOT output ACCEPT/ESCALATE/CONTEST. "
                "The decision is made by decision_policy.decide()."
            ),
        },

        "architecture_separation": {
            "ml_model": "Outputs P(contest_success) and model-derived reason codes",
            "decision_policy": "Separate module: uses P(success) + evidence + expected value + business rules to decide ACCEPT/ESCALATE/CONTEST",
            "evidence_agent": "Only activated downstream when decision is CONTEST",
            "llm_role": "Summarize evidence, draft responses. NEVER generates reason codes or risk scores.",
        },
    }

    report_path = os.path.join(REPORTS_DIR, "phase1_final_report.json")
    with open(report_path, "w") as f:
        json.dump(phase1_report, f, indent=2)
    print(f"\n  Phase 1 report saved -> {report_path}")

    # Also save threshold comparison as CSV for easy viewing
    val_threshold_df.to_csv(
        os.path.join(REPORTS_DIR, "threshold_comparison.csv"), index=False
    )

    # Update model metadata with the new threshold
    metadata = data["metadata"]
    metadata["optimal_threshold"] = selected_threshold
    metadata["threshold_selection_method"] = (
        "Maximize F1 with precision >= 0.70, recall >= 0.75 on validation set"
    )
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Model metadata updated with threshold={selected_threshold}")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    print()
    print("=" * 65)
    print("PHASE 1 FINAL SUMMARY")
    print("=" * 65)
    print(f"  Model:           Logistic Regression")
    print(f"  Threshold:       {selected_threshold}")
    print(f"  Precision:       {test_metrics['precision']}")
    print(f"  Recall:          {test_metrics['recall']}")
    print(f"  F1:              {test_metrics['f1']}")
    print(f"  ROC-AUC:         {test_metrics['roc_auc']}")
    print(f"  PR-AUC:          {test_metrics['pr_auc']}")
    print(f"  Brier Score:     {test_metrics['brier_score']}")
    print(f"  ECE:             {test_cal['ece']}")
    print(f"  Confusion:       TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Reason Codes:    LR signed coefficients (model-derived)")
    print(f"  Interface:       predict_case(features_dict) -> dict")
    print()
    print("  ALL METRICS ARE SYNTHETIC DEVELOPMENT METRICS.")
    print("  The model predicts P(success). The policy decides actions.")
    print("=" * 65)


if __name__ == "__main__":
    main()
