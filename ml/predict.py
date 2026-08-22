"""
RazorPay Sentinel -- Prediction Interface
==========================================
Provides predict_case(features_dict) -> {probability, risk_level, reason_codes}

ARCHITECTURE SEPARATION:
  - This module outputs P(success) and model-derived reason codes ONLY.
  - It does NOT decide ACCEPT / ESCALATE / CONTEST.
  - The decision is made by the SEPARATE decision_policy.py module.

Reason codes are derived from:
  - Logistic Regression: signed coefficients x feature values
  - XGBoost: SHAP TreeExplainer values

The LLM does NOT generate reason codes. They come from the model.
"""

import os
import json
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_model():
    """Load the winning model, scaler, and metadata."""
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    model_path = os.path.join(MODELS_DIR, "risk_model.joblib")
    model = joblib.load(model_path)

    # Always load scaler (needed for LR-based models, including calibrated ones)
    scaler = None
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    # Load raw model for reason code extraction (calibrated wrapper doesn't
    # expose coefficients/feature_importances directly)
    raw_model = None
    model_type = metadata.get("model_type", "")
    if model_type == "Logistic Regression":
        raw_path = os.path.join(MODELS_DIR, "lr_model.joblib")
        if os.path.exists(raw_path):
            raw_model = joblib.load(raw_path)
    elif model_type == "XGBoost":
        raw_path = os.path.join(MODELS_DIR, "xgb_model.joblib")
        if os.path.exists(raw_path):
            raw_model = joblib.load(raw_path)

    return model, scaler, metadata, raw_model


def _map_feature_to_reason(feature_name: str, raw_value: float, direction: str) -> tuple:
    """Map feature value and direction to a standardized reason code and human-readable text."""
    sign = "+" if direction == "positive" else "-"

    if feature_name == "delivery_confirmed":
        if raw_value >= 1:
            code = "delivery_confirmed"
            display_text = "+ Delivery confirmed"
        else:
            code = "delivery_not_confirmed"
            display_text = "- Delivery not confirmed"

    elif feature_name == "evidence_completeness":
        if direction == "positive":
            code = "strong_evidence"
            display_text = f"+ Strong evidence availability ({raw_value*100:.0f}% complete)"
        else:
            code = "insufficient_evidence"
            display_text = f"- Insufficient evidence availability ({raw_value*100:.0f}% complete)"

    elif feature_name == "evidence_items_available":
        if direction == "positive":
            code = "high_evidence_count"
            display_text = f"+ Strong evidence availability ({int(raw_value)} items available)"
        else:
            code = "low_evidence_count"
            display_text = f"- Minimal evidence items ({int(raw_value)} available)"

    elif feature_name == "evidence_items_missing":
        if direction == "positive":
            code = "minimal_missing_evidence"
            display_text = f"+ Minimal missing evidence ({int(raw_value)} missing)"
        else:
            code = "multiple_missing_evidence"
            display_text = f"- Multiple missing evidence items ({int(raw_value)} missing)"

    elif feature_name == "previous_disputes":
        if raw_value == 0:
            code = "no_previous_disputes"
            display_text = "+ No previous dispute history"
        else:
            code = "prior_dispute_history"
            display_text = f"- Prior dispute history ({int(raw_value)} previous disputes)"

    elif feature_name == "previous_refunds":
        if raw_value == 0:
            code = "no_previous_refunds"
            display_text = "+ No previous refund history"
        elif direction == "negative":
            code = "high_previous_refund_activity"
            display_text = f"- Previous refund history ({int(raw_value)} prior refunds)"
        else:
            code = "low_previous_refunds"
            display_text = f"+ Low previous refund activity ({int(raw_value)} prior refund)"

    elif feature_name == "refund_amount_ratio":
        if direction == "positive":
            code = "low_refund_ratio"
            display_text = f"+ Low customer refund ratio ({raw_value*100:.1f}%)"
        else:
            code = "high_refund_ratio"
            display_text = f"- High customer refund ratio ({raw_value*100:.1f}% of order value)"

    elif feature_name == "reason_item_not_received":
        code = "dispute_reason_item_not_received"
        display_text = "+ Dispute reason: Item Not Received (winnable with delivery proof)"

    elif feature_name == "reason_duplicate":
        code = "dispute_reason_duplicate"
        display_text = "+ Dispute reason: Duplicate charge claim"

    elif feature_name == "reason_unauthorized":
        code = "dispute_reason_unauthorized"
        display_text = "- Dispute reason: Unauthorized transaction (higher burden of proof)"

    elif feature_name == "reason_defective":
        code = "dispute_reason_defective"
        display_text = "- Dispute reason: Defective / quality dispute"

    elif feature_name == "customer_order_count":
        if direction == "positive":
            code = "established_customer"
            display_text = f"+ Established customer ({int(raw_value)} previous orders)"
        else:
            code = "minimal_customer_history"
            display_text = f"- Minimal customer order history ({int(raw_value)} orders)"

    elif feature_name == "communication_count":
        if direction == "positive":
            code = "active_communication"
            display_text = f"+ Active customer communication ({int(raw_value)} logs)"
        else:
            code = "minimal_communication"
            display_text = f"- Minimal customer communication ({int(raw_value)} logs)"

    elif feature_name == "payment_failures":
        if raw_value == 0:
            code = "clean_payment_history"
            display_text = "+ Clean payment history (0 payment failures)"
        else:
            code = "prior_payment_failures"
            display_text = f"- Prior payment failures detected ({int(raw_value)})"

    elif feature_name == "transaction_amount":
        if direction == "negative":
            code = "high_transaction_amount_scrutiny"
            display_text = f"- High-value transaction (INR {raw_value:,.0f}) requires rigorous evidence"
        else:
            code = "standard_transaction_amount"
            display_text = f"+ Standard transaction value (INR {raw_value:,.0f})"

    elif feature_name == "dispute_delay_days":
        if direction == "positive":
            code = "prompt_dispute_filing"
            display_text = f"+ Prompt dispute filing ({int(raw_value)} days)"
        else:
            code = "delayed_dispute_filing"
            display_text = f"- Delayed dispute filing ({int(raw_value)} days)"

    elif feature_name == "delivery_delay_days":
        if direction == "positive":
            code = "on_time_delivery"
            display_text = f"+ On-time delivery ({int(raw_value)} delay days)"
        else:
            code = "delivery_delay"
            display_text = f"- Delivery delay recorded ({int(raw_value)} days)"

    else:
        code = feature_name
        display_text = f"{sign} {feature_name}: {raw_value}"

    return code, display_text


def _get_reason_codes_lr(raw_model, scaler, feature_values, feature_names, top_n=5):
    """
    Extract model-derived reason codes from Logistic Regression using signed coefficients.

    contribution = coefficient x scaled_feature_value
    """
    scaled = scaler.transform([feature_values])[0]
    coefficients = raw_model.coef_[0]

    contributions = []
    for i, (fname, coeff, scaled_val) in enumerate(
        zip(feature_names, coefficients, scaled)
    ):
        raw_val = float(feature_values[i])
        # Skip dispute_reason dummy features that are 0 (not selected)
        if fname.startswith("reason_") and raw_val == 0:
            continue

        contribution = coeff * scaled_val
        direction = "positive" if contribution > 0 else "negative"
        code, display_text = _map_feature_to_reason(fname, raw_val, direction)

        contributions.append({
            "code": code,
            "feature": fname,
            "direction": direction,
            "contribution": round(float(abs(contribution)), 4),
            "signed_contribution": round(float(contribution), 4),
            "raw_value": round(raw_val, 4),
            "display_text": display_text,
        })

    # Sort by absolute contribution magnitude
    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    top_items = contributions[:top_n]

    reason_codes = [item["code"] for item in top_items]
    positive_factors = [
        item["display_text"] for item in contributions if item["direction"] == "positive"
    ][:top_n]
    negative_factors = [
        item["display_text"] for item in contributions if item["direction"] == "negative"
    ][:top_n]

    return reason_codes, positive_factors, negative_factors, top_items


def _get_reason_codes_xgb(raw_model, feature_values, feature_names, top_n=5):
    """Extract reason codes from XGBoost using SHAP TreeExplainer."""
    try:
        import shap
        explainer = shap.TreeExplainer(raw_model)
        shap_values = explainer.shap_values(np.array([feature_values]))

        contributions = []
        for i, (fname, sv) in enumerate(zip(feature_names, shap_values[0])):
            raw_val = float(feature_values[i])
            if fname.startswith("reason_") and raw_val == 0:
                continue
            direction = "positive" if sv > 0 else "negative"
            code, display_text = _map_feature_to_reason(fname, raw_val, direction)
            contributions.append({
                "code": code,
                "feature": fname,
                "direction": direction,
                "contribution": round(float(abs(sv)), 4),
                "signed_contribution": round(float(sv), 4),
                "raw_value": round(raw_val, 4),
                "display_text": display_text,
            })

        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        top_items = contributions[:top_n]
        reason_codes = [item["code"] for item in top_items]
        positive_factors = [
            item["display_text"] for item in contributions if item["direction"] == "positive"
        ][:top_n]
        negative_factors = [
            item["display_text"] for item in contributions if item["direction"] == "negative"
        ][:top_n]

        return reason_codes, positive_factors, negative_factors, top_items

    except ImportError:
        importances = raw_model.feature_importances_
        contributions = []
        for i, (fname, imp) in enumerate(zip(feature_names, importances)):
            raw_val = float(feature_values[i])
            if fname.startswith("reason_") and raw_val == 0:
                continue
            code, display_text = _map_feature_to_reason(fname, raw_val, "positive")
            contributions.append({
                "code": code,
                "feature": fname,
                "direction": "positive",
                "contribution": round(float(imp), 4),
                "signed_contribution": round(float(imp), 4),
                "raw_value": round(raw_val, 4),
                "display_text": display_text,
            })
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        top_items = contributions[:top_n]
        reason_codes = [item["code"] for item in top_items]
        positive_factors = [item["display_text"] for item in top_items]
        negative_factors = []
        return reason_codes, positive_factors, negative_factors, top_items


def _classify_risk_level(probability: float) -> str:
    """
    Derive risk level display label from contest probability.

    Higher P(success) = HIGH likelihood / priority to contest.
    This is a display label, NOT an action decision.
    """
    if probability >= 0.75:
        return "HIGH"
    elif probability >= 0.45:
        return "MEDIUM"
    else:
        return "LOW"


def predict_case(features_dict: dict, top_n_reasons: int = 5) -> dict:
    """
    Predict contest success probability and generate model-derived reason codes.

    This function outputs P(success) and reason codes ONLY.
    It does NOT output ACCEPT / ESCALATE / CONTEST.
    Use decision_policy.decide() for action decisions.

    Args:
        features_dict: dict containing dispute/transaction features.
                       Accepts booleans, integers, floats, and strings.
        top_n_reasons: number of top reason codes to return (default: 5)

    Returns:
        {
            "contest_probability": float (e.g. 0.91),
            "risk_level": "HIGH" | "MEDIUM" | "LOW",
            "reason_codes": list[str],
            "positive_factors": list[str],
            "negative_factors": list[str],
            "reason_details": list[dict],
            "model_type": str,
            "recalibrated": bool,
            "optimal_threshold": float,
            "data_disclaimer": str,
            "note": str,
        }
    """
    model, scaler, metadata, raw_model = load_model()
    feature_names = metadata["features"]
    model_type = metadata["model_type"]
    recalibrated = metadata.get("recalibrated", False)

    # Normalize input dictionary (convert booleans/strings to floats where applicable)
    normalized_input = {}
    for k, v in features_dict.items():
        if isinstance(v, bool):
            normalized_input[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            normalized_input[k] = float(v)
        else:
            normalized_input[k] = v

    # Extract dispute reason (default to "item_not_received" if delivery_confirmed else "other")
    dispute_reason = str(normalized_input.get("dispute_reason", "item_not_received")).lower()

    # Evidence completeness calculation
    if "evidence_completeness" not in normalized_input:
        if "evidence_items_available" in normalized_input and "evidence_items_missing" in normalized_input:
            avail = normalized_input["evidence_items_available"]
            missing = normalized_input["evidence_items_missing"]
            total = avail + missing
            normalized_input["evidence_completeness"] = avail / total if total > 0 else 0.5
        elif normalized_input.get("delivery_confirmed", 0.0) >= 1.0:
            # Default strong evidence profile when delivery confirmed
            normalized_input.setdefault("evidence_items_available", 4.0)
            normalized_input.setdefault("evidence_items_missing", 2.0)
            normalized_input["evidence_completeness"] = 4.0 / 6.0
        else:
            normalized_input.setdefault("evidence_items_available", 2.0)
            normalized_input.setdefault("evidence_items_missing", 4.0)
            normalized_input["evidence_completeness"] = 2.0 / 6.0

    # Default order and transaction metrics if missing
    tx_amount = normalized_input.get("transaction_amount", 5000.0)
    normalized_input.setdefault("transaction_amount", tx_amount)
    normalized_input.setdefault("customer_order_count", 5.0)
    normalized_input.setdefault("customer_avg_order_value", tx_amount)
    normalized_input.setdefault("previous_refunds", 0.0)
    normalized_input.setdefault("previous_disputes", 0.0)
    normalized_input.setdefault("delivery_confirmed", 0.0)
    normalized_input.setdefault("delivery_delay_days", 0.0)
    normalized_input.setdefault("dispute_delay_days", 10.0)
    normalized_input.setdefault("communication_count", 2.0)
    normalized_input.setdefault("refund_amount_ratio", 0.05)
    normalized_input.setdefault("payment_failures", 0.0)

    feature_values = []
    for fname in feature_names:
        if fname.startswith("reason_"):
            reason_name = fname.replace("reason_", "")
            feature_values.append(1.0 if dispute_reason == reason_name else 0.0)
        elif fname in normalized_input:
            feature_values.append(float(normalized_input[fname]))
        else:
            feature_values.append(0.0)

    feature_values = np.array(feature_values)

    # Model inference
    if model_type == "Logistic Regression":
        prob = model.predict_proba(scaler.transform([feature_values]))[0][1]
    else:
        prob = model.predict_proba([feature_values])[0][1]

    probability = round(float(prob), 4)
    risk_level = _classify_risk_level(probability)

    # Reason code extraction from raw model
    if raw_model is None:
        raw_model = model

    if model_type == "Logistic Regression":
        reason_codes, pos_factors, neg_factors, details = _get_reason_codes_lr(
            raw_model, scaler, feature_values, feature_names, top_n_reasons
        )
    else:
        reason_codes, pos_factors, neg_factors, details = _get_reason_codes_xgb(
            raw_model, feature_values, feature_names, top_n_reasons
        )

    return {
        "contest_probability": probability,
        "risk_level": risk_level,
        "reason_codes": reason_codes,
        "positive_factors": pos_factors,
        "negative_factors": neg_factors,
        "reason_details": details,
        "model_type": model_type,
        "recalibrated": recalibrated,
        "optimal_threshold": metadata.get("optimal_threshold", 0.4),
        "data_disclaimer": "Prediction based on model trained with SYNTHETIC DEVELOPMENT DATA.",
        "note": "This is P(success) only. Use decision_policy.decide() for ACCEPT/ESCALATE/CONTEST.",
    }


def main():
    """Demo: predict sample cases and verify model-derived explanations."""
    print("=" * 65)
    print("RazorPay Sentinel -- Prediction Interface Demo")
    print("=" * 65)
    print("NOTE: predict_case() outputs P(success) + model-derived reason codes.")
    print("      The ACCEPT/ESCALATE/CONTEST decision is made by decision_policy.py.")
    print()

    # Case 1: Minimal input as in specification
    spec_input = {
        "transaction_amount": 84999,
        "delivery_confirmed": True,
        "previous_disputes": 0,
        "previous_refunds": 1,
    }

    print("--- Case 1: Spec Minimal Input ---")
    print(f"Input: {json.dumps(spec_input, indent=2)}")
    res1 = predict_case(spec_input)
    print("\nResult:")
    print(f"  Contest probability: {res1['contest_probability']:.2%}")
    print(f"  Risk level:          {res1['risk_level']}")
    print(f"  Reason codes:        {res1['reason_codes']}")
    print("\n  Positive factors:")
    for pf in res1["positive_factors"]:
        print(f"    {pf}")
    print("  Negative factors:")
    for nf in res1["negative_factors"]:
        print(f"    {nf}")

    # Case 2: Full feature case (weak case)
    print("\n" + "-" * 50)
    weak_case = {
        "transaction_amount": 1200.0,
        "customer_order_count": 2,
        "previous_refunds": 3,
        "previous_disputes": 2,
        "delivery_confirmed": False,
        "delivery_delay_days": 15,
        "dispute_delay_days": 45,
        "customer_avg_order_value": 800.0,
        "communication_count": 0,
        "refund_amount_ratio": 0.45,
        "payment_failures": 2,
        "evidence_items_available": 1,
        "evidence_items_missing": 5,
        "dispute_reason": "unauthorized",
    }
    print("--- Case 2: Weak Case (Unauthorized, No Delivery Proof) ---")
    res2 = predict_case(weak_case)
    print(f"  Contest probability: {res2['contest_probability']:.2%}")
    print(f"  Risk level:          {res2['risk_level']}")
    print(f"  Reason codes:        {res2['reason_codes']}")
    print("\n  Positive factors:")
    for pf in res2["positive_factors"]:
        print(f"    {pf}")
    print("  Negative factors:")
    for nf in res2["negative_factors"]:
        print(f"    {nf}")


if __name__ == "__main__":
    main()
