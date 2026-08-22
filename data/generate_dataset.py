"""
RazorPay Sentinel — Synthetic Development Dataset Generator
============================================================
Generates a realistic synthetic dispute dataset for Phase 1 ML development.

IMPORTANT: All metrics derived from this data must be labelled
"Synthetic Development Metrics" -- NOT production performance.

Target variable: contest_success (binary)
  1 = dispute FULLY REVERSED in merchant's favor (full recovery)
  0 = otherwise (merchant lost, partial recovery, or accepted)

This is a deliberate Phase 1 simplification. Partial recoveries are
treated as losses. The target can be refined in later phases if needed.

Correlations injected (domain-realistic):
  - Delivery confirmed → higher win rate
  - More evidence items → higher win rate
  - Prior dispute pattern → lower win rate
  - Higher refund ratio → lower win rate
  - Item not received + delivery confirmed → high win rate
  - Unauthorized + low evidence → low win rate
  - Noise added to prevent perfect separability
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_SAMPLES = 10_000  # 10k gives more stable experiments; synthetic data is cheap to generate
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "disputes.csv")

DISPUTE_REASONS = [
    "item_not_received",
    "unauthorized",
    "defective",
    "duplicate",
    "other",
]

# Evidence items that can be present/absent (6 total)
EVIDENCE_TYPES = [
    "delivery_proof",
    "order_confirmation",
    "customer_communication",
    "refund_records",
    "product_description",
    "usage_logs",
]


def generate_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic dispute dataset with realistic correlations."""
    rng = np.random.default_rng(seed)

    records = []
    for i in range(n):
        case_id = f"CASE-{i+1:04d}"

        # --- Base features ---
        transaction_amount = float(
            rng.choice(
                [
                    rng.uniform(200, 2000),      # low value (40%)
                    rng.uniform(2000, 20000),     # mid value (35%)
                    rng.uniform(20000, 150000),   # high value (25%)
                ],
                p=[0.40, 0.35, 0.25],
            )
        )
        transaction_amount = round(transaction_amount, 2)

        customer_order_count = int(rng.choice(
            [rng.integers(1, 5), rng.integers(5, 20), rng.integers(20, 100)],
            p=[0.35, 0.45, 0.20],
        ))

        previous_refunds = int(rng.choice(
            [0, rng.integers(1, 4), rng.integers(4, 10)],
            p=[0.50, 0.35, 0.15],
        ))

        previous_disputes = int(rng.choice(
            [0, rng.integers(1, 3), rng.integers(3, 8)],
            p=[0.60, 0.30, 0.10],
        ))

        delivery_confirmed = int(rng.random() < 0.70)  # 70% have delivery confirmed

        delivery_delay_days = int(
            0 if delivery_confirmed and rng.random() < 0.60
            else rng.integers(0, 30)
        )

        dispute_delay_days = int(rng.choice(
            [rng.integers(1, 7), rng.integers(7, 30), rng.integers(30, 90)],
            p=[0.30, 0.50, 0.20],
        ))

        customer_avg_order_value = round(
            transaction_amount * rng.uniform(0.3, 2.5), 2
        )

        communication_count = int(rng.choice(
            [0, rng.integers(1, 5), rng.integers(5, 15)],
            p=[0.25, 0.50, 0.25],
        ))

        # Refund amount ratio — higher for problematic customers
        refund_amount_ratio = round(
            min(1.0, max(0.0, rng.beta(2 + previous_refunds, 10) * (1 + previous_refunds * 0.1))),
            3,
        )

        payment_failures = int(rng.choice(
            [0, rng.integers(1, 3), rng.integers(3, 7)],
            p=[0.65, 0.25, 0.10],
        ))

        dispute_reason = rng.choice(DISPUTE_REASONS, p=[0.35, 0.25, 0.20, 0.10, 0.10])

        # Evidence items — correlated with delivery confirmation and communication
        base_evidence_prob = 0.5
        if delivery_confirmed:
            base_evidence_prob += 0.2
        if communication_count > 3:
            base_evidence_prob += 0.1

        evidence_mask = rng.random(len(EVIDENCE_TYPES)) < base_evidence_prob
        # Delivery proof is more likely if delivery is confirmed
        if delivery_confirmed:
            evidence_mask[0] = rng.random() < 0.90
        else:
            evidence_mask[0] = rng.random() < 0.15

        evidence_items_available = int(evidence_mask.sum())
        evidence_items_missing = len(EVIDENCE_TYPES) - evidence_items_available

        # --- Generate target with realistic correlations ---
        # Build a log-odds score from features
        log_odds = 0.0

        # Delivery confirmed is the strongest predictor
        log_odds += 1.8 if delivery_confirmed else -1.5

        # Evidence availability
        evidence_completeness = evidence_items_available / len(EVIDENCE_TYPES)
        log_odds += 2.0 * (evidence_completeness - 0.5)

        # Prior disputes are negative signal
        log_odds -= 0.4 * previous_disputes

        # Prior refunds are moderate negative signal
        log_odds -= 0.2 * previous_refunds

        # High refund ratio is negative
        log_odds -= 1.5 * refund_amount_ratio

        # Dispute reason effects
        if dispute_reason == "item_not_received" and delivery_confirmed:
            log_odds += 1.2  # Strong case: claimed not received but delivery confirmed
        elif dispute_reason == "unauthorized":
            log_odds -= 0.8  # Harder to contest
        elif dispute_reason == "defective":
            log_odds -= 0.3  # Moderate difficulty
        elif dispute_reason == "duplicate":
            log_odds += 0.5  # Usually easy to prove

        # Communication helps
        log_odds += 0.1 * min(communication_count, 5)

        # Customer history: more orders = more established = slight positive
        log_odds += 0.02 * min(customer_order_count, 20)

        # Late disputes are harder to win
        if dispute_delay_days > 30:
            log_odds -= 0.5

        # High value transactions get more scrutiny (slight negative)
        if transaction_amount > 50000:
            log_odds -= 0.3

        # Payment failures suggest issues
        log_odds -= 0.15 * payment_failures

        # Add noise to prevent perfect separability
        noise = rng.normal(0, 0.8)
        log_odds += noise

        # Convert to probability and sample outcome
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        contest_success = int(rng.random() < prob)

        records.append({
            "case_id": case_id,
            "transaction_amount": transaction_amount,
            "customer_order_count": customer_order_count,
            "previous_refunds": previous_refunds,
            "previous_disputes": previous_disputes,
            "delivery_confirmed": delivery_confirmed,
            "delivery_delay_days": delivery_delay_days,
            "dispute_delay_days": dispute_delay_days,
            "customer_avg_order_value": customer_avg_order_value,
            "communication_count": communication_count,
            "refund_amount_ratio": refund_amount_ratio,
            "payment_failures": payment_failures,
            "evidence_items_available": evidence_items_available,
            "evidence_items_missing": evidence_items_missing,
            "dispute_reason": dispute_reason,
            "contest_success": contest_success,
        })

    df = pd.DataFrame(records)
    return df


def main():
    print("=" * 60)
    print("RazorPay Sentinel — Synthetic Dataset Generator")
    print("=" * 60)

    df = generate_dataset()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nGenerated {len(df)} records -> {OUTPUT_PATH}")
    print(f"\nShape: {df.shape}")
    print(f"\nTarget distribution:")
    print(df["contest_success"].value_counts().to_string())
    print(f"\nSuccess rate: {df['contest_success'].mean():.2%}")
    print(f"\nDispute reason distribution:")
    print(df["dispute_reason"].value_counts().to_string())
    print(f"\nFeature summary:")
    print(df.describe().round(2).to_string())
    print("\n[OK] Dataset generation complete.")


if __name__ == "__main__":
    main()
