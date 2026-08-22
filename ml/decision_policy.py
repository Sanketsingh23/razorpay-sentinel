"""
RazorPay Sentinel — Decision Policy
=====================================
Separate module from the ML model. The model outputs P(success) and reason codes.
This module makes the ACCEPT / ESCALATE / CONTEST decision.

Decision logic:
  1. Evidence completeness guardrail (< threshold → ESCALATE regardless)
  2. Expected value calculation: E[recovery] = amount × P(success)
  3. Net value = E[recovery] - operational cost
  4. Three-way decision based on probability zones + evidence + net value

The decision policy is configurable — thresholds are not blindly hardcoded.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


@dataclass
class PolicyConfig:
    """Configurable decision policy parameters."""
    # Probability thresholds (two-threshold system for 3-way decision)
    contest_threshold: float = 0.65        # P(success) above this → likely CONTEST
    accept_threshold: float = 0.30         # P(success) below this → likely ACCEPT
    # Between accept_threshold and contest_threshold → ESCALATE zone

    # Evidence completeness guardrail
    evidence_threshold: float = 0.50       # Evidence below this → ESCALATE regardless

    # Cost parameters (₹)
    contest_operational_cost: float = 500.0
    false_contest_cost: float = 1000.0
    escalation_cost: float = 200.0

    # Minimum net expected value to justify contest
    min_net_value: float = 100.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Decision:
    """Result of the decision policy."""
    action: str                  # ACCEPT | ESCALATE | CONTEST
    probability: float
    evidence_completeness: float
    transaction_amount: float
    expected_recovery: float
    net_expected_value: float
    reasoning: list              # list of strings explaining the decision
    guardrail_triggered: bool    # True if evidence guardrail forced ESCALATE
    config_used: dict

    def to_dict(self):
        return asdict(self)


def load_optimal_threshold() -> float:
    """Load the optimal threshold from model metadata."""
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata.get("optimal_threshold", 0.5)
    except FileNotFoundError:
        return 0.5


def decide(
    probability: float,
    evidence_completeness: float,
    transaction_amount: float,
    config: Optional[PolicyConfig] = None,
) -> Decision:
    """
    Make the ACCEPT / ESCALATE / CONTEST decision.

    This function is SEPARATE from the ML model.
    The model only provides probability and reason codes.
    This policy layer uses probability + evidence + economics to decide.

    Args:
        probability: P(successful contest) from the ML model
        evidence_completeness: ratio of available evidence items (0.0 to 1.0)
        transaction_amount: dispute amount in ₹
        config: PolicyConfig (uses defaults if None)

    Returns:
        Decision object with action, reasoning, and all supporting data
    """
    if config is None:
        config = PolicyConfig()

    reasoning = []

    # Calculate expected value
    expected_recovery = transaction_amount * probability
    net_expected_value = expected_recovery - config.contest_operational_cost

    reasoning.append(
        f"Expected recovery: INR {transaction_amount:,.0f} x {probability:.2f} = INR {expected_recovery:,.0f}"
    )
    reasoning.append(
        f"Net expected value: INR {expected_recovery:,.0f} - INR {config.contest_operational_cost:,.0f} = INR {net_expected_value:,.0f}"
    )

    guardrail_triggered = False

    # --- Guardrail: Evidence completeness ---
    if evidence_completeness < config.evidence_threshold:
        guardrail_triggered = True
        action = "ESCALATE"
        reasoning.append(
            f"[!] Evidence completeness ({evidence_completeness:.0%}) below threshold "
            f"({config.evidence_threshold:.0%}) -> ESCALATE (guardrail)"
        )
        return Decision(
            action=action,
            probability=probability,
            evidence_completeness=evidence_completeness,
            transaction_amount=transaction_amount,
            expected_recovery=round(expected_recovery, 2),
            net_expected_value=round(net_expected_value, 2),
            reasoning=reasoning,
            guardrail_triggered=guardrail_triggered,
            config_used=config.to_dict(),
        )

    # --- Decision zones ---

    # Zone 1: Low probability -> ACCEPT
    if probability < config.accept_threshold:
        action = "ACCEPT"
        reasoning.append(
            f"P(success) = {probability:.2f} < accept threshold {config.accept_threshold:.2f} -> ACCEPT"
        )

    # Zone 2: Negative expected value -> ACCEPT
    elif net_expected_value < config.min_net_value:
        action = "ACCEPT"
        reasoning.append(
            f"Net expected value INR {net_expected_value:,.0f} < minimum INR {config.min_net_value:,.0f} -> ACCEPT"
        )

    # Zone 3: High probability + positive value + sufficient evidence -> CONTEST
    elif probability >= config.contest_threshold and net_expected_value >= config.min_net_value:
        action = "CONTEST"
        reasoning.append(
            f"P(success) = {probability:.2f} >= contest threshold {config.contest_threshold:.2f}"
        )
        reasoning.append(
            f"Evidence = {evidence_completeness:.0%} >= threshold {config.evidence_threshold:.0%}"
        )
        reasoning.append(
            f"Net expected value INR {net_expected_value:,.0f} >= minimum INR {config.min_net_value:,.0f} -> CONTEST"
        )

    # Zone 4: Uncertain middle zone -> ESCALATE
    else:
        action = "ESCALATE"
        reasoning.append(
            f"P(success) = {probability:.2f} is between accept ({config.accept_threshold:.2f}) "
            f"and contest ({config.contest_threshold:.2f}) thresholds -> ESCALATE (uncertain)"
        )

    return Decision(
        action=action,
        probability=probability,
        evidence_completeness=evidence_completeness,
        transaction_amount=transaction_amount,
        expected_recovery=round(expected_recovery, 2),
        net_expected_value=round(net_expected_value, 2),
        reasoning=reasoning,
        guardrail_triggered=guardrail_triggered,
        config_used=config.to_dict(),
    )


def main():
    """Demo: run decision policy on sample cases from the spec."""
    print("=" * 60)
    print("RazorPay Sentinel -- Decision Policy Demo")
    print("=" * 60)
    print()

    config = PolicyConfig()
    print("Policy Configuration:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
    print()

    # --- Case 1: Strong CONTEST ---
    print("-" * 40)
    print("Case 1: High probability, strong evidence, high value")
    d1 = decide(
        probability=0.91,
        evidence_completeness=0.92,
        transaction_amount=85000,
        config=config,
    )
    print(f"  Decision: {d1.action}")
    for r in d1.reasoning:
        print(f"    {r}")
    print()

    # --- Case 2: Clear ACCEPT ---
    print("-" * 40)
    print("Case 2: Low probability, weak economics")
    d2 = decide(
        probability=0.12,
        evidence_completeness=0.50,
        transaction_amount=1200,
        config=config,
    )
    print(f"  Decision: {d2.action}")
    for r in d2.reasoning:
        print(f"    {r}")
    print()

    # --- Case 3: ESCALATE (evidence guardrail) ---
    print("-" * 40)
    print("Case 3: High probability but insufficient evidence")
    d3 = decide(
        probability=0.91,
        evidence_completeness=0.33,
        transaction_amount=50000,
        config=config,
    )
    print(f"  Decision: {d3.action}")
    print(f"  Guardrail triggered: {d3.guardrail_triggered}")
    for r in d3.reasoning:
        print(f"    {r}")
    print()

    # --- Case 4: ESCALATE (uncertain zone) ---
    print("-" * 40)
    print("Case 4: Medium probability, uncertain zone")
    d4 = decide(
        probability=0.50,
        evidence_completeness=0.67,
        transaction_amount=25000,
        config=config,
    )
    print(f"  Decision: {d4.action}")
    for r in d4.reasoning:
        print(f"    {r}")
    print()

    # --- Case 5: ACCEPT (low value despite high probability) ---
    print("-" * 40)
    print("Case 5: High probability but very low transaction value")
    d5 = decide(
        probability=0.80,
        evidence_completeness=0.83,
        transaction_amount=400,
        config=config,
    )
    print(f"  Decision: {d5.action}")
    for r in d5.reasoning:
        print(f"    {r}")
    print()

    # Summary
    print("=" * 40)
    print("Summary of demo cases:")
    for i, d in enumerate([d1, d2, d3, d4, d5], 1):
        print(f"  Case {i}: {d.action} "
              f"(P={d.probability:.2f}, "
              f"Ev={d.evidence_completeness:.0%}, "
              f"INR {d.transaction_amount:,.0f}, "
              f"Net=INR {d.net_expected_value:,.0f})")


if __name__ == "__main__":
    main()
