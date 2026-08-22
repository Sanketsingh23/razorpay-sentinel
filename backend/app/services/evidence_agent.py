import logging
from typing import Dict, Any, List, Tuple, Optional
from backend.app.models.entities import Case, Prediction, Decision, EvidencePacket

logger = logging.getLogger(__name__)

class EvidenceAgent:
    @staticmethod
    def collect_evidence(case: Case) -> List[Dict[str, Any]]:
        """Extract structured evidence items from factual case fields."""
        items = []

        delivery_confirmed = bool(getattr(case, "delivery_confirmed", False))
        delivery_delay = float(getattr(case, "delivery_delay_days", 0.0) or 0.0)
        tx_amount = float(getattr(case, "transaction_amount", 0.0) or 0.0)
        dispute_reason = str(getattr(case, "dispute_reason", "other") or "other")
        order_count = int(getattr(case, "customer_order_count", 1) or 1)
        prev_disputes = int(getattr(case, "previous_disputes", 0) or 0)
        prev_refunds = int(getattr(case, "previous_refunds", 0) or 0)
        refund_ratio = float(getattr(case, "refund_amount_ratio", 0.0) or 0.0)
        comm_count = int(getattr(case, "communication_count", 0) or 0)
        pay_failures = int(getattr(case, "payment_failures", 0) or 0)

        # 1. Delivery Proof
        if delivery_confirmed:
            if delivery_delay > 60:
                status = "CONFLICTING"
                summary = f"Delivery recorded as confirmed, but fulfillment delay is excessively long ({delivery_delay:.0f} days)."
            else:
                status = "AVAILABLE"
                summary = f"Carrier delivery confirmation on record with {delivery_delay:.0f} days fulfillment delay."
        else:
            status = "MISSING"
            summary = "No carrier delivery confirmation or proof of delivery on record."

        items.append({
            "evidence_id": "EVID-001",
            "type": "delivery_proof",
            "source": "case.delivery_confirmed",
            "status": status,
            "summary": summary,
            "relevance": "HIGH",
        })

        # 2. Order Confirmation
        items.append({
            "evidence_id": "EVID-002",
            "type": "order_confirmation",
            "source": "case.transaction_amount",
            "status": "AVAILABLE",
            "summary": f"Order transaction record for INR {tx_amount:,.2f} under category '{dispute_reason}'.",
            "relevance": "HIGH",
        })

        # 3. Customer Account History
        items.append({
            "evidence_id": "EVID-003",
            "type": "customer_account_history",
            "source": "case.customer_order_count",
            "status": "AVAILABLE",
            "summary": f"Customer account history verified: {order_count} previous orders, {prev_disputes} prior disputes.",
            "relevance": "MEDIUM",
        })

        # 4. Refund Records
        if refund_ratio > 0.8:
            status_refund = "CONFLICTING"
            summary_refund = f"Conflicting refund profile: high refund ratio ({refund_ratio:.1%}) with {prev_refunds} refunds."
        else:
            status_refund = "AVAILABLE"
            summary_refund = f"Merchant refund record: {prev_refunds} prior refunds (refund ratio: {refund_ratio:.1%})."

        items.append({
            "evidence_id": "EVID-004",
            "type": "refund_records",
            "source": "case.previous_refunds",
            "status": status_refund,
            "summary": summary_refund,
            "relevance": "MEDIUM",
        })

        # 5. Customer Communication Logs
        if comm_count > 0:
            status_comm = "AVAILABLE"
            summary_comm = f"{comm_count} customer communication interactions logged prior to dispute."
        else:
            status_comm = "MISSING"
            summary_comm = "No pre-dispute customer communication logs found."

        items.append({
            "evidence_id": "EVID-005",
            "type": "customer_communication",
            "source": "case.communication_count",
            "status": status_comm,
            "summary": summary_comm,
            "relevance": "MEDIUM" if comm_count > 0 else "LOW",
        })

        # 6. Payment Authorization & Settlement
        if pay_failures > 3:
            status_pay = "CONFLICTING"
            summary_pay = f"Conflicting payment history: {pay_failures} payment failure retries on record."
        else:
            status_pay = "AVAILABLE"
            summary_pay = f"Payment settlement verified with {pay_failures} payment failure retries."

        items.append({
            "evidence_id": "EVID-006",
            "type": "payment_verification",
            "source": "case.payment_failures",
            "status": status_pay,
            "summary": summary_pay,
            "relevance": "HIGH",
        })

        return items

    @staticmethod
    def validate_evidence(evidence_items: List[Dict[str, Any]], case: Case) -> Tuple[List[str], List[str]]:
        """Detect missing and conflicting evidence items."""
        missing = [item["type"] for item in evidence_items if item["status"] == "MISSING"]
        conflicts = [item["summary"] for item in evidence_items if item["status"] == "CONFLICTING"]

        # Check for invalid data constraints
        tx_amount = float(getattr(case, "transaction_amount", 0.0) or 0.0)
        if tx_amount <= 0:
            for item in evidence_items:
                if item["evidence_id"] == "EVID-002":
                    item["status"] = "INVALID"
                    item["summary"] = "Transaction amount is non-positive or invalid."

        return missing, conflicts

    @staticmethod
    def calculate_completeness(evidence_items: List[Dict[str, Any]]) -> float:
        """Deterministic calculation of evidence completeness."""
        valid_available = sum(1 for item in evidence_items if item["status"] == "AVAILABLE")
        total_items = len(evidence_items)
        if total_items == 0:
            return 0.0
        return round(valid_available / total_items, 4)

    @staticmethod
    def draft_dispute_response(
        case: Case,
        evidence_items: List[Dict[str, Any]],
        reason_codes: List[str],
    ) -> Dict[str, Any]:
        """Generate a factual, professional dispute response with claims mapped to evidence IDs."""
        claims = []
        tx_amount = float(getattr(case, "transaction_amount", 0.0) or 0.0)
        dispute_reason = str(getattr(case, "dispute_reason", "other") or "other")
        delivery_delay = float(getattr(case, "delivery_delay_days", 0.0) or 0.0)
        order_count = int(getattr(case, "customer_order_count", 1) or 1)
        prev_disputes = int(getattr(case, "previous_disputes", 0) or 0)
        prev_refunds = int(getattr(case, "previous_refunds", 0) or 0)
        refund_ratio = float(getattr(case, "refund_amount_ratio", 0.0) or 0.0)
        comm_count = int(getattr(case, "communication_count", 0) or 0)
        pay_failures = int(getattr(case, "payment_failures", 0) or 0)
        case_id = str(getattr(case, "case_id", "CASE-UNKNOWN"))

        # Map available evidence items into strict factual claims
        for item in evidence_items:
            if item["status"] == "AVAILABLE":
                if item["type"] == "order_confirmation":
                    claims.append({
                        "claim": f"Order transaction of INR {tx_amount:,.2f} under category '{dispute_reason}' is verified and documented.",
                        "source_evidence_id": item["evidence_id"],
                    })
                elif item["type"] == "delivery_proof":
                    claims.append({
                        "claim": f"The order was fulfilled and delivered with carrier confirmation (Fulfillment delay: {delivery_delay:.0f} days).",
                        "source_evidence_id": item["evidence_id"],
                    })
                elif item["type"] == "customer_account_history":
                    claims.append({
                        "claim": f"The customer account shows an established history of {order_count} previous orders and {prev_disputes} prior disputes.",
                        "source_evidence_id": item["evidence_id"],
                    })
                elif item["type"] == "refund_records":
                    claims.append({
                        "claim": f"Merchant records confirm {prev_refunds} prior refunds (refund ratio: {refund_ratio:.1%}).",
                        "source_evidence_id": item["evidence_id"],
                    })
                elif item["type"] == "customer_communication":
                    claims.append({
                        "claim": f"Support logs record {comm_count} customer communication entries prior to dispute initiation.",
                        "source_evidence_id": item["evidence_id"],
                    })
                elif item["type"] == "payment_verification":
                    claims.append({
                        "claim": f"Payment settlement was verified with {pay_failures} payment failure retries.",
                        "source_evidence_id": item["evidence_id"],
                    })

        # Assemble full statement text
        statement_lines = [
            f"DISPUTE CONTEST RESPONSE — CASE {case_id}",
            f"Dispute Category: {dispute_reason} | Disputed Amount: INR {tx_amount:,.2f}",
            "",
            "Factual Justifications:",
        ]
        for i, c in enumerate(claims, 1):
            statement_lines.append(f"{i}. [{c['source_evidence_id']}] {c['claim']}")

        statement_lines.extend([
            "",
            "Conclusion:",
            "Based on the documented evidence items listed above, the merchant respectfully requests reversal of this chargeback.",
        ])

        return {
            "statement": "\n".join(statement_lines),
            "claims": claims,
        }

    @classmethod
    def build_packet(
        cls,
        case: Case,
        prediction: Optional[Prediction] = None,
        decision: Optional[Decision] = None,
    ) -> EvidencePacket:
        """Orchestrate evidence collection, validation, completeness calculation, and response drafting."""
        if decision is not None and decision.action != "CONTEST":
            raise ValueError(f"Evidence workflow cannot be initiated for decision action '{decision.action}'. Only CONTEST decisions qualify.")

        evidence_items = cls.collect_evidence(case)
        missing_evidence, conflicting_evidence = cls.validate_evidence(evidence_items, case)
        completeness = cls.calculate_completeness(evidence_items)

        reason_codes = prediction.reason_codes if prediction else []
        response_draft = cls.draft_dispute_response(case, evidence_items, reason_codes)

        # Determine review status
        if conflicting_evidence:
            packet_status = "CONFLICT_DETECTED"
        elif completeness < 0.50:
            packet_status = "INSUFFICIENT_EVIDENCE"
        else:
            packet_status = "READY_FOR_REVIEW"

        return EvidencePacket(
            case_id=case.case_id,
            status=packet_status,
            evidence_completeness=completeness,
            evidence_items=evidence_items,
            missing_evidence=missing_evidence,
            conflicting_evidence=conflicting_evidence,
            response_draft=response_draft,
            requires_human_review=True,
        )
