import json
import logging
from typing import Dict, Any, Optional, Tuple
import httpx
from backend.app.core.config import settings
from backend.app.models.entities import Case, Prediction
from ml.decision_policy import decide, PolicyConfig, Decision as PolicyDecision

logger = logging.getLogger(__name__)

class DecisionAgent:
    @staticmethod
    def gather_context(case: Case, prediction: Prediction) -> Dict[str, Any]:
        """Extract structured context from case and ML prediction."""
        return {
            "case_id": case.case_id,
            "transaction_amount": float(case.transaction_amount),
            "dispute_reason": case.dispute_reason,
            "delivery_confirmed": bool(case.delivery_confirmed),
            "customer_order_count": int(case.customer_order_count),
            "previous_refunds": int(case.previous_refunds),
            "previous_disputes": int(case.previous_disputes),
            "evidence_items_available": int(case.evidence_items_available),
            "evidence_items_missing": int(case.evidence_items_missing),
            "evidence_completeness": float(case.evidence_completeness),
            "contest_probability": float(prediction.contest_probability),
            "risk_level": prediction.risk_level,
            "reason_codes": prediction.reason_codes or [],
            "positive_factors": prediction.positive_factors or [],
            "negative_factors": prediction.negative_factors or [],
            "model_version": prediction.model_version,
        }

    @staticmethod
    def evaluate_deterministic(context: Dict[str, Any], config: Optional[PolicyConfig] = None) -> PolicyDecision:
        """Run authoritative deterministic decision policy and guardrails."""
        return decide(
            probability=context["contest_probability"],
            evidence_completeness=context["evidence_completeness"],
            transaction_amount=context["transaction_amount"],
            config=config,
        )

    @staticmethod
    def _call_llm(context: Dict[str, Any], deterministic_action: str) -> Optional[Dict[str, Any]]:
        """Query LLM for structured reasoning if configured."""
        if not settings.LLM_ENABLED or not settings.LLM_API_KEY:
            return None

        prompt = (
            f"You are the Decision Reasoning Assistant for RazorPay Sentinel.\n"
            f"Evaluate the following case context:\n"
            f"Transaction Amount: INR {context['transaction_amount']:,.2f}\n"
            f"Dispute Reason: {context['dispute_reason']}\n"
            f"Delivery Confirmed: {context['delivery_confirmed']}\n"
            f"Evidence Completeness: {context['evidence_completeness']:.1%}\n"
            f"Contest Probability: {context['contest_probability']:.2f}\n"
            f"Risk Level: {context['risk_level']}\n"
            f"Model Reason Codes: {', '.join(context['reason_codes'])}\n"
            f"Baseline Policy Action: {deterministic_action}\n\n"
            f"Respond STRICTLY with a valid JSON object matching this schema:\n"
            f'{{"recommended_action": "ACCEPT|ESCALATE|CONTEST", "rationale": "short explanation", "concerns": ["item1"]}}'
        )

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.LLM_MODEL}:generateContent?key={settings.LLM_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json"},
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.post(url, json=payload)
                if res.status_code != 200:
                    logger.warning("LLM API returned status %s: %s", res.status_code, res.text)
                    return None
                data = res.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(raw_text)
        except Exception as e:
            logger.warning("LLM reasoning call failed: %s", str(e))
            return None

    @classmethod
    def decide(
        cls,
        case: Case,
        prediction: Prediction,
        config: Optional[PolicyConfig] = None,
        llm_response_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[PolicyDecision, Dict[str, Any]]:
        """
        Orchestrate the Decision Agent evaluation.
        
        Returns:
            Tuple of (PolicyDecision, metadata_dict)
        """
        if config is None:
            config = PolicyConfig()

        context = cls.gather_context(case, prediction)
        det_result = cls.evaluate_deterministic(context, config)

        llm_used = False
        fallback_used = False
        final_action = det_result.action
        reasoning = list(det_result.reasoning)

        # Optional LLM reasoning step with exception containment
        llm_output = None
        if llm_response_override is not None:
            llm_output = llm_response_override
        else:
            try:
                llm_output = cls._call_llm(context, det_result.action)
            except Exception as e:
                logger.warning("LLM call failed with exception: %s", str(e))
                llm_output = None
                fallback_used = True
                reasoning.append(f"[Fallback] LLM reasoning service unavailable ({str(e)}). Used deterministic policy.")

        if llm_output is not None:
            llm_used = True
            rec_action = str(llm_output.get("recommended_action", "")).upper()
            rationale = llm_output.get("rationale")
            concerns = llm_output.get("concerns", [])

            # Hard deterministic guardrail validation
            if det_result.guardrail_triggered and rec_action == "CONTEST":
                # LLM output cannot override policy guardrails.
                fallback_used = True
                reasoning.append(
                    f"[Guardrail Override] LLM proposed CONTEST, but evidence completeness "
                    f"({context['evidence_completeness']:.0%}) < {config.evidence_threshold:.0%}. Enforcing ESCALATE."
                )
                final_action = "ESCALATE"
            elif det_result.action == "ACCEPT" and rec_action == "CONTEST":
                # LLM cannot contest when expected value is non-viable
                fallback_used = True
                reasoning.append(
                    f"[Economics Override] LLM proposed CONTEST, but net expected value "
                    f"(INR {det_result.net_expected_value:,.0f}) < minimum INR {config.min_net_value:,.0f}. Enforcing ACCEPT."
                )
                final_action = "ACCEPT"
            elif rec_action in ["ACCEPT", "ESCALATE", "CONTEST"]:
                final_action = rec_action
                if rationale:
                    reasoning.append(f"Agent reasoning: {rationale}")
                for concern in concerns:
                    reasoning.append(f"Agent concern: {concern}")
            else:
                # Invalid action from LLM -> Safe fallback to deterministic action
                fallback_used = True
                reasoning.append(f"[Fallback] LLM returned invalid action '{rec_action}'. Falling back to deterministic decision.")
                final_action = det_result.action

        decision_obj = PolicyDecision(
            action=final_action,
            probability=det_result.probability,
            evidence_completeness=det_result.evidence_completeness,
            transaction_amount=det_result.transaction_amount,
            expected_recovery=det_result.expected_recovery,
            net_expected_value=det_result.net_expected_value,
            reasoning=reasoning,
            guardrail_triggered=det_result.guardrail_triggered,
            config_used=det_result.config_used,
        )

        agent_meta = {
            "llm_used": llm_used,
            "fallback_used": fallback_used,
            "policy_version": "v1.0",
        }
        return decision_obj, agent_meta
