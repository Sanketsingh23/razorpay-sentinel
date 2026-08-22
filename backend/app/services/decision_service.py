from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.app.models.entities import Case, Decision, Prediction
from backend.app.services.risk_service import RiskService
from backend.app.services.audit_service import AuditService
from backend.app.services.decision_agent import DecisionAgent
from ml.decision_policy import PolicyConfig

class DecisionService:
    @staticmethod
    def run_decision(
        db: Session,
        case: Case,
        config: Optional[PolicyConfig] = None,
        llm_response_override: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        # Get latest prediction or run it if not present
        prediction = case.predictions[0] if case.predictions else None
        if prediction is None:
            prediction = RiskService.run_prediction(db, case)

        try:
            decision_result, agent_meta = DecisionAgent.decide(
                case=case,
                prediction=prediction,
                config=config,
                llm_response_override=llm_response_override,
            )
        except Exception as e:
            AuditService.log_event(
                db=db,
                event_type="DECISION_FAILED",
                case_id=case.case_id,
                metadata_payload={"error": str(e)},
            )
            raise RuntimeError(f"Decision agent evaluation failed: {str(e)}") from e

        decision = Decision(
            case_id=case.case_id,
            action=decision_result.action,
            expected_recovery=decision_result.expected_recovery,
            expected_value=decision_result.net_expected_value,
            evidence_completeness=decision_result.evidence_completeness,
            guardrail_triggered=decision_result.guardrail_triggered,
            reasoning=decision_result.reasoning,
            policy_version=agent_meta.get("policy_version", "v1.0"),
        )
        db.add(decision)
        case.status = "DECIDED"
        db.commit()
        db.refresh(decision)

        AuditService.log_event(
            db=db,
            event_type="DECISION_MADE",
            case_id=case.case_id,
            metadata_payload={
                "action": decision.action,
                "expected_recovery": decision.expected_recovery,
                "expected_value": decision.expected_value,
                "guardrail_triggered": decision.guardrail_triggered,
                "llm_used": agent_meta.get("llm_used", False),
                "fallback_used": agent_meta.get("fallback_used", False),
            },
        )
        return decision
