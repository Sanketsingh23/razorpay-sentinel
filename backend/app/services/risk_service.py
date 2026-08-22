from sqlalchemy.orm import Session
from backend.app.models.entities import Case, Prediction
from backend.app.services.audit_service import AuditService
from ml.predict import predict_case

class RiskService:
    @staticmethod
    def run_prediction(db: Session, case: Case) -> Prediction:
        features = {
            "transaction_amount": case.transaction_amount,
            "dispute_reason": case.dispute_reason,
            "delivery_confirmed": case.delivery_confirmed,
            "customer_order_count": case.customer_order_count,
            "customer_avg_order_value": case.customer_avg_order_value,
            "previous_refunds": case.previous_refunds,
            "previous_disputes": case.previous_disputes,
            "delivery_delay_days": case.delivery_delay_days,
            "dispute_delay_days": case.dispute_delay_days,
            "communication_count": case.communication_count,
            "refund_amount_ratio": case.refund_amount_ratio,
            "payment_failures": case.payment_failures,
            "evidence_items_available": case.evidence_items_available,
            "evidence_items_missing": case.evidence_items_missing,
            "evidence_completeness": case.evidence_completeness,
        }

        try:
            result = predict_case(features)
        except Exception as e:
            AuditService.log_event(
                db=db,
                event_type="PREDICTION_FAILED",
                case_id=case.case_id,
                metadata_payload={"error": str(e)},
            )
            raise RuntimeError(f"Risk engine inference failed: {str(e)}") from e

        prediction = Prediction(
            case_id=case.case_id,
            model_version=result.get("model_type", "Logistic Regression"),
            contest_probability=result["contest_probability"],
            risk_level=result["risk_level"],
            reason_codes=result.get("reason_codes", []),
            positive_factors=result.get("positive_factors", []),
            negative_factors=result.get("negative_factors", []),
        )
        db.add(prediction)
        case.status = "PREDICTED"
        db.commit()
        db.refresh(prediction)

        AuditService.log_event(
            db=db,
            event_type="PREDICTION_GENERATED",
            case_id=case.case_id,
            metadata_payload={
                "contest_probability": prediction.contest_probability,
                "risk_level": prediction.risk_level,
                "model_version": prediction.model_version,
            },
        )
        return prediction
