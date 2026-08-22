import uuid
from typing import Optional, List
from sqlalchemy.orm import Session
from backend.app.models.entities import Case
from backend.app.schemas.case import CaseCreate
from backend.app.services.audit_service import AuditService

class CaseService:
    @staticmethod
    def create_case(db: Session, case_in: CaseCreate) -> Case:
        case_id = case_in.case_id or f"CASE-{uuid.uuid4().hex[:8].upper()}"
        avg_order_val = case_in.customer_avg_order_value if case_in.customer_avg_order_value is not None else case_in.transaction_amount
        
        # Determine evidence completeness
        evidence_comp = case_in.evidence_completeness
        if evidence_comp is None:
            total_items = case_in.evidence_items_available + case_in.evidence_items_missing
            if total_items > 0:
                evidence_comp = round(case_in.evidence_items_available / total_items, 4)
            elif case_in.delivery_confirmed:
                evidence_comp = round(4.0 / 6.0, 4)
            else:
                evidence_comp = round(2.0 / 6.0, 4)

        db_case = Case(
            case_id=case_id,
            transaction_amount=case_in.transaction_amount,
            dispute_reason=case_in.dispute_reason,
            delivery_confirmed=case_in.delivery_confirmed,
            customer_order_count=case_in.customer_order_count,
            customer_avg_order_value=avg_order_val,
            previous_refunds=case_in.previous_refunds,
            previous_disputes=case_in.previous_disputes,
            delivery_delay_days=case_in.delivery_delay_days,
            dispute_delay_days=case_in.dispute_delay_days,
            communication_count=case_in.communication_count,
            refund_amount_ratio=case_in.refund_amount_ratio,
            payment_failures=case_in.payment_failures,
            evidence_items_available=case_in.evidence_items_available,
            evidence_items_missing=case_in.evidence_items_missing,
            evidence_completeness=evidence_comp,
            status="CREATED",
        )
        db.add(db_case)
        db.commit()
        db.refresh(db_case)

        AuditService.log_event(
            db=db,
            event_type="CASE_CREATED",
            case_id=case_id,
            metadata_payload={"transaction_amount": db_case.transaction_amount, "dispute_reason": db_case.dispute_reason},
        )
        return db_case

    @staticmethod
    def get_case(db: Session, case_id: str) -> Optional[Case]:
        return db.query(Case).filter(Case.case_id == case_id).first()

    @staticmethod
    def list_cases(db: Session, limit: int = 50, offset: int = 0) -> List[Case]:
        return db.query(Case).order_by(Case.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    def format_case_response(case: Case) -> dict:
        latest_pred = case.predictions[0] if case.predictions else None
        latest_dec = case.decisions[0] if case.decisions else None
        latest_evid = case.evidence_packets[0] if case.evidence_packets else None
        return {
            "case_id": case.case_id,
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
            "status": case.status,
            "created_at": case.created_at,
            "updated_at": case.updated_at,
            "latest_prediction": latest_pred,
            "latest_decision": latest_dec,
            "latest_evidence_packet": latest_evid,
        }
