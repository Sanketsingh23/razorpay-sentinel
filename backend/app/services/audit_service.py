from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.app.models.entities import AuditEvent

class AuditService:
    @staticmethod
    def log_event(
        db: Session,
        event_type: str,
        case_id: Optional[str] = None,
        metadata_payload: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        event = AuditEvent(
            case_id=case_id,
            event_type=event_type,
            metadata_payload=metadata_payload or {},
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    @staticmethod
    def get_events_for_case(db: Session, case_id: str):
        return (
            db.query(AuditEvent)
            .filter(AuditEvent.case_id == case_id)
            .order_by(AuditEvent.event_timestamp.asc())
            .all()
        )
